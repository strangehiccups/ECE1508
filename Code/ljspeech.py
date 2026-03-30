import csv
import logging
import os
import random
from pathlib import Path
from typing import Optional

import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import pandas as pd
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset
from config import TOKENIZER, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class LJSpeechDataset(Dataset):
    """LJSpeech-1.1 dataset.

    Expected layout::

        data_dir/
            wavs/           # .wav audio files
            metadata.csv    # pipe-separated: file_id|raw_text|normalised_text
    """

    def __init__(self, data_dir: str, augment: bool = False, cache_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.file_dir = self.data_dir / 'wavs'
        self.augment  = augment
        # Cache is disabled when augment=True — random time-stretching makes samples non-deterministic.
        self._cache_dir: Optional[Path] = None
        if cache_dir is not None and not augment:
            self._cache_dir = Path(cache_dir).resolve()
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.file_dir.is_dir():
            raise FileNotFoundError(f"Audio files directory 'wavs' not found in {data_dir}")

        logger.info(f"Loading LJSpeech dataset from {self.file_dir}...")

        self.file_paths = sorted([
            self.file_dir / f
            for f in os.listdir(self.file_dir)
            if f.endswith('.wav')
        ])

        label_file = self.data_dir / 'metadata.csv'
        if not label_file.exists():
            raise FileNotFoundError(f"Label file 'metadata.csv' not found in {data_dir}")

        labels_df = pd.read_csv(
            label_file, header=None, sep='|',
            encoding='utf-8', quoting=csv.QUOTE_NONE, engine='python'
        )
        self.labels = {row[0]: row[2] for _, row in labels_df.iterrows()}

        # Build the mel pipeline once — reused for every sample.
        self._to_mel = nn.Sequential(
            T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0),
            T.AmplitudeToDB(top_db=80.0),
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Cache hit — skip all audio I/O and spectrogram computation.
        if self._cache_dir is not None:
            cache_path = self._cache_dir / f"{idx}.pt"
            if cache_path.exists():
                try:
                    saved = torch.load(cache_path, weights_only=True)
                    return saved['mel'], saved['tokens']
                except Exception:
                    cache_path.unlink(missing_ok=True)  # corrupt file — recompute below

        file_path = self.file_paths[idx]
        audio, sr = torchaudio.load(file_path)

        # Optional time-stretching augmentation (speed-up/slow-down) on raw audio.
        if self.augment:
            rate = random.choice([0.9, 1.0, 1.1])
            if rate != 1.0:
                audio = AF.resample(audio, orig_freq=int(sr * rate), new_freq=sr)

        if audio.dim() > 1:
            audio = audio.mean(dim=0)
        if sr != SAMPLE_RATE:
            audio = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(audio)
        mel = self._to_mel(audio)

        raw_text = self.labels.get(file_path.stem)
        try:
            raw_text = raw_text.upper()
            tokenized_text = TOKENIZER.encode(raw_text)
        except AttributeError:
            logger.warning(f"Missing label for file: {file_path.name}. Skipping sample.")
            return None

        tokens = torch.tensor(tokenized_text, dtype=torch.long)

        if self._cache_dir is not None:
            torch.save({'mel': mel, 'tokens': tokens}, cache_path)

        return mel, tokens
    
def load_ljspeech(data_dir: str, augment: bool = False, cache_dir: Optional[str] = None) -> LJSpeechDataset:
    """Helper function to load LJSpeech dataset."""
    return LJSpeechDataset(data_dir, augment=augment, cache_dir=cache_dir)
