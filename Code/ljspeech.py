import csv
import logging
import os
import random
from pathlib import Path

import torchaudio
import numpy as np
import pandas as pd
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset
from config import TOKENIZER, SAMPLE_RATE

from utils import AudioSample, get_audio_mel_spectrogram, spec_augment

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

    def __init__(self, data_dir: str, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.file_dir = self.data_dir / 'wavs'
        self.augment  = augment

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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx) -> AudioSample:
        file_path = self.file_paths[idx]
        audio, sr = torchaudio.load(file_path)

        # Spec Augment suggestions for preventing overfitting on smaller datasets
        # Essentially a time-distortion (speed-up/slow-down)
        # SpecAugment paper: https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf
        if self.augment:
            rate = random.choice([0.9, 1.0, 1.1])
            if rate != 1.0:
                audio = AF.resample(audio, orig_freq=int(sr * rate), new_freq=sr)

        raw_mel_audio = get_audio_mel_spectrogram(audio, sr)
        mel_spec_augment_audio = spec_augment(raw_mel_audio) if self.augment else None

        raw_text = self.labels.get(file_path.stem)
        try:
            raw_text = raw_text.upper()
            tokenized_text = TOKENIZER.encode(raw_text)
        except AttributeError:
            logger.warning(f"Missing label for file: {file_path.name}. Skipping sample.")
            return None

        return AudioSample(
            raw_audio=audio,
            raw_mel_audio=raw_mel_audio,
            mel_audio_spec_augment=mel_spec_augment_audio,
            sample_rate=sr,
            file_path=str(file_path),
            raw_text=raw_text,
            tokenized_text=torch.tensor(tokenized_text, dtype=torch.long),
        )
    
def load_ljspeech(data_dir: str, augment: bool = True) -> LJSpeechDataset:
    """Helper function to load LJSpeech dataset."""
    return LJSpeechDataset(data_dir, augment=augment)
