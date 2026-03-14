import csv
import logging
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2CTCTokenizer

from utils import AudioSample, get_audio_mel_spectrogram

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

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

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.file_dir = self.data_dir / 'wavs'

        if not self.file_dir.is_dir():
            raise FileNotFoundError(f"Audio files directory 'wavs' not found in {data_dir}")

        logger.info(f"Loading LJSpeech dataset from {self.file_dir}...")

        self.file_paths = [
            self.file_dir / f
            for f in os.listdir(self.file_dir)
            if f.endswith('.wav')
        ]

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
        audio, sr = librosa.load(file_path, sr=None)
        mel_audio = get_audio_mel_spectrogram(audio, sr)

        raw_text = self.labels.get(file_path.stem)
        try:
            raw_text = raw_text.upper()
            tokenized_text = tokenizer.encode(raw_text)
        except AttributeError:
            logger.warning(f"Missing label for file: {file_path.name}. Skipping sample.")
            return None

        return AudioSample(
            raw_audio=audio,
            mel_audio=torch.tensor(mel_audio, dtype=torch.float32),
            sample_rate=sr,
            file_path=str(file_path),
            raw_text=raw_text,
            tokenized_text=torch.tensor(tokenized_text, dtype=torch.long),
        )
    
def load_ljspeech(data_dir: str) -> LJSpeechDataset:
    """Helper function to load LJSpeech dataset."""
    return LJSpeechDataset(data_dir)
