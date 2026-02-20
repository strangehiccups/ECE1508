
from dataclasses import dataclass
from torch.utils.data import Dataset
from pathlib import Path

import librosa
import pandas as pd
import numpy as np
import os

import logging


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass(frozen=True)
class AudioSample:
    audio: np.ndarray
    sample_rate: int
    file_path: str
    text: str
    
class LJSpeechDataset(Dataset):
    def __init__(self, data_dir: Path):
        # Initialize dataset, e.g., load file paths and labels
        self.data_dir = data_dir
        self.file_dir = data_dir / 'wavs/'

        if not self.file_dir.exists():
            raise FileNotFoundError(f"Audio files directory 'wavs/' not found in {data_dir}")
        
        self.file_paths = []  # List to store file paths
        self.labels = {}      # Dictionary to store labels (if applicable)
        # Load file paths and labels from the dataset directory
        logger.info(f"Loading dataset from {self.file_dir}...")
        
        for file_name in os.listdir(self.file_dir):
            if file_name.endswith('.wav'):
                self.file_paths.append(self.file_dir / file_name)

        label_file = self.data_dir / 'metadata.csv'
        if label_file.exists():
            labels_df = pd.read_csv(label_file, header=None, sep='|', encoding='utf-8', )
            for _, row in labels_df.iterrows():
                file_name = row[0]  # The first column contains the file name
                label = row[2]      # The third column contains the "normalized" text
                self.labels[file_name] = label
        else:
            raise FileNotFoundError(f"Label file 'metadata.csv' not found in {data_dir}")

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.file_paths)

    def __getitem__(self, idx) -> AudioSample:
        # Load and return a sample from the dataset at the given index
        file_path = self.file_paths[idx]
        audio, sr = librosa.load(file_path, sr=None)  # Load audio file with original sampling rate
        sample = AudioSample(
            audio=audio,
            sample_rate=sr,
            file_path=file_path,
            text=self.labels[file_path.stem]
        )
        return sample