
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from utils import get_audio_mel_spectrogram

import librosa
import pandas as pd
import numpy as np
import os

import logging


from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass(frozen=True)
class AudioSample:
    raw_audio: np.ndarray
    mel_audio: np.ndarray
    sample_rate: int
    file_path: str
    raw_text: str
    tokenized_text: Optional[np.ndarray] = None
    
class LJSpeechDataset(Dataset):
    def __init__(self, data_dir: str):
        # Initialize dataset, e.g., load file paths and labels
        self.data_dir = Path(data_dir)
        self.file_dir = self.data_dir / 'wavs'

        if not self.file_dir.is_dir():
            raise FileNotFoundError(f"Audio files directory 'wavs' not found in {data_dir}")
        
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
        
        self.audio2mel = get_audio_mel_spectrogram

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.file_paths)

    def __getitem__(self, idx) -> AudioSample:
        # Load and return a sample from the dataset at the given index
        file_path = self.file_paths[idx]
        audio, sr = librosa.load(file_path, sr=None)  # Load audio file with original sampling rate

        mel_audio: np.ndarray = self.audio2mel(audio, sr)

        raw_text = self.labels[file_path.stem]  # Handle missing labels gracefully

        try:
            raw_text = raw_text.upper()  # Get the raw text label for the audio file, convert to uppercase to match tokenizer vocab
            tokenized_text = tokenizer.encode(raw_text)
        except AttributeError:
            logger.warning(f"Missing label for file: {file_path.name}. Using empty string as label.")
            return None  # Skip this sample if label is missing
    
        sample = AudioSample(
            raw_audio=audio,
            mel_audio=torch.tensor(mel_audio, dtype=torch.float32),
            sample_rate=sr,
            file_path=file_path,
            raw_text=raw_text,
            tokenized_text=torch.tensor(tokenized_text, dtype=torch.long)
        )
        return sample


def collate_fn(batch) -> Optional[dict]:
    # Custom collate function to handle batching of variable-length audio samples
    batch = [sample for sample in batch if sample is not None]  # Filter out None samples
    if len(batch) == 0:
        return None  # Return None if all samples were invalid
    
    audios = [sample.mel_audio for sample in batch]

    tokenized_texts = [sample.tokenized_text for sample in batch]

    # Sort by descending spectrogram length (required for pack_padded_sequence)
    batch = sorted(zip(audios, tokenized_texts), key=lambda x: x[0].shape[1], reverse=True)
    audios, tokenized_texts = zip(*batch)

    # --- Spectrograms ---
    input_lengths = torch.tensor([a.shape[1] for a in audios], dtype=torch.long)  # time_frames per sample

    # Pad to (batch, n_mels, max_time_frames), then permute to (max_time_frames, batch, n_mels)
    # pack_padded_sequence expects (time, batch, features)
    padded_spectrograms = torch.nn.utils.rnn.pad_sequence(
        [a.transpose(0, 1) for a in audios],  # each becomes (time_frames, n_mels)
        batch_first=True                       # output: (batch, max_time_frames, n_mels)
    )

    # Add dimension for channel
    padded_spectrograms = padded_spectrograms.unsqueeze(1)  # (batch, 1, max_time_frames, n_mels)

    # Convert to (batch, channel, n_mels, max_time_frames) for CNN input
    # Permute is basically switching order of dimensions
    padded_spectrograms = padded_spectrograms.permute(0, 1, 3, 2)  # (batch, 1, n_mels, max_time_frames)

    # --- Transcripts ---
    # CTC loss expects transcripts concatenated (not padded) into a single 1D tensor
    target_lengths = torch.tensor([len(t) for t in tokenized_texts], dtype=torch.long)
    packed_transcripts = torch.cat([t.clone().detach().long() for t in tokenized_texts])

    batch = {
        'padded_spectrograms': padded_spectrograms,   # Required for CNN input: (batch, channel, n_mels, max_time_frames)
        'input_lengths': input_lengths,  # Time frames per sample (before padding), Required for
        'packed_transcripts': packed_transcripts,  # Concatenated tokenized transcripts for CTC loss
        'target_lengths': target_lengths  # Lengths of each transcript (before packing), Required for CTC loss
    }

    return batch
