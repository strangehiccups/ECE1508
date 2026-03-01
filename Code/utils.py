import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import transformers
from transformers import Wav2Vec2CTCTokenizer
from tqdm.notebook import tqdm

HOP_LENGTH = 256
N_FFT = 512
N_MELS = 80
SAMPLE_RATE = 22050

def get_audio_duration(audio, sample_rate):
    return len(audio) / sample_rate

# n_FFT algorithms are optimized for powers of two (e.g., 512, 1024, 2048).
# 512 is a common choice for speech to balance time and frequency resolution.
# if the nfft is too small, the spectrogram will have poor frequency resolution, making it harder to distinguish between different phonemes. 
# If it's too large, the time resolution will suffer, which can also negatively impact ASR performance.
# Hop length is a percentage of n_fft (e.g., 25-50%) to ensure sufficient overlap between frames,
# capturing rapid changes in speech.
# n_mels is typically set to 80 for ASR tasks, providing a good balance between frequency resolution and computational efficiency.
# 80 is also the default in many ASR models, including OpenAI's Whisper, and is widely used in the research community for speech recognition tasks.

def get_audio_mel_spectrogram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS) -> np.ndarray:  # Hop length set to 512 as recommended for Audio processing
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6) # Normalize to [0, 1]

def plot_waveform(audio, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_audio_mel_spectrogram(mel_spectrogram: torch.Tensor, sample_rate: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_spectrogram.numpy(), sr=sample_rate, hop_length=hop_length, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()

def train(model: nn.Module,
          optimiser: optim.Optimizer,
          train_loader: DataLoader,
          val_loader: DataLoader=None,
          tokenizer: transformers.PreTrainedTokenizerBase=Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base"),
          loss_fn: nn.modules.loss._Loss=nn.CTCLoss(blank=Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base").pad_token_id, reduction='mean'),
          loss_threshold: float=0.0,
          max_epochs: int=20,
          device: torch.device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device=device)
    train_risk = []
    val_risk = []
    num_train_batches = len(train_loader)
    train_set_size = len(train_loader.dataset)
    for epoch in tqdm(range(max_epochs), desc="epoch", position=0):
        # 1. train
        risk = 0.0
        model.train()
        for batch in tqdm(train_loader, desc="batch", position=1, leave=False):
            specs = batch['padded_spectrograms']
            seq_lens = batch['input_lengths']
            targets = batch['packed_transcripts']
            target_lens = batch['target_lengths']
            # move tensors to device
            specs = specs.to(device=device)
            seq_lens = seq_lens.to(device=device)
            targets = targets.to(device=device)
            # forward pass
            outputs, seq_lens = model.forward(specs, seq_lens)
            loss = loss_fn(outputs, seq_lens, targets, target_lens)
            # collect the training loss
            risk += loss.item()
            # backward pass
            optimiser.zero_grad()
            loss.backward()
            # gradient descent step
            optimiser.step()
            
        train_risk.append(risk/train_set_size)
        
        # 2. validate
        if val_loader is not None:
            val_risk.append(test(model, val_loader, loss_fn, device))
        
        if loss <= loss_threshold: # early termination
            break

    return train_risk, val_risk

def test(model: nn.Module,
         test_loader: DataLoader,
         loss_fn: nn.modules.loss._Loss,
         device: torch.device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device = device)
    model.eval()
    with torch.no_grad():
        risk = 0.0
        for batch in test_loader:
            specs = batch['padded_spectrograms']
            seq_lens = batch['input_lengths']
            targets = batch['packed_transcripts']
            target_lens = batch['target_lengths']
            # move tensors to device
            specs = specs.to(device=device)
            seq_lens = seq_lens.to(device=device)
            targets = targets.to(device=device)
            # forward pass
            outputs = model.forward(specs, seq_lens)
            loss = loss_fn(outputs, seq_lens, targets, target_lens)
            # collect the training loss
            risk += loss.item()

    return risk/len(test_loader.dataset)