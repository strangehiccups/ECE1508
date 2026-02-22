import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch


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