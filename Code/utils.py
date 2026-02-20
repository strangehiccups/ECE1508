import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def get_audio_duration(audio, sample_rate):
    return len(audio) / sample_rate

def get_audio_stft(audio, n_fft=2048, hop_length=512):
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

def get_audio_mel_spectrogram(audio, sample_rate, n_fft=2048, hop_length=512, n_mels=128):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(S, ref=np.max)

def plot_waveform(audio, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_audio_mel_spectrogram(audio, sample_rate):
    mel_spectrogram = get_audio_mel_spectrogram(audio, sample_rate)
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time', sr=sample_rate)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()