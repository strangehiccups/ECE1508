import matplotlib.pyplot as plt
import torch

from config import HOP_LENGTH, SAMPLE_RATE


def plot_waveform(audio: torch.Tensor, sample_rate: int):
    plt.figure(figsize=(12, 4))
    time_axis = torch.arange(audio.shape[-1]) / sample_rate
    plt.plot(time_axis, audio.squeeze().numpy())
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_audio_mel_spectrogram(
    mel_spectrogram: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
):
    plt.figure(figsize=(12, 8))
    plt.imshow(
        mel_spectrogram.squeeze().numpy(),
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram (Sample Rate: {sample_rate} Hz, Hop Length: {hop_length} samples)")
    plt.show()


def log_audio_sample(
    raw_audio: torch.Tensor,
    sample_rate: int,
    mel: torch.Tensor,
    raw_text: str,
    tokenized_text=None,
    title: str = "",
):
    from IPython.display import Audio, display

    print(f"--- {title} ---")
    plot_audio_mel_spectrogram(mel.float(), sample_rate=sample_rate)
    print(f"Raw text: {raw_text}")
    if tokenized_text is not None:
        print(f"Tokenized text: {tokenized_text}")
    print(f"Audio shape: {raw_audio.shape}, Sample rate: {sample_rate} Hz")
    display(Audio(raw_audio.squeeze().numpy(), rate=sample_rate))


def plot_training_loss_history(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("CTC Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_training_cer_history(history: dict):
    epochs = range(1, len(history["train_cer"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_cer"], label="Train CER")
    if "val_cer" in history:
        plt.plot(epochs, history["val_cer"], label="Val CER")
    plt.title("Character Error Rate (CER) History")
    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.legend()
    plt.show()


def plot_training_wer_history(history: dict):
    epochs = range(1, len(history["train_wer"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_wer"], label="Train WER")
    if "val_wer" in history:
        plt.plot(epochs, history["val_wer"], label="Val WER")
    plt.title("Word Error Rate (WER) History")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.legend()
    plt.show()
