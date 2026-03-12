import struct
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
from torchmetrics.text import CharErrorRate, WordErrorRate
from tqdm.auto import tqdm
import os
import h5py

HOP_LENGTH = 256
N_FFT = 512
N_MELS = 80
SAMPLE_RATE = 22050
SAVE_MODEL_PATH = "model.pth"
SAVE_HISTORY_PATH = "history.h5"
HISTORY_KEYS = ["train_loss", "val_loss", "train_cer", "val_cer", "train_wer", "val_wer"]

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

# Save the trained model
def save_model(model, optimizer, epoch, loss=None, filepath=SAVE_MODEL_PATH):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} at epoch {epoch}")

# Load the saved model.
def load_model(model, device, filepath=SAVE_MODEL_PATH, optimizer=None):

    if not os.path.exists(filepath):
        print(f"Checkpoint '{filepath}' not found.")
        return 0, None
    
    print(f"Loading checkpoint from '{filepath}'...")
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state to the proper device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    
    print(f"Checkpoint loaded successfully. Resuming from epoch {epoch}.")
    return epoch, loss

def save_history(history_values: list, path: str=SAVE_HISTORY_PATH):
    if len(history_values) != len(HISTORY_KEYS):
        print(f'no. of history values {len(history_values)} does not match no. of history keys {len(HISTORY_KEYS)}')
        return None
    
    with h5py.File(path, "a") as f: # a: append
        if HISTORY_KEYS[0] not in f:
            for key in HISTORY_KEYS:
                f.create_dataset(key, shape=(0,), maxshape=(None,))

        n = f[HISTORY_KEYS[0]].shape[0]
        new_size = n + 1
        for i, key in enumerate(HISTORY_KEYS):
            arr = f[key]
            arr.resize((new_size,))
            arr[n] = history_values[i]

def load_h5_struct(filename: str) -> struct:
    history = {}
    with h5py.File(SAVE_HISTORY_PATH, "r") as f: # r: read
        for key in f.keys():
            history[key] = f[key][:]
    return history

def ctc_greedy_decode(log_probs: torch.Tensor,
                      output_lengths: torch.Tensor,
                      tokenizer) -> list:
    """Greedy CTC decode: argmax -> collapse repeats -> remove blanks -> decode tokens.

    Args:
        log_probs:      (batch, time, vocab) tensor (log-probs or softmax probs).
        output_lengths: (batch,) tensor of valid time-steps per sample.
        tokenizer:      tokenizer whose pad_token_id is the CTC blank label.

    Returns:
        List of decoded strings, one per sample in the batch.
    """
    blank_id = tokenizer.pad_token_id
    decoded = []
    for b in range(log_probs.shape[0]):
        T = output_lengths[b].item()
        pred_ids = log_probs[b, :T].argmax(dim=-1).tolist()
        collapsed = []
        prev = None
        for idx in pred_ids:
            if idx != prev:
                if idx != blank_id:
                    collapsed.append(idx)
                prev = idx
        decoded.append(tokenizer.decode(collapsed))
    return decoded


def _decode_targets(packed_transcripts: torch.Tensor,
                    target_lengths: torch.Tensor,
                    tokenizer) -> list:
    """Reconstruct the ground-truth strings from packed CTC targets."""
    texts = []
    offset = 0
    for length in target_lengths.tolist():
        ids = packed_transcripts[offset: offset + length].tolist()
        texts.append(tokenizer.decode(ids))
        offset += length
    return texts


def train(model: nn.Module,
          optimiser: optim.Optimizer,
          train_loader: DataLoader,
          val_loader: DataLoader=None,
          tokenizer: transformers.PreTrainedTokenizerBase=Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base"),
          loss_fn: nn.modules.loss._Loss=nn.CTCLoss(blank=Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base").pad_token_id, reduction='mean'),
          loss_threshold: float=0.0,
          start_epoch: int=0,
          max_epochs: int=20,
          device: torch.device=None):

    model = model.to(device=device)
    use_amp = device is not None and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    num_train_batches = len(train_loader)
    train_set_size    = len(train_loader.dataset)

    for epoch in tqdm(range(start_epoch, max_epochs), desc="epoch", position=0):
        # ------------------------------------------------------------------ #
        # 1. Training pass                                                    #
        # ------------------------------------------------------------------ #
        risk = 0.0
        epoch_refs, epoch_hyps = [], []
        model.train()
        for i, batch in tqdm(enumerate(train_loader), desc="batch", position=1, leave=False, total=num_train_batches):
            specs      = batch['padded_spectrograms']
            seq_lens   = batch['input_lengths']
            targets    = batch['packed_transcripts']
            target_lens = batch['target_lengths']
            # move tensors to device (lengths must stay on CPU for CTCLoss)
            specs = specs.to(device=device)
            targets = targets.to(device=device)
            # forward pass with mixed precision
            optimiser.zero_grad()
            # collect the training loss
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs, out_lens = model.forward(specs, seq_lens)
                loss = loss_fn(outputs, out_lens.cpu(), targets, target_lens.cpu())

            risk += loss.item()
            # backward pass (scaled for mixed precision)
            scaler.scale(loss).backward()
            # gradient clipping to stabilise CTC training
            scaler.unscale_(optimiser)
            # gradient descent step
            scaler.step(optimiser)
            scaler.update()

            # Accumulate predictions/references for CER/WER (detach from graph)
            with torch.no_grad():
                refs = _decode_targets(targets.cpu(), target_lens.cpu(), tokenizer)
                hyps = ctc_greedy_decode(outputs.float().cpu(), out_lens.cpu(), tokenizer)
            epoch_refs.extend(refs)
            epoch_hyps.extend(hyps)

            if i % 10 == 0:
                print(f"Epoch {epoch}/{max_epochs}, Batch {i+1}/{num_train_batches}, Loss: {loss.item():.4f}")

        epoch_loss = risk / train_set_size
        epoch_cer  = CharErrorRate()(epoch_hyps, epoch_refs).item()
        epoch_wer  = WordErrorRate()(epoch_hyps, epoch_refs).item()
        print(f"[Train] Epoch {epoch}/{max_epochs}  Loss: {epoch_loss:.6f}  CER: {epoch_cer:.4f}  WER: {epoch_wer:.4f}")

        # ------------------------------------------------------------------ #
        # 2. Validation pass                                                  #
        # ------------------------------------------------------------------ #
        if val_loader is not None:
            v_loss, v_cer, v_wer = test(model, val_loader, loss_fn, device, tokenizer)
            print(f"[ Val ] Epoch {epoch+1}/{max_epochs}  Loss: {v_loss:.6f}  CER: {v_cer:.4f}  WER: {v_wer:.4f}")
        else:
            v_loss, v_cer, v_wer = [-1,-1,-1] # negative values indicate undefined (for saving history)

        save_model(
            model=model,
            optimizer=optimiser,
            epoch=epoch,
            loss=loss,
            filepath=SAVE_MODEL_PATH
        )

        save_history(
            [epoch_loss,
             v_loss,
             epoch_cer,
             v_cer,
             epoch_wer,
             v_wer]
        )

        if loss <= loss_threshold:  # early termination
            break

    return load_h5_struct(SAVE_HISTORY_PATH)

def test(model: nn.Module,
         test_loader: DataLoader,
         loss_fn: nn.modules.loss._Loss,
         device: torch.device = None,
         tokenizer=None) -> tuple:
    """Evaluate the model on a DataLoader.

    Returns:
        (loss, cer, wer) averaged over the full dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

    model = model.to(device=device)
    model.eval()

    risk = 0.0
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for batch in test_loader:
            specs       = batch['padded_spectrograms']
            seq_lens    = batch['input_lengths']
            targets     = batch['packed_transcripts']
            target_lens = batch['target_lengths']
            # move tensors to device (lengths must stay on CPU for CTCLoss)
            specs = specs.to(device=device)
            targets = targets.to(device=device)

            # forward pass
            log_probs, out_lens = model.forward(specs, seq_lens)
            loss = loss_fn(log_probs, out_lens.cpu(), targets, target_lens.cpu())
            risk += loss.item()

            refs = _decode_targets(targets.cpu(), target_lens.cpu(), tokenizer)
            hyps = ctc_greedy_decode(log_probs.cpu(), out_lens.cpu(), tokenizer)
            all_refs.extend(refs)
            all_hyps.extend(hyps)

    avg_loss = risk / len(test_loader.dataset)
    cer = CharErrorRate()(all_hyps, all_refs).item()
    wer = WordErrorRate()(all_hyps, all_refs).item()
    return avg_loss, cer, wer


def plot_training_loss_history(history: dict):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    if history['val_loss']:
        plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('CTC Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_training_cer_history(history: dict):
    epochs = range(1, len(history['train_cer']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_cer'], label='Train CER')
    if history['val_cer']:
        plt.plot(epochs, history['val_cer'], label='Val CER')
    plt.title('Character Error Rate (CER) History')
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.legend()
    plt.show()

def plot_training_wer_history(history: dict):
    epochs = range(1, len(history['train_wer']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_wer'], label='Train WER')
    if history['val_wer']:
        plt.plot(epochs, history['val_wer'], label='Val WER')
    plt.title('Word Error Rate (WER) History')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.legend()
    plt.show()
