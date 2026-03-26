import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import os
import time

from torch.utils.data import DataLoader
from transformers import Wav2Vec2CTCTokenizer
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional

import h5py

from config import (
    TOKENIZER,
    BLANK_TOKEN_ID,
    HISTORY_KEYS,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    SAVE_MODEL_PATH,
    SAVE_BEST_MODEL_PATH,
    SAVE_HISTORY_PATH
)

from torchmetrics.text import CharErrorRate, WordErrorRate


@dataclass(frozen=True)
class AudioSample:
    raw_audio: Optional[torch.Tensor]  # only needed for debug display; None on cache hits
    raw_mel_audio: torch.Tensor
    mel_audio_spec_augment: Optional[torch.Tensor]
    sample_rate: int
    file_path: str
    raw_text: str
    tokenized_text: Optional[torch.Tensor] = None


def get_audio_duration(audio, sample_rate):
    return audio.shape[-1] / sample_rate


# n_FFT algorithms are optimized for powers of two (e.g., 512, 1024, 2048).
# 512 is a common choice for speech to balance time and frequency resolution.
# if the nfft is too small, the spectrogram will have poor frequency resolution, making it harder to distinguish between different phonemes. 
# If it's too large, the time resolution will suffer, which can also negatively impact ASR performance.
# Hop length is a percentage of n_fft (e.g., 25-50%) to ensure sufficient overlap between frames,
# capturing rapid changes in speech.
# n_mels is typically set to 80 for ASR tasks, providing a good balance between frequency resolution and computational efficiency.
# 80 is also the default in many ASR models, including OpenAI's Whisper, and is widely used in the research community for speech recognition tasks.
def get_audio_mel_spectrogram(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS) -> torch.Tensor:
    # Librosa defaults to mono=True, which forcibly averages multiple channels into 1D (time,).
    # Since torchaudio preserves channels e.g. (channels, time), we must average them to 
    # prevent the extra dimension from breaking the downstream pad_sequence!
    if audio.dim() > 1:
        audio = audio.mean(dim=0)

    # Resample if the audio sample rate is different from the expected SAMPLE_RATE
    if sample_rate != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        audio = resampler(audio)
        sample_rate = SAMPLE_RATE
        
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    S = mel_transform(audio)
    return torch.log(S + 1e-9)  # log-mel; BatchNorm in the CNN handles normalisation


# SpecAugment paper: https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf
# time_mask_param: maximum width of the time mask (in frames)
# freq_mask_param: maximum width of the frequency mask (in mel bins)
# Check Table 1. in this paper, set the # of params based on Libri Double (which is basically)
# for multi-speaker (and I've set it to this currently in preparation for libri-small)
def spec_augment(mel_spectrogram: torch.Tensor, time_mask_param: int = 30, freq_mask_param: int = 13) -> torch.Tensor:
    # Clone to avoid in-place mutation of the original mel spectrogram.
    augmented = mel_spectrogram.clone()

    # Guard against short utterances so mask widths do not exceed dimensions.
    freq_bins = augmented.shape[-2]
    time_steps = augmented.shape[-1]
    safe_freq_mask = min(freq_mask_param, max(1, freq_bins - 1))
    safe_time_mask = min(time_mask_param, max(1, time_steps - 1))

    augmented = T.FrequencyMasking(safe_freq_mask)(augmented)
    augmented = T.FrequencyMasking(safe_freq_mask)(augmented)
    augmented = T.TimeMasking(safe_time_mask)(augmented)
    augmented = T.TimeMasking(safe_time_mask)(augmented)
    return augmented


def plot_waveform(audio: torch.Tensor, sample_rate: int):
    plt.figure(figsize=(12, 4))
    time_axis = torch.arange(audio.shape[-1]) / sample_rate
    plt.plot(time_axis, audio.squeeze().numpy())
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def plot_audio_mel_spectrogram(mel_spectrogram: torch.Tensor, sample_rate: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH):
    plt.figure(figsize=(12, 8))
    plt.imshow(mel_spectrogram.squeeze().numpy(), aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram (Sample Rate: {sample_rate} Hz, Hop Length: {hop_length} samples)')
    plt.show()


# Collate function for DataLoader to handle variable-length spectrograms and transcripts
def _collate_from_mels(mels: list[torch.Tensor], tokenized_texts: list[torch.Tensor]) -> dict:
    # Sort by descending spectrogram length (required for pack_padded_sequence)
    batch = sorted(zip(mels, tokenized_texts), key=lambda x: x[0].shape[1], reverse=True)
    audios, tokenized_texts = zip(*batch)

    input_lengths = torch.tensor([a.shape[1] for a in audios], dtype=torch.long)

    # Pad to (batch, max_time_frames, n_mels), then reshape for CNN input
    padded_spectrograms = torch.nn.utils.rnn.pad_sequence(
        [a.transpose(0, 1) for a in audios],  # each: (time_frames, n_mels)
        batch_first=True                       # output: (batch, max_time_frames, n_mels)
    )
    padded_spectrograms = padded_spectrograms.unsqueeze(1)         # (batch, 1, max_time_frames, n_mels)
    padded_spectrograms = padded_spectrograms.permute(0, 1, 3, 2)   # (batch, 1, n_mels, max_time_frames)

    # CTC loss expects transcripts concatenated into a single 1-D tensor
    target_lengths = torch.tensor([len(t) for t in tokenized_texts], dtype=torch.long)
    packed_transcripts = torch.cat([t.clone().detach().long() for t in tokenized_texts])

    return {
        'padded_spectrograms': padded_spectrograms,  # (batch, 1, n_mels, max_time_frames)
        'input_lengths': input_lengths,
        'packed_transcripts': packed_transcripts,
        'target_lengths': target_lengths,
    }


def _pick_mel(sample, attrs):
    for attr in attrs:
        val = getattr(sample, attr, None)
        if val is not None:
            return val
    return None


# We have different collate functions for train/eval because
# we cannot run SpecAugment pre-processing on validation sets,
# and this was easier than figuring out how to parametrize the
# dataloader
def collate_fn_train(batch) -> Optional[dict]:
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    mels = [_pick_mel(s, ('mel_audio_spec_augment',)) for s in batch]
    tokenized_texts = [sample.tokenized_text for sample in batch]
    return _collate_from_mels(mels, tokenized_texts)


def collate_fn_eval(batch) -> Optional[dict]:
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    mels = [_pick_mel(s, ('raw_mel_audio',)) for s in batch]
    tokenized_texts = [sample.tokenized_text for sample in batch]
    return _collate_from_mels(mels, tokenized_texts)


# Backward-compatible default: training collate (includes SpecAugment when present).
def collate_fn(batch) -> Optional[dict]:
    return collate_fn_train(batch)


def log_audio_sample(sample: AudioSample, title: str):
    from IPython.display import display, Audio
    print(f"--- {title} ---")
    # True audio
    plot_audio_mel_spectrogram(sample.raw_mel_audio.float(), sample_rate=sample.sample_rate)
    # Time/Frequency-masked sample
    plot_audio_mel_spectrogram(sample.mel_audio_spec_augment.float(), sample_rate=sample.sample_rate)
    print(f"Raw text: {sample.raw_text}")
    print(f"Tokenized text: {sample.tokenized_text}")
    print(f"Audio shape: {sample.raw_audio.shape}, Sample rate: {sample.sample_rate} Hz")
    display(Audio(sample.raw_audio.squeeze().numpy(), rate=sample.sample_rate))


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


def load_h5_struct(filename: str):
    history = {}
    try:
        with h5py.File(SAVE_HISTORY_PATH, "r") as f: # r: read
            for key in f.keys():
                history[key] = f[key][:]
    except FileNotFoundError:
        history = None
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
          tokenizer: transformers.PreTrainedTokenizerBase=TOKENIZER,
          loss_fn: nn.modules.loss._Loss=nn.CTCLoss(blank=BLANK_TOKEN_ID, reduction='mean'),
          loss_threshold: float=0.0,
          start_epoch: int=0,
          max_epochs: int=20,
          scheduler: optim.lr_scheduler.LRScheduler=None,
          batch_scheduler: optim.lr_scheduler.LRScheduler=None,
          device: torch.device=None):

    model = model.to(device=device)
    use_amp = device is not None and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    num_train_batches = len(train_loader)
    train_set_size    = len(train_loader.dataset)
    best_val_wer      = float('inf')

    for epoch in tqdm(range(start_epoch, max_epochs+1), desc="epoch", position=0):
        # ------------------------------------------------------------------ #
        # 1. Training pass                                                   #
        # ------------------------------------------------------------------ #
        risk = 0.0
        epoch_refs, epoch_hyps = [], []
        train_epoch_time = 0.0; val_epoch_time = 0.0; # in seconds
        model.train()
        for i, batch in tqdm(enumerate(train_loader), desc="batch", position=1, leave=False, total=num_train_batches):
            batch_start_time = time.time()
            
            specs      = batch['padded_spectrograms']
            seq_lens   = batch['input_lengths']
            targets    = batch['packed_transcripts']
            target_lens = batch['target_lengths']
            # move tensors to device (lengths must stay on CPU for CTCLoss)
            specs = specs.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            # forward pass with mixed precision
            optimiser.zero_grad()
            # collect the training loss
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs, out_lens = model.forward(specs, seq_lens)
                loss = loss_fn(
                    blank=BLANK_TOKEN_ID,
                    log_probs=outputs,
                    seq_lens=out_lens.cpu(),
                    targets=targets,
                    target_lens=target_lens.cpu(),
                )

            risk += loss.item()
            # backward pass (scaled for mixed precision)
            scaler.scale(loss).backward()
            # gradient clipping to stabilise CTC training
            scaler.unscale_(optimiser)
            # gradient descent step
            scaler.step(optimiser)
            scaler.update()

            # step the batch-level scheduler (e.g., OneCycleLR) after each batch, if provided
            # this allows for more fine-grained control over the learning rate within an epoch
            # without this, we were noticing that the learning rate only updated after each epoch,
            # and was actually contributing to overfitting
            # Basically Epoch-level PlateauLR reacts after each epoch based on val loss, while
            # batch-level OneCycleLR updates after each batch based on the total number of batches and epochs.
            if batch_scheduler is not None:
                batch_scheduler.step()

            train_epoch_time += time.time() - batch_start_time
            
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
            epoch_start_time = time.time()
            v_loss, v_cer, v_wer = test(model, val_loader, loss_fn, device, tokenizer)
            val_epoch_time = time.time() - epoch_start_time
            print(f"[ Val ] Epoch {epoch}/{max_epochs}  Loss: {v_loss:.6f}  CER: {v_cer:.4f}  WER: {v_wer:.4f}")
            if scheduler is not None:
                scheduler.step(v_loss)
                print(f"[ LR  ] {scheduler.get_last_lr()[0]:.2e}")
            if v_wer < best_val_wer:
                best_val_wer = v_wer
                save_model(model, optimiser, epoch, loss=epoch_loss, filepath=SAVE_BEST_MODEL_PATH)
                print(f"[ Best] New best val WER {best_val_wer:.4f} — checkpoint saved to {SAVE_BEST_MODEL_PATH}")
        else:
            v_loss, v_cer, v_wer = [-1,-1,-1] # negative values indicate undefined (for saving history)

        save_model(
            model=model,
            optimizer=optimiser,
            epoch=epoch,
            loss=epoch_loss,
            filepath=SAVE_MODEL_PATH
        )

        save_history(
            [epoch_loss,
             v_loss,
             epoch_cer,
             v_cer,
             epoch_wer,
             v_wer,
             train_epoch_time,
             val_epoch_time]
        )

        if epoch_loss <= loss_threshold:  # early termination
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
            loss = loss_fn(
                    blank=BLANK_TOKEN_ID,
                    log_probs=log_probs,
                    seq_lens=out_lens.cpu(),
                    targets=targets,
                    target_lens=target_lens.cpu(),
                )
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
    if 'val_loss' in history:
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
    if 'val_cer' in history:
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
    if 'val_wer' in history:
        plt.plot(epochs, history['val_wer'], label='Val WER')
    plt.title('Word Error Rate (WER) History')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.legend()
    plt.show()
