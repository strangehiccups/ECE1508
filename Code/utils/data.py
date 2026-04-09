from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR


def _collate_from_mels(mels: list, tokenized_texts: list) -> dict:
    # Sort by descending spectrogram length (required for pack_padded_sequence)
    batch = sorted(zip(mels, tokenized_texts), key=lambda x: x[0].shape[1], reverse=True)
    audios, tokenized_texts = zip(*batch)

    input_lengths = torch.tensor([a.shape[1] for a in audios], dtype=torch.long)

    # Pad to (batch, max_time_frames, n_mels), then reshape for CNN input
    padded_spectrograms = torch.nn.utils.rnn.pad_sequence(
        [a.transpose(0, 1) for a in audios],
        batch_first=True,
    )
    padded_spectrograms = padded_spectrograms.unsqueeze(1)          # (B, 1, T, F)
    padded_spectrograms = padded_spectrograms.permute(0, 1, 3, 2)   # (B, 1, F, T)

    target_lengths = torch.tensor([len(t) for t in tokenized_texts], dtype=torch.long)
    packed_transcripts = torch.cat([t.clone().detach().long() for t in tokenized_texts])

    return {
        "padded_spectrograms": padded_spectrograms,
        "input_lengths": input_lengths,
        "packed_transcripts": packed_transcripts,
        "target_lengths": target_lengths,
    }


def collate_fn_train(batch) -> Optional[dict]:
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    mels = [mel for mel, _ in batch]
    tokens = [tok for _, tok in batch]
    return _collate_from_mels(mels, tokens)


def collate_fn_eval(batch) -> Optional[dict]:
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    mels = [mel for mel, _ in batch]
    tokens = [tok for _, tok in batch]
    return _collate_from_mels(mels, tokens)


def build_dataloaders(
    train_set,
    val_set,
    test_set,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    prefetch_factor: int = PREFETCH_FACTOR,
    pin_memory: bool = PIN_MEMORY,
):
    _persistent = num_workers > 0
    _prefetch = prefetch_factor if num_workers > 0 else None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        collate_fn=collate_fn_train,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        collate_fn=collate_fn_eval,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        collate_fn=collate_fn_eval,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
        pin_memory=pin_memory,
    )
    print(
        f"DataLoaders ready — "
        f"train: {len(train_loader)} batches, "
        f"val: {len(val_loader)} batches, "
        f"test: {len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader
