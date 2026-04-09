import os

import h5py
import torch
import torch.nn as nn
import torch.optim as optim

from config import HISTORY_KEYS, SAVE_BEST_MODEL_PATH, SAVE_HISTORY_PATH, SAVE_MODEL_PATH


def save_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss=None,
    filepath: str = SAVE_MODEL_PATH,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} at epoch {epoch}")


def load_model(
    model: nn.Module,
    device: torch.device,
    filepath: str = SAVE_MODEL_PATH,
    optimizer: optim.Optimizer = None,
):
    if not os.path.exists(filepath):
        print(f"Checkpoint '{filepath}' not found.")
        return 0, None

    print(f"Loading checkpoint from '{filepath}'...")
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    print(f"Checkpoint loaded successfully. Resuming from epoch {epoch}.")
    return epoch, loss


def save_history(history_values: list, path: str = SAVE_HISTORY_PATH):
    if len(history_values) != len(HISTORY_KEYS):
        print(
            f"no. of history values {len(history_values)} does not match "
            f"no. of history keys {len(HISTORY_KEYS)}"
        )
        return None

    with h5py.File(path, "a") as f:
        if HISTORY_KEYS[0] not in f:
            for key in HISTORY_KEYS:
                f.create_dataset(key, shape=(0,), maxshape=(None,))
        n = f[HISTORY_KEYS[0]].shape[0]
        new_size = n + 1
        for i, key in enumerate(HISTORY_KEYS):
            arr = f[key]
            arr.resize((new_size,))
            arr[n] = history_values[i]


def load_h5_struct(file_path: str = SAVE_HISTORY_PATH):
    history = {}
    try:
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                history[key] = f[key][:]
    except FileNotFoundError:
        history = None
    return history
