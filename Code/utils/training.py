import time

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate
from tqdm.auto import tqdm
from transformers import Wav2Vec2CTCTokenizer

from config import (
    BLANK_TOKEN_ID,
    SAVE_BEST_MODEL_PATH,
    SAVE_HISTORY_PATH,
    SAVE_MODEL_PATH,
    TOKENIZER,
)
from utils.checkpointing import load_h5_struct, save_history, save_model


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer,
) -> list:
    """Greedy CTC decode: argmax → collapse repeats → remove blanks → decode tokens."""
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


def _decode_targets(
    packed_transcripts: torch.Tensor,
    target_lengths: torch.Tensor,
    tokenizer,
) -> list:
    """Reconstruct ground-truth strings from packed CTC targets."""
    texts = []
    offset = 0
    for length in target_lengths.tolist():
        ids = packed_transcripts[offset: offset + length].tolist()
        texts.append(tokenizer.decode(ids))
        offset += length
    return texts


def train(
    model: nn.Module,
    optimiser: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    tokenizer: transformers.PreTrainedTokenizerBase = TOKENIZER,
    loss_fn: nn.modules.loss._Loss = nn.CTCLoss(blank=BLANK_TOKEN_ID, reduction="mean"),
    loss_threshold: float = 0.0,
    start_epoch: int = 0,
    max_epochs: int = 20,
    scheduler: optim.lr_scheduler.LRScheduler = None,
    batch_scheduler: optim.lr_scheduler.LRScheduler = None,
    save_model_path: str = SAVE_MODEL_PATH,
    save_best_model_path: str = SAVE_BEST_MODEL_PATH,
    save_history_path: str = SAVE_HISTORY_PATH,
    device: torch.device = None,
):
    model = model.to(device=device)
    use_amp = device is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    num_train_batches = len(train_loader)
    best_val_wer = float("inf")

    for epoch in tqdm(range(start_epoch, max_epochs + 1), desc="epoch", position=0):
        risk = 0.0
        epoch_refs, epoch_hyps = [], []
        epoch_decode_queue = []
        train_epoch_time = 0.0
        val_epoch_time = 0.0

        model.train()
        for i, batch in tqdm(
            enumerate(train_loader),
            desc="batch",
            position=1,
            leave=False,
            total=num_train_batches,
        ):
            batch_start_time = time.time()

            specs = batch["padded_spectrograms"]
            seq_lens = batch["input_lengths"]
            targets = batch["packed_transcripts"]
            target_lens = batch["target_lengths"]

            specs = specs.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
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
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            scaler.step(optimiser)
            scaler.update()

            if batch_scheduler is not None:
                batch_scheduler.step()

            train_epoch_time += time.time() - batch_start_time

            # Stash CPU copies for end-of-epoch decode — avoids blocking the GPU per batch
            epoch_decode_queue.append((
                outputs.detach().float().cpu(),
                out_lens.cpu(),
                targets.detach().cpu(),
                target_lens.cpu(),
            ))
            if i % 10 == 0:
                print(
                    f"Epoch {epoch}/{max_epochs}, "
                    f"Batch {i + 1}/{num_train_batches}, "
                    f"Loss: {loss.item():.4f}"
                )

        for outputs_cpu, out_lens_cpu, targets_cpu, tlens_cpu in epoch_decode_queue:
            epoch_refs.extend(_decode_targets(targets_cpu, tlens_cpu, tokenizer))
            epoch_hyps.extend(ctc_greedy_decode(outputs_cpu, out_lens_cpu, tokenizer))

        epoch_loss = risk / num_train_batches
        epoch_cer = CharErrorRate()(epoch_hyps, epoch_refs).item()
        epoch_wer = WordErrorRate()(epoch_hyps, epoch_refs).item()
        print(
            f"[Train] Epoch {epoch}/{max_epochs}  "
            f"Loss: {epoch_loss:.6f}  CER: {epoch_cer:.4f}  WER: {epoch_wer:.4f}"
        )

        if val_loader is not None:
            epoch_start_time = time.time()
            v_loss, v_cer, v_wer = test(model, val_loader, loss_fn, device, tokenizer)
            val_epoch_time = time.time() - epoch_start_time
            print(
                f"[ Val ] Epoch {epoch}/{max_epochs}  "
                f"Loss: {v_loss:.6f}  CER: {v_cer:.4f}  WER: {v_wer:.4f}"
            )
            if scheduler is not None:
                scheduler.step(v_loss)
                print(f"[ LR  ] {scheduler.get_last_lr()[0]:.2e}")
            if v_wer < best_val_wer:
                best_val_wer = v_wer
                save_model(model, optimiser, epoch, loss=epoch_loss, filepath=save_best_model_path)
                print(
                    f"[ Best] New best val WER {best_val_wer:.4f} — "
                    f"checkpoint saved to {save_best_model_path}"
                )
        else:
            v_loss, v_cer, v_wer = -1, -1, -1

        save_model(
            model=model,
            optimizer=optimiser,
            epoch=epoch,
            loss=epoch_loss,
            filepath=save_model_path,
        )
        save_history(
            [epoch_loss, v_loss, epoch_cer, v_cer, epoch_wer, v_wer, train_epoch_time, val_epoch_time],
            path=save_history_path,
        )

        if epoch_loss <= loss_threshold:
            break

    return load_h5_struct(save_history_path)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.modules.loss._Loss,
    device: torch.device = None,
    tokenizer=None,
) -> tuple:
    """Evaluate the model on a DataLoader. Returns (loss, cer, wer)."""
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
            specs = batch["padded_spectrograms"]
            seq_lens = batch["input_lengths"]
            targets = batch["packed_transcripts"]
            target_lens = batch["target_lengths"]

            specs = specs.to(device=device)
            targets = targets.to(device=device)

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

    avg_loss = risk / len(test_loader)
    cer = CharErrorRate()(all_hyps, all_refs).item()
    wer = WordErrorRate()(all_hyps, all_refs).item()
    return avg_loss, cer, wer
