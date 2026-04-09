import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

import utils
from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_experiments(
    train_set,
    val_set,
    test_set,
    model_builder: Callable[[], torch.nn.Module],
    learning_rate: float,
    num_epochs: int,
    device: torch.device,
    seeds: list[int],
    out_dir: Path,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    seed_histories: dict[int, dict[str, np.ndarray] | None] = {}
    seed_models: dict[int, torch.nn.Module] = {}

    _persistent = NUM_WORKERS > 0
    _prefetch = PREFETCH_FACTOR if NUM_WORKERS > 0 else None

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n================ Run {run_idx}/{len(seeds)} | seed={seed} ================")
        _set_all_seeds(seed)

        model = model_builder().to(device)
        loss_fn = model.loss_fn

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=BATCH_SIZE,
            collate_fn=utils.collate_fn_train,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=_persistent,
            prefetch_factor=_prefetch,
            pin_memory=PIN_MEMORY,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=BATCH_SIZE,
            collate_fn=utils.collate_fn_eval,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=_persistent,
            prefetch_factor=_prefetch,
            pin_memory=PIN_MEMORY,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=BATCH_SIZE,
            collate_fn=utils.collate_fn_eval,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=_persistent,
            prefetch_factor=_prefetch,
            pin_memory=PIN_MEMORY,
        )

        optimiser = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
        batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.05,
            anneal_strategy="cos",
        )

        run_dir = out_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.pth"
        best_model_path = run_dir / "model_best.pth"
        history_path = run_dir / "history.h5"

        history = utils.train(
            model=model,
            optimiser=optimiser,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            start_epoch=1,
            max_epochs=num_epochs,
            batch_scheduler=batch_scheduler,
            save_model_path=str(model_path),
            save_best_model_path=str(best_model_path),
            save_history_path=str(history_path),
            device=device,
        )

        test_loss, test_cer, test_wer = utils.test(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            tokenizer=model.tokenizer,
        )

        best_val_wer = float(np.min(history["val_wer"])) if history is not None and "val_wer" in history else float("nan")
        best_val_cer = float(np.min(history["val_cer"])) if history is not None and "val_cer" in history else float("nan")
        best_epoch = int(np.argmin(history["val_wer"]) + 1) if history is not None and "val_wer" in history else -1

        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_cer": best_val_cer,
            "best_val_wer": best_val_wer,
            "test_loss": float(test_loss),
            "test_cer": float(test_cer),
            "test_wer": float(test_wer),
            "model_path": str(model_path),
            "best_model_path": str(best_model_path),
            "history_path": str(history_path),
        }
        seed_results.append(result)
        seed_histories[seed] = history
        seed_models[seed] = model
        print(f"[Seed {seed}] best_val_wer={best_val_wer:.4f} | test_wer={test_wer:.4f}")

    seed_results_df = pd.DataFrame(seed_results).sort_values("seed").reset_index(drop=True)

    summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "best_val_wer_mean": float(seed_results_df["best_val_wer"].mean()),
        "best_val_wer_std": float(seed_results_df["best_val_wer"].std(ddof=1)) if len(seed_results_df) > 1 else 0.0,
        "test_wer_mean": float(seed_results_df["test_wer"].mean()),
        "test_wer_std": float(seed_results_df["test_wer"].std(ddof=1)) if len(seed_results_df) > 1 else 0.0,
        "test_cer_mean": float(seed_results_df["test_cer"].mean()),
        "test_cer_std": float(seed_results_df["test_cer"].std(ddof=1)) if len(seed_results_df) > 1 else 0.0,
    }

    seed_results_csv = out_dir / "seed_results.csv"
    seed_summary_json = out_dir / "seed_summary.json"

    seed_results_df.to_csv(seed_results_csv, index=False)
    with open(seed_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    best_seed = int(seed_results_df.sort_values("best_val_wer").iloc[0]["seed"])

    return {
        "seed_results_df": seed_results_df,
        "summary": summary,
        "seed_results_csv": str(seed_results_csv),
        "seed_summary_json": str(seed_summary_json),
        "best_seed": best_seed,
        "best_model": seed_models[best_seed],
        "best_history": seed_histories[best_seed],
    }


def run_multi_seed_experiment(
    train_set,
    val_set,
    test_set,
    learning_rate: float,
    num_epochs: int,
    device: torch.device,
    model: torch.nn.Module | None = None,
    model_builder: Callable[[], torch.nn.Module] | None = None,
    seeds: list[int] | None = None,
    results_dir: str = "../models/seed_runs",
) -> dict[str, Any]:
    if seeds is None:
        seeds = [1508, 2603, 9102]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_builder is None:
        if model is None:
            raise ValueError("Either `model_builder` or `model` must be provided.")
        model_template = copy.deepcopy(model).cpu()

        def model_builder() -> torch.nn.Module:
            return copy.deepcopy(model_template)

    return _run_experiments(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        model_builder=model_builder,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
        seeds=seeds,
        out_dir=out_dir,
    )
