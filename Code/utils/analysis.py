from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.checkpointing import load_h5_struct


@dataclass
class Result:
    num_params: int
    histories: dict
    test: pd.core.frame.DataFrame
    stats: dict = field(default_factory=dict)


@dataclass
class Stats:
    mean: float
    std: float


def load_results(results_path: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    results = {}
    for model_dir in Path(results_path).iterdir():
        num_params = None
        if model_dir.is_dir():
            histories = {}
            for run_dir in Path(model_dir).iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("seed_"):
                    histories[run_dir.name] = load_h5_struct(run_dir / "history.h5")
                    if num_params is None:
                        state_dict = torch.load(
                            run_dir / "model_best.pth",
                            weights_only=True,
                            map_location=device,
                        )["model_state_dict"]
                        num_params = sum(v.numel() for v in state_dict.values())
            try:
                test_results = pd.read_csv(model_dir / "seed_results.csv")
            except Exception:
                test_results = None
            results[model_dir.name] = Result(
                num_params=num_params, histories=histories, test=test_results
            )
    return results


def compute_stats(results: dict):
    for model, result in results.items():
        history_metrics = {}
        for run, history in result.histories.items():
            for metric, values in history.items():
                if metric in history_metrics:
                    history_metrics[metric] = np.vstack((history_metrics[metric], values))
                else:
                    history_metrics[metric] = values
        stats = {}
        for metric, values in history_metrics.items():
            stats[metric] = Stats(mean=values.mean(axis=0), std=values.std(axis=0))
        if result.test is not None:
            stats["test_loss"] = Stats(
                mean=result.test.test_loss.mean(axis=0),
                std=result.test.test_loss.std(axis=0),
            )
            stats["test_cer"] = Stats(
                mean=result.test.test_cer.mean(axis=0),
                std=result.test.test_cer.std(axis=0),
            )
            stats["test_wer"] = Stats(
                mean=result.test.test_wer.mean(axis=0),
                std=result.test.test_wer.std(axis=0),
            )
        results[model].stats = stats
    return results


def visualise_stats(results: dict):
    axes = {}
    for model, result in results.items():
        print(model + " params: " + str(result.num_params))
        for stat, values in result.stats.items():
            if stat.startswith("test_"):
                print(model + " " + stat + " mean: " + str(values.mean))
                print(model + " " + stat + " std: " + str(values.std))
            else:
                epochs = np.arange(1, len(values.mean) + 1)
                if stat in axes:
                    ax = axes[stat]
                else:
                    _, ax = plt.subplots()
                    ax.set_title(stat)
                    ax.set_xlabel("epoch")
                    if stat.endswith("_time"):
                        ax.set_ylabel("time (s)")
                    axes[stat] = ax
                ax.plot(epochs, values.mean, label=model)
                ax.fill_between(
                    epochs,
                    values.mean - values.std,
                    values.mean + values.std,
                    alpha=0.3,
                    label=None,
                )
                ax.legend()
