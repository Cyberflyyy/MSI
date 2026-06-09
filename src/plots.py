"""Wizualizacje wyników."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src import config


def plot_metrics_bars(scores: dict, filename: str) -> None:
    """
    scores: {model_name: {metric: [fold_scores]}}
    Rysuje 4 wykresy słupkowe (jeden per metryka) z error barami (std).
    """
    metrics = list(next(iter(scores.values())).keys())
    model_names = list(scores.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))

    for ax, metric in zip(axes, metrics):
        means = [np.mean(scores[m][metric]) for m in model_names]
        stds  = [np.std(scores[m][metric])  for m in model_names]
        bars = ax.bar(model_names, means, yerr=stds, capsize=5)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=6)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Porównanie modeli (CV mean ± std)", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{filename}.png", dpi=150)
    plt.close(fig)
    print(f"Wykres: {config.FIGURES_DIR / filename}.png")
