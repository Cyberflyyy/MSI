"""Porównanie sklearn KNN, własnego KNN i Gaussian NB.

Walidacja: RepeatedKFold (2 powtórzenia × 5 splitów = 10 foldów).
Metryki: Balanced Accuracy, F1, Precision, Recall.
Analiza statystyczna: Shapiro-Wilk na różnicach → t-test sparowany lub Wilcoxon.

Uruchomienie:
    python scripts/compare_knn.py [--sample N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import shapiro, ttest_rel, wilcoxon

from src import config, data
from src.knn import KNearestNeighbors

DEFAULT_SAMPLE = 500
K = 5
N_SPLITS = 5
N_REPEATS = 2
ALPHA = 0.05

METRICS = ["balanced_accuracy", "f1", "precision", "recall"]
METRIC_LABELS = {
    "balanced_accuracy": "Balanced Accuracy",
    "f1":                "F1",
    "precision":         "Precision",
    "recall":            "Recall",
}


def stat_test(a, b):
    """Shapiro-Wilk na różnicach → t-test lub Wilcoxon. Zwraca (test_name, stat, p)."""
    diff = np.array(a) - np.array(b)
    if np.std(diff) == 0:
        return "brak (identyczne)", float("nan"), float("nan")
    _, p_sw = shapiro(diff)
    if p_sw > ALPHA:
        stat, p = ttest_rel(a, b)
        return "t-test", stat, p
    else:
        stat, p = wilcoxon(a, b)
        return "Wilcoxon", stat, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    config.ensure_output_dirs()

    print(f"Wczytywanie danych (sample={args.sample})...")
    df = data.load_train(sample=args.sample)
    X_text, y = data.get_xy(df)
    X_text = list(X_text)

    clfs = [
        KNeighborsClassifier(n_neighbors=K, metric="euclidean", algorithm="brute"),
        KNearestNeighbors(k=K),
        GaussianNB(),
    ]
    names = ["KNN sklearn", "KNN własny", "Gaussian NB"]

    kf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=config.RANDOM_STATE)
    n_folds = kf.get_n_splits()
    scores = {m: np.full((len(clfs), n_folds), np.nan) for m in METRICS}

    for i, (train_idx, test_idx) in enumerate(kf.split(X_text, y)):
        print(f"Fold {i + 1}/{n_folds}...")
        X_train_text = [X_text[j] for j in train_idx]
        X_test_text  = [X_text[j] for j in test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        vec = TfidfVectorizer(**config.VECTORIZER_PARAMS)
        X_train = vec.fit_transform(X_train_text)
        X_test  = vec.transform(X_test_text)

        for c, clf in enumerate(clfs):
            clf_c = clone(clf)
            if isinstance(clf_c, GaussianNB):
                clf_c.fit(X_train.toarray(), y_train)
                preds = clf_c.predict(X_test.toarray())
            else:
                clf_c.fit(X_train, y_train)
                preds = clf_c.predict(X_test)
            scores["balanced_accuracy"][c, i] = balanced_accuracy_score(y_test, preds)
            scores["f1"][c, i]                = f1_score(y_test, preds, zero_division=0)
            scores["precision"][c, i]         = precision_score(y_test, preds, zero_division=0)
            scores["recall"][c, i]            = recall_score(y_test, preds, zero_division=0)

    # ── Wyniki średnie ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WYNIKI (mean ± std po foldach):")
    print("=" * 60)
    header = f"{'Model':<16}" + "".join(f"{METRIC_LABELS[m]:>20}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for c, name in enumerate(names):
        row = f"{name:<16}"
        for m in METRICS:
            mean = np.mean(scores[m][c])
            std  = np.std(scores[m][c])
            row += f"  {mean:.3f} ± {std:.3f}    "
        print(row)

    # ── Analiza statystyczna dla każdej metryki ───────────────────────────────
    print("\n" + "=" * 60)
    print("ANALIZA STATYSTYCZNA (alfa = 0.05):")
    print("=" * 60)

    n = len(clfs)
    for metric in METRICS:
        means = np.mean(scores[metric], axis=1)
        print(f"\n--- {METRIC_LABELS[metric]} ---")
        for i in range(n):
            for j in range(n):
                if i >= j:
                    continue
                test_name, stat, p = stat_test(scores[metric][i], scores[metric][j])
                if np.isnan(p):
                    print(f"  {names[i]} vs {names[j]}: brak różnicy (identyczne wyniki)")
                    continue
                sig = p < ALPHA
                better = i if means[i] > means[j] else j
                worse  = j if better == i else i
                if sig:
                    print(
                        f"  {names[better]} ({means[better]:.3f}) jest statystycznie znacząco "
                        f"lepszy niż {names[worse]} ({means[worse]:.3f})  "
                        f"[{test_name}: stat={stat:.3f}, p={p:.4f}]"
                    )
                else:
                    print(
                        f"  {names[i]} vs {names[j]}: brak różnicy istotnej statystycznie  "
                        f"[{test_name}: stat={stat:.3f}, p={p:.4f}]"
                    )

    # ── Wykresy słupkowe ──────────────────────────────────────────────────────
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    fig.suptitle(f"Porównanie klasyfikatorów (sample={args.sample}, {n_folds} foldów)", fontsize=11)

    for ax, metric in zip(axes, METRICS):
        m_means = np.mean(scores[metric], axis=1)
        m_stds  = np.std(scores[metric], axis=1)
        bars = ax.bar(names, m_means, yerr=m_stds, capsize=6, color=colors)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, m_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_fig = config.FIGURES_DIR / "comparison.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"\nWykres zapisany: {out_fig}")


if __name__ == "__main__":
    main()
