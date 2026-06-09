"""Porównanie sklearn KNN, własnego KNN i Gaussian NB.

Walidacja: RepeatedStratifiedKFold (2 powtórzenia × 5 splitów = 10 foldów).
Metryki: Balanced Accuracy, Precision, Recall, F1.
Analiza statystyczna: Shapiro-Wilk → t-test sparowany lub Wilcoxon.

Uruchomienie:
    python scripts/compare_knn.py [--sample N]
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src import config, data, evaluate, plots
from src.knn import KNearestNeighbors

DEFAULT_SAMPLE = 500
K = 5
N_SPLITS = 5
N_REPEATS = 2


class DenseWrapper:
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X.toarray(), y)
        return self

    def predict(self, X):
        return self.clf.predict(X.toarray())


def make_models():
    return {
        "KNN sklearn": KNeighborsClassifier(n_neighbors=K, metric="euclidean", algorithm="brute"),
        "KNN własny":  KNearestNeighbors(k=K),
        "Gaussian NB": DenseWrapper(GaussianNB()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    config.ensure_output_dirs()

    print(f"Wczytywanie danych (sample={args.sample})...")
    df = data.load_train(sample=args.sample)
    X_text, y = data.get_xy(df)
    X_text = list(X_text)

    model_names = list(make_models().keys())
    # scores[model][metric] = lista wyników z foldów
    scores = {name: {m: [] for m in evaluate.METRICS} for name in model_names}

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=config.RANDOM_STATE
    )
    total_folds = N_SPLITS * N_REPEATS

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_text, y)):
        print(f"Fold {fold_idx + 1}/{total_folds}...")

        X_train_text = [X_text[i] for i in train_idx]
        X_test_text  = [X_text[i] for i in test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        vec = TfidfVectorizer(**config.VECTORIZER_PARAMS)
        X_train = vec.fit_transform(X_train_text)
        X_test  = vec.transform(X_test_text)

        for name, model in make_models().items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fold_metrics = evaluate.compute_metrics(y_test, pred)
            for metric, val in fold_metrics.items():
                scores[name][metric].append(val)

    # ── Tabela wyników (mean ± std) ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WYNIKI (mean ± std po foldach):")
    print("=" * 60)
    summary_rows = []
    for name in model_names:
        row = {"Model": name}
        for metric in evaluate.METRICS:
            vals = scores[name][metric]
            row[metric] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).set_index("Model")
    print(df_summary.to_string())
    df_summary.to_csv(config.TABLES_DIR / "comparison.csv", encoding="utf-8-sig")

    # ── Analiza statystyczna ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALIZA STATYSTYCZNA (porównania parami):")
    print("=" * 60)

    pairs = list(itertools.combinations(model_names, 2))
    stat_rows = []

    for metric in evaluate.METRICS:
        print(f"\n-- {metric} --")
        for a, b in pairs:
            result = evaluate.statistical_test(scores[a][metric], scores[b][metric])
            sig = "ISTOTNA" if result["significant"] else "nieistotna"
            print(
                f"  {a} vs {b}:\n"
                f"    Shapiro-Wilk p={result['shapiro_p']:.4f} "
                f"({'normalny' if result['normal'] else 'nienormalny'})\n"
                f"    Test: {result['test']}, statystyka={result['statistic']:.4f}, "
                f"p={result['p_value']:.4f} -> {sig}"
            )
            stat_rows.append({
                "Metryka": metric,
                "Model A": a,
                "Model B": b,
                "Test": result["test"],
                "Statystyka": round(result["statistic"], 4),
                "p-value": round(result["p_value"], 4),
                "Shapiro p": round(result["shapiro_p"], 4),
                "Istotna": result["significant"],
            })

    pd.DataFrame(stat_rows).to_csv(
        config.TABLES_DIR / "statistical_tests.csv", index=False, encoding="utf-8-sig"
    )

    # ── Wykresy ──────────────────────────────────────────────────────────────
    plots.plot_metrics_bars(scores, "comparison")

    print(f"\nTabele: {config.TABLES_DIR}")


if __name__ == "__main__":
    main()
