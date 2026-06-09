"""Weryfikacja zgodności własnego KNN ze sklearn KNN.

Porównuje predykcje obu implementacji na identycznej próbce danych
i oblicza procent zgodności. Oczekiwane: ≥ 99%.

Uruchomienie:
    python scripts/verify_knn.py [--sample N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from src import config, data
from src.knn import KNearestNeighbors

DEFAULT_SAMPLE = 2_000
K = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    config.ensure_output_dirs()

    print(f"Wczytywanie danych (sample={args.sample})...")
    train_df = data.load_train(sample=args.sample)
    test_df = data.load_test(sample=500)
    X_train_text, y_train = data.get_xy(train_df)
    X_test_text, _ = data.get_xy(test_df)

    vec = TfidfVectorizer(**config.VECTORIZER_PARAMS)
    X_train = vec.fit_transform(X_train_text)
    X_test = vec.transform(X_test_text)

    sklearn_knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean", algorithm="brute")
    own_knn = KNearestNeighbors(k=K)

    sklearn_knn.fit(X_train, y_train)
    own_knn.fit(X_train, y_train)

    pred_sklearn = sklearn_knn.predict(X_test)
    pred_own = own_knn.predict(X_test)

    agreement = float(np.mean(pred_sklearn == pred_own)) * 100
    print(f"\nZgodność predykcji: {agreement:.2f}%")

    # Wykres porównania prawdopodobieństw.
    proba_sklearn = sklearn_knn.predict_proba(X_test)[:, 1]
    proba_own = own_knn.predict_proba(X_test)[:, 1]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(proba_sklearn, proba_own, alpha=0.4, s=10)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("sklearn KNN — P(toxic)")
    ax.set_ylabel("Własny KNN — P(toxic)")
    ax.set_title(f"Zgodność predykcji: {agreement:.1f}%")
    fig.tight_layout()
    out_fig = config.FIGURES_DIR / "knn_agreement.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Wykres: {out_fig}")


if __name__ == "__main__":
    main()
