"""Metryki ewaluacji i testy statystyczne."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

METRICS = ["Balanced Accuracy", "Precision", "Recall", "F1"]


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":                float(f1_score(y_true, y_pred, zero_division=0)),
    }


def statistical_test(scores_a: list, scores_b: list) -> dict:
    """
    Porównuje dwa modele na wynikach z foldów CV.
    1. Shapiro-Wilk na różnicach → sprawdza normalność
    2. Jeśli p_shapiro > 0.05: sparowany t-test
       Jeśli p_shapiro <= 0.05: Wilcoxon signed-rank
    Zwraca słownik z wynikami.
    """
    diff = np.array(scores_a) - np.array(scores_b)

    # Shapiro-Wilk wymaga min 3 obserwacji i nie wszystkie identyczne
    if len(diff) >= 3 and np.std(diff) > 0:
        stat_sw, p_sw = stats.shapiro(diff)
    else:
        stat_sw, p_sw = float("nan"), float("nan")

    normal = (not np.isnan(p_sw)) and (p_sw > 0.05)

    if np.std(diff) == 0:
        # Identyczne wyniki — brak różnicy
        return {
            "test": "brak (identyczne wyniki)",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "shapiro_p": p_sw,
            "normal": normal,
            "significant": False,
        }

    if normal:
        stat, p = stats.ttest_rel(scores_a, scores_b)
        test_name = "t-test (sparowany)"
    else:
        stat, p = stats.wilcoxon(scores_a, scores_b)
        test_name = "Wilcoxon"

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "shapiro_p": float(p_sw),
        "normal": normal,
        "significant": float(p) < 0.05,
    }
