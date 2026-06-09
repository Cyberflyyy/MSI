"""Konfiguracja projektu: ścieżki i stałe."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_LABELS_CSV = DATA_DIR / "test_labels.csv"

BINARY_LABEL = "toxic"

VECTORIZER_PARAMS = {
    "ngram_range": (1, 1),
    "min_df": 3,
    "max_features": 5_000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}

RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
