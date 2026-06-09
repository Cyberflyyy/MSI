"""Wczytywanie i przygotowanie danych."""
from __future__ import annotations

import pandas as pd

from src import config


def load_train(sample: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(config.TRAIN_CSV)
    if sample is not None:
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    return df


def load_test(sample: int | None = None) -> pd.DataFrame:
    text_df = pd.read_csv(config.TEST_CSV)
    labels_df = pd.read_csv(config.TEST_LABELS_CSV)
    df = text_df.merge(labels_df, on="id")
    # Jigsaw: wiersze z etykietą -1 to dane bez oceny — pomijamy.
    df = df[df[config.BINARY_LABEL] != -1].reset_index(drop=True)
    if sample is not None:
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    return df


def get_xy(df: pd.DataFrame) -> tuple:
    """Zwraca (X_text, y) dla klasyfikacji binarnej (kolumna BINARY_LABEL)."""
    X = df["comment_text"].astype(str).tolist()
    y = df[config.BINARY_LABEL].to_numpy()
    return X, y
