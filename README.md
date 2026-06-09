# Wykrywanie toksycznych treści — porównanie klasycznych metod klasyfikacji

Projekt na przedmiot **MSI (ścieżka aplikacyjna)**. Porównujemy klasyczne (bez sieci
neuronowych) metody klasyfikacji toksycznych komentarzy internetowych na zbiorze
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).

Problem jest **wieloetykietowy** (6 etykiet) i silnie **niezbalansowany**.

## Element pracy własnej
1. **Własna implementacja Multinomial Naive Bayes** (`src/naive_bayes.py`, NumPy + scipy),
   zweryfikowana względem `scikit-learn` (skrypt `04`).
2. **Eksperymenty z niezbalansowaniem klas**: ważenie klas, under/oversampling oraz dobór
   progu decyzyjnego (`src/imbalance.py`, skrypt `03`).

## Instalacja
```bash
pip install -r requirements.txt
```

## Dane
Pobierz dane z Kaggle (wymaga konta i akceptacji regulaminu konkursu) i umieść pliki
`train.csv`, `test.csv`, `test_labels.csv` w katalogu `data/`.

Przez API Kaggle:
```bash
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p data/
# następnie rozpakuj archiwum do data/
```

## Uruchomienie eksperymentów
Każdy skrypt zapisuje wykresy do `outputs/figures/` i tabele do `outputs/tables/`.
Opcjonalny `--sample N` ogranicza dane (szybki smoke test).

```bash
python scripts/01_eda.py                 # analiza eksploracyjna
python scripts/02_compare_models.py      # porównanie modeli (główny eksperyment)
python scripts/03_imbalance.py           # obsługa niezbalansowania
python scripts/04_verify_own_nb.py       # weryfikacja własnego Naive Bayes
```

Szybki sprawdzian poprawności na podpróbce:
```bash
python scripts/04_verify_own_nb.py --sample 5000
```

## Struktura
```
src/        # kod: dane, modele, metryki, wykresy
scripts/    # skrypty eksperymentów (01–04)
outputs/    # wygenerowane wykresy (PNG) i tabele (CSV)
poster/     # szkic treści plakatu (poster_draft.md)
data/       # pliki CSV z Kaggle (nie w repozytorium)
```

## Protokół eksperymentalny (skrót)
- **Reprezentacja:** TF-IDF (słowa + bigramy, `min_df=5`, `max_features=50000`).
- **Modele:** klasyfikator większościowy (baseline), Multinomial NB (sklearn),
  Multinomial NB (własny), regresja logistyczna. Tryb One-vs-Rest (osobny model na etykietę).
- **Ewaluacja:** trening na `train.csv`, test na oficjalnym `test.csv`/`test_labels.csv`
  (z odfiltrowaniem wierszy `-1`).
- **Metryki:** średnie kolumnowe ROC AUC (metryka konkursowa) + PR-AUC, F1, precyzja, czułość.
# MSI
