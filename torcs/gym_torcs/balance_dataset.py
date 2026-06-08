"""
Bilanciamento del dataset per training MLP (Imitation Learning).

PROBLEMA: il dataset prodotto da filter_dataset.py e' fortemente sbilanciato:
~55% dei sample ha sterzo praticamente zero (rettilinei). Un MLP addestrato
con loss MSE tenderebbe a predire sempre 0 per minimizzare l'errore medio,
fallendo proprio nelle curve dove servirebbe sterzare.

SOLUZIONE: ridistribuiamo i sample applicando in cascata:
  1. SUBSAMPLING dei rettilinei (sample con |steer| piccolo)
  2. OVERSAMPLING delle curve (sample con |steer| medio/forte)
  3. Stratificazione finale per controllare la distribuzione

Uso:
    python balance_dataset.py dataset_clean.csv -o dataset_balanced.csv
    python balance_dataset.py dataset_clean.csv -o out.csv --plot

L'output mantiene la stessa struttura del CSV in input (stesse colonne).
Solo il numero di righe cambia.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

# Parametri di bilanciamento
# Subsampling rettilinei: |steer| < SOGLIA_DRITTO -> tieni solo KEEP_DRITTO
SOGLIA_DRITTO = 0.05
KEEP_DRITTO = 0.30          # tieni solo il 30% (~scarta 70%)

# Subsampling intermedi (curve dolci, gia' abbastanza rappresentate)
SOGLIA_DOLCE = 0.10
KEEP_DOLCE = 0.70           # tieni il 70%

# Oversampling curve medie
SOGLIA_MEDIA = 0.30
DUP_MEDIA = 2               # duplica x2

# Oversampling curve forti (rare e cruciali)
DUP_FORTE = 3               # duplica x3 i sample con |steer| >= SOGLIA_MEDIA


def balance_by_steer(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Applica subsampling/oversampling in base al valore di 'steer'.

    Bins:
      - |steer| < SOGLIA_DRITTO       --> KEEP_DRITTO
      - |steer| in [DRITTO, DOLCE)    --> KEEP_DOLCE
      - |steer| in [DOLCE, MEDIA)     --> tieni tutto (no resampling)
      - |steer| >= MEDIA              --> oversampling DUP_MEDIA o DUP_FORTE
    """
    rng = np.random.default_rng(seed)
    abs_steer = df['steer'].abs()

    # Maschere
    mask_dritto = abs_steer < SOGLIA_DRITTO
    mask_dolce = (abs_steer >= SOGLIA_DRITTO) & (abs_steer < SOGLIA_DOLCE)
    mask_normale = (abs_steer >= SOGLIA_DOLCE) & (abs_steer < SOGLIA_MEDIA)
    mask_media = (abs_steer >= SOGLIA_MEDIA) & (abs_steer < 0.6)
    mask_forte = abs_steer >= 0.6

    parts = []

    # 1. Rettilinei: subsampling al KEEP_DRITTO
    dritto = df[mask_dritto]
    n_keep = int(len(dritto) * KEEP_DRITTO)
    idx = rng.choice(len(dritto), n_keep, replace=False)
    parts.append(dritto.iloc[idx])

    # 2. Curve dolci: subsampling al KEEP_DOLCE
    dolce = df[mask_dolce]
    n_keep = int(len(dolce) * KEEP_DOLCE)
    idx = rng.choice(len(dolce), n_keep, replace=False)
    parts.append(dolce.iloc[idx])

    # 3. Curve normali: tieni tutto
    parts.append(df[mask_normale])

    # 4. Curve medie: oversampling DUP_MEDIA
    media = df[mask_media]
    for _ in range(DUP_MEDIA):
        parts.append(media)

    # 5. Curve forti: oversampling DUP_FORTE
    forte = df[mask_forte]
    for _ in range(DUP_FORTE):
        parts.append(forte)

    out = pd.concat(parts, ignore_index=True)

    # Shuffle finale (importante per il training)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def print_steer_distribution(df: pd.DataFrame, label: str = ""):
    """Stampa la distribuzione dello sterzo a schermo (istogramma testuale)."""
    bins = [-1.01, -0.6, -0.3, -0.10, -0.05, 0.05, 0.10, 0.30, 0.6, 1.01]
    labels = ['<-0.6', '-0.6..-0.3', '-0.3..-0.1', '-0.1..-0.05',
              '-0.05..0.05', '0.05..0.1', '0.1..0.3', '0.3..0.6', '>0.6']
    hist, _ = np.histogram(df['steer'], bins=bins)
    print(f"\n=== Distribuzione steer{(' ' + label) if label else ''} ===")
    print(f"Totale sample: {len(df)}")
    maxv = max(hist) if max(hist) > 0 else 1
    for lbl, count in zip(labels, hist):
        pct = 100 * count / len(df)
        bar = '#' * int(50 * count / maxv)
        print(f"  {lbl:>14}: {count:6d} ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(description="Bilanciamento dataset per MLP")
    parser.add_argument('input', help='CSV in input (output di filter_dataset.py)')
    parser.add_argument('-o', '--output', default='dataset_balanced.csv',
                        help='CSV in output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed per random subsampling (default: 42)')
    parser.add_argument('--plot', action='store_true',
                        help='Mostra istogramma prima/dopo')
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"ERROR: {args.input} non trovato")
        sys.exit(1)

    print(f"Leggo {args.input}...")
    df = pd.read_csv(p)
    print(f"Sample input: {len(df)}")

    print_steer_distribution(df, "PRIMA")

    df_bal = balance_by_steer(df, seed=args.seed)

    print_steer_distribution(df_bal, "DOPO")

    df_bal.to_csv(args.output, index=False)

    print(f"\n=== SALVATO ===")
    print(f"  File: {args.output}")
    print(f"  Sample finali: {len(df_bal)}")
    print(f"  Cambio: {len(df)} -> {len(df_bal)} ({100*(len(df_bal)-len(df))/len(df):+.0f}%)")


if __name__ == '__main__':
    main()
