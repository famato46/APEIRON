"""
grafico_dataset.py
------------------
Confronta medie mobili sullo sterzo grezzo, per scegliere una finestra di
smoothing. Usa la colonna 'target_steer' (nuovo formato lap_*).

Uso:
    python grafico_dataset.py lap_001_time_01-08-010_20260530_010919.csv
    python grafico_dataset.py dataset_clean.csv --col target_steer --n 500
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Confronto medie mobili sterzo")
    parser.add_argument('input', help='CSV in input')
    parser.add_argument('--col', default='target_steer',
                        help='Colonna sterzo (default: target_steer)')
    parser.add_argument('--n', type=int, default=500,
                        help='Numero di step da plottare (default: 500)')
    parser.add_argument('-o', '--output', default=None,
                        help='Se indicato, salva PNG invece di mostrare')
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"ERROR: {args.input} non trovato")
        sys.exit(1)

    df = pd.read_csv(p)
    if args.col not in df.columns:
        print(f"ERROR: colonna '{args.col}' mancante. Colonne: {list(df.columns)}")
        sys.exit(1)

    df_subset = df[args.col].head(args.n)

    plt.figure(figsize=(12, 6))
    plt.plot(df_subset.values, label='Originale (Tastiera)', color='lightgray', alpha=0.7)
    plt.plot(df_subset.rolling(window=5).mean().values, label='Finestra = 5', color='green')
    plt.plot(df_subset.rolling(window=10).mean().values, label='Finestra = 10', color='blue')
    plt.plot(df_subset.rolling(window=20).mean().values, label='Finestra = 20', color='red')

    plt.title(f"Confronto Medie Mobili sullo Sterzo ({args.col})")
    plt.xlabel("Step")
    plt.ylabel(args.col)
    plt.legend()
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=120)
        print(f"Salvato: {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
