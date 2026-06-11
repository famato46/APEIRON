import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import glob

SOGLIA_DRITTO = 0.05
KEEP_DRITTO = 0.30

SOGLIA_DOLCE = 0.10
KEEP_DOLCE = 0.70

SOGLIA_MEDIA = 0.30
DUP_MEDIA = 2

DUP_FORTE = 3

def balance_by_steer(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    abs_steer = df['steer'].abs()

    mask_dritto = abs_steer < SOGLIA_DRITTO
    mask_dolce = (abs_steer >= SOGLIA_DRITTO) & (abs_steer < SOGLIA_DOLCE)
    mask_normale = (abs_steer >= SOGLIA_DOLCE) & (abs_steer < SOGLIA_MEDIA)
    mask_media = (abs_steer >= SOGLIA_MEDIA) & (abs_steer < 0.6)
    mask_forte = abs_steer >= 0.6

    parts = []

    dritto = df[mask_dritto]
    n_keep = int(len(dritto) * KEEP_DRITTO)
    idx = rng.choice(len(dritto), n_keep, replace=False)
    parts.append(dritto.iloc[idx])

    dolce = df[mask_dolce]
    n_keep = int(len(dolce) * KEEP_DOLCE)
    idx = rng.choice(len(dolce), n_keep, replace=False)
    parts.append(dolce.iloc[idx])

    parts.append(df[mask_normale])

    media = df[mask_media]
    for _ in range(DUP_MEDIA):
        parts.append(media)

    forte = df[mask_forte]
    for _ in range(DUP_FORTE):
        parts.append(forte)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def print_steer_distribution(df: pd.DataFrame, label: str = ""):
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
    parser = argparse.ArgumentParser(description="Bilanciamento dataset multi-file")
    parser.add_argument('inputs', nargs='+', help='File CSV in input (supporta espressioni come data/*.csv)')
    parser.add_argument('-o', '--output', default='dataset_balanced.csv', help='CSV in output')
    parser.add_argument('--seed', type=int, default=42, help='Seed per random (default: 42)')
    args = parser.parse_args()

    all_files = []
    for path in args.inputs:
        all_files.extend(glob.glob(path))
    
    if not all_files:
        print("ERRORE: Nessun file trovato corrispondente agli input specificati.")
        sys.exit(1)

    print(f"Trovati {len(all_files)} file da processare. Inizio caricamento...\n")
    
    dfs = []
    righe_totali = 0
    
    for f in all_files:
        try:
            df_temp = pd.read_csv(f)
            righe = len(df_temp)
            dfs.append(df_temp)
            righe_totali += righe
            print(f"  [OK] {f:30s} -> {righe:6d} sample")
        except Exception as e:
            print(f"  [ERROR] Impossibile leggere {f}: {e}")
            
    if not dfs:
        print("ERRORE CRITICO: Nessun dato caricato. Esco.")
        sys.exit(1)

    print("\nConcatenazione dei dataset in corso...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Dataset combinato: {len(df)} sample totali.")

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