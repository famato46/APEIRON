"""
Script di pulizia dei CSV raccolti dal bot TORCS.

QUESTO SCRIPT NON va integrato nel bot.
La raccolta deve registrare TUTTO (anche sample sporchi, anche giri brutti)
con il flag is_clean a marcare la qualita'. Il filtraggio si fa qui, in una
fase separata, perche':
  - le soglie sono euristiche e vanno regolate dopo l'osservazione dei dati
  - i sample "scartati oggi" potrebbero servire domani (es. recovery per RL)
  - raccolta e pulizia sono fasi distinte: tieni i CSV grezzi come backup

Uso:
    python filter_dataset.py file1.csv file2.csv ... -o output.csv
    python filter_dataset.py file1.csv file2.csv ... -o output.csv --only-good

Per ogni CSV:
  1. Identifica i giri (basato sui reset di cur_lap_time)
  2. Classifica ogni giro in tre categorie:
       - BUONO:    tempo<LAP_TIME_MAX, puliti>=CLEAN_PCT_MIN, v_max>=SPEED_MAX_MIN
                   --> tieni tutti i sample con is_clean=1
       - DECENTE:  giro completato ma sotto soglia (es. piu' lento o con piu'
                   rumore). NON frozen, NON disastroso.
                   --> tieni i sample con is_clean=1 (sono comunque guida
                       valida, anche se sub-ottimale)
       - SCARTATO: frozen (fine quick race), disastroso, incompleto
                   --> butta tutto
  3. Concatena tutto in un unico CSV finale

Flag opzionale --only-good per usare solo giri BUONI (modalita' stretta).
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

# Soglie BUONO: requisiti rigorosi per giri ottimali
LAP_TIME_MAX = 115.0
CLEAN_PCT_MIN = 92.0
SPEED_MAX_MIN = 215.0

# Soglie DISASTRO: oltre cui buttiamo via tutto
DISASTER_LAP_TIME = 125.0
DISASTER_CLEAN_PCT = 75.0
DISASTER_V_MAX = 180.0

# Frozen detection (TORCS bloccato a fine quick race)
FROZEN_PCT_MAX = 10.0


def split_into_laps(df: pd.DataFrame):
    """Divide il DataFrame in liste di DataFrame, uno per giro."""
    lap_resets = []
    prev_t = -999
    for i, t in enumerate(df['cur_lap_time'].values):
        if t < prev_t - 5:
            lap_resets.append(i)
        prev_t = t

    laps = []
    prev = 0
    for r in lap_resets:
        laps.append(df.iloc[prev:r].copy())
        prev = r
    laps.append(df.iloc[prev:].copy())
    return laps


def evaluate_lap(lap_df: pd.DataFrame) -> dict:
    """Calcola le metriche di un giro."""
    frozen_count = 0
    if len(lap_df) > 10:
        same = (lap_df['speedX'].diff().abs() < 0.001) & \
               (lap_df['trackPos'].diff().abs() < 0.0001)
        frozen_count = int(same.sum())

    return {
        'n_steps': len(lap_df),
        'lap_time': lap_df['cur_lap_time'].max(),
        'clean_pct': 100 * lap_df['is_clean'].mean(),
        'v_max': lap_df['speedX'].max(),
        'v_mean': lap_df['speedX'].mean(),
        'frozen_pct': 100 * frozen_count / max(len(lap_df), 1),
    }


def classify_lap(metrics: dict):
    """
    Classifica il giro: 'BUONO', 'DECENTE', 'SCARTATO'.
    """
    if metrics['n_steps'] < 1000:
        return 'SCARTATO', "troppo corto (probabilmente incompleto)"
    if metrics['frozen_pct'] > FROZEN_PCT_MAX:
        return 'SCARTATO', f"troppi frame congelati ({metrics['frozen_pct']:.1f}%)"
    if metrics['lap_time'] > DISASTER_LAP_TIME:
        return 'SCARTATO', f"giro disastroso ({metrics['lap_time']:.1f}s > {DISASTER_LAP_TIME}s)"
    if metrics['clean_pct'] < DISASTER_CLEAN_PCT:
        return 'SCARTATO', f"sample sporchi disastrosi ({metrics['clean_pct']:.1f}% < {DISASTER_CLEAN_PCT}%)"
    if metrics['v_max'] < DISASTER_V_MAX:
        return 'SCARTATO', f"velocita troppo bassa ({metrics['v_max']:.0f} < {DISASTER_V_MAX})"

    if (metrics['lap_time'] <= LAP_TIME_MAX
            and metrics['clean_pct'] >= CLEAN_PCT_MIN
            and metrics['v_max'] >= SPEED_MAX_MIN):
        return 'BUONO', "ok"

    reasons = []
    if metrics['lap_time'] > LAP_TIME_MAX:
        reasons.append(f"tempo {metrics['lap_time']:.1f}s")
    if metrics['clean_pct'] < CLEAN_PCT_MIN:
        reasons.append(f"clean {metrics['clean_pct']:.1f}%")
    if metrics['v_max'] < SPEED_MAX_MIN:
        reasons.append(f"v_max {metrics['v_max']:.0f}")
    return 'DECENTE', "sub-ottimale (" + ", ".join(reasons) + ") ma sample tenuti"


def process_file(path: Path):
    df = pd.read_csv(path)
    laps = split_into_laps(df)

    buoni = []
    decenti = []
    print(f"\n--- {path.name} ---")
    for i, lap in enumerate(laps):
        m = evaluate_lap(lap)
        cat, reason = classify_lap(m)
        print(f"  Giro {i}: {m['lap_time']:6.2f}s  v_max={m['v_max']:5.0f}  "
              f"clean={m['clean_pct']:5.1f}%  frozen={m['frozen_pct']:5.1f}%  "
              f"step={m['n_steps']:5d}  [{cat:8s}] {reason}")
        if cat == 'BUONO':
            buoni.append(lap[lap['is_clean'] == 1])
        elif cat == 'DECENTE':
            decenti.append(lap[lap['is_clean'] == 1])

    df_b = pd.concat(buoni, ignore_index=True) if buoni else None
    df_d = pd.concat(decenti, ignore_index=True) if decenti else None
    return df_b, df_d


def main():
    parser = argparse.ArgumentParser(description="Filtro CSV dataset TORCS")
    parser.add_argument('inputs', nargs='+', help='CSV in input')
    parser.add_argument('-o', '--output', default='dataset_filtered.csv',
                        help='CSV in output (default: dataset_filtered.csv)')
    parser.add_argument('--only-good', action='store_true',
                        help='Tieni solo giri BUONI (modalita stretta)')
    args = parser.parse_args()

    all_buoni = []
    all_decenti = []
    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"WARN: {inp} non trovato")
            continue
        df_b, df_d = process_file(p)
        if df_b is not None:
            all_buoni.append(df_b)
        if df_d is not None:
            all_decenti.append(df_d)

    n_b = sum(len(d) for d in all_buoni)
    n_d = sum(len(d) for d in all_decenti)

    if args.only_good:
        if not all_buoni:
            print("\nNESSUN giro BUONO. Usa senza --only-good per includere i DECENTI.")
            sys.exit(1)
        merged = pd.concat(all_buoni, ignore_index=True)
    else:
        all_kept = all_buoni + all_decenti
        if not all_kept:
            print("\nNESSUN sample utilizzabile.")
            sys.exit(1)
        merged = pd.concat(all_kept, ignore_index=True)

    merged.to_csv(args.output, index=False)

    print(f"\n=== RISULTATO FINALE ===")
    print(f"Sample da giri BUONI:   {n_b}")
    print(f"Sample da giri DECENTI: {n_d}")
    print(f"Sample totali in output: {len(merged)}")
    print(f"File: {args.output}")
    print(f"Velocita: mean={merged['speedX'].mean():.1f}, max={merged['speedX'].max():.1f}")
    print(f"Marcia usata: {dict(merged['gear_in'].value_counts().sort_index())}")


if __name__ == '__main__':
    main()
