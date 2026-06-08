"""
Script di pulizia dei CSV TORCS (nuovo formato: 1 file = 1 giro).

DIFFERENZE rispetto al formato precedente:
  - Ogni file = un singolo giro (nome: lap_NNN_time_MM-SS-mmm_*.csv)
  - Nessuna colonna 'is_clean' o 'is_manual_override': il sample-quality
    viene derivato qui sample-per-sample.
  - 'timestamp' (sessione) invece di 'cur_lap_time' (giro)
  - Azioni: 'target_steer/accel/brake/gear' invece di 'steer/accel/brake/gear_out'
  - Stato marcia: 'gear' invece di 'gear_in'

Uso:
    python filter_dataset.py "lap_*.csv" -o dataset_filtered.csv
    python filter_dataset.py file1.csv file2.csv ... -o output.csv --only-good

Pipeline per ogni file (= un giro):
  1. Calcola lap_time dalla durata di 'timestamp'
  2. Calcola is_clean per ogni riga (warmup, velocita, trackPos, angle, track sensors)
  3. Calcola le metriche del giro e classifica BUONO / DECENTE / SCARTATO
  4. Da giri tenuti, esporta solo i sample is_clean=1
  5. Concatena su un unico CSV finale.

L'output mantiene le colonne originali + una colonna aggiunta 'is_clean'
per coerenza con la pipeline (balance_dataset, analyze_features).
"""

import pandas as pd
import numpy as np
import argparse
import sys
import glob
import re
from pathlib import Path

# Soglie BUONO
LAP_TIME_MAX = 115.0
CLEAN_PCT_MIN = 92.0
SPEED_MAX_MIN = 215.0

# Soglie DISASTRO
DISASTER_LAP_TIME = 125.0
DISASTER_CLEAN_PCT = 75.0
DISASTER_V_MAX = 180.0

# Frozen detection
FROZEN_PCT_MAX = 10.0

# Sample-quality (ex is_clean del bot)
QUALITY_MIN_SPEED = 5.0          # km/h
QUALITY_MAX_TRACKPOS = 0.85
QUALITY_MAX_ANGLE = 0.35         # rad
WARMUP_SECONDS = 1.0             # primo secondo del giro escluso

# Colonne sensori track
TRACK_COLS = [f'track_{i}' for i in range(19)]


def compute_is_clean(df: pd.DataFrame) -> pd.Series:
    """
    Deriva un flag is_clean per ogni sample.

    Scarta:
      - warmup: primi WARMUP_SECONDS dall'inizio del file
      - speedX < QUALITY_MIN_SPEED
      - |trackPos| > QUALITY_MAX_TRACKPOS
      - |angle|    > QUALITY_MAX_ANGLE
      - qualunque sensore track con valore negativo (auto fuori pista)
    """
    t0 = df['timestamp'].iloc[0]
    not_warmup = (df['timestamp'] - t0) > WARMUP_SECONDS
    speed_ok = df['speedX'] >= QUALITY_MIN_SPEED
    tp_ok = df['trackPos'].abs() <= QUALITY_MAX_TRACKPOS
    ang_ok = df['angle'].abs() <= QUALITY_MAX_ANGLE
    track_ok = (df[TRACK_COLS] >= 0).all(axis=1)
    return (not_warmup & speed_ok & tp_ok & ang_ok & track_ok).astype(int)


def parse_lap_time_from_filename(path: Path):
    """
    Estrae il tempo dal nome file: lap_001_time_01-08-010_*.csv -> 68.010s
    Ritorna None se il pattern non matcha.
    """
    m = re.search(r'time_(\d+)-(\d+)-(\d+)', path.name)
    if not m:
        return None
    minutes, seconds, millis = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return minutes * 60 + seconds + millis / 1000.0


def evaluate_lap(df: pd.DataFrame, path: Path) -> dict:
    """Calcola le metriche del giro (l'intero file = un giro)."""
    # Preferisci il tempo dal nome file (preciso); fallback alla durata timestamp
    lap_time = parse_lap_time_from_filename(path)
    if lap_time is None:
        lap_time = float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])

    frozen_count = 0
    if len(df) > 10:
        same = (df['speedX'].diff().abs() < 0.001) & \
               (df['trackPos'].diff().abs() < 0.0001)
        frozen_count = int(same.sum())

    return {
        'n_steps': len(df),
        'lap_time': lap_time,
        'clean_pct': 100 * df['is_clean'].mean(),
        'v_max': df['speedX'].max(),
        'v_mean': df['speedX'].mean(),
        'frozen_pct': 100 * frozen_count / max(len(df), 1),
    }


def classify_lap(m: dict):
    """Ritorna ('BUONO'|'DECENTE'|'SCARTATO', motivo)."""
    if m['n_steps'] < 1000:
        return 'SCARTATO', "troppo corto (probabilmente incompleto)"
    if m['frozen_pct'] > FROZEN_PCT_MAX:
        return 'SCARTATO', f"troppi frame congelati ({m['frozen_pct']:.1f}%)"
    if m['lap_time'] > DISASTER_LAP_TIME:
        return 'SCARTATO', f"giro disastroso ({m['lap_time']:.1f}s > {DISASTER_LAP_TIME}s)"
    if m['clean_pct'] < DISASTER_CLEAN_PCT:
        return 'SCARTATO', f"sample sporchi disastrosi ({m['clean_pct']:.1f}% < {DISASTER_CLEAN_PCT}%)"
    if m['v_max'] < DISASTER_V_MAX:
        return 'SCARTATO', f"velocita troppo bassa ({m['v_max']:.0f} < {DISASTER_V_MAX})"

    if (m['lap_time'] <= LAP_TIME_MAX
            and m['clean_pct'] >= CLEAN_PCT_MIN
            and m['v_max'] >= SPEED_MAX_MIN):
        return 'BUONO', "ok"

    reasons = []
    if m['lap_time'] > LAP_TIME_MAX:
        reasons.append(f"tempo {m['lap_time']:.1f}s")
    if m['clean_pct'] < CLEAN_PCT_MIN:
        reasons.append(f"clean {m['clean_pct']:.1f}%")
    if m['v_max'] < SPEED_MAX_MIN:
        reasons.append(f"v_max {m['v_max']:.0f}")
    return 'DECENTE', "sub-ottimale (" + ", ".join(reasons) + ") ma sample tenuti"


def process_file(path: Path):
    df = pd.read_csv(path)
    df['is_clean'] = compute_is_clean(df)
    m = evaluate_lap(df, path)
    cat, reason = classify_lap(m)
    print(f"  {path.name}: {m['lap_time']:6.2f}s  v_max={m['v_max']:5.0f}  "
          f"clean={m['clean_pct']:5.1f}%  frozen={m['frozen_pct']:5.1f}%  "
          f"step={m['n_steps']:5d}  [{cat:8s}] {reason}")
    if cat == 'BUONO':
        return df[df['is_clean'] == 1], None
    if cat == 'DECENTE':
        return None, df[df['is_clean'] == 1]
    return None, None


def expand_inputs(inputs):
    """Espande wildcard (Windows compat)."""
    out = []
    for inp in inputs:
        if any(c in inp for c in '*?['):
            matches = sorted(glob.glob(inp))
            if matches:
                out.extend(matches)
            else:
                print(f"WARN: nessun file corrisponde al pattern '{inp}'")
        else:
            out.append(inp)
    return out


def main():
    parser = argparse.ArgumentParser(description="Filtro CSV dataset TORCS (nuovo formato lap_*)")
    parser.add_argument('inputs', nargs='+', help='CSV in input (uno per giro)')
    parser.add_argument('-o', '--output', default='dataset_filtered.csv')
    parser.add_argument('--only-good', action='store_true',
                        help='Tieni solo giri BUONI (modalita stretta)')
    args = parser.parse_args()

    inputs = expand_inputs(args.inputs)

    all_buoni, all_decenti = [], []
    print("\n--- Analisi giri ---")
    for inp in inputs:
        p = Path(inp)
        if not p.exists():
            print(f"WARN: {inp} non trovato")
            continue
        df_b, df_d = process_file(p)
        if df_b is not None: all_buoni.append(df_b)
        if df_d is not None: all_decenti.append(df_d)

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
    print(f"Marcia usata: {dict(merged['gear'].value_counts().sort_index())}")


if __name__ == '__main__':
    main()