"""
build_dataset_bc.py
-------------------
Genera il dataset finale per il Behavioral Cloning (Fase 4) a partire dal CSV
ripulito di Fase 2. Esegue:

  1. Drop delle colonne inutili (opponents, sensori a varianza nulla, ecc.).
  2. Selezione delle 10 feature di input + 3 target.
  3. Feature engineering: delta_track = track_18 - track_0.
  4. Split train/val/test 80/10/10 (PRIMA del bilanciamento -> niente data leakage).
       - Train: addestra i pesi
       - Val  : early stopping e tuning iperparametri durante il training
       - Test : valutazione finale, NON TOCCARE fino a fine progetto
  5. Bilanciamento dello sterzo SOLO sul training set
     (subsampling rettilinei + oversampling curve forti).
  6. Normalizzazione StandardScaler fittata SOLO sul training set.
  7. Salvataggio di:
       - dataset_bc.csv               (versione human-readable, non scalata)
       - dataset_bc.npz               (array numpy pronti per il training)
       - scaler.joblib                (StandardScaler salvato per inferenza)
       - feature_config.json          (lista feature, target, parametri)

Uso:
    python build_dataset_bc.py dataset_clean.csv
    python build_dataset_bc.py dataset_clean.csv -o ./out/ --plot
    python build_dataset_bc.py dataset_clean.csv --extra-tracks   # +track_1/17
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================ CONFIG ============================

# Feature di input scelte in Fase 3 (l'ordine viene preservato fino all'inferenza:
# l'agente di guida deve costruire il vettore di stato esattamente in questo ordine).
INPUT_FEATURES_BASE = [
    'speedX',
    'angle',
    'trackPos',
    'dist_from_start',
    'track_0',
    'track_4',
    'track_9',
    'track_14',
    'track_18',
    'delta_track',
]

# Feature opzionali: aggiunte con --extra-tracks
# track_1 e track_17 hanno r=+0.45/-0.46 con steer ma sono parzialmente ridondanti
# con track_0/4/14/18. Da provare in Fase 4 se l'MLP fatica nelle curve di apertura.
INPUT_FEATURES_EXTRA = ['track_1', 'track_17']

TARGETS = ['steer', 'accel', 'brake']

# Parametri di bilanciamento (applicati SOLO al training set)
SOGLIA_DRITTO  = 0.05
KEEP_DRITTO    = 0.30
SOGLIA_DOLCE   = 0.10
KEEP_DOLCE     = 0.70
SOGLIA_MEDIA   = 0.30
SOGLIA_FORTE   = 0.60
DUP_MEDIA      = 2
DUP_FORTE      = 3

# Split: 80% train, 10% val, 10% test
VAL_FRAC      = 0.10
TEST_FRAC     = 0.10
RANDOM_STATE  = 42


# ============================ HELPERS ============================

def stampa_distrib_steer(df: pd.DataFrame, label: str = ""):
    bins = [-1.01, -0.6, -0.3, -0.10, -0.05, 0.05, 0.10, 0.30, 0.6, 1.01]
    labels = ['<-0.6', '-0.6..-0.3', '-0.3..-0.1', '-0.1..-0.05',
              '-0.05..0.05', '0.05..0.1', '0.1..0.3', '0.3..0.6', '>0.6']
    hist, _ = np.histogram(df['steer'], bins=bins)
    print(f"\n--- Distribuzione steer {label} (N={len(df)}) ---")
    maxv = max(hist) if max(hist) > 0 else 1
    for lbl, count in zip(labels, hist):
        pct = 100 * count / max(len(df), 1)
        bar = '#' * int(40 * count / maxv)
        print(f"  {lbl:>14}: {count:6d} ({pct:5.1f}%) {bar}")


def bilancia_steer(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Subsampling rettilinei + oversampling curve. SOLO sul training set."""
    rng = np.random.default_rng(seed)
    a = df['steer'].abs()

    parts = []

    dritto = df[a < SOGLIA_DRITTO]
    if len(dritto) > 0:
        n = max(1, int(len(dritto) * KEEP_DRITTO))
        idx = rng.choice(len(dritto), n, replace=False)
        parts.append(dritto.iloc[idx])

    dolce = df[(a >= SOGLIA_DRITTO) & (a < SOGLIA_DOLCE)]
    if len(dolce) > 0:
        n = max(1, int(len(dolce) * KEEP_DOLCE))
        idx = rng.choice(len(dolce), n, replace=False)
        parts.append(dolce.iloc[idx])

    parts.append(df[(a >= SOGLIA_DOLCE) & (a < SOGLIA_MEDIA)])

    media = df[(a >= SOGLIA_MEDIA) & (a < SOGLIA_FORTE)]
    for _ in range(DUP_MEDIA):
        parts.append(media)

    forte = df[a >= SOGLIA_FORTE]
    for _ in range(DUP_FORTE):
        parts.append(forte)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def split_train_val_test(df, val_frac, test_frac, seed=42):
    """Prima estrae test (mai più toccato), poi divide il resto in train/val."""
    df_trainval, df_test = train_test_split(
        df, test_size=test_frac, random_state=seed, shuffle=True)
    val_size_relative = val_frac / (1.0 - test_frac)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_size_relative, random_state=seed, shuffle=True)
    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


# ============================ MAIN ============================

def main():
    parser = argparse.ArgumentParser(description="Genera il dataset BC finale.")
    parser.add_argument('input', help='CSV in input (es. dataset_clean.csv)')
    parser.add_argument('-o', '--outdir', default='./out_bc',
                        help='Cartella di output (default: ./out_bc)')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disabilita il bilanciamento del training set')
    parser.add_argument('--extra-tracks', action='store_true',
                        help='Aggiunge track_1 e track_17 alle feature di input (12 totali)')
    parser.add_argument('--plot', action='store_true',
                        help='Salva istogrammi PNG prima/dopo')
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} non trovato")
        sys.exit(1)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set feature finale
    if args.extra_tracks:
        input_features = ['speedX','angle','trackPos','dist_from_start',
                          'track_0','track_1','track_4','track_9','track_14',
                          'track_17','track_18','delta_track']
        print(">> Modalita' --extra-tracks: 12 feature di input (aggiunti track_1, track_17)")
    else:
        input_features = list(INPUT_FEATURES_BASE)

    # ------ 1. Caricamento + feature engineering ------
    print(f"[1/7] Carico {in_path}...")
    df = pd.read_csv(in_path)
    print(f"      Shape grezza: {df.shape}")

    for f in input_features:
        if f == 'delta_track':
            continue
        if f not in df.columns:
            print(f"ERROR: colonna mancante: {f}")
            sys.exit(1)

    df['delta_track'] = df['track_18'] - df['track_0']
    print(f"      Aggiunta feature 'delta_track' = track_18 - track_0")

    # ------ 2. Selezione colonne ------
    print(f"[2/7] Seleziono {len(input_features)} feature + {len(TARGETS)} target...")
    df_sel = df[input_features + TARGETS].copy()
    print(f"      Shape selezionata: {df_sel.shape}")

    n_nan = df_sel.isna().sum().sum()
    if n_nan > 0:
        print(f"      ATTENZIONE: {n_nan} NaN trovati, droppo.")
        df_sel = df_sel.dropna().reset_index(drop=True)

    # ------ 3. Split 3-way ------
    print(f"[3/7] Split train/val/test "
          f"({int((1-VAL_FRAC-TEST_FRAC)*100)}/{int(VAL_FRAC*100)}/{int(TEST_FRAC*100)})...")
    df_train, df_val, df_test = split_train_val_test(
        df_sel, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=RANDOM_STATE)
    print(f"      Train: {len(df_train)}   Val: {len(df_val)}   Test: {len(df_test)}")

    # ------ 4. Bilanciamento (solo training) ------
    if not args.no_balance:
        print(f"[4/7] Bilancio il TRAINING set (val e test intatti)...")
        stampa_distrib_steer(df_train, "train PRIMA")
        df_train = bilancia_steer(df_train, seed=RANDOM_STATE)
        stampa_distrib_steer(df_train, "train DOPO")
        print(f"      Train dopo bilanciamento: {len(df_train)}")
    else:
        print(f"[4/7] Bilanciamento DISABILITATO (--no-balance)")

    # ------ 5. Normalizzazione ------
    print(f"[5/7] StandardScaler fittato sul training set...")
    X_train = df_train[input_features].values.astype(np.float32)
    y_train = df_train[TARGETS].values.astype(np.float32)
    X_val   = df_val[input_features].values.astype(np.float32)
    y_val   = df_val[TARGETS].values.astype(np.float32)
    X_test  = df_test[input_features].values.astype(np.float32)
    y_test  = df_test[TARGETS].values.astype(np.float32)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    print(f"      Train scalato:  mean={X_train_s.mean():+.4f}  std={X_train_s.std():.4f}  (atteso 0/1)")
    print(f"      Val   scalato:  mean={X_val_s.mean():+.4f}  std={X_val_s.std():.4f}")
    print(f"      Test  scalato:  mean={X_test_s.mean():+.4f}  std={X_test_s.std():.4f}")

    # ------ 6. CSV human-readable con split ------
    print(f"[6/7] Genero CSV human-readable con flag split...")
    df_tr = df_train.copy(); df_tr['split'] = 'train'
    df_va = df_val.copy();   df_va['split'] = 'val'
    df_te = df_test.copy();  df_te['split'] = 'test'
    df_full = pd.concat([df_tr, df_va, df_te], ignore_index=True)
    csv_path = out_dir / 'dataset_bc.csv'
    df_full.to_csv(csv_path, index=False)
    print(f"      {csv_path}  ({len(df_full)} righe totali)")

    # ------ 7. Salvataggio artefatti per training/inferenza ------
    print(f"[7/7] Salvo artefatti in {out_dir}/")

    npz_path = out_dir / 'dataset_bc.npz'
    np.savez_compressed(
        npz_path,
        X_train=X_train_s, y_train=y_train,
        X_val=X_val_s,     y_val=y_val,
        X_test=X_test_s,   y_test=y_test,
        feature_names=np.array(input_features),
        target_names =np.array(TARGETS),
    )
    print(f"      {npz_path}")

    scaler_path = out_dir / 'scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"      {scaler_path}   <-- CARICARE IN INFERENZA")

    config = {
        'input_features': input_features,
        'targets': TARGETS,
        'splits': {
            'train_frac': 1 - VAL_FRAC - TEST_FRAC,
            'val_frac':   VAL_FRAC,
            'test_frac':  TEST_FRAC,
        },
        'random_state': RANDOM_STATE,
        'extra_tracks': args.extra_tracks,
        'balancing': None if args.no_balance else {
            'soglia_dritto': SOGLIA_DRITTO, 'keep_dritto': KEEP_DRITTO,
            'soglia_dolce':  SOGLIA_DOLCE,  'keep_dolce':  KEEP_DOLCE,
            'soglia_media':  SOGLIA_MEDIA,  'dup_media':   DUP_MEDIA,
            'soglia_forte':  SOGLIA_FORTE,  'dup_forte':   DUP_FORTE,
        },
        'n_train': int(len(df_train)),
        'n_val':   int(len(df_val)),
        'n_test':  int(len(df_test)),
    }
    cfg_path = out_dir / 'feature_config.json'
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"      {cfg_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
            for a, d, title, color in zip(
                ax,
                [df_sel, df_train, df_test],
                ['steer — dataset completo', 'steer — train (bilanciato)', 'steer — test (intatto)'],
                ['gray','steelblue','darkorange']):
                a.hist(d['steer'], bins=80, color=color)
                a.set_yscale('log'); a.set_title(title)
                a.axvline(-SOGLIA_DRITTO, color='r', ls='--', lw=1)
                a.axvline( SOGLIA_DRITTO, color='r', ls='--', lw=1)
            fig.tight_layout()
            plot_path = out_dir / 'steer_distribution.png'
            fig.savefig(plot_path, dpi=120)
            print(f"      {plot_path}")
        except ImportError:
            print("      (matplotlib non disponibile, salto il plot)")

    print(f"\n=== FATTO ===")
    print(f"In Fase 4 (training MLP) carica con:")
    print(f"  import numpy as np, joblib")
    print(f"  data   = np.load('{npz_path}')")
    print(f"  scaler = joblib.load('{scaler_path}')")
    print(f"  X_train, y_train = data['X_train'], data['y_train']")
    print(f"  X_val,   y_val   = data['X_val'],   data['y_val']")
    print(f"  X_test,  y_test  = data['X_test'],  data['y_test']   # NON usare per tuning")


if __name__ == '__main__':
    main()
