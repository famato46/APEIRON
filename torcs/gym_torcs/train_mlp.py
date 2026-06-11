import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ARCHITECTURES = [
    (64,),          
    (64, 32),        
    (128, 64),        
    (128, 64, 32),    
]
LEARNING_RATES = [1e-3, 5e-4]
ACTIVATION     = 'tanh'      
SOLVER         = 'adam'


MAX_ITER_GRID  = 200          
MAX_ITER_FINAL = 500          
PATIENCE       = 15         
VAL_FRAC_ES    = 0.10         
RANDOM_STATE   = 42


def carica_dataset(data_dir: Path):
    npz_path = data_dir / 'dataset_bc.npz'
    cfg_path = data_dir / 'feature_config.json'
    if not npz_path.exists():
        sys.exit(f"ERROR: {npz_path} non trovato. Hai lanciato build_dataset_bc.py?")
    data = np.load(npz_path)
    with open(cfg_path) as f:
        cfg = json.load(f)
    return data, cfg


def metriche(y_true, y_pred, prefix=""):
    targets = ['steer', 'accel', 'brake']
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    out = {
        f'{prefix}mse_total': float(mse),
        f'{prefix}mae_total': float(mae),
    }
    for i, name in enumerate(targets):
        out[f'{prefix}mae_{name}'] = float(np.abs(y_pred[:, i] - y_true[:, i]).mean())
        out[f'{prefix}r2_{name}']  = float(r2_score(y_true[:, i], y_pred[:, i]))
    return out


def stampa_metriche(m, label):
    print(f"\n--- {label} ---")
    print(f"  MSE totale: {m['mse_total' if 'mse_total' in m else list(m.keys())[0]]:.5f}")
    print(f"  MAE totale: {m['mae_total' if 'mae_total' in m else list(m.keys())[1]]:.5f}")
    print(f"  Per target:")
    for name in ['steer', 'accel', 'brake']:
        mae = m.get(f'mae_{name}')
        r2  = m.get(f'r2_{name}')
        print(f"    {name:5s}  MAE={mae:.4f}   R²={r2:+.3f}")


def fit_mlp(X_train, y_train, arch, lr, max_iter, verbose=False):
    mlp = MLPRegressor(
        hidden_layer_sizes=arch,
        activation=ACTIVATION,
        solver=SOLVER,
        learning_rate_init=lr,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=VAL_FRAC_ES,
        n_iter_no_change=PATIENCE,
        random_state=RANDOM_STATE,
        verbose=verbose,
    )
    mlp.fit(X_train, y_train)
    return mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./out_bc',
                        help='Cartella prodotta da build_dataset_bc.py')
    parser.add_argument('--out',  default='./models',
                        help='Cartella per salvare modello e report')
    parser.add_argument('--quick', action='store_true',
                        help='Solo baseline + 2 architetture (test rapido)')
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Carico dataset da {data_dir}")
    data, cfg = carica_dataset(data_dir)
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    feature_names = list(data['feature_names'])

    print(f"      Train: {X_train.shape}   Val: {X_val.shape}   Test: {X_test.shape}")
    print(f"      Feature: {len(feature_names)} input, {y_train.shape[1]} target")

    print(f"\n[2/5] Baseline: (64, 32), lr=1e-3, max_iter=100")
    t0 = time.time()
    baseline = fit_mlp(X_train, y_train, (64, 32), 1e-3, max_iter=100)
    t_base = time.time() - t0
    base_pred = baseline.predict(X_val)
    base_m = metriche(y_val, base_pred)
    print(f"      Training: {t_base:.1f}s, converged @ epoch {baseline.n_iter_}")
    stampa_metriche(base_m, "BASELINE Val")

    archs = ARCHITECTURES if not args.quick else [(64,), (64, 32)]
    lrs   = LEARNING_RATES if not args.quick else [1e-3]
    total = len(archs) * len(lrs)
    print(f"\n[3/5] Grid search ({total} combinazioni)")

    risultati = []
    t0 = time.time()
    for i, arch in enumerate(archs):
        for j, lr in enumerate(lrs):
            k = i * len(lrs) + j + 1
            t_run = time.time()
            mlp = fit_mlp(X_train, y_train, arch, lr, MAX_ITER_GRID)
            pred = mlp.predict(X_val)
            m = metriche(y_val, pred)
            dt = time.time() - t_run
            print(f"   [{k}/{total}] arch={arch}  lr={lr:.0e}  "
                  f"epochs={mlp.n_iter_:3d}  Val MAE={m['mae_total']:.4f}  "
                  f"R²(steer)={m['r2_steer']:+.3f}  {dt:.1f}s")
            risultati.append({
                'arch': list(arch), 'lr': lr,
                'epochs': mlp.n_iter_,
                'metrics': m,
                'time_s': dt,
            })
    print(f"      Grid completata in {time.time()-t0:.1f}s")

    best = min(risultati, key=lambda r: r['metrics']['mae_total'])
    print(f"\n   ===> MIGLIORE: arch={tuple(best['arch'])}  lr={best['lr']:.0e}")
    print(f"        Val MAE: {best['metrics']['mae_total']:.4f}")


    print(f"\n[4/5] Training finale con max_iter={MAX_ITER_FINAL}")
    t0 = time.time()
    final = fit_mlp(X_train, y_train,
                    tuple(best['arch']), best['lr'],
                    max_iter=MAX_ITER_FINAL, verbose=False)
    t_final = time.time() - t0
    final_val_pred = final.predict(X_val)
    final_val_m = metriche(y_val, final_val_pred)
    print(f"      Training: {t_final:.1f}s, converged @ epoch {final.n_iter_}")
    stampa_metriche(final_val_m, "MODELLO FINALE Val")

    print(f"\n[5/5] Valutazione TEST set (mai usato per tuning)")
    test_pred = final.predict(X_test)
    test_m = metriche(y_test, test_pred)
    stampa_metriche(test_m, "MODELLO FINALE Test")

    gap = test_m['mae_total'] - final_val_m['mae_total']
    print(f"\n   Gap MAE val→test: {gap:+.4f}  "
          f"({'overfitting sospetto' if gap > 0.01 else 'generalizzazione OK'})")

    model_path = out_dir / 'model_bc.joblib'
    joblib.dump(final, model_path)
    print(f"\n   Modello salvato: {model_path}")

    report = {
        'feature_names':  feature_names,
        'target_names':   list(data['target_names']),
        'best_arch':      best['arch'],
        'best_lr':        best['lr'],
        'best_epochs':    final.n_iter_,
        'baseline':       base_m,
        'grid_results':   risultati,
        'final_val':      final_val_m,
        'final_test':     test_m,
        'val_test_gap':   gap,
        'config': {
            'activation': ACTIVATION, 'solver': SOLVER,
            'max_iter_grid': MAX_ITER_GRID, 'max_iter_final': MAX_ITER_FINAL,
            'patience': PATIENCE, 'random_state': RANDOM_STATE,
        },
    }
    report_path = out_dir / 'training_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    print(f"   Report salvato: {report_path}")

    print(f"\n   === Errore MAE per regime di sterzo (su test set) ===")
    abs_steer = np.abs(y_test[:, 0])
    masks = [
        ('rettilineo  |s|<0.05',  abs_steer < 0.05),
        ('curva dolce |s|<0.10',  (abs_steer >= 0.05) & (abs_steer < 0.10)),
        ('curva medio|s|<0.30',   (abs_steer >= 0.10) & (abs_steer < 0.30)),
        ('curva forte |s|>=0.30', abs_steer >= 0.30),
    ]
    for label, mask in masks:
        if mask.sum() == 0:
            continue
        mae_steer = np.abs(test_pred[mask, 0] - y_test[mask, 0]).mean()
        print(f"     {label:25s}  N={mask.sum():4d}   MAE(steer)={mae_steer:.4f}")

    print(f"\n=== FATTO ===")
    print(f"Per usare il modello in inferenza:")
    print(f"  model  = joblib.load('{model_path}')")
    print(f"  scaler = joblib.load('{data_dir}/scaler.joblib')")
    print(f"  y_pred = model.predict(scaler.transform(x))")


if __name__ == '__main__':
    main()
