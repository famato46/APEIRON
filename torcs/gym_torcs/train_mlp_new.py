import numpy as np
import joblib
import os
import json
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURAZIONE PERCORSI ---
INPUT_FOLDER = "./out_bc"
NPZ_PATH = os.path.join(INPUT_FOLDER, "dataset_bc.npz")
CONFIG_PATH = os.path.join(INPUT_FOLDER, "feature_config.json")
MODEL_OUTPUT_PATH = "model_bc.joblib"

def main():
    print("\n==================================================")
    print("   TORCS AI - AGENTE MLP TRAINING (APEIRON)")
    print("==================================================")

    if not os.path.exists(NPZ_PATH):
        print(f"[ERRORE CRITICO] Impossibile trovare il file dei dati in: {NPZ_PATH}")
        print("Assicurati di aver lanciato prima: python build_dataset.py")
        return

    print(f"[1/4] Caricamento del dataset compresso da {NPZ_PATH}...")
    data = np.load(NPZ_PATH)

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    # Nomi target dai dati salvati (formato nuovo: 'target_steer', ...)
    target_keys = [str(t) for t in data['target_names']]

    print(f"  -> Training Set:   {X_train.shape[0]} campioni")
    print(f"  -> Validation Set: {X_val.shape[0]} campioni")
    print(f"  -> Test Set:       {X_test.shape[0]} campioni")
    print(f"  -> Numero Feature di Input: {X_train.shape[1]}")
    print(f"  -> Target:         {target_keys}")

    # FIX: build_dataset.py salva la chiave come 'input_features'
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            feats = config.get('input_features', config.get('features', 'Non specificate'))
            print(f"  -> Feature in uso: {feats}")

    # 2. ARCHITETTURA
    print("\n[2/4] Configurazione dell'architettura di rete...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True
    )

    # 3. TRAINING
    print("\n[3/4] Avvio del processo di addestramento (Imitation Learning)...")
    mlp.fit(X_train, y_train)
    print("\n[OK] Addestramento completato o interrotto dall'Early Stopping!")

    # 4. VALUTAZIONE
    print("\n[4/4] Valutazione delle performance sul Test Set...")
    y_pred = mlp.predict(X_test)

    # Etichette di stampa coerenti con l'ordine dei target salvati
    label_map = {
        'target_steer': 'STERZO (steer)',
        'target_accel': 'ACCELERATORE (accel)',
        'target_brake': 'FRENO (brake)',
    }
    target_names = [label_map.get(k, k) for k in target_keys]

    print("\n=== METRICHE DI PERFORMANCE VALUTATE ===")
    for i, name in enumerate(target_names):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"\n Target: {name}")
        print(f"  -> Mean Squared Error (MSE): {mse:.6f} (Più è vicino a 0, meglio è)")
        print(f"  -> R² Score (Accuratezza):   {r2*100:.2f}% (Punteggio di imitazione del pilota)")

    # 5. SALVATAGGIO
    joblib.dump(mlp, MODEL_OUTPUT_PATH)
    print(f"\n[SALVATO] Cervello AI salvato con successo in: {MODEL_OUTPUT_PATH}")
    print("Ora puoi lanciare il bot con: python ai_driver_new.py\n")

if __name__ == "__main__":
    main()