import numpy as np
import joblib
import os
import json
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURAZIONE PERCORSI ---
INPUT_FOLDER = "./out_bc"  # La cartella creata da build_dataset.py
NPZ_PATH = os.path.join(INPUT_FOLDER, "dataset_bc.npz")
CONFIG_PATH = os.path.join(INPUT_FOLDER, "feature_config.json")
MODEL_OUTPUT_PATH = "model_bc.joblib"  # Salvato nella root per ai_driver.py

def main():
    print("\n==================================================")
    print("   TORCS AI - AGENTE MLP TRAINING (APEIRON)")
    print("==================================================")

    # 1. CARICAMENTO DEI DATI PREPARATI
    if not os.path.exists(NPZ_PATH):
        print(f"[ERRORE CRITICO] Impossibile trovare il file dei dati in: {NPZ_PATH}")
        print("Assicurati di aver lanciato prima: python build_dataset.py")
        return

    print(f"[1/4] Caricamento del dataset compresso da {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    print(f"  -> Training Set:   {X_train.shape[0]} campioni")
    print(f"  -> Validation Set: {X_val.shape[0]} campioni")
    print(f"  -> Test Set:       {X_test.shape[0]} campioni")
    print(f"  -> Numero Feature di Input: {X_train.shape[1]}")

    # Leggiamo i nomi delle feature per il log visivo
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            print(f"  -> Feature in uso: {config.get('features', 'Non specificate')}")

    # 2. CONFIGURAZIONE DELLA RETE NEURALE (MLP)
    print("\n[2/4] Configurazione dell'architettura di rete...")
    # Architettura profonda a 3 livelli nascosti (128, 64, 32 neuroni)
    # Ottima per mappare le relazioni non lineari dei sensori di pista
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',         # Standard industriale per evitare la sparizione del gradiente
        solver='adam',             # Ottimizzatore robusto e veloce a convergere
        alpha=0.0001,              # Regolarizzazione L2 per evitare overfitting
        batch_size=256,            # Dimensione del blocco di dati ad ogni step
        learning_rate_init=0.001,  # Passo di apprendimento iniziale
        max_iter=200,              # Numero massimo di epoche
        random_state=42,           # Riproducibilità del seed
        early_stopping=True,       # Interrompe il training se la validation loss smette di migliorare
        validation_fraction=0.1,   # Quota interna per l'early stopping
        verbose=True               # Stampa i progressi della loss ad ogni epoca
    )

    # 3. ADDESTRAMENTO
    print("\n[3/4] Avvio del processo di addestramento (Imitation Learning)...")
    mlp.fit(X_train, y_train)
    print("\n[OK] Addestramento completato o interrotto dall'Early Stopping!")

    # 4. VALUTAZIONE DETTAGLIATA (TEST SET SEGRETATO)
    print("\n[4/4] Valutazione delle performance sul Test Set...")
    y_pred = mlp.predict(X_test)

    # Nomi dei target nell'ordine esatto salvato nel dataset
    target_names = ['STERZO (steer)', 'ACCELERATORE (accel)', 'FRENO (brake)']
    
    print("\n=== METRICHE DI PERFORMANCE VALUTATE ===")
    for i, name in enumerate(target_names):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        print(f"\n Target: {name}")
        print(f"  -> Mean Squared Error (MSE): {mse:.6f} (Più è vicino a 0, meglio è)")
        print(f"  -> R² Score (Accuratezza):   {r2*100:.2f}% (Punteggio di imitazione del pilota)")

    # 5. SALVATAGGIO DEL MODELLO
    joblib.dump(mlp, MODEL_OUTPUT_PATH)
    print(f"\n[SALVATO] Cervello AI salvato con successo in: {MODEL_OUTPUT_PATH}")
    print("Ora puoi lanciare il bot con: python ai_driver.py\n")

if __name__ == "__main__":
    main()