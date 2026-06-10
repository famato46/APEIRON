import time
import json
import joblib
import numpy as np
import snakeoil3_jm2 as snakeoil3

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
# Assicurati che questi percorsi corrispondano a quelli usati dal tuo train_mlp.py
MODEL_PATH       = 'models/model_bc.joblib'
SCALER_PATH      = 'out_bc/scaler.joblib'
FEATURE_CFG_PATH = 'out_bc/feature_config.json'

print("[drive_bc] Caricamento file di configurazione...")
try:
    with open(FEATURE_CFG_PATH, 'r') as f:
        cfg = json.load(f)
        # Cerca la lista delle feature nel JSON (si adatta a diversi formati)
        if 'feature_names' in cfg:
            FEATURES = cfg['feature_names']
        elif 'input_features' in cfg:
            FEATURES = cfg['input_features']
        else:
            FEATURES = next(v for v in cfg.values() if isinstance(v, list))
    print(f"[drive_bc] Feature attese: {len(FEATURES)}")
except Exception as e:
    print(f"ERRORE CRITICO: Impossibile leggere {FEATURE_CFG_PATH}. {e}")
    FEATURES = []

def build_state(S):
    """
    Costruisce l'array dei sensori nello stesso esatto ordine 
    usato durante l'addestramento, leggendo da FEATURES.
    """
    track = S.get('track', [200.0] * 19)
    # Raccogliamo i dati grezzi da TORCS
    stato = {
        'speedX':          S.get('speedX', 0.0),
        'speedY':          S.get('speedY', 0.0),
        'speedZ':          S.get('speedZ', 0.0),
        'angle':           S.get('angle', 0.0),
        'trackPos':        S.get('trackPos', 0.0),
        'rpm':             S.get('rpm', 0.0),
        'distFromStart':   S.get('distFromStart', S.get('distRaced', 0.0)),
        'distRaced':       S.get('distRaced', 0.0),
        'delta_track':     float(track[18]) - float(track[0]),
    }
    # Aggiungiamo i sensori di traccia
    for i in range(19):
        stato[f'track_{i}'] = float(track[i])
        
    # Creiamo l'array usando la lista FEATURES come stampo
    x_array = [stato.get(name, 0.0) for name in FEATURES]
    return np.array([x_array], dtype=np.float32)

def automatic_gear(speed_x):
    """
    Dato che MLPRegressor in train_mlp.py predice solo steer, accel, brake,
    dobbiamo gestire le marce manualmente in base alla velocità.
    """
    if speed_x < -5:  return -1
    if speed_x < 50:  return 1
    if speed_x < 90:  return 2
    if speed_x < 140: return 3
    if speed_x < 190: return 4
    if speed_x < 240: return 5
    return 6

def main():
    print("[drive_bc] Caricamento Modello e Scaler (Scikit-Learn)...")
    try:
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        print("[drive_bc] OK! Modello MLPRegressor caricato.")
    except Exception as e:
        print(f"ERRORE CRITICO: File modello o scaler mancanti. {e}")
        return

    while True:
        client = None
        while client is None:
            try:
                # Disabilitata vision=True per renderlo più leggero come in vecchia cfg
                client = snakeoil3.Client(p=3001, vision=False)
            except Exception:
                time.sleep(2)
                
        print("Connesso a TORCS — Guida autonoma (Scikit-Learn)")
        step = 0
        
        while True:
            client.get_servers_input()
            S = client.S.d
            if client.so is None:
                break

            # 1. Costruisce lo stato in base al JSON
            raw_state = build_state(S)
            
            # 2. Applica lo scaler calcolato sui dati di addestramento
            state_scaled = scaler.transform(raw_state)
            
            # 3. Chiede al modello la predizione
            out = model.predict(state_scaled)[0]

            # 4. Clipping di sicurezza (Scikit-Learn non ha sigmoid nativo nell'output)
            steer = float(np.clip(out[0], -1.0, 1.0))
            accel = float(np.clip(out[1],  0.0, 1.0))
            brake = float(np.clip(out[2],  0.0, 1.0))
            
            # 5. Gestione manuale della marcia
            gear = automatic_gear(S.get('speedX', 0.0))

            # 6. Invio comandi a TORCS
            R = client.R.d
            R["steer"]  = steer
            R["accel"]  = accel
            R["brake"]  = brake
            R["gear"]   = gear
            R["clutch"] = 0.0
            R["meta"]   = 0
            
            client.respond_to_server()
            
            # Stampa di log ogni 30 step per monitorare
            if step % 30 == 0:
                print(
                    f"step={step:05d} | str={steer:+.2f} acc={accel:.2f} "
                    f"brk={brake:.2f} gr={gear} v={S.get('speedX', 0):.1f}km/h"
                )
            step += 1

if __name__ == "__main__":
    main()