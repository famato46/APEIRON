import time
import json
import math
import joblib
import numpy as np
import snakeoil3_jm2 as snakeoil3

# ==========================================
# CONFIGURAZIONE MODELLI
# ==========================================
MODEL_PATH       = 'models/model_bc.joblib'
SCALER_PATH      = 'out_bc/scaler.joblib'
FEATURE_CFG_PATH = 'out_bc/feature_config.json'

# ==========================================
# REGOLE DI SICUREZZA (Dal tuo vecchio script)
# ==========================================
PREBRAKE_T9_FAR    = 110.0; PREBRAKE_V_FAR     = 150.0; PREBRAKE_FORCE_FAR = 0.45
PREBRAKE_T9_MID    = 80.0;  PREBRAKE_V_MID     = 120.0; PREBRAKE_FORCE_MID = 0.65
PREBRAKE_T9_NEAR   = 50.0;  PREBRAKE_V_NEAR    = 95.0;  PREBRAKE_FORCE_NEAR= 0.90

SPIN_ANGLE         = 1.4
UNSTICK_SPEED      = 5.0
UNSTICK_PATIENCE   = 30

# Lettura features
try:
    with open(FEATURE_CFG_PATH, 'r') as f:
        cfg = json.load(f)
        if 'feature_names' in cfg: FEATURES = cfg['feature_names']
        elif 'input_features' in cfg: FEATURES = cfg['input_features']
        else: FEATURES = next(v for v in cfg.values() if isinstance(v, list))
except Exception as e:
    print(f"ERRORE CRITICO: Impossibile leggere config. {e}")
    FEATURES = []

def build_state(S):
    track = S.get('track', [200.0] * 19)
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
    for i in range(19): stato[f'track_{i}'] = float(track[i])
    x_array = [stato.get(name, 0.0) for name in FEATURES]
    return np.array([x_array], dtype=np.float32)

def automatic_gear(speed_x, is_reversing=False):
    if is_reversing or speed_x < -2.0: return -1
    if speed_x < 50:  return 1
    if speed_x < 90:  return 2
    if speed_x < 140: return 3
    if speed_x < 190: return 4
    if speed_x < 240: return 5
    return 6

def corkscrew_prebrake(track_9, speed_x):
    """Calcola se forzare una frenata di sicurezza."""
    if track_9 <= 0: return 0.0
    if track_9 < PREBRAKE_T9_NEAR and speed_x > PREBRAKE_V_NEAR: return PREBRAKE_FORCE_NEAR
    if track_9 < PREBRAKE_T9_MID and speed_x > PREBRAKE_V_MID:   return PREBRAKE_FORCE_MID
    if track_9 < PREBRAKE_T9_FAR and speed_x > PREBRAKE_V_FAR:   return PREBRAKE_FORCE_FAR
    return 0.0

def main():
    print("[drive_bc_hybrid] Caricamento Modello MLP Ibrido...")
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    print("[drive_bc_hybrid] OK! Motore acceso.")

    while True:
        client = None
        while client is None:
            try:
                client = snakeoil3.Client(p=3001, vision=False)
            except Exception:
                time.sleep(2)
                
        print("Connesso a TORCS — AI Ibrida (MLP + Heuristics)")
        step = 0
        slow_counter = 0
        
        while True:
            client.get_servers_input()
            S = client.S.d
            if client.so is None: break

            speed_x = S.get('speedX', 0.0)
            angle = S.get('angle', 0.0)
            track = S.get('track', [200.0] * 19)
            track_9 = track[9] if len(track) > 9 else 200.0

            # 1. PREDIZIONE DELL'AI
            raw_state = build_state(S)
            state_scaled = scaler.transform(raw_state)
            out = model.predict(state_scaled)[0]

            steer = float(np.clip(out[0], -1.0, 1.0))
            accel = float(np.clip(out[1],  0.0, 1.0))
            brake = float(np.clip(out[2],  0.0, 1.0))
            gear = automatic_gear(speed_x)
            tag = "AI_PURA"

            # ----------------------------------------------------
            # 2. SISTEMI DI SICUREZZA (OVERRIDE)
            # ----------------------------------------------------
            
            # A) Pre-Brake (L'Assistenza alla Frenata per il Corkscrew)
            pb = corkscrew_prebrake(track_9, speed_x)
            if pb > 0.0:
                if pb > brake: # L'AI sta frenando troppo poco
                    brake = pb
                    accel = 0.0
                    tag = f"PREBRAKE({pb:.2f})"
            
            # B) Recovery da Testacoda / Incidente
            if abs(angle) > SPIN_ANGLE or speed_x < -5.0:
                steer = float(np.clip(np.sign(angle) * 0.6, -1.0, 1.0))
                accel = 0.35
                brake = 0.0
                gear = -1
                tag = "RECOVERY_SPIN"
                
            # C) Unstick (Se siamo fermi a muro)
            if abs(speed_x) < UNSTICK_SPEED and abs(angle) < SPIN_ANGLE:
                slow_counter += 1
                if slow_counter > UNSTICK_PATIENCE:
                    steer = float(np.clip(-angle, -1.0, 1.0)) # Cerca di raddrizzarsi
                    accel = 1.0
                    brake = 0.0
                    gear = -1 if track[9] < 10.0 else 1
                    tag = "UNSTICK"
            else:
                slow_counter = 0

            # ----------------------------------------------------

            # Invio comandi a TORCS
            R = client.R.d
            R["steer"]  = steer
            R["accel"]  = accel
            R["brake"]  = brake
            R["gear"]   = gear
            R["clutch"] = 0.0
            R["meta"]   = 0
            
            client.respond_to_server()
            
            if step % 30 == 0:
                print(
                    f"step={step:05d} | str={steer:+.2f} acc={accel:.2f} "
                    f"brk={brake:.2f} gr={gear:2d} v={speed_x:+6.1f}km/h | {tag}"
                )
            step += 1

if __name__ == "__main__":
    main()