"""
ai_driver.py — Bot TORCS basato su MLP addestrato in Fase 4

Differenze rispetto alla versione base:
  - Carica feature_names dal JSON E le mappa correttamente, INCLUSO dist_from_start.
  - Coerenza accel/brake: niente acceleratore se il modello chiede freno.
  - Safety net geometrico: quando l'auto e' troppo vicino al bordo o storta,
    il modello viene parzialmente sovrascritto da un controller proporzionale
    che riporta l'auto al centro. Risolve il distributional shift del BC.
  - Anti-stallo: se la velocita' resta sotto 5 km/h per troppo tempo, riavvio.
  - Cambio marce piu' conservativo e con isteresi (no oscillazioni a 60 km/h).
"""

import socket
import sys
import json
import time

import numpy as np
import joblib


# =================================================================
# CONFIG
# =================================================================

HOST = 'localhost'
PORT = 3001
DATA_SIZE = 2**17

# Path agli artefatti di Fase 3 e Fase 4
MODEL_PATH       = 'model_bc.joblib'
SCALER_PATH      = 'scaler.joblib'
FEATURE_CFG_PATH = 'feature_config.json'    # <-- aggiusta il path se serve

# Soglie del safety net
TRACKPOS_SAFE   = 0.85    # oltre questa |trackPos| attiviamo il recupero proporzionale
TRACKPOS_BLEND  = 0.70    # da 0.70 a 0.85 facciamo blend MLP + recovery
ANGLE_SAFE      = 0.30    # rad, oltre questo entriamo in modalita' raddrizzamento
RECOVERY_STEER_GAIN = 0.5 # peso del recupero proporzionale
RECOVERY_ANGLE_GAIN = 2.0 # peso del raddrizzamento angolo

# Anti-stallo
MIN_SPEED_STALL = 5.0     # km/h
STALL_PATIENCE  = 100     # tick (a 50 Hz = 2 secondi)


# =================================================================
# CARICAMENTO MODELLO
# =================================================================

print("[ai_driver] Caricamento modelli...")
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_CFG_PATH, 'r') as f:
        cfg = json.load(f)

    # build_dataset_bc.py salva la lista come 'input_features'
    if 'input_features' in cfg:
        FEATURES = cfg['input_features']
    elif 'features' in cfg:
        FEATURES = cfg['features']
    else:
        FEATURES = next(v for v in cfg.values() if isinstance(v, list))

    print(f"[ai_driver] Modello caricato. Feature richieste ({len(FEATURES)}):")
    for i, f in enumerate(FEATURES):
        print(f"   [{i:2d}] {f}")

except Exception as e:
    print(f"[ai_driver] ERRORE caricamento: {e}")
    sys.exit(1)


# =================================================================
# CONNESSIONE TORCS
# =================================================================

def setup_connection():
    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1)
    initmsg = 'SCR(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)'
    while True:
        try:
            so.sendto(initmsg.encode(), (HOST, PORT))
            data, _ = so.recvfrom(DATA_SIZE)
            if '***identified***' in data.decode('utf-8'):
                print("[ai_driver] Connesso a TORCS.")
                return so
        except socket.error:
            print("[ai_driver] In attesa di TORCS sulla porta 3001...")


def parse_server_str(server_string):
    d = {}
    s = server_string.strip()[:-1]
    parts = s.lstrip('(').rstrip(')').split(')(')
    for token in parts:
        w = token.split(' ')
        if len(w) > 1:
            try:
                d[w[0]] = [float(x) for x in w[1:]] if len(w[1:]) > 1 else float(w[1])
            except ValueError:
                d[w[0]] = w[1:]
    return d


# =================================================================
# COSTRUZIONE VETTORE STATO (rispetta l'ordine del config)
# =================================================================

def build_state(S):
    """
    Costruisce il vettore di input nell'ordine richiesto dal modello.
    Mappa esplicita: nome_feature_nel_config -> nome_sensore_TORCS.
    """
    track = S.get('track', [200.0] * 19)

    # Mappa COMPLETA con TUTTI i nomi possibili che potrebbe richiedere il config.
    # Notare i nomi camelCase usati da TORCS/SnakeOil.
    stato = {
        # Cinematica
        'speedX':         S.get('speedX', 0.0),
        'speedY':         S.get('speedY', 0.0),
        'speedZ':         S.get('speedZ', 0.0),
        'angle':          S.get('angle', 0.0),
        'trackPos':       S.get('trackPos', 0.0),
        'rpm':            S.get('rpm', 0.0),

        # GPS: TORCS lo chiama distFromStart (camelCase!),
        # il nostro dataset l'ha rinominato dist_from_start.
        'dist_from_start': S.get('distFromStart', 0.0),
        'distFromStart':   S.get('distFromStart', 0.0),
        'distRaced':       S.get('distRaced', 0.0),

        # Feature engineered
        'delta_track':    float(track[18]) - float(track[0]),
    }
    # Singoli telemetri
    for i in range(19):
        stato[f'track_{i}'] = float(track[i])

    # Pesco solo le feature richieste, nell'ordine giusto
    x = np.array([[stato.get(name, 0.0) for name in FEATURES]],
                 dtype=np.float32)
    return x


# =================================================================
# SAFETY NET: controller geometrico di backup
# =================================================================

def recovery_steer(track_pos, angle):
    """
    Sterzata proporzionale per riportare l'auto al centro pista quando
    siamo fuori dalla distribuzione vista in training.

    Formula identica a quella che usavi nel bot manuale di raccolta:
      steer = -trackPos * gain + angle * gain
    """
    return np.clip(
        -track_pos * RECOVERY_STEER_GAIN + angle * RECOVERY_ANGLE_GAIN,
        -1.0, 1.0,
    )


def blend_factor(track_pos, angle):
    """
    Quanto pesare il safety net (1.0) vs il modello MLP (0.0).
    Cresce gradualmente da 0 a 1 man mano che ci si avvicina al bordo
    o si raddrizza male, evitando salti bruschi nelle predizioni.
    """
    abs_tp = abs(track_pos)
    abs_an = abs(angle)

    # Componente trackPos: 0 fino a TRACKPOS_BLEND, sale lineare fino a 1 a TRACKPOS_SAFE
    if abs_tp <= TRACKPOS_BLEND:
        w_tp = 0.0
    elif abs_tp >= TRACKPOS_SAFE:
        w_tp = 1.0
    else:
        w_tp = (abs_tp - TRACKPOS_BLEND) / (TRACKPOS_SAFE - TRACKPOS_BLEND)

    # Componente angle: 0 fino a meta' soglia, 1 a soglia piena
    if abs_an >= ANGLE_SAFE:
        w_an = 1.0
    else:
        w_an = max(0.0, (abs_an - ANGLE_SAFE * 0.5) / (ANGLE_SAFE * 0.5))

    return max(w_tp, w_an)


# =================================================================
# CAMBIO MARCE con isteresi
# =================================================================

def gear_with_hysteresis(speed_kmh, current_gear, rpm):
    """
    Cambio marce semplice ma con margini diversi per shift up/down
    (evita oscillazioni continue intorno alle soglie).
    """
    if speed_kmh < -2:
        return -1
    # Shift up: soglie standard
    up_thresholds   = [0, 55, 100, 145, 190, 230]   # km/h soglie per marcia 1..6
    # Shift down: soglie 10 km/h sotto (isteresi)
    down_thresholds = [0, 45,  90, 135, 180, 220]

    target = current_gear
    # Salgo?
    if current_gear < 6 and speed_kmh > up_thresholds[current_gear]:
        target = current_gear + 1
    # Scendo?
    elif current_gear > 1 and speed_kmh < down_thresholds[current_gear - 1]:
        target = current_gear - 1
    # Caso speciale: ingranamento iniziale da fermo
    if current_gear == 0 or current_gear == -1 and speed_kmh > 0:
        target = 1
    return max(1, target) if speed_kmh >= 0 else -1


# =================================================================
# LOOP PRINCIPALE
# =================================================================

def run_ai():
    so = setup_connection()
    gear = 1
    stall_counter = 0
    last_log_t = 0.0

    while True:
        try:
            raw, _ = so.recvfrom(DATA_SIZE)
            msg = raw.decode('utf-8')

            if '***shutdown***' in msg:
                print("[ai_driver] Server in shutdown.")
                break
            if '***restart***' in msg:
                print("[ai_driver] Restart richiesto dal server.")
                stall_counter = 0
                continue
            if not msg:
                continue

            S = parse_server_str(msg)

            # ---- 1. Predizione MLP ----
            x = build_state(S)
            x_scaled = scaler.transform(x)
            y = model.predict(x_scaled)[0]
            steer_mlp = float(np.clip(y[0], -1.0, 1.0))
            accel     = float(np.clip(y[1],  0.0, 1.0))
            brake     = float(np.clip(y[2],  0.0, 1.0))

            # ---- 2. Safety net: blend con recovery geometrico ----
            track_pos = S.get('trackPos', 0.0)
            angle     = S.get('angle',    0.0)
            speed_x   = S.get('speedX',   0.0)

            steer_rec = recovery_steer(track_pos, angle)
            w = blend_factor(track_pos, angle)
            steer = (1.0 - w) * steer_mlp + w * steer_rec

            # Se siamo decisamente fuori pista, abbassiamo anche il gas
            if abs(track_pos) > TRACKPOS_SAFE:
                accel = min(accel, 0.3)

            # ---- 3. Coerenza accel/brake ----
            if brake > 0.1:
                accel = 0.0

            # ---- 4. Cambio marce con isteresi ----
            rpm = S.get('rpm', 0.0)
            gear = gear_with_hysteresis(speed_x, gear, rpm)

            # ---- 5. Anti-stallo ----
            if speed_x < MIN_SPEED_STALL:
                stall_counter += 1
                if stall_counter > STALL_PATIENCE:
                    # Diamo una spinta di gas
                    accel = 1.0
                    brake = 0.0
                    if stall_counter > STALL_PATIENCE * 3:
                        # Decisamente bloccati: chiediamo restart
                        print("[ai_driver] STALLO PROLUNGATO: meta=1 (restart)")
                        out = "(meta 1)"
                        so.sendto(out.encode(), (HOST, PORT))
                        stall_counter = 0
                        continue
            else:
                stall_counter = 0

            # ---- 6. Invio comandi ----
            out = (f"(accel {accel:.3f})(brake {brake:.3f})"
                   f"(gear {gear})(steer {steer:.3f})"
                   f"(clutch 0)(focus 0)(meta 0)")
            so.sendto(out.encode(), (HOST, PORT))

            # ---- 7. Log periodico (1 volta al secondo) ----
            now = time.time()
            if now - last_log_t > 1.0:
                print(f"v={speed_x:5.1f} km/h  tp={track_pos:+.2f}  ang={angle:+.2f}  "
                      f"steer_mlp={steer_mlp:+.2f}  steer={steer:+.2f}  "
                      f"a={accel:.2f}  b={brake:.2f}  g={gear}  w_safe={w:.2f}")
                last_log_t = now

        except socket.timeout:
            print("[ai_driver] Timeout socket, ritento...")
            continue
        except Exception as e:
            print(f"[ai_driver] ERRORE LOOP: {e}")


if __name__ == "__main__":
    run_ai()