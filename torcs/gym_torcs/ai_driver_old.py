"""
ai_driver.py — Bot TORCS basato su MLP addestrato in Fase 4

v2: fix dei bug del log del 14/05
  - FIX: gear logic con parentesi corrette + initial gear=1 stabile
  - FIX: anti-overspeed in curva (prefrenata se track_9 si accorcia troppo veloce)
  - FIX: recovery testacoda (|angle|>1.5 rad) con sequenza dedicata
  - Resto invariato: blend MLP + safety geometrico, mapping camelCase, ecc.
"""

import socket
import sys
import json
import time
import math

import numpy as np
import joblib


# =================================================================
# CONFIG
# =================================================================

HOST = 'localhost'
PORT = 3001
DATA_SIZE = 2**17

MODEL_PATH       = 'model_bc.joblib'
SCALER_PATH      = 'scaler.joblib'
FEATURE_CFG_PATH = 'feature_config.json'

# Soglie safety net standard
TRACKPOS_SAFE   = 0.85
TRACKPOS_BLEND  = 0.70
ANGLE_SAFE      = 0.30
RECOVERY_STEER_GAIN = 0.5
RECOVERY_ANGLE_GAIN = 2.0

# Soglie testacoda
SPIN_ANGLE      = 1.2     # rad ~ 70 gradi: oltre, siamo decisamente di traverso
SPIN_RECOVERY_MAX_SPEED = 30.0  # km/h

# Anti-overspeed in curva
PREBRAKE_TRACK9_THRESHOLD = 50.0   # m: track_9 sotto questo = "curva vicina"
PREBRAKE_SPEED_THRESHOLD  = 100.0  # km/h
PREBRAKE_FORCE            = 0.5

# Anti-stallo
MIN_SPEED_STALL = 5.0
STALL_PATIENCE  = 100


# =================================================================
# CARICAMENTO MODELLO
# =================================================================

print("[ai_driver] Caricamento modelli...")
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_CFG_PATH, 'r') as f:
        cfg = json.load(f)

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
# COSTRUZIONE VETTORE STATO
# =================================================================

def build_state(S):
    track = S.get('track', [200.0] * 19)
    stato = {
        'speedX':         S.get('speedX', 0.0),
        'speedY':         S.get('speedY', 0.0),
        'speedZ':         S.get('speedZ', 0.0),
        'angle':          S.get('angle', 0.0),
        'trackPos':       S.get('trackPos', 0.0),
        'rpm':            S.get('rpm', 0.0),
        'dist_from_start': S.get('distFromStart', 0.0),
        'distFromStart':   S.get('distFromStart', 0.0),
        'distRaced':       S.get('distRaced', 0.0),
        'delta_track':    float(track[18]) - float(track[0]),
    }
    for i in range(19):
        stato[f'track_{i}'] = float(track[i])
    x = np.array([[stato.get(name, 0.0) for name in FEATURES]],
                 dtype=np.float32)
    return x


# =================================================================
# SAFETY NET geometrico
# =================================================================

def recovery_steer(track_pos, angle):
    return float(np.clip(
        -track_pos * RECOVERY_STEER_GAIN + angle * RECOVERY_ANGLE_GAIN,
        -1.0, 1.0,
    ))


def blend_factor(track_pos, angle):
    abs_tp = abs(track_pos)
    abs_an = abs(angle)
    if abs_tp <= TRACKPOS_BLEND:
        w_tp = 0.0
    elif abs_tp >= TRACKPOS_SAFE:
        w_tp = 1.0
    else:
        w_tp = (abs_tp - TRACKPOS_BLEND) / (TRACKPOS_SAFE - TRACKPOS_BLEND)
    if abs_an >= ANGLE_SAFE:
        w_an = 1.0
    else:
        w_an = max(0.0, (abs_an - ANGLE_SAFE * 0.5) / (ANGLE_SAFE * 0.5))
    return max(w_tp, w_an)


# =================================================================
# CAMBIO MARCE — FIX precedenza operatori
# =================================================================

def gear_logic(speed_kmh, current_gear):
    """
    Cambio marce con isteresi. Logica corretta:
      - speed < -5 km/h e auto chiaramente all'indietro -> retro
      - altrimenti scelgo marcia in base alla velocita' assoluta
    """
    if speed_kmh < -5.0:
        return -1

    if speed_kmh < 5.0:
        return max(1, current_gear) if current_gear > 0 else 1

    down_thresh = {2: 35, 3: 75, 4: 115, 5: 160, 6: 200}
    up_thresh   = {1: 55, 2: 95, 3: 135, 4: 180, 5: 215}

    g = current_gear if current_gear >= 1 else 1

    if g < 6 and speed_kmh > up_thresh.get(g, 999):
        return g + 1
    if g > 1 and speed_kmh < down_thresh.get(g, 0):
        return g - 1
    return g


# =================================================================
# RECOVERY TESTACODA
# =================================================================

def is_spinning(angle, speed_x):
    return abs(angle) > SPIN_ANGLE


def spin_recovery_action(angle, track_pos, speed_x):
    """
    1. Se ancora veloci: freno duro.
    2. Se lenti e girati al contrario (|angle|>pi/2): retromarcia controllata.
    3. Se solo di traverso ma orientati avanti: gas leggero + sterzo correttivo.
    """
    if speed_x > SPIN_RECOVERY_MAX_SPEED:
        return {
            'steer': float(np.clip(-angle * 0.5, -1.0, 1.0)),
            'accel': 0.0,
            'brake': 0.8,
            'gear':  None,
        }

    if abs(angle) > math.pi / 2:
        steer = float(np.clip(np.sign(angle) * 0.7, -1.0, 1.0))
        return {
            'steer': steer,
            'accel': 0.4,
            'brake': 0.0,
            'gear':  -1,
        }
    else:
        return {
            'steer': float(np.clip(-angle * 1.5, -1.0, 1.0)),
            'accel': 0.3,
            'brake': 0.0,
            'gear':  1,
        }


# =================================================================
# PREFRENATA AUTOMATICA
# =================================================================

def needs_prebrake(track_9, speed_x):
    return track_9 < PREBRAKE_TRACK9_THRESHOLD and speed_x > PREBRAKE_SPEED_THRESHOLD


# =================================================================
# LOOP PRINCIPALE
# =================================================================

def run_ai():
    so = setup_connection()
    gear = 1
    stall_counter = 0
    last_log_t = 0.0
    in_recovery = False
    recovery_counter = 0

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
                in_recovery = False
                gear = 1
                continue
            if not msg:
                continue

            S = parse_server_str(msg)
            track_pos = S.get('trackPos', 0.0)
            angle     = S.get('angle',    0.0)
            speed_x   = S.get('speedX',   0.0)
            rpm       = S.get('rpm',      0.0)
            track     = S.get('track',    [200.0] * 19)
            track_9   = track[9] if len(track) > 9 else 200.0

            # MODALITA' RECOVERY TESTACODA
            if is_spinning(angle, speed_x):
                in_recovery = True
                recovery_counter += 1
            elif in_recovery and abs(angle) < 0.5 and abs(track_pos) < 1.0:
                print(f"[ai_driver] Recovery completato dopo {recovery_counter} tick")
                in_recovery = False
                recovery_counter = 0
                gear = 1

            if in_recovery:
                act = spin_recovery_action(angle, track_pos, speed_x)
                steer = act['steer']
                accel = act['accel']
                brake = act['brake']
                gear  = act['gear'] if act['gear'] is not None else gear
                w_log = 99.0
                steer_mlp = float('nan')
            else:
                # GUIDA NORMALE
                x = build_state(S)
                x_scaled = scaler.transform(x)
                y = model.predict(x_scaled)[0]
                steer_mlp = float(np.clip(y[0], -1.0, 1.0))
                accel     = float(np.clip(y[1],  0.0, 1.0))
                brake     = float(np.clip(y[2],  0.0, 1.0))

                steer_rec = recovery_steer(track_pos, angle)
                w = blend_factor(track_pos, angle)
                steer = (1.0 - w) * steer_mlp + w * steer_rec

                if abs(track_pos) > TRACKPOS_SAFE:
                    accel = min(accel, 0.3)

                if needs_prebrake(track_9, speed_x):
                    brake = max(brake, PREBRAKE_FORCE)

                if brake > 0.1:
                    accel = 0.0

                gear = gear_logic(speed_x, gear)
                w_log = w

            # Anti-stallo
            if not in_recovery and abs(speed_x) < MIN_SPEED_STALL:
                stall_counter += 1
                if stall_counter > STALL_PATIENCE:
                    accel = 1.0
                    brake = 0.0
                    gear  = 1
                    if stall_counter > STALL_PATIENCE * 3:
                        print("[ai_driver] STALLO PROLUNGATO: meta=1")
                        so.sendto(b"(meta 1)", (HOST, PORT))
                        stall_counter = 0
                        continue
            else:
                stall_counter = 0

            out = (f"(accel {accel:.3f})(brake {brake:.3f})"
                   f"(gear {gear})(steer {steer:.3f})"
                   f"(clutch 0)(focus 0)(meta 0)")
            so.sendto(out.encode(), (HOST, PORT))

            now = time.time()
            if now - last_log_t > 1.0:
                tag = "REC" if in_recovery else f"w={w_log:.2f}"
                print(f"v={speed_x:+6.1f} km/h  tp={track_pos:+.2f}  "
                      f"ang={angle:+.2f}  tr9={track_9:5.1f}  "
                      f"s_mlp={steer_mlp:+.2f}  s={steer:+.2f}  "
                      f"a={accel:.2f}  b={brake:.2f}  g={gear}  {tag}")
                last_log_t = now

        except socket.timeout:
            print("[ai_driver] Timeout socket, ritento...")
            continue
        except Exception as e:
            print(f"[ai_driver] ERRORE LOOP: {e}")


if __name__ == "__main__":
    run_ai()