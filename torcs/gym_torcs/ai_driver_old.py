import socket
import sys
import json
import time
import math

import numpy as np
import joblib

HOST = '127.0.0.1'
PORT = 3001
DATA_SIZE = 2**17

MODEL_PATH       = 'models/model_bc.joblib'
SCALER_PATH      = 'scaler.joblib'
FEATURE_CFG_PATH = 'feature_config.json'

TRACKPOS_SAFE   = 0.92
TRACKPOS_BLEND  = 0.80
ANGLE_SAFE      = 0.60
RECOVERY_STEER_GAIN = 0.4
RECOVERY_ANGLE_GAIN = 1.5

STEER_EMA_ALPHA = 0.50
STEER_DEADBAND  = 0.07
STEER_MAX_DELTA = 0.10

PREBRAKE_T9_FAR    = 110.0
PREBRAKE_V_FAR     = 150.0
PREBRAKE_FORCE_FAR = 0.45
PREBRAKE_T9_MID    = 80.0
PREBRAKE_V_MID     = 120.0
PREBRAKE_FORCE_MID = 0.65
PREBRAKE_T9_NEAR   = 50.0
PREBRAKE_V_NEAR    = 95.0
PREBRAKE_FORCE_NEAR= 0.90

SPIN_ANGLE             = 1.4
SPIN_EXIT_ANGLE        = 0.40
SPIN_HYSTERESIS_TICKS  = 6
SPIN_STOP_SPEED        = 8.0
SPIN_MAX_TICKS         = 120

UNSTICK_SPEED   = 12.0
UNSTICK_PATIENCE = 8
STALL_HARD_PATIENCE = 100

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


def build_state(S):
    track = S.get('track', [200.0] * 19)
    stato = {
        'speedX':          S.get('speedX', 0.0),
        'speedY':          S.get('speedY', 0.0),
        'speedZ':          S.get('speedZ', 0.0),
        'angle':           S.get('angle', 0.0),
        'trackPos':        S.get('trackPos', 0.0),
        'rpm':             S.get('rpm', 0.0),
        'dist_from_start': S.get('distFromStart', 0.0),
        'distFromStart':   S.get('distFromStart', 0.0),
        'distRaced':       S.get('distRaced', 0.0),
        'delta_track':     float(track[18]) - float(track[0]),
    }
    for i in range(19):
        stato[f'track_{i}'] = float(track[i])
    return np.array([[stato.get(name, 0.0) for name in FEATURES]], dtype=np.float32)


_prev_track_pos = 0.0

def recovery_steer(track_pos, angle, speed_x=0.0):
    v_gain = 1.0 + max(0.0, speed_x - 80.0) / 150.0
    return float(np.clip(
        -track_pos * RECOVERY_STEER_GAIN * v_gain + angle * RECOVERY_ANGLE_GAIN,
        -1.0, 1.0))


def blend_factor(track_pos, angle, drift_rate=0.0):
    abs_tp = abs(track_pos); abs_an = abs(angle)
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
    w_drift = 0.0
    if abs_tp > 0.55 and drift_rate * track_pos > 0.003:
        w_drift = min(1.0, drift_rate * track_pos * 30.0)
    return max(w_tp, w_an, w_drift)


def gear_logic(speed_kmh, current_gear):
    if speed_kmh < -5.0:
        return -1
    if speed_kmh < 5.0:
        return max(1, current_gear) if current_gear > 0 else 1
    down_thresh = {2: 25, 3: 60, 4: 95, 5: 140, 6: 180}
    up_thresh   = {1: 60, 2: 100, 3: 145, 4: 190, 5: 225}
    g = current_gear if current_gear >= 1 else 1
    if g < 6 and speed_kmh > up_thresh.get(g, 999):
        return g + 1
    if g > 1 and speed_kmh < down_thresh.get(g, 0):
        return g - 1
    return g


def spin_recovery_action(angle, track_pos, speed_x, abs_speed):
    if abs_speed > SPIN_STOP_SPEED:
        return {'steer': float(np.clip(-np.sign(angle) * 0.2, -1.0, 1.0)),
                'accel': 0.0, 'brake': 1.0, 'gear': None}
    if abs(angle) > math.pi / 2:
        return {'steer': float(np.clip(np.sign(angle) * 0.6, -1.0, 1.0)),
                'accel': 0.35, 'brake': 0.0, 'gear': -1}
    return {'steer': float(np.clip(-angle * 1.0, -1.0, 1.0)),
            'accel': 0.30, 'brake': 0.0, 'gear': 1}


def corkscrew_prebrake(track_9, speed_x):
    if track_9 <= 0:
        return 0.0
    if track_9 < PREBRAKE_T9_NEAR and speed_x > PREBRAKE_V_NEAR:
        return PREBRAKE_FORCE_NEAR
    if track_9 < PREBRAKE_T9_MID and speed_x > PREBRAKE_V_MID:
        return PREBRAKE_FORCE_MID
    if track_9 < PREBRAKE_T9_FAR and speed_x > PREBRAKE_V_FAR:
        return PREBRAKE_FORCE_FAR
    return 0.0


def run_ai():
    so = setup_connection()
    gear = 1
    last_log_t = 0.0
    in_recovery = False
    recovery_counter = 0
    spin_exit_counter = 0
    prev_steer = 0.0
    slow_counter = 0
    global _prev_track_pos
    _prev_track_pos = 0.0

    while True:
        try:
            raw, _ = so.recvfrom(DATA_SIZE)
            msg = raw.decode('utf-8')

            if '***shutdown***' in msg:
                print("[ai_driver] Server in shutdown.")
                break
            if '***restart***' in msg:
                print("[ai_driver] Restart richiesto dal server.")
                in_recovery = False; recovery_counter = 0; spin_exit_counter = 0
                gear = 1; prev_steer = 0.0; slow_counter = 0
                continue
            if not msg:
                continue

            S = parse_server_str(msg)
            track_pos = S.get('trackPos', 0.0)
            angle     = S.get('angle',    0.0)
            speed_x   = S.get('speedX',   0.0)
            track     = S.get('track',    [200.0] * 19)
            track_9   = track[9] if len(track) > 9 else 200.0
            abs_speed = abs(speed_x)

            if abs(angle) > SPIN_ANGLE:
                in_recovery = True; recovery_counter += 1; spin_exit_counter = 0
            elif in_recovery:
                recovery_counter += 1
                if (abs(angle) < SPIN_EXIT_ANGLE and abs(track_pos) < 0.92
                        and speed_x > 1.0):
                    spin_exit_counter += 1
                    if spin_exit_counter >= SPIN_HYSTERESIS_TICKS:
                        print(f"[ai_driver] Recovery OK dopo {recovery_counter} tick")
                        in_recovery = False; recovery_counter = 0
                        spin_exit_counter = 0; prev_steer = 0.0; gear = 1
                else:
                    spin_exit_counter = 0

            if in_recovery and (recovery_counter > SPIN_MAX_TICKS
                                or abs(track_pos) > 2.0):
                print("[ai_driver] Recovery FALLITO: meta=1")
                so.sendto(b"(meta 1)", (HOST, PORT))
                in_recovery = False; recovery_counter = 0
                spin_exit_counter = 0; gear = 1; prev_steer = 0.0; slow_counter = 0
                continue

            if in_recovery:
                act = spin_recovery_action(angle, track_pos, speed_x, abs_speed)
                steer = act['steer']; accel = act['accel']; brake = act['brake']
                gear  = act['gear'] if act['gear'] is not None else gear
                w_log = 99.0; steer_mlp = float('nan')
            else:
                x = build_state(S)
                y = model.predict(scaler.transform(x))[0]
                steer_mlp = float(np.clip(y[0], -1.0, 1.0))
                accel     = float(np.clip(y[1],  0.0, 1.0))
                brake     = float(np.clip(y[2],  0.0, 1.0))

                steer_rec = recovery_steer(track_pos, angle, speed_x)
                drift_rate = track_pos - _prev_track_pos
                _prev_track_pos = track_pos
                w = blend_factor(track_pos, angle, drift_rate)
                if track_9 > 0 and track_9 < 15.0:
                    w = max(w, 0.7)
                steer_raw = (1.0 - w) * steer_mlp + w * steer_rec

                steer_smooth = STEER_EMA_ALPHA * steer_raw + (1.0 - STEER_EMA_ALPHA) * prev_steer
                if abs(steer_smooth) < STEER_DEADBAND and w < 0.3:
                    steer_smooth = 0.0
                delta = steer_smooth - prev_steer
                if delta > STEER_MAX_DELTA:
                    steer_smooth = prev_steer + STEER_MAX_DELTA
                elif delta < -STEER_MAX_DELTA:
                    steer_smooth = prev_steer - STEER_MAX_DELTA
                steer = float(np.clip(steer_smooth, -1.0, 1.0))
                prev_steer = steer

                pb = corkscrew_prebrake(track_9, speed_x)
                if pb > 0.0:
                    brake = max(brake, pb); accel = 0.0

                if w > 0.3 and speed_x > 60.0:
                    accel = min(accel, max(0.0, 1.0 - w))
                    brake = max(brake, min(0.6, w * 0.5))

                if abs(track_pos) > TRACKPOS_SAFE:
                    accel = min(accel, 0.2); brake = max(brake, 0.3)

                if track_9 > 0 and track_9 < 80.0 and speed_x > 80.0:
                    accel = min(accel, max(0.0, (track_9 - 20.0) / 60.0))

                if brake > 0.20:
                    accel = 0.0
                else:
                    brake = 0.0
                    if abs(track_pos) < 0.90:
                        if speed_x < 30.0:   accel = max(accel, 1.00)
                        elif speed_x < 55.0: accel = max(accel, 0.85)
                        elif speed_x < 80.0: accel = max(accel, 0.65)
                        elif speed_x < 110.0:accel = max(accel, 0.42)
                        elif speed_x < 140.0:accel = max(accel, 0.22)

                gear = gear_logic(speed_x, gear)
                w_log = w

                # UNSTICK: bloccato lento in pista col muso dritto -> forza gas
                # ha priorita' ASSOLUTA su qualsiasi freno residuo del modello
                if abs_speed < UNSTICK_SPEED and abs(track_pos) < 0.95 and abs(angle) < 0.5:
                    slow_counter += 1
                    if slow_counter >= UNSTICK_PATIENCE:
                        accel = 1.0; brake = 0.0; gear = 1
                    if slow_counter >= STALL_HARD_PATIENCE:
                        print("[ai_driver] STALLO DURO: meta=1")
                        so.sendto(b"(meta 1)", (HOST, PORT))
                        slow_counter = 0
                        continue
                else:
                    slow_counter = 0

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