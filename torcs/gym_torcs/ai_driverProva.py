"""
ai_driver.py — Bot TORCS basato su MLP addestrato in Fase 4

v5.1 TIGHT RACING LINE per car-ow-1 / Corkscrew:
  FIX della racing line: in v5 era invertita (out-in-out classico, ma sbagliato
  per F1 con grip alto). Ora segue strategia "tight": SEMPRE sul lato interno.

  Curva a sinistra (delta_track > 0):
    tp_target sempre POSITIVO (bordo sinistro, lato interno della curva)
  Curva a destra (delta_track < 0):
    tp_target sempre NEGATIVO (bordo destro, lato interno)

  Le 3 fasi:
    - Entrata/lontano (tr9 > 80): tp_target = ±0.40 (un po' verso interno)
    - Avvicinamento  (40 < tr9 < 80): tp_target = ±0.50 (piu' stretto)
    - Apice          (tr9 < 40): tp_target = ±0.60 (sul cordolo)

  Altri parametri invariati rispetto v5:
    - Late braking force 0.65 + cooldown 20 tick
    - Steer amplify x1.25 in curve lente
    - Accel boost permissivo
"""

import socket
import sys
import json
import time
import math
from collections import deque

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

# Safety net standard
TRACKPOS_SAFE   = 0.95          # ALZATO da 0.85: tolleriamo linea piu' larga
TRACKPOS_BLEND  = 0.85          # ALZATO da 0.70: lasciamo che la racing line lavori
ANGLE_SAFE      = 0.30
RECOVERY_STEER_GAIN = 0.5
RECOVERY_ANGLE_GAIN = 2.0

SPIN_ANGLE      = 1.2
SPIN_RECOVERY_MAX_SPEED = 30.0

# Prebrake
PREBRAKE_TRACK9_THRESHOLD = 35.0
PREBRAKE_SPEED_THRESHOLD  = 150.0
PREBRAKE_FORCE            = 0.6      # ridotto da 0.7

MIN_SPEED_STALL = 5.0
STALL_PATIENCE  = 100

# Smoothing
USE_STEER_SMOOTHING = True
STEER_EMA_ALPHA     = 0.65

# Accel boost
USE_ACCEL_BOOST     = True
BOOST_TRACK9_MIN    = 80.0
BOOST_ANGLE_MAX     = 0.10           # tollera piu' angolo (eravamo in racing line)
BOOST_TRACKPOS_MAX  = 0.60           # ALZATO da 0.35: tolleriamo essere sul cordolo
BOOST_SIDE_RATIO    = 0.70

# === LATE BRAKING piu' dolce ===
USE_LATE_BRAKING       = True
LATE_BRAKE_TRACK9_MAX  = 55.0        # ridotto da 60: si attiva piu' tardi
LATE_BRAKE_SPEED_MIN   = 150.0       # alzato da 130: solo se davvero veloce
LATE_BRAKE_FORCE       = 0.65        # RIDOTTO da 0.85: il problema principale
LATE_BRAKE_COOLDOWN    = 20          # tick di "cooldown" dopo aver frenato forte

# Steer amplify
USE_STEER_AMPLIFY      = True
AMP_SPEED_MAX          = 100.0
AMP_STEER_THRESHOLD    = 0.15
AMP_FACTOR             = 1.25         # ridotto da 1.30 (era un po' troppo)


# =================================================================
# NUOVO: RACING LINE
# =================================================================
USE_RACING_LINE  = True

# trackPos: -1 = bordo destro, +1 = bordo sinistro
# In curva a sinistra (delta_track > 0):
#   entrata: tp target = +0.6 (largo a destra)
#   apice:   tp target = -0.5 (sul cordolo sinistro)
#   uscita:  tp target = +0.6 (largo a destra di nuovo)
# In curva a destra (delta_track < 0): tutto specchiato

DELTA_TRACK_CURVE_THRESHOLD = 30.0   # |delta_track| > 30 -> curva significativa
ENTRY_DETECT_TRACK9         = 100.0  # tr9 sotto questo = avviciniamoci a curva
APEX_DETECT_TRACK9          = 40.0   # tr9 sotto questo = siamo all'apice
EXIT_DETECT_TRACK9          = 80.0   # tr9 risale = stiamo uscendo

# Pesi della racing line sull'angolo di sterzo
RL_STEER_GAIN_ENTRY = 0.4           # quanto correggere verso la linea ideale
RL_STEER_GAIN_APEX  = 0.3
RL_STEER_GAIN_EXIT  = 0.4


# =================================================================
# CARICAMENTO MODELLO
# =================================================================

print("[ai_driver v5 racing line] Caricamento modelli...")
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
    print(f"[ai_driver] PROFILE: v5.1 TIGHT RACING LINE for car-ow-1 / Corkscrew")
    print(f"[ai_driver]   Late braking : {USE_LATE_BRAKING} (force={LATE_BRAKE_FORCE})")
    print(f"[ai_driver]   Steer amplify: {USE_STEER_AMPLIFY}")
    print(f"[ai_driver]   Accel boost  : {USE_ACCEL_BOOST}")
    print(f"[ai_driver]   Racing line  : {USE_RACING_LINE}")
except Exception as e:
    print(f"[ai_driver] ERRORE caricamento: {e}")
    sys.exit(1)


# =================================================================
# CONNESSIONE / PARSING (invariate)
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


def gear_logic(speed_kmh, current_gear):
    if speed_kmh < -5.0:
        return -1
    if speed_kmh < 5.0:
        return max(1, current_gear) if current_gear > 0 else 1
    down_thresh = {2: 30, 3: 60, 4: 100, 5: 140, 6: 190}
    up_thresh   = {1: 45, 2: 80, 3: 120, 4: 165, 5: 215}
    g = current_gear if current_gear >= 1 else 1
    if g < 6 and speed_kmh > up_thresh.get(g, 999):
        return g + 1
    if g > 1 and speed_kmh < down_thresh.get(g, 0):
        return g - 1
    return g


def is_spinning(angle, speed_x):
    return abs(angle) > SPIN_ANGLE


def spin_recovery_action(angle, track_pos, speed_x):
    if speed_x > SPIN_RECOVERY_MAX_SPEED:
        return {'steer': float(np.clip(-angle * 0.5, -1.0, 1.0)),
                'accel': 0.0, 'brake': 0.8, 'gear': None}
    if abs(angle) > math.pi / 2:
        steer = float(np.clip(np.sign(angle) * 0.7, -1.0, 1.0))
        return {'steer': steer, 'accel': 0.4, 'brake': 0.0, 'gear': -1}
    return {'steer': float(np.clip(-angle * 1.5, -1.0, 1.0)),
            'accel': 0.3, 'brake': 0.0, 'gear': 1}


def needs_prebrake(track_9, speed_x):
    return track_9 < PREBRAKE_TRACK9_THRESHOLD and speed_x > PREBRAKE_SPEED_THRESHOLD


def is_open_track(track, angle, track_pos):
    if len(track) < 19:
        return False
    t8, t9, t10 = float(track[8]), float(track[9]), float(track[10])
    if t9 < BOOST_TRACK9_MIN:
        return False
    if t8 < BOOST_SIDE_RATIO * t9 or t10 < BOOST_SIDE_RATIO * t9:
        return False
    if abs(angle) > BOOST_ANGLE_MAX:
        return False
    if abs(track_pos) > BOOST_TRACKPOS_MAX:
        return False
    return True


def amplify_steer_if_needed(steer_mlp, speed_x):
    if not USE_STEER_AMPLIFY:
        return steer_mlp
    if speed_x > AMP_SPEED_MAX:
        return steer_mlp
    abs_steer = abs(steer_mlp)
    if AMP_STEER_THRESHOLD < abs_steer < 0.5:
        return float(np.clip(steer_mlp * AMP_FACTOR, -1.0, 1.0))
    return steer_mlp


# =================================================================
# === RACING LINE ===
# =================================================================

def compute_racing_line_target(track, track_pos):
    """
    Calcola la posizione laterale target sulla pista in base alla forma
    della curva vista dai sensori track.

    STRATEGIA "TIGHT LINE" (massimo taglio):
      In curva si sta sempre vicini al lato INTERNO. Per car-ow-1 (F1) con
      enorme grip, tagliare la curva e' piu' veloce di mantenere raggio
      largo (out-in-out classico). Cioe':

      Curva a SINISTRA (delta_track > 0):
        target = trackPos POSITIVO (bordo sinistro, lato interno)

      Curva a DESTRA (delta_track < 0):
        target = trackPos NEGATIVO (bordo destro, lato interno)

      All'apice si va piu' stretti, in entrata/uscita un po' meno per
      avere margine.

    Restituisce: trackPos_target in [-1, +1], oppure None se rettilineo.
    """
    if len(track) < 19:
        return None
    t0, t9, t18 = float(track[0]), float(track[9]), float(track[18])
    delta = t18 - t0

    # Rettilineo o curva trascurabile -> nessuna racing line, MLP decide
    if abs(delta) < DELTA_TRACK_CURVE_THRESHOLD:
        return None

    # Direzione della curva = direzione in cui devo andare con trackPos
    # delta > 0: curva sx, voglio tp > 0 (bordo sx interno)
    # delta < 0: curva dx, voglio tp < 0 (bordo dx interno)
    curve_dir = 1 if delta > 0 else -1

    # Fase della curva: tutti i target sul LATO INTERNO
    if t9 > EXIT_DETECT_TRACK9:
        # Uscita o lontano: leggermente meno stretto, per riprendere velocita'
        target = curve_dir * 0.40
    elif t9 > APEX_DETECT_TRACK9:
        # Avvicinamento: gia' verso il cordolo interno
        target = curve_dir * 0.50
    else:
        # Apice: massimo taglio, sul cordolo
        target = curve_dir * 0.60

    return float(np.clip(target, -0.7, 0.7))


def racing_line_steer_correction(track_pos, target_tp, speed_x):
    """
    Calcola una correzione di sterzo proporzionale alla distanza tra
    posizione attuale e target di racing line.

    A bassa velocita' la correzione e' maggiore (l'auto cambia direzione facilmente).
    Ad alta velocita' la correzione e' attenuata (per non destabilizzare).
    """
    error = target_tp - track_pos
    # Riduzione del gain con la velocita'
    speed_factor = max(0.3, 1.0 - speed_x / 300.0)
    # Convenzione: trackPos +1 = sinistra, steer +1 = sinistra
    # Se target > current (devo spostarmi a sinistra), steer positivo
    correction = error * 0.3 * speed_factor
    return float(np.clip(correction, -0.3, 0.3))


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
    steer_prev = 0.0
    late_brake_cooldown = 0   # tick di cooldown dopo late braking

    while True:
        try:
            raw, _ = so.recvfrom(DATA_SIZE)
            msg = raw.decode('utf-8')

            if '***shutdown***' in msg:
                print("[ai_driver] Server in shutdown.")
                break
            if '***restart***' in msg:
                print("[ai_driver] Restart richiesto.")
                stall_counter = 0
                in_recovery = False
                gear = 1
                steer_prev = 0.0
                late_brake_cooldown = 0
                continue
            if not msg:
                continue

            S = parse_server_str(msg)
            track_pos = S.get('trackPos', 0.0)
            angle     = S.get('angle',    0.0)
            speed_x   = S.get('speedX',   0.0)
            track     = S.get('track',    [200.0] * 19)
            track_9   = track[9] if len(track) > 9 else 200.0

            if is_spinning(angle, speed_x):
                in_recovery = True
                recovery_counter += 1
            elif in_recovery and abs(angle) < 0.5 and abs(track_pos) < 1.0:
                print(f"[ai_driver] Recovery completato dopo {recovery_counter} tick")
                in_recovery = False
                recovery_counter = 0
                gear = 1
                late_brake_cooldown = 50   # cooldown lungo dopo recovery

            # Cooldown del late braking
            if late_brake_cooldown > 0:
                late_brake_cooldown -= 1

            boost_active = False
            late_brake_active = False
            amp_active = False
            rl_active = False
            rl_target = None

            if in_recovery:
                act = spin_recovery_action(angle, track_pos, speed_x)
                steer = act['steer']
                accel = act['accel']
                brake = act['brake']
                gear  = act['gear'] if act['gear'] is not None else gear
                w_log = 99.0
                steer_mlp = float('nan')
                steer_prev = steer

            else:
                # ====== GUIDA NORMALE ======
                x = build_state(S)
                x_scaled = scaler.transform(x)
                y = model.predict(x_scaled)[0]
                steer_mlp = float(np.clip(y[0], -1.0, 1.0))
                accel     = float(np.clip(y[1],  0.0, 1.0))
                brake     = float(np.clip(y[2],  0.0, 1.0))

                # Anti-sotto-sterzo
                steer_amp = amplify_steer_if_needed(steer_mlp, speed_x)
                if steer_amp != steer_mlp:
                    amp_active = True
                steer_mlp = steer_amp

                # === NUOVO: Racing line ===
                steer_correction = 0.0
                if USE_RACING_LINE:
                    rl_target = compute_racing_line_target(track, track_pos)
                    if rl_target is not None:
                        steer_correction = racing_line_steer_correction(
                            track_pos, rl_target, speed_x)
                        rl_active = abs(steer_correction) > 0.02

                # Blend con safety
                steer_rec = recovery_steer(track_pos, angle)
                w = blend_factor(track_pos, angle)
                steer_target_base = (1.0 - w) * steer_mlp + w * steer_rec
                # Aggiungo la correzione di racing line PRIMA della safety
                # se siamo in zona sicura, dopo se siamo in safety
                if w < 0.3:
                    steer_target = steer_target_base + steer_correction
                else:
                    # safety prevale, racing line attenuata
                    steer_target = steer_target_base + steer_correction * (1.0 - w)

                if USE_STEER_SMOOTHING and w < 0.5:
                    steer = STEER_EMA_ALPHA * steer_target + (1.0 - STEER_EMA_ALPHA) * steer_prev
                else:
                    steer = steer_target
                steer_prev = steer
                steer = float(np.clip(steer, -1.0, 1.0))

                if abs(track_pos) > TRACKPOS_SAFE:
                    accel = min(accel, 0.3)

                # Late braking con cooldown
                if (USE_LATE_BRAKING
                        and late_brake_cooldown == 0
                        and track_9 < LATE_BRAKE_TRACK9_MAX
                        and speed_x > LATE_BRAKE_SPEED_MIN):
                    brake = max(brake, LATE_BRAKE_FORCE)
                    late_brake_active = True
                    late_brake_cooldown = LATE_BRAKE_COOLDOWN
                elif needs_prebrake(track_9, speed_x):
                    brake = max(brake, PREBRAKE_FORCE)

                if USE_ACCEL_BOOST and brake < 0.05 and is_open_track(track, angle, track_pos):
                    accel = 1.0
                    boost_active = True

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
                if in_recovery:
                    tag = "REC"
                elif late_brake_active:
                    tag = "LBR"
                elif rl_active:
                    tag = f"RL{rl_target:+.1f}" if rl_target is not None else "RL"
                elif boost_active:
                    tag = "BST"
                elif amp_active:
                    tag = "AMP"
                else:
                    tag = f"w={w_log:.2f}"
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