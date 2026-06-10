"""
ai_driver_v16_brake_release.py — TORCS MLP driver, versione STABILE per Corkscrew

Obiettivo della v9:
  - evitare giravolta nella doppia curva del Corkscrew;
  - evitare zig-zag sul rettilineo finale;
  - evitare pista in contromano dopo testacoda;
  - mantenere velocita' buona, ma con un "governatore" di velocita' stabile.

Idea:
  1. L'MLP resta il pilota principale.
  2. Sopra l'MLP aggiungiamo un controllore di stabilita':
       - target speed in base a curvatura + distanza frontale;
       - frenata anticipata e progressiva, non all'ultimo;
       - gas minimo in uscita curva solo se l'auto e' stabile.
  3. Lo sterzo ha un filtro anti-zig-zag che NON resta bloccato col segno sbagliato.
  4. La recovery non usa retro permanente e non continua in contromano.
"""

import socket
import sys
import json
import time
import math

import numpy as np
import joblib


# =================================================================
# CONFIG BASE
# =================================================================

HOST = '127.0.0.1'
PORT = 3001
DATA_SIZE = 2**17

MODEL_PATH       = 'models/model_bc.joblib'
SCALER_PATH      = 'out_bc/scaler.joblib'
FEATURE_CFG_PATH = 'feature_config.json'


# =================================================================
# SAFETY NET: più attivo sulla posizione, meno nervoso sull'angolo
# =================================================================

TRACKPOS_BLEND = 0.48     # prima entra il richiamo al centro
TRACKPOS_SAFE  = 0.88
ANGLE_BLEND    = 0.34
ANGLE_SAFE     = 0.62     # non scattare troppo presto solo per angle

RECOVERY_STEER_GAIN = 0.70
RECOVERY_ANGLE_GAIN = 0.95

USE_SAFETY_W_SMOOTHING = True
SAFETY_W_ALPHA = 0.55


# =================================================================
# RECOVERY / TESTACODA
# =================================================================

SPIN_ANGLE = 1.10

ANTI_REVERSE_SPEED = -3.0
ANTI_REVERSE_BRAKE = 0.85

RECOVERY_MAX_TICKS = 130

# Se True, quando la macchina è chiaramente irrecuperabile evita di fare pista contromano.
# Se il regolamento non permette restart automatici, mettilo False.
USE_META_RESTART_ON_IRRECOVERABLE = False

# Curve difficili: se siamo vicini al bordo o in curva cieca, rallenta PRIMA.
CRITICAL_CURV = 0.48
CRITICAL_FRONT = 16.0
EDGE_WARN_TP = 0.62
OFFTRACK_GUARD_TP = 0.82

# Se dopo un rientro si ferma dentro una curva, non deve restare piantata.
USE_LOW_SPEED_RELAUNCH = True
RELAUNCH_SPEED_MAX = 16.0
RELAUNCH_ANGLE_MAX = 0.95
RELAUNCH_TP_MAX = 0.92
RELAUNCH_FRONT_MIN = 6.0

# Early-brake assist:
# Dai log si vede che entra ancora troppo forte: a ~200 km/h con track_9 ~70
# il vecchio target era ancora circa 195, quindi frenava tardi.
# Queste soglie abbassano la velocita' target PRIMA che track_9 crolli.
EARLY_FRONT_1 = 115.0
EARLY_FRONT_2 = 95.0
EARLY_FRONT_3 = 78.0
EARLY_FRONT_4 = 62.0

EARLY_TARGET_1 = 188.0
EARLY_TARGET_2 = 164.0
EARLY_TARGET_3 = 139.0
EARLY_TARGET_4 = 116.0

# Brake minimo se arrivi troppo veloce verso una chiusura frontale.
EARLY_BRAKE_SPEED_1 = 175.0
EARLY_BRAKE_SPEED_2 = 165.0
EARLY_BRAKE_SPEED_3 = 150.0
EARLY_BRAKE_MIN_1 = 0.14
EARLY_BRAKE_MIN_2 = 0.25
EARLY_BRAKE_MIN_3 = 0.38

# Hard anti-stop finale: viene applicato subito prima dell'invio comando.
HARD_RELAUNCH_SPEED = 8.0
HARD_RELAUNCH_FRONT = 6.0
HARD_RELAUNCH_ANGLE = 1.05
HARD_RELAUNCH_TP = 0.98

# V14 wall guard:
# La V13 prendeva il muro quando curvatura alta + tr9 basso e poi dava ancora gas.
# Questo blocco NON serve per fare il giro piu' veloce: serve per evitare il muro.
USE_WALL_GUARD = True
WALL_CURV_STRONG = 0.48
WALL_CURV_MED    = 0.40
WALL_FRONT_DANGER = 13.0
WALL_FRONT_WARN   = 38.0
WALL_TP_WARN      = 0.68
WALL_ANGLE_WARN   = 0.55

# Piccolo gain rispetto alla V11 solo in curve controllabili.
USE_SAFE_CORNER_GAIN = True
SAFE_GAIN_MAX_CURV = 0.34
SAFE_GAIN_MIN_FRONT = 18.0
SAFE_GAIN_MAX_TP = 0.48
SAFE_GAIN_MAX_ANGLE = 0.26

# V15 last curve tuning:
# La V14 chiude il giro in 1:47.54, ma nell'ultima curva il wall_guard frena troppo
# a bassa velocita'. Qui NON alziamo la velocita' generale: rendiamo solo piu'
# fluido l'apice e l'uscita della curva stretta.
USE_LAST_CURVE_APEX_TUNING = True
APEX_FRONT_MAX = 14.0
APEX_CURV_MIN = 0.40
APEX_SPEED_MAX = 58.0
APEX_TP_MAX = 0.88
APEX_ANGLE_MAX = 0.58
APEX_BRAKE_CAP = 0.08
APEX_EXIT_ACCEL = 0.32
APEX_EXIT_SPEED = 48.0

# V16 brake release:
# Dai log V15 il freno rimane spesso attivo anche quando v <= vt.
# Questo taglia il freno residuo e ridà gas SOLO se siamo sotto target e stabili.
USE_BRAKE_RELEASE_UNDER_TARGET = True
RELEASE_FRONT_MIN = 14.0
RELEASE_MAX_CURV = 0.48
RELEASE_MAX_TP = 0.68
RELEASE_MAX_ANGLE = 0.44
RELEASE_MARGIN = 3.0
RELEASE_BRAKE_CAP = 0.03
NEAR_TARGET_BRAKE_CAP = 0.08


# =================================================================
# SPEED GOVERNOR: stabilità in curva
# =================================================================

USE_SPEED_GOVERNOR = True

# Quanto aggressivamente il governor può frenare.
# Non sono valori "da qualifica": sono per finire stabile.
GOV_BRAKE_VERY_HIGH = 0.60
GOV_BRAKE_HIGH      = 0.44
GOV_BRAKE_MED       = 0.28
GOV_BRAKE_LOW       = 0.12
GOV_BRAKE_TINY      = 0.04

# Rampa freno: impedisce inchiodate istantanee.
USE_BRAKE_RATE_LIMIT = True
BRAKE_RISE_MAX = 0.18
BRAKE_FALL_MAX = 0.50

# In uscita curva, se sei troppo lento e la macchina è orientata bene, forza un po' di gas.
USE_CORNER_EXIT_ASSIST = True
EXIT_MIN_FRONT = 12.0
EXIT_MAX_ANGLE = 0.46
EXIT_MAX_TRACKPOS = 0.82


# =================================================================
# PREFRENATA LEGGERA
# =================================================================
# La vera logica ora è nel governor. Questa è solo una safety extra.

PREBRAKE_TRACK9_THRESHOLD = 26.0
PREBRAKE_SPEED_THRESHOLD  = 135.0
PREBRAKE_FORCE            = 0.14


# =================================================================
# STERZO: anti-zig-zag ma senza ritardo di segno
# =================================================================

USE_STEER_FILTER = True

# Rate limiter base
STEER_RATE_LOW_SPEED  = 0.17
STEER_RATE_MID_SPEED  = 0.12
STEER_RATE_HIGH_SPEED = 0.085
STEER_RATE_VERY_HIGH  = 0.060

# Sui rettilinei ad alta velocità riduciamo i micro-comandi.
HIGH_SPEED_STRAIGHT_DAMPING = True
STRAIGHT_DAMP_SPEED = 175.0
STRAIGHT_DAMP_CURV  = 0.10
STRAIGHT_DAMP_TP    = 0.34
STRAIGHT_DAMP_ANGLE = 0.10
STRAIGHT_STEER_GAIN = 0.30

STEER_DEADBAND = 0.006


# =================================================================
# ACCEL BOOST: molto prudente
# =================================================================

USE_ACCEL_BOOST = True
BOOST_TRACK9_MIN   = 138.0
BOOST_SIDE_RATIO   = 0.86
BOOST_CURV_MAX     = 0.050
BOOST_ANGLE_MAX    = 0.045
BOOST_TRACKPOS_MAX = 0.18


# =================================================================
# ANTI-STALLO
# =================================================================

MIN_SPEED_STALL = 4.0
STALL_PATIENCE  = 35


# =================================================================
# CARICAMENTO MODELLO
# =================================================================

print("[ai_driver_v16] Caricamento modelli...")
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

    print(f"[ai_driver_v16] Modello caricato. Feature richieste ({len(FEATURES)}):")
    for i, f in enumerate(FEATURES):
        print(f"   [{i:2d}] {f}")

    print("[ai_driver_v16] Modalità: V15 BASE / BRAKE RELEASE")
    print(f"[ai_driver_v16] Speed governor: {USE_SPEED_GOVERNOR}")
    print(f"[ai_driver_v16] Steer filter: {USE_STEER_FILTER}")
    print(f"[ai_driver_v16] Meta restart irrecoverable: {USE_META_RESTART_ON_IRRECOVERABLE}")

except Exception as e:
    print(f"[ai_driver_v16] ERRORE caricamento: {e}")
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
                print("[ai_driver_v16] Connesso a TORCS.")
                return so
        except socket.error:
            print("[ai_driver_v16] In attesa di TORCS sulla porta 3001...")


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
    x = np.array([[stato.get(name, 0.0) for name in FEATURES]], dtype=np.float32)
    return x


# =================================================================
# GEOMETRIA
# =================================================================

def curvature_score(track):
    """
    Stima curvatura tramite coppie simmetriche.
    0 = dritto, alto = curva/asimmetria.
    """
    if len(track) < 19:
        return 0.0

    pairs = [
        (0, 18, 0.40),
        (1, 17, 0.35),
        (2, 16, 0.15),
        (3, 15, 0.07),
        (4, 14, 0.03),
    ]

    score = 0.0
    for r_idx, l_idx, w in pairs:
        r = float(track[r_idx])
        l = float(track[l_idx])
        score += w * ((l - r) / (l + r + 1e-6))
    return float(score)


def speed_hint_is_fast(front, curv):
    """
    Helper prudente: se front e' sotto 115 m, anche con curvatura bassa
    conviene iniziare a preparare la frenata. Serve per la prima curva.
    """
    return front < 115.0 or curv > 0.08


def target_speed(track, angle, track_pos):
    """
    Velocità target stabile.
    Serve per evitare esattamente il problema dei log:
      - 200+ sul dritto va bene;
      - ma non devi arrivare a 160-180 dentro una curva con tr9 45 e curv 0.35+.
    """
    if len(track) < 19:
        return 120.0

    front = float(track[9])
    curv = abs(curvature_score(track))

    # Base da curvatura
    if curv < 0.06:
        v = 225.0
    elif curv < 0.13:
        v = 195.0
    elif curv < 0.22:
        v = 155.0
    elif curv < 0.32:
        v = 126.0
    elif curv < 0.45:
        v = 96.0
    elif curv < 0.60:
        v = 70.0
    else:
        v = 52.0

    # Base da spazio davanti
    if front < 9.0:
        v = min(v, 36.0)
    elif front < 12.0:
        v = min(v, 42.0)
    elif front < 16.0:
        v = min(v, 50.0)
    elif front < 20.0:
        v = min(v, 62.0)
    elif front < 30.0:
        v = min(v, 82.0)
    elif front < 42.0:
        v = min(v, 100.0)
    elif front < 58.0:
        if curv < 0.10:
            v = min(v, 150.0)
        elif curv < 0.22:
            v = min(v, 128.0)
        else:
            v = min(v, 100.0)
    elif front < 75.0 and curv > 0.22:
        v = min(v, 114.0)

    # Penalità se non siamo centrati/allineati.
    v -= 50.0 * max(0.0, abs(track_pos) - 0.16)
    v -= 42.0 * max(0.0, abs(angle) - 0.08)

    # Early braking anche se curv sembra bassa: su Corkscrew il frontale crolla
    # molto prima che la curvatura stimata diventi enorme.
    if front < EARLY_FRONT_1 and speed_hint_is_fast(front, curv):
        v = min(v, EARLY_TARGET_1)
    if front < EARLY_FRONT_2:
        v = min(v, EARLY_TARGET_2)
    if front < EARLY_FRONT_3:
        v = min(v, EARLY_TARGET_3)
    if front < EARLY_FRONT_4:
        v = min(v, EARLY_TARGET_4)

    # Situazioni viste nei log: tr9 molto basso + curv alta + macchina verso il bordo.
    # Qui bisogna salvare stabilità, non tempo.
    if (front < CRITICAL_FRONT and curv > CRITICAL_CURV) or abs(track_pos) > EDGE_WARN_TP:
        v = min(v, 54.0)
    if abs(track_pos) > OFFTRACK_GUARD_TP:
        v = min(v, 38.0)

    # Mini-gain rispetto alla V11: solo quando siamo ancora ben dentro la pista.
    # Non si applica alle curve strette dove la V13 ha preso il muro.
    if (
        USE_SAFE_CORNER_GAIN
        and front > SAFE_GAIN_MIN_FRONT
        and curv < SAFE_GAIN_MAX_CURV
        and abs(track_pos) < SAFE_GAIN_MAX_TP
        and abs(angle) < SAFE_GAIN_MAX_ANGLE
    ):
        v += 5.0

    return float(np.clip(v, 35.0, 235.0))


# =================================================================
# SAFETY STERZO
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

    if abs_an <= ANGLE_BLEND:
        w_an = 0.0
    elif abs_an >= ANGLE_SAFE:
        w_an = 1.0
    else:
        w_an = (abs_an - ANGLE_BLEND) / (ANGLE_SAFE - ANGLE_BLEND)

    return float(np.clip(max(w_tp, w_an), 0.0, 1.0))


def smooth_safety_w(w_raw, w_prev):
    if not USE_SAFETY_W_SMOOTHING:
        return w_raw
    if w_raw > 0.85:
        return w_raw
    return float(SAFETY_W_ALPHA * w_raw + (1.0 - SAFETY_W_ALPHA) * w_prev)


# =================================================================
# CAMBIO MARCE
# =================================================================

def gear_logic(speed_kmh, current_gear):
    """
    Niente retro automatica.
    """
    if speed_kmh < 8.0:
        return 1

    down_thresh = {2: 35, 3: 75, 4: 115, 5: 160, 6: 200}
    up_thresh   = {1: 55, 2: 95, 3: 135, 4: 180, 5: 215}

    g = current_gear if current_gear >= 1 else 1

    if g < 6 and speed_kmh > up_thresh.get(g, 999):
        return g + 1
    if g > 1 and speed_kmh < down_thresh.get(g, 0):
        return g - 1
    return g


# =================================================================
# STERZO
# =================================================================

def steer_rate_limit(steer_target, steer_prev, speed_x):
    speed = abs(speed_x)

    if speed > 180:
        max_delta = STEER_RATE_VERY_HIGH
    elif speed > 125:
        max_delta = STEER_RATE_HIGH_SPEED
    elif speed > 70:
        max_delta = STEER_RATE_MID_SPEED
    else:
        max_delta = STEER_RATE_LOW_SPEED

    delta = float(np.clip(steer_target - steer_prev, -max_delta, max_delta))
    return float(steer_prev + delta)


def filter_steer(steer_target, steer_prev, speed_x, safety_w, curv, angle, track_pos):
    """
    Anti-zig-zag:
      - filtra micro-movimenti ad alta velocità;
      - ma se il modello cambia segno in modo netto, passa subito al segno corretto.
    """
    steer_target = float(np.clip(steer_target, -1.0, 1.0))

    if not USE_STEER_FILTER or safety_w > 0.90:
        steer = steer_target
    else:
        sign_change = (steer_target * steer_prev) < 0.0 and abs(steer_target) > 0.08

        if sign_change:
            # Non restare col segno sbagliato: attraversa subito lo zero,
            # ma con ampiezza controllata.
            if abs(speed_x) > 170:
                steer = float(np.sign(steer_target) * min(abs(steer_target), 0.16))
            elif abs(speed_x) > 120:
                steer = float(np.sign(steer_target) * min(abs(steer_target), 0.22))
            else:
                steer = 0.75 * steer_target + 0.25 * steer_prev
        else:
            steer = steer_rate_limit(steer_target, steer_prev, speed_x)

    # Sul rettilineo veloce, niente micro zig-zag.
    if (
        HIGH_SPEED_STRAIGHT_DAMPING
        and speed_x > STRAIGHT_DAMP_SPEED
        and abs(curv) < STRAIGHT_DAMP_CURV
        and abs(track_pos) < STRAIGHT_DAMP_TP
        and abs(angle) < STRAIGHT_DAMP_ANGLE
    ):
        steer *= STRAIGHT_STEER_GAIN

    if abs(steer) < STEER_DEADBAND and speed_x > 70:
        steer = 0.0

    return float(np.clip(steer, -1.0, 1.0))


# =================================================================
# FRENO / GAS
# =================================================================

def needs_prebrake(track_9, speed_x):
    return track_9 < PREBRAKE_TRACK9_THRESHOLD and speed_x > PREBRAKE_SPEED_THRESHOLD


def limit_brake_rate(brake, brake_prev, critical=False):
    if not USE_BRAKE_RATE_LIMIT:
        return float(np.clip(brake, 0.0, 1.0))

    # In curva critica il freno deve salire più rapidamente, altrimenti
    # frena troppo tardi e l'auto va larga.
    rise = 0.30 if critical else BRAKE_RISE_MAX
    fall = BRAKE_FALL_MAX

    if brake > brake_prev:
        brake = min(brake, brake_prev + rise)
    else:
        brake = max(brake, brake_prev - fall)

    return float(np.clip(brake, 0.0, 1.0))


def apply_speed_governor(accel, brake, speed_x, track, angle, track_pos):
    """
    Governatore V16:
    - resta prudente quando sei davvero sopra target;
    - ma rilascia il freno quando sei gia' sotto target o vicino al target.
    Questo e' il collo di bottiglia visto nei log V15: brk% circa 30+ e GOV circa 50%.
    """
    if not USE_SPEED_GOVERNOR or len(track) < 19:
        return accel, brake, 999.0

    front = float(track[9])
    curv = abs(curvature_score(track))
    v_target = target_speed(track, angle, track_pos)
    over = speed_x - v_target

    desired_brake = 0.0

    # Brake progressivo, leggermente meno invasivo della V15.
    if over > 70.0:
        desired_brake = GOV_BRAKE_VERY_HIGH
    elif over > 50.0:
        desired_brake = GOV_BRAKE_HIGH
    elif over > 32.0:
        desired_brake = GOV_BRAKE_MED
    elif over > 16.0:
        desired_brake = GOV_BRAKE_LOW
    elif over > 7.0:
        desired_brake = GOV_BRAKE_TINY

    # Early brake: non togliamo la sicurezza, ma non deve diventare "panic brake" troppo presto.
    if front < EARLY_FRONT_1 and speed_x > EARLY_BRAKE_SPEED_1:
        desired_brake = max(desired_brake, EARLY_BRAKE_MIN_1)
        accel = min(accel, 0.18)
    if front < EARLY_FRONT_2 and speed_x > EARLY_BRAKE_SPEED_2:
        desired_brake = max(desired_brake, EARLY_BRAKE_MIN_2)
        accel = 0.0
    if front < EARLY_FRONT_3 and speed_x > EARLY_BRAKE_SPEED_3:
        desired_brake = max(desired_brake, EARLY_BRAKE_MIN_3)
        accel = 0.0

    # Critico solo se siamo davvero in pericolo.
    # La V15 era ancora troppo severa in molti tratti medio-lenti.
    critical = (
        (front < CRITICAL_FRONT and curv > CRITICAL_CURV)
        or abs(track_pos) > EDGE_WARN_TP
        or (front < 12.0 and speed_x > 48.0)
        or (front < 68.0 and speed_x > 172.0 and curv > 0.08)
    )

    if critical:
        if speed_x > v_target + 24.0:
            desired_brake = max(desired_brake, 0.52)
        elif speed_x > v_target + 12.0:
            desired_brake = max(desired_brake, 0.36)
        elif speed_x > v_target + 4.0:
            desired_brake = max(desired_brake, 0.20)
        accel = min(accel, 0.16)

    # Se siamo chiaramente sopra target, niente gas.
    # Se siamo appena sopra, non serve tagliare tutto.
    if over > 8.0:
        accel = 0.0
    elif over > 3.0:
        accel = min(accel, 0.22)

    # Cap al freno del modello.
    if not critical:
        if over < 0.0:
            brake = min(brake, RELEASE_BRAKE_CAP)
        elif over < 10.0:
            brake = min(brake, NEAR_TARGET_BRAKE_CAP)
        elif over < 24.0:
            brake = min(brake, 0.14)
        elif over < 42.0:
            brake = min(brake, 0.26)
        elif over < 62.0:
            brake = min(brake, 0.40)

    brake = max(brake, desired_brake)

    # Brake-release quando siamo sotto target e in controllo.
    if USE_BRAKE_RELEASE_UNDER_TARGET:
        safe_release = (
            front > RELEASE_FRONT_MIN
            and curv < RELEASE_MAX_CURV
            and abs(track_pos) < RELEASE_MAX_TP
            and abs(angle) < RELEASE_MAX_ANGLE
            and speed_x < v_target - RELEASE_MARGIN
            and not critical
        )

        if safe_release:
            brake = min(brake, RELEASE_BRAKE_CAP)

            # Gas progressivo in base alla curva.
            if curv < 0.18:
                if speed_x < 140.0:
                    accel = max(accel, 0.76)
                else:
                    accel = max(accel, 0.55)
            elif curv < 0.34:
                if speed_x < 115.0:
                    accel = max(accel, 0.64)
                else:
                    accel = max(accel, 0.42)
            else:
                if speed_x < 78.0:
                    accel = max(accel, 0.48)
                else:
                    accel = max(accel, 0.28)

    # Uscita curva: gas controllato solo se siamo stabili.
    if USE_CORNER_EXIT_ASSIST:
        stable_exit = (
            front > EXIT_MIN_FRONT
            and abs(angle) < EXIT_MAX_ANGLE
            and abs(track_pos) < EXIT_MAX_TRACKPOS
            and speed_x < v_target - 10.0
        )

        if stable_exit:
            brake = min(brake, 0.03)
            if speed_x < 45.0:
                accel = max(accel, 0.42)
            elif speed_x < 90.0:
                accel = max(accel, 0.58)
            else:
                accel = max(accel, 0.66)

    return accel, brake, v_target

def is_open_track(track, angle, track_pos):
    if len(track) < 19:
        return False

    t8, t9, t10 = float(track[8]), float(track[9]), float(track[10])
    curv = abs(curvature_score(track))

    if t9 < BOOST_TRACK9_MIN:
        return False
    if t8 < BOOST_SIDE_RATIO * t9 or t10 < BOOST_SIDE_RATIO * t9:
        return False
    if curv > BOOST_CURV_MAX:
        return False
    if abs(angle) > BOOST_ANGLE_MAX:
        return False
    if abs(track_pos) > BOOST_TRACKPOS_MAX:
        return False

    return True


# =================================================================
# RECOVERY
# =================================================================

def is_spinning(angle):
    return abs(angle) > SPIN_ANGLE


def spin_recovery_action(angle, track_pos, speed_x, recovery_dir, recovery_counter):
    """
    Recovery senza pista in contromano.

    Se siamo girati quasi a 180°, non acceleriamo come pazzi:
    prima rallentiamo, poi proviamo a ruotare piano in una direzione stabile.
    """
    abs_angle = abs(angle)

    # Se va indietro, frena.
    if speed_x < ANTI_REVERSE_SPEED:
        return {
            'steer': float(recovery_dir),
            'accel': 0.0,
            'brake': ANTI_REVERSE_BRAKE,
            'gear': 1,
        }

    # Se è quasi contromano, non farlo correre: fermalo/raddrizzalo.
    if abs_angle > 2.20:
        if speed_x > 8.0:
            return {
                'steer': float(recovery_dir),
                'accel': 0.0,
                'brake': 0.85,
                'gear': 1,
            }
        return {
            'steer': float(recovery_dir),
            'accel': 0.12,
            'brake': 0.0,
            'gear': 1,
        }

    # Testacoda ma non 180 pieno: raddrizza piano.
    if speed_x > 35.0:
        return {
            'steer': float(np.clip(-angle * 0.45, -1.0, 1.0)),
            'accel': 0.0,
            'brake': 0.65,
            'gear': 1,
        }

    steer = float(np.clip(-angle * 1.10 - track_pos * 0.20, -1.0, 1.0))
    return {
        'steer': steer,
        'accel': 0.20,
        'brake': 0.0,
        'gear': 1,
    }


def offtrack_guard(accel, brake, steer, gear, track_pos, angle, speed_x):
    """
    Guard bordo/fuori pista.
    v10 entra già da |tp| > 0.82, non aspetta |tp| > 1.05.
    """
    abs_tp = abs(track_pos)

    if abs_tp <= OFFTRACK_GUARD_TP:
        return accel, brake, steer, gear

    # Sterza verso il centro, non seguire più l'MLP.
    steer_center = float(np.clip(-track_pos * 0.95 + angle * 0.45, -1.0, 1.0))

    if abs_tp > 1.05:
        # Fuori pista pieno: rallenta e rientra piano.
        if speed_x > 28.0:
            accel = 0.0
            brake = max(brake, 0.55)
        else:
            accel = 0.22
            brake = 0.0
    else:
        # Bordo pista: evita di uscire completamente.
        if speed_x > 70.0:
            accel = 0.0
            brake = max(brake, 0.38)
        elif speed_x > 45.0:
            accel = 0.0
            brake = max(brake, 0.24)
        else:
            accel = min(max(accel, 0.18), 0.28)
            brake = min(brake, 0.05)

    return accel, brake, steer_center, 1


def low_speed_relaunch_guard(accel, brake, steer, gear, speed_x, track_pos, angle, track):
    """
    Risolve il problema del log finale:
    dopo essere rientrata in pista, l'auto resta a 0 km/h con brake residuo
    e accel=0 dentro l'ultima curva.
    """
    if not USE_LOW_SPEED_RELAUNCH or len(track) < 19:
        return accel, brake, steer, gear

    front = float(track[9])

    if (
        speed_x < RELAUNCH_SPEED_MAX
        and abs(track_pos) < RELAUNCH_TP_MAX
        and abs(angle) < RELAUNCH_ANGLE_MAX
        and front > RELAUNCH_FRONT_MIN
    ):
        brake = 0.0
        gear = 1

        # Se siamo quasi fermi, serve una spinta chiara ma non violenta.
        if speed_x < 4.0:
            accel = max(accel, 0.36)
        elif speed_x < 10.0:
            accel = max(accel, 0.30)
        else:
            accel = max(accel, 0.24)

    return accel, brake, steer, gear


def hard_final_relaunch(accel, brake, steer, gear, speed_x, track_pos, angle, track):
    """
    Ultima barriera anti-stop.
    La mettiamo immediatamente prima dell'invio comando, cosi' nessun altro blocco
    puo' rimettere accel=0/brake>0 quando la macchina e' quasi ferma ma recuperabile.
    """
    if len(track) < 19:
        return accel, brake, steer, gear, False

    front = float(track[9])
    recoverable = (
        speed_x < HARD_RELAUNCH_SPEED
        and abs(track_pos) < HARD_RELAUNCH_TP
        and abs(angle) < HARD_RELAUNCH_ANGLE
        and front > HARD_RELAUNCH_FRONT
    )

    if not recoverable:
        return accel, brake, steer, gear, False

    gear = 1
    brake = 0.0

    if speed_x < 1.5:
        accel = 0.48
    elif speed_x < 4.0:
        accel = 0.42
    else:
        accel = 0.34

    return accel, brake, steer, gear, True


def last_curve_apex_tuning(accel, brake, speed_x, track_pos, angle, track):
    """
    Micro-tuning per l'ultima curva/hairpin:
    se siamo gia' lenti, non ha senso inchiodare ancora.
    Serve a evitare il caso V14:
      v~46, tr9~8, curv~0.44, brake~0.42.
    """
    if not USE_LAST_CURVE_APEX_TUNING or len(track) < 19:
        return accel, brake, False

    front = float(track[9])
    curv = abs(curvature_score(track))

    active = (
        front < APEX_FRONT_MAX
        and curv > APEX_CURV_MIN
        and speed_x < APEX_SPEED_MAX
        and abs(track_pos) < APEX_TP_MAX
        and abs(angle) < APEX_ANGLE_MAX
    )

    if not active:
        return accel, brake, False

    # Se siamo gia' a velocita' da hairpin, rilascia il freno.
    brake = min(brake, APEX_BRAKE_CAP)

    # Uscita apice: gas dolce, non pieno.
    if speed_x < APEX_EXIT_SPEED:
        accel = max(accel, APEX_EXIT_ACCEL)

    return accel, brake, True


def wall_guard(accel, brake, steer, gear, speed_x, track_pos, angle, track):
    """
    Guard anti-muro V16.
    Tiene la protezione della V15, ma lascia più scorrimento a bassa velocità
    nell'ultima curva. Il problema non e' piu' il danno: e' il tempo perso.
    """
    if not USE_WALL_GUARD or len(track) < 19:
        return accel, brake, steer, gear, False

    front = float(track[9])
    curv = abs(curvature_score(track))
    active = False

    danger = (
        (curv > WALL_CURV_STRONG and front < WALL_FRONT_WARN)
        or front < WALL_FRONT_DANGER
        or (abs(track_pos) > WALL_TP_WARN and (curv > 0.25 or front < 45.0))
        or (abs(angle) > WALL_ANGLE_WARN and front < 30.0)
    )

    if not danger:
        return accel, brake, steer, gear, False

    active = True

    if abs(track_pos) > 0.55:
        steer_center = float(np.clip(-track_pos * 0.75 + angle * 0.40, -1.0, 1.0))
        steer = 0.65 * steer + 0.35 * steer_center

    # Curva stretta: conserva stabilità, ma sotto 50 km/h non bloccare più.
    if curv > WALL_CURV_STRONG:
        if speed_x > 72.0:
            accel = 0.0
            brake = max(brake, 0.26)
        elif speed_x > 54.0:
            accel = min(accel, 0.12)
            brake = max(brake, 0.16)
        elif speed_x > 38.0:
            accel = max(min(accel, 0.24), 0.16)
            brake = min(brake, 0.06)
        else:
            accel = max(min(accel, 0.34), 0.26)
            brake = min(brake, 0.03)

    # Frontale chiuso: proteggi solo quando la velocità è ancora alta.
    if front < WALL_FRONT_DANGER:
        if speed_x > 62.0:
            accel = 0.0
            brake = max(brake, 0.34)
        elif speed_x > 45.0:
            accel = min(accel, 0.10)
            brake = min(max(brake, 0.14), 0.20)
        elif speed_x > 30.0:
            accel = max(min(accel, 0.28), 0.20)
            brake = min(brake, 0.05)
        else:
            accel = max(min(accel, 0.34), 0.26)
            brake = min(brake, 0.03)

    if abs(angle) > WALL_ANGLE_WARN and speed_x > 48.0:
        accel = min(accel, 0.08)
        brake = max(brake, 0.18)
    elif abs(angle) > WALL_ANGLE_WARN and speed_x <= 48.0:
        accel = min(accel, 0.24)
        brake = min(brake, 0.06)

    accel, brake, apex_active = last_curve_apex_tuning(
        accel, brake, speed_x, track_pos, angle, track
    )

    return accel, brake, float(np.clip(steer, -1.0, 1.0)), gear, active


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
    recovery_dir = 1.0

    steer_prev = 0.0
    brake_prev = 0.0
    safety_w_prev = 0.0

    # diagnostica
    tick_count = 0
    brake_count = 0
    safety_count = 0
    boost_count = 0
    recovery_count = 0
    governor_count = 0
    speed_sum = 0.0
    speed_max = 0.0

    while True:
        try:
            raw, _ = so.recvfrom(DATA_SIZE)
            msg = raw.decode('utf-8')

            if '***shutdown***' in msg:
                print("[ai_driver_v16] Server in shutdown.")
                break

            if '***restart***' in msg:
                print("[ai_driver_v16] Restart richiesto dal server.")
                stall_counter = 0
                in_recovery = False
                recovery_counter = 0
                recovery_dir = 1.0
                gear = 1
                steer_prev = 0.0
                brake_prev = 0.0
                safety_w_prev = 0.0
                continue

            if not msg:
                continue

            S = parse_server_str(msg)

            track_pos = S.get('trackPos', 0.0)
            angle     = S.get('angle',    0.0)
            speed_x   = S.get('speedX',   0.0)
            track     = S.get('track',    [200.0] * 19)
            track_9   = track[9] if len(track) > 9 else 200.0

            curv = abs(curvature_score(track))
            v_target_log = target_speed(track, angle, track_pos)

            # =====================================================
            # RECOVERY
            # =====================================================
            if is_spinning(angle):
                if not in_recovery:
                    recovery_dir = -1.0 if angle > 0.0 else 1.0
                    recovery_counter = 0
                in_recovery = True
                recovery_counter += 1
            elif in_recovery and abs(angle) < 0.45 and abs(track_pos) < 0.95 and speed_x > -1.0:
                print(f"[ai_driver_v16] Recovery completato dopo {recovery_counter} tick")
                in_recovery = False
                recovery_counter = 0
                gear = 1
                safety_w_prev = 0.0

            if in_recovery and recovery_counter > RECOVERY_MAX_TICKS:
                print(f"[ai_driver_v16] Recovery troppo lungo ({recovery_counter} tick). Blocco contromano.")
                if USE_META_RESTART_ON_IRRECOVERABLE:
                    so.sendto(b"(meta 1)", (HOST, PORT))
                    in_recovery = False
                    recovery_counter = 0
                    continue

            boost_active = False
            governor_active = False

            if in_recovery:
                act = spin_recovery_action(angle, track_pos, speed_x, recovery_dir, recovery_counter)

                steer = act['steer']
                accel = act['accel']
                brake = act['brake']
                gear  = act['gear']

                w_log = 99.0
                steer_mlp = float('nan')

                steer_prev = steer
                brake_prev = brake
                safety_w_prev = 1.0

            else:
                # =================================================
                # GUIDA NORMALE
                # =================================================
                x = build_state(S)
                x_scaled = scaler.transform(x)
                y = model.predict(x_scaled)[0]

                steer_mlp = float(np.clip(y[0], -1.0, 1.0))
                accel     = float(np.clip(y[1],  0.0, 1.0))
                brake     = float(np.clip(y[2],  0.0, 1.0))

                # Safety steering
                steer_rec = recovery_steer(track_pos, angle)
                w_raw = blend_factor(track_pos, angle)

                # Boost safety nelle curve difficili o vicino al bordo.
                # Serve a evitare il caso del log: entra in curva, va largo,
                # tp supera 1.0 e poi deve recuperare fuori pista.
                if (curv > CRITICAL_CURV or track_9 < CRITICAL_FRONT) and abs(track_pos) > 0.36:
                    w_raw = max(w_raw, min(0.75, (abs(track_pos) - 0.30) / 0.55))
                if abs(track_pos) > EDGE_WARN_TP:
                    w_raw = max(w_raw, 0.70)

                w = smooth_safety_w(w_raw, safety_w_prev)
                safety_w_prev = w

                steer_target = (1.0 - w) * steer_mlp + w * steer_rec
                steer = filter_steer(
                    steer_target, steer_prev, speed_x, w, curv, angle, track_pos
                )
                steer_prev = steer

                # Se vicini al bordo, non serve gas pieno.
                if abs(track_pos) > 0.78:
                    accel = min(accel, 0.35)

                # Prefrenata extra leggera
                if needs_prebrake(track_9, speed_x):
                    brake = max(brake, PREBRAKE_FORCE)

                # Speed governor: cuore della v9
                accel_before_gov = accel
                brake_before_gov = brake
                accel, brake, v_target_log = apply_speed_governor(
                    accel, brake, speed_x, track, angle, track_pos
                )
                if abs(accel - accel_before_gov) > 1e-3 or abs(brake - brake_before_gov) > 1e-3:
                    governor_active = True

                # Boost solo su rettilineo molto chiaro e sotto target.
                if (
                    USE_ACCEL_BOOST
                    and brake < 0.04
                    and speed_x < v_target_log - 8.0
                    and is_open_track(track, angle, track_pos)
                ):
                    accel = max(accel, 0.96)
                    boost_active = True

                # Non accelerare se stai frenando davvero.
                if brake > 0.10:
                    accel = 0.0

                # Rate limit freno alla fine.
                critical_brake_zone = (
                    (track_9 < CRITICAL_FRONT and curv > CRITICAL_CURV)
                    or abs(track_pos) > EDGE_WARN_TP
                    or (track_9 < 12.0 and speed_x > 45.0)
                )
                brake = limit_brake_rate(brake, brake_prev, critical=critical_brake_zone)
                brake_prev = brake

                gear = gear_logic(speed_x, gear)
                w_log = w

            # =====================================================
            # GUARD FINALI
            # =====================================================

            # Anti-retromarcia: non andare contromano all'indietro.
            if speed_x < ANTI_REVERSE_SPEED:
                accel = 0.0
                brake = max(brake, ANTI_REVERSE_BRAKE)
                gear = 1

            # Fuori pista / bordo pista: prima rientra, poi pensa al tempo.
            accel, brake, steer, gear = offtrack_guard(
                accel, brake, steer, gear, track_pos, angle, speed_x
            )

            # Se dopo un rientro si pianta a 0 km/h dentro l'ultima curva, riparte piano.
            accel, brake, steer, gear = low_speed_relaunch_guard(
                accel, brake, steer, gear, speed_x, track_pos, angle, track
            )

            # Anti-muro: impedisce gas pieno in curve cieche/strette.
            wall_guard_active = False
            accel, brake, steer, gear, wall_guard_active = wall_guard(
                accel, brake, steer, gear, speed_x, track_pos, angle, track
            )

            # Anti-stallo
            if not in_recovery and abs(speed_x) < MIN_SPEED_STALL:
                stall_counter += 1
                if stall_counter > STALL_PATIENCE:
                    accel = 0.45
                    brake = 0.0
                    gear  = 1

                    if stall_counter > STALL_PATIENCE * 3:
                        print("[ai_driver_v16] STALLO PROLUNGATO: meta=1")
                        so.sendto(b"(meta 1)", (HOST, PORT))
                        stall_counter = 0
                        continue
            else:
                stall_counter = 0

            # Hard anti-stop finale: deve stare subito prima dell'output.
            hard_relaunch_active = False
            accel, brake, steer, gear, hard_relaunch_active = hard_final_relaunch(
                accel, brake, steer, gear, speed_x, track_pos, angle, track
            )
            if hard_relaunch_active:
                brake_prev = 0.0

            # =====================================================
            # INVIO COMANDO
            # =====================================================
            out = (f"(accel {accel:.3f})(brake {brake:.3f})"
                   f"(gear {gear})(steer {steer:.3f})"
                   f"(clutch 0)(focus 0)(meta 0)")
            so.sendto(out.encode(), (HOST, PORT))

            # =====================================================
            # DIAGNOSTICA
            # =====================================================
            tick_count += 1
            speed_sum += speed_x
            speed_max = max(speed_max, speed_x)

            if brake > 0.10:
                brake_count += 1
            if not in_recovery and w_log > 0.50:
                safety_count += 1
            if boost_active:
                boost_count += 1
            if in_recovery:
                recovery_count += 1
            if governor_active:
                governor_count += 1

            now = time.time()
            if now - last_log_t > 1.0:
                if in_recovery:
                    tag = "REC"
                elif wall_guard_active:
                    tag = "WALL"
                elif hard_relaunch_active:
                    tag = "REL"
                elif governor_active:
                    tag = "GOV"
                elif boost_active:
                    tag = "BST"
                else:
                    tag = f"w={w_log:.2f}"

                avg_v = speed_sum / max(1, tick_count)
                brake_pct = 100.0 * brake_count / max(1, tick_count)
                safety_pct = 100.0 * safety_count / max(1, tick_count)
                boost_pct = 100.0 * boost_count / max(1, tick_count)
                rec_pct = 100.0 * recovery_count / max(1, tick_count)
                gov_pct = 100.0 * governor_count / max(1, tick_count)

                print(
                    f"v={speed_x:+6.1f} km/h  avg={avg_v:5.1f}  max={speed_max:5.1f}  "
                    f"vt={v_target_log:5.1f}  tp={track_pos:+.2f}  ang={angle:+.2f}  "
                    f"curv={curv:.2f}  tr9={track_9:5.1f}  "
                    f"s_mlp={steer_mlp:+.2f}  s={steer:+.2f}  "
                    f"a={accel:.2f}  b={brake:.2f}  g={gear}  {tag}  "
                    f"brk%={brake_pct:4.1f} saf%={safety_pct:4.1f} "
                    f"bst%={boost_pct:4.1f} gov%={gov_pct:4.1f} rec%={rec_pct:4.1f}"
                )

                last_log_t = now

        except socket.timeout:
            print("[ai_driver_v16] Timeout socket, ritento...")
            continue
        except Exception as e:
            print(f"[ai_driver_v16] ERRORE LOOP: {e}")


if __name__ == "__main__":
    run_ai()
