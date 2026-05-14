"""
TORCS bot di guida.

Pilota la macchina sul circuito e logga in un CSV, per ogni step, lo stato
del simulatore e l'azione decisa.

Architettura del controllore:
  - sterzo: heading feedback + cross-track feedback + feedforward sui sensori
  - velocita' target dipendente dalla distanza vista davanti e dalla curvatura
  - throttle/brake con deadband, trail braking e riaccelerazione anticipata
  - traction control via wheelSpinVel
  - cambio marce con isteresi e cooldown
"""

import socket
import sys
import getopt
import os
import time
import csv
import math
import threading
from pynput.keyboard import Key, Listener

PI = 3.14159265359
data_size = 2**17

ophelp  = 'Options:\n'
ophelp += ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp += ' --port, -p <port>    TORCS port. [3001]\n'
ophelp += ' --id, -i <id>        ID for server. [SCR]\n'
ophelp += ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp += ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp += ' --track, -t <track>  Your name for this track. [unknown]\n'
ophelp += ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp += ' --debug, -d          Output full telemetry.\n'
ophelp += ' --help, -h           Show this help.\n'
ophelp += ' --version, -v        Show current version.'
usage   = 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage   = usage + ophelp
version = "il-dataset-1.0"


def clip(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v


# =====================================================================
# NETWORKING / PROTOCOLLO SCR
# =====================================================================

class Client():
    def __init__(self, H=None, p=None, i=None, e=None, t=None, s=None, d=None, vision=False):
        self.vision = vision
        self.host = 'localhost'
        self.port = 3001
        self.sid = 'SCR'
        self.maxEpisodes = 1
        self.trackname = 'unknown'
        self.stage = 3
        self.debug = False
        self.maxSteps = 100000
        self.parse_the_command_line()
        if H: self.host = H
        if p: self.port = p
        if i: self.sid = i
        if e: self.maxEpisodes = e
        if t: self.trackname = t
        if s: self.stage = s
        if d: self.debug = d
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, a)
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except (socket.error, ConnectionResetError):
                print("In attesa che TORCS avvii la gara sulla porta %d..." % self.port)
                if n_fail < 0:
                    print("Riavvio TORCS (Windows mode)...")
                    os.system('taskkill /IM torcs.exe /F')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('start torcs -nofuel -nodamage -nolaptime')
                    else:
                        os.system('start torcs -nofuel -nodamage -nolaptime -vision')
                    time.sleep(1.0)
                    n_fail = 5
                n_fail -= 1
            if '***identified***' in sockdata:
                print("Client connected on %d." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(
                sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                ['host=', 'port=', 'id=', 'steps=',
                 'episodes=', 'track=', 'stage=',
                 'debug', 'help', 'version'])
        except getopt.error:
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] in ('-h', '--help'):
                    sys.exit(0)
                if opt[0] in ('-d', '--debug'):
                    self.debug = True
                if opt[0] in ('-p', '--port'):
                    self.port = int(opt[1])
                if opt[0] in ('-m', '--steps'):
                    self.maxSteps = int(opt[1])
                if opt[0] in ('-t', '--track'):
                    self.trackname = opt[1]
        except ValueError:
            sys.exit(-1)

    def get_servers_input(self):
        if not self.so: return
        sockdata = str()
        while True:
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error:
                pass
            if '***identified***' in sockdata:
                continue
            elif '***shutdown***' in sockdata:
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                self.shutdown()
                return
            elif not sockdata:
                continue
            else:
                self.S.parse_server_str(sockdata)
                break

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error:
            sys.exit(-1)

    def shutdown(self):
        if not self.so: return
        self.so.close()
        self.so = None


class ServerState():
    def __init__(self):
        self.d = dict()
    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])


class DriverAction():
    def __init__(self):
        self.d = {'accel': 0.2, 'brake': 0, 'clutch': 0, 'gear': 1,
                  'steer': 0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0}
    def clip_to_limits(self):
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'], 0, 1)
        self.d['accel'] = clip(self.d['accel'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not type(v) is list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out


def destringify(s):
    if not s: return s
    if type(s) is str:
        try: return float(s)
        except ValueError: return s
    elif type(s) is list:
        if len(s) < 2: return destringify(s[0])
        return [destringify(i) for i in s]


# =====================================================================
# OVERRIDE MANUALE DA TASTIERA
# I sample raccolti durante l'override NON vengono salvati nel CSV.
# =====================================================================

manual_steer = 0.0
manual_accel_override = None
manual_brake_override = None
is_manual_override = False

def on_press(key):
    global manual_steer, manual_accel_override, manual_brake_override, is_manual_override
    if key == Key.left:
        manual_steer = 0.6
        is_manual_override = True
    elif key == Key.right:
        manual_steer = -0.6
        is_manual_override = True
    elif key == Key.up:
        manual_accel_override = 1.0
        manual_brake_override = 0.0
        is_manual_override = True
    elif key == Key.down:
        manual_brake_override = 1.0
        manual_accel_override = 0.0
        is_manual_override = True

def on_release(key):
    global manual_steer, manual_accel_override, manual_brake_override, is_manual_override
    if key in (Key.left, Key.right):
        manual_steer = 0.0
    if key in (Key.up, Key.down):
        manual_accel_override = None
        manual_brake_override = None
    if manual_steer == 0.0 and manual_accel_override is None and manual_brake_override is None:
        is_manual_override = False

listener = Listener(on_press=on_press, on_release=on_release)
listener.start()


# =====================================================================
# LOGICA DI GUIDA
# =====================================================================

# --- Parametri sterzo ---
STEER_K_E              = 1.0
STEER_K_SOFT           = 5.0
STEER_K_HEADING        = 1.3
STEER_K_LOOKAHEAD      = 0.55
STEER_K_LOOKAHEAD_VSCALE = 0.006
ANGLE_FILTER_ALPHA     = 0.4
STEER_MAX_DELTA        = 0.13
STEER_RAD_TO_CMD       = 1.0

# --- Debug ---
DEBUG_STEERING       = True
DEBUG_PRINT_EVERY    = 25

# --- Parametri velocita' ---
SPEED_MAP = [
    (200.0, 310.0),
    (150.0, 250.0),
    (100.0, 195.0),
    ( 70.0, 155.0),
    ( 45.0, 118.0),
    ( 25.0,  85.0),
    (  0.0,  55.0),
]

CURV_THRESHOLD     = 0.08
CURV_FULL_CUT      = 0.45
CURV_MAX_REDUCTION = 0.35

# --- Cambio marce ---
RPM_UP   = 9000
RPM_DOWN = 6500
GEAR_MIN_SPEED = [0, 0, 35, 65, 100, 130, 175]

# --- Quality gate ---
QUALITY_MAX_TRACKPOS = 0.85
QUALITY_MAX_ANGLE    = 0.35
QUALITY_MIN_SPEED    = 5.0
WARMUP_STEPS         = 50

_state = {
    'prev_steer': 0.0,
    'steer_ema_slow': 0.0,
    'filtered_angle': 0.0,
    'prev_gear': 1,
    'gear_change_cooldown': 0,
    'debug_step': 0,
}


def effective_front_distance(track):
    """
    Distanza utile davanti per decidere la velocita'.
    Uso il MAX tra i sensori a piccolo angolo per evitare che un singolo
    sensore "pescando" sul muro laterale strangoli il target di velocita'.
    """
    candidates = []
    for idx in (3, 4, 9, 14, 15):
        v = track[idx]
        if v >= 0:
            candidates.append(v)
    if not candidates:
        return 50.0
    return max(candidates)


def lookup_target_speed(track):
    """Velocita' target = SPEED_MAP(distanza vista) modulata per curvatura."""
    front_dist = effective_front_distance(track)
    pts = SPEED_MAP

    if front_dist >= pts[0][0]:
        base_speed = pts[0][1]
    elif front_dist <= pts[-1][0]:
        base_speed = pts[-1][1]
    else:
        base_speed = pts[-1][1]
        for i in range(len(pts) - 1):
            d_hi, v_hi = pts[i]
            d_lo, v_lo = pts[i + 1]
            if d_lo <= front_dist <= d_hi:
                t = (front_dist - d_lo) / (d_hi - d_lo)
                base_speed = v_lo + t * (v_hi - v_lo)
                break

    curv = abs(estimate_curvature(track))
    if curv > CURV_THRESHOLD:
        t = clip((curv - CURV_THRESHOLD) / (CURV_FULL_CUT - CURV_THRESHOLD), 0.0, 1.0)
        reduction = CURV_MAX_REDUCTION * t
        base_speed *= (1.0 - reduction)

    return base_speed


def estimate_curvature(track):
    """
    Stima la curvatura imminente confrontando i sensori track laterali lontani.
    Sensori SCR (default angles): -45 -19 -12 -7 -4 -2.5 -1.7 -1 -0.5 0 0.5 1 1.7 2.5 4 7 12 19 45
    indici:                          0   1   2  3  4   5    6   7   8  9 10 11  12  13 14 15 16 17 18
    Convenzione: ritorno > 0 => sterzo POSITIVO richiesto (sinistra).
    """
    pairs = [
        ( 0, 18, 0.40),
        ( 1, 17, 0.35),
        ( 2, 16, 0.15),
        ( 3, 15, 0.07),
        ( 4, 14, 0.03),
    ]
    total = 0.0
    weight_sum = 0.0
    for i_left, i_right, w in pairs:
        l = track[i_left]
        r = track[i_right]
        if l < 0 or r < 0:
            continue
        if l > 195 and r > 195:
            continue
        denom = max(l + r, 1.0)
        total += w * (l - r) / denom
        weight_sum += w

    if weight_sum < 1e-6:
        return 0.0
    return total / weight_sum


def calculate_steering(S):
    """
    Controller di sterzo a tre componenti: heading feedback, cross-track
    feedback (Stanley), lookahead feedforward. In modalita' recovery
    (|angle|>0.5 rad o |trackPos|>0.9) disabilita il feedforward.
    """
    angle      = S.get('angle', 0.0)
    track_pos  = S.get('trackPos', 0.0)
    track      = S.get('track', [200.0] * 19)
    speedX_kmh = S.get('speedX', 0.0)

    is_anomalous = abs(angle) > 0.5 or abs(track_pos) > 0.9

    af = (1.0 - ANGLE_FILTER_ALPHA) * angle + ANGLE_FILTER_ALPHA * _state['filtered_angle']
    _state['filtered_angle'] = af

    k_h = STEER_K_HEADING * (1.8 if is_anomalous else 1.0)
    heading_term = k_h * af

    if is_anomalous:
        curvature = 0.0
    else:
        curvature = estimate_curvature(track)

    speed_ms = max(speedX_kmh / 3.6, 0.1)
    cross_track_term = -math.atan2(STEER_K_E * track_pos, STEER_K_SOFT + speed_ms)

    k_lookahead_eff = STEER_K_LOOKAHEAD + STEER_K_LOOKAHEAD_VSCALE * max(0.0, speedX_kmh)
    lookahead_term = k_lookahead_eff * curvature

    raw_steer = heading_term + cross_track_term + lookahead_term
    target_steer = clip(raw_steer * STEER_RAD_TO_CMD, -1.0, 1.0)

    # Slew rate limit scalato sulla velocita'
    if speedX_kmh < 60:
        max_delta = 0.18
    elif speedX_kmh < 100:
        max_delta = 0.12
    elif speedX_kmh < 140:
        max_delta = 0.08
    else:
        max_delta = 0.05
    delta = target_steer - _state['prev_steer']
    if delta >  max_delta: delta =  max_delta
    if delta < -max_delta: delta = -max_delta
    final_steer = _state['prev_steer'] + delta

    if speedX_kmh > 180.0:
        final_steer = clip(final_steer, -0.6, 0.6)

    final_steer = clip(final_steer, -1.0, 1.0)
    _state['steer_ema_slow'] = 0.90 * _state['steer_ema_slow'] + 0.10 * final_steer
    _state['prev_steer'] = final_steer

    if DEBUG_STEERING:
        _state['debug_step'] += 1
        if _state['debug_step'] % DEBUG_PRINT_EVERY == 0:
            front_eff = effective_front_distance(track)
            tgt_v = lookup_target_speed(track)
            print(
                f"[ctrl] v={speedX_kmh:6.1f} tgt={tgt_v:6.1f} d={tgt_v-speedX_kmh:+5.1f} "
                f"rpm={S.get('rpm',0):5.0f} gear={int(S.get('gear',0))} "
                f"front={front_eff:5.1f} ang={angle:+.2f} tp={track_pos:+.2f} "
                f"curv={curvature:+.2f} steer={final_steer:+.2f}"
            )

    return final_steer


def calculate_throttle_and_brake(S, target_speed):
    """
    Throttle/brake con deadband, trail braking e riaccelerazione anticipata
    in uscita di curva. In modalita' recovery: niente gas.
    """
    speedX = S.get('speedX', 0.0)
    angle = S.get('angle', 0.0)
    track_pos = S.get('trackPos', 0.0)

    is_anomalous = abs(angle) > 0.5 or abs(track_pos) > 0.9
    if is_anomalous:
        if speedX > 30:
            return (0.0, 0.15)
        return (0.0, 0.0)

    delta = target_speed - speedX
    BRAKE_DEADBAND = 8.0

    # Unwinding: sterzo attuale piu' vicino a zero della EMA lenta
    prev = _state['prev_steer']
    ema = _state['steer_ema_slow']
    is_unwinding = (abs(ema) > 0.20) and (abs(prev) < abs(ema) - 0.08)

    if delta > -BRAKE_DEADBAND:
        if delta > 0:
            accel = clip(0.70 + 0.07 * delta, 0.0, 1.0)
        else:
            if is_unwinding:
                accel = 0.95
            else:
                accel = clip(0.70 + 0.09 * delta, 0.30, 0.70)
        brake = 0.0
    else:
        overspeed = -delta - BRAKE_DEADBAND
        brake = clip(0.032 * overspeed, 0.0, 0.93)
        if brake < 0.4:
            accel = 0.22 * (1.0 - brake / 0.4)
        else:
            accel = 0.0

    # Lift-off in curva: solo a sterzo > 0.35
    steer_abs = abs(prev)
    if steer_abs > 0.35 and accel > 0:
        accel *= (1.0 - 0.25 * (steer_abs - 0.35) / 0.65)
        accel = clip(accel, 0.0, 1.0)

    return accel, brake


def traction_control(S, accel):
    """Riduce gas se le ruote posteriori girano molto piu' delle anteriori."""
    wsv = S.get('wheelSpinVel', None)
    if not wsv or len(wsv) != 4:
        return accel
    front = wsv[0] + wsv[1]
    rear  = wsv[2] + wsv[3]
    spin = rear - front
    if spin > 5.0:
        accel = clip(accel - 0.20, 0.0, 1.0)
    elif spin > 3.0:
        accel = clip(accel - 0.10, 0.0, 1.0)
    return accel


def shift_gears(S):
    """Cambio basato su RPM, con guard sulla velocita' minima per marcia e cooldown."""
    speedX = S.get('speedX', 0.0)
    rpm    = S.get('rpm', 0.0)

    if _state['gear_change_cooldown'] > 0:
        _state['gear_change_cooldown'] -= 1
        return _state['prev_gear']

    if speedX < -2:
        gear = -1
    else:
        gear = _state['prev_gear']
        if gear < 1:
            gear = 1
        if gear < 6 and rpm > RPM_UP and speedX > GEAR_MIN_SPEED[gear + 1] - 5:
            gear += 1
            _state['gear_change_cooldown'] = 6
        elif gear > 1 and (rpm < RPM_DOWN or speedX < GEAR_MIN_SPEED[gear] - 10):
            gear -= 1
            _state['gear_change_cooldown'] = 6

    _state['prev_gear'] = gear
    return gear


def drive(c):
    """Funzione principale di guida. Modifica c.R.d in-place."""
    S = c.S.d
    R = c.R.d

    track = S.get('track', [200.0] * 19)
    target_speed = lookup_target_speed(track)

    R['steer'] = calculate_steering(S)
    accel, brake = calculate_throttle_and_brake(S, target_speed)
    accel = traction_control(S, accel)
    R['accel'] = accel
    R['brake'] = brake
    R['gear']  = shift_gears(S)

    if is_manual_override:
        if manual_steer != 0.0:
            R['steer'] = manual_steer
        if manual_accel_override is not None:
            R['accel'] = manual_accel_override
        if manual_brake_override is not None:
            R['brake'] = manual_brake_override


# =====================================================================
# QUALITY GATE: decide se il sample va salvato nel CSV.
# =====================================================================

def is_sample_clean(S, step_index):
    """Ritorna True se il sample e' adatto al training."""
    if step_index < WARMUP_STEPS:
        return False
    if is_manual_override:
        return False

    speedX    = S.get('speedX', 0.0)
    track_pos = S.get('trackPos', 0.0)
    angle     = S.get('angle', 0.0)
    track     = S.get('track', [200.0] * 19)

    if speedX < QUALITY_MIN_SPEED:
        return False
    if abs(track_pos) > QUALITY_MAX_TRACKPOS:
        return False
    if abs(angle) > QUALITY_MAX_ANGLE:
        return False
    if min(track) < 0:
        return False
    return True


# =====================================================================
# MAIN LOOP
# =====================================================================

if __name__ == "__main__":
    C = Client(p=3001)
    track_name = C.trackname if C.trackname != 'unknown' else 'track'
    csv_filename = f'dataset_{track_name}_{int(time.time())}.csv'

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)

        headers = [
            'step', 'cur_lap_time', 'dist_from_start', 'dist_raced',
            'speedX', 'speedY', 'speedZ',
            'angle', 'trackPos',
            'rpm', 'gear_in',
            'steer', 'accel', 'brake', 'gear_out',
            'is_clean',
        ]
        for i in range(19):
            headers.append(f'track_{i}')
        for i in range(4):
            headers.append(f'wheelSpinVel_{i}')
        for i in range(36):
            headers.append(f'opponents_{i}')
        writer.writerow(headers)

        print("=" * 60)
        print(" TORCS bot v6 (rollback - giro 1:42 stabile)")
        print(f"   RPM_UP={RPM_UP}, RPM_DOWN={RPM_DOWN}")
        print(f"   GEAR_MIN_SPEED={GEAR_MIN_SPEED}")
        print(f"   SPEED_MAP top={SPEED_MAP[0][1]:.0f} km/h")
        print(f"   CURV thresh={CURV_THRESHOLD}, full_cut={CURV_FULL_CUT}, max_red={CURV_MAX_REDUCTION}")
        print(f" Output: {csv_filename}")
        print(" Tasti: frecce sx/dx = nudge sterzo, su/giu = gas/freno")
        print(" (i sample durante l'override manuale NON vengono salvati)")
        print("=" * 60)

        clean_count = 0
        total_count = 0

        for step in range(C.maxSteps, 0, -1):
            C.get_servers_input()
            drive(C)

            S = C.S.d
            R = C.R.d
            step_index = C.maxSteps - step

            clean = is_sample_clean(S, step_index)
            total_count += 1
            if clean:
                clean_count += 1

            track_sensors = S.get('track', [0.0] * 19)
            wsv = S.get('wheelSpinVel', [0.0] * 4)
            opp = S.get('opponents', [200.0] * 36)
            if len(track_sensors) < 19: track_sensors = list(track_sensors) + [0.0] * (19 - len(track_sensors))
            if len(wsv) < 4:            wsv = list(wsv) + [0.0] * (4 - len(wsv))
            if len(opp) < 36:           opp = list(opp) + [200.0] * (36 - len(opp))

            row = [
                step_index,
                S.get('curLapTime', 0.0),
                S.get('distFromStart', 0.0),
                S.get('distRaced', 0.0),
                S.get('speedX', 0.0),
                S.get('speedY', 0.0),
                S.get('speedZ', 0.0),
                S.get('angle', 0.0),
                S.get('trackPos', 0.0),
                S.get('rpm', 0.0),
                S.get('gear', 0),
                R['steer'],
                R['accel'],
                R['brake'],
                R['gear'],
                int(clean),
            ]
            row.extend(track_sensors[:19])
            row.extend(wsv[:4])
            row.extend(opp[:36])
            writer.writerow(row)

            C.respond_to_server()

            if step_index % 1000 == 0 and step_index > 0:
                ratio = clean_count / max(1, total_count) * 100
                print(f"[step {step_index}] sample puliti: {clean_count}/{total_count} ({ratio:.1f}%)")

        print(f"\nFatto. Sample totali: {total_count}, puliti: {clean_count}")
        print(f"In addestramento usa SOLO le righe con is_clean=1.")

    C.shutdown()