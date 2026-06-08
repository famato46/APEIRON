import numpy as np
import joblib
import socket
import sys
import os
import time

# --- CONFIGURAZIONE TORCS ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17

# --- CONFIGURAZIONE AI ---
MODEL_PATH = "model_bc.joblib"
SCALER_PATH = "out_bc/scaler.joblib"

# --- POST-PROCESSING ---
STEER_SMOOTH_ALPHA   = 0.85          # piu' smoothing (era 0.95)

# Amplificazione (ridotta: 1.15 era troppo)
STEER_GAIN_BASE      = 1.00          # nessuna amplificazione, fidiamoci del modello

# Lookahead curva (ridotto: 0.30 era troppo)
CURVE_LOOKAHEAD_GAIN = 0.10          # contributo molto leggero

# Cross-track come correzione SIMMETRICA per il recovery
CROSS_TRACK_GAIN_INNER = 0.15        # quando lo sterzo del modello e' coerente
CROSS_TRACK_GAIN_OUTER = 0.50        # quando siamo fuori dalla parte opposta (RECOVERY)
CROSS_TRACK_DEADBAND   = 0.25
OPPOSITE_TP_THRESHOLD  = 0.50        # |tp| oltre cui la correzione opposta diventa forte

# Slew rate limit: blocca cambi bruschi di sterzo (anti-oversterzo)
STEER_MAX_DELTA      = 0.10          # max variazione tra frame consecutivi

# Safety
SAFETY_BRAKE_TP      = 0.90
SAFETY_BRAKE_FORCE   = 0.50
SAFETY_FRONT_MIN     = 30.0

# Lift-off
LIFTOFF_STEER        = 0.25
LIFTOFF_MAX_REDUCT   = 0.40

# Auto-brake in entrata curva veloce
AUTOBRAKE_STEER      = 0.40
AUTOBRAKE_SPEED      = 100.0
AUTOBRAKE_FORCE      = 0.40

# Throttle floor
THROTTLE_FLOOR_LOW_V_KMH     = 60.0
THROTTLE_FLOOR_LOW_V_ACCEL   = 0.70
THROTTLE_FLOOR_MID_V_KMH     = 130.0
THROTTLE_FLOOR_MID_V_ACCEL   = 0.40
THROTTLE_FLOOR_HIGH_V_ACCEL  = 0.20
THROTTLE_FLOOR_STEER_KILL    = 0.30

BRAKE_MIN_TRUST      = 0.25
LAUNCH_SPEED         = 5.0
STALL_SPEED          = 20.0

PRINT_EVERY = 30


class ServerState():
    def __init__(self):
        self.d = dict()
    def parse_server_str(self, server_string):
        servstr = server_string.strip()[:-1]
        sslisted = servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = self.destringify(w[1:])
    def destringify(self, s):
        if not s: return s
        if type(s) is str:
            try: return float(s)
            except ValueError: return s
        elif type(s) is list:
            if len(s) < 2: return self.destringify(s[0])
            else: return [self.destringify(i) for i in s]


class DriverAction():
    def __init__(self):
        self.d = {'accel': 0, 'brake': 0, 'clutch': 0, 'gear': 1, 'steer': 0,
                  'focus': [-90, -45, 0, 45, 90], 'meta': 0}
    def __repr__(self):
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not isinstance(v, list): out += '%.3f' % v
            else: out += ' '.join([str(x) for x in v])
            out += ')'
        return out


def smart_gear(speedX, current_gear):
    if speedX < -2:  return -1
    if speedX < 35:  return 1
    if speedX < 75:  return 2
    if speedX < 115: return 3
    if speedX < 160: return 4
    if speedX < 215: return 5
    return 6


def throttle_floor(speedX_kmh, abs_steer):
    if abs_steer > THROTTLE_FLOOR_STEER_KILL:
        return 0.0
    if speedX_kmh < THROTTLE_FLOOR_LOW_V_KMH:
        return THROTTLE_FLOOR_LOW_V_ACCEL
    if speedX_kmh < THROTTLE_FLOOR_MID_V_KMH:
        return THROTTLE_FLOOR_MID_V_ACCEL
    return THROTTLE_FLOOR_HIGH_V_ACCEL


def run_ai():
    print("\n==================================================")
    print("   TORCS AI - PILOTA AUTOMATICO (INFERENZA)")
    print("==================================================")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"\n[ERRORE] File mancanti!")
        print(f"Cerco il modello in: {MODEL_PATH}")
        print(f"Cerco lo scaler in:  {SCALER_PATH}")
        return

    print("Caricamento Rete Neurale e Scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Modello caricato. n_layers={getattr(model, 'n_layers_', '?')}, "
          f"n_outputs={getattr(model, 'n_outputs_', '?')}\n")

    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1.0)
    initmsg = f"{SID}(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)"

    print("In attesa di TORCS (Avvia la gara in Practice mode)...")
    while True:
        try:
            so.sendto(initmsg.encode(), (HOST, PORT))
            sockdata, _ = so.recvfrom(DATA_SIZE)
            if '***identified***' in sockdata.decode():
                print(">>> [OK] Connesso a TORCS!\n")
                break
        except:
            pass

    S = ServerState()
    R = DriverAction()

    prev_steer = 0.0
    step_count = 0

    try:
        while True:
            try:
                sockdata, _ = so.recvfrom(DATA_SIZE)
                sockstr = sockdata.decode()

                if '***restart***' in sockstr:
                    print("\n[RESET] Gara riavviata.")
                    R.d['meta'] = 0
                    prev_steer = 0.0
                    while True:
                        so.sendto(initmsg.encode(), (HOST, PORT))
                        try:
                            so.settimeout(0.5)
                            resp, _ = so.recvfrom(DATA_SIZE)
                            if '***identified***' in resp.decode(): break
                        except: pass
                    so.settimeout(1.0)
                    continue
                S.parse_server_str(sockstr)
            except socket.timeout:
                continue

            # --- 1. SENSORI ---
            speedX = S.d.get('speedX', 0)
            angle = S.d.get('angle', 0)
            trackPos = S.d.get('trackPos', 0)
            track = S.d.get('track', [0]*19)

            track_0  = track[0]
            track_4  = track[4]
            track_9  = track[9]
            track_14 = track[14]
            track_18 = track[18]
            delta_track = track_18 - track_0

            X_raw = np.array([[speedX, angle, trackPos,
                               track_0, track_4, track_9, track_14, track_18,
                               delta_track]])

            # --- 2. PREDIZIONE RAW ---
            X_scaled = scaler.transform(X_raw)
            pred = model.predict(X_scaled)[0]
            raw_steer = float(pred[0])
            raw_accel = float(pred[1])
            raw_brake = float(pred[2])

            # --- 3. STERZO ---

            # 3a. Modello (no amplificazione)
            ai_steer = STEER_GAIN_BASE * raw_steer

            # 3b. Lookahead leggero
            dt_norm = max(-1.0, min(1.0, delta_track / 100.0))
            ai_steer += CURVE_LOOKAHEAD_GAIN * dt_norm

            # 3c. CROSS-TRACK SIMMETRICO: agisce SEMPRE quando siamo fuori
            # dalla deadband. Forza proporzionale alla distanza dal centro.
            # Quando trackPos > 0 (auto a destra) -> correzione negativa (sterza sx).
            # IMPORTANTE: questo deve riportare al centro ANCHE quando il modello
            # sterza nella direzione sbagliata (sovrasterzo all'esterno).
            if abs(trackPos) > CROSS_TRACK_DEADBAND:
                if abs(trackPos) > OPPOSITE_TP_THRESHOLD:
                    # Recovery forte: ci stiamo allontanando dal centro
                    gain = CROSS_TRACK_GAIN_OUTER
                else:
                    gain = CROSS_TRACK_GAIN_INNER
                # correzione che punta SEMPRE verso il centro
                correction = -gain * trackPos
                ai_steer += correction

            # 3d. Smoothing
            ai_steer = STEER_SMOOTH_ALPHA * ai_steer + (1.0 - STEER_SMOOTH_ALPHA) * prev_steer

            # 3e. SLEW RATE LIMIT: limita la variazione tra frame.
            # Evita scatti bruschi che provocano l'oscillazione dx<->sx.
            delta = ai_steer - prev_steer
            if delta > STEER_MAX_DELTA:
                ai_steer = prev_steer + STEER_MAX_DELTA
            elif delta < -STEER_MAX_DELTA:
                ai_steer = prev_steer - STEER_MAX_DELTA

            # --- 4. FRENO / GAS ---
            ai_accel = raw_accel
            ai_brake = raw_brake

            if ai_brake < BRAKE_MIN_TRUST:
                ai_brake = 0.0

            on_track = (track_9 > 0)
            if abs(trackPos) > SAFETY_BRAKE_TP:
                ai_brake = max(ai_brake, SAFETY_BRAKE_FORCE)
                ai_accel = 0.0
            elif on_track and speedX > 100 and track_9 < SAFETY_FRONT_MIN:
                ai_brake = max(ai_brake, 0.55)
                ai_accel = 0.0

            if speedX > AUTOBRAKE_SPEED and abs(ai_steer) > AUTOBRAKE_STEER:
                ai_brake = max(ai_brake, AUTOBRAKE_FORCE)

            if ai_brake > 0.1:
                ai_accel = 0.0

            if speedX < LAUNCH_SPEED:
                ai_accel = 1.0
                ai_brake = 0.0
            elif speedX < STALL_SPEED and on_track and track_9 > 40 \
                    and abs(trackPos) < SAFETY_BRAKE_TP and abs(ai_steer) < 0.30:
                ai_accel = 1.0
                ai_brake = 0.0

            if ai_brake < 0.05:
                ai_accel = max(ai_accel, throttle_floor(speedX, abs(ai_steer)))

            if speedX > 60 and abs(ai_steer) > LIFTOFF_STEER:
                excess = (abs(ai_steer) - LIFTOFF_STEER) / (1.0 - LIFTOFF_STEER)
                ai_accel *= (1.0 - LIFTOFF_MAX_REDUCT * min(1.0, excess))

            # --- 5. CLAMPING ---
            ai_steer = max(-1.0, min(1.0, ai_steer))
            ai_accel = max(0.0, min(1.0, ai_accel))
            ai_brake = max(0.0, min(1.0, ai_brake))

            prev_steer = ai_steer

            # --- 6. MARCE ---
            current_server_gear = int(S.d.get('gear', 1))
            target_gear = smart_gear(speedX, current_server_gear)
            ai_gear = current_server_gear if abs(ai_steer) > 0.5 else target_gear

            # --- 7. INVIO ---
            R.d['steer'] = ai_steer
            R.d['accel'] = ai_accel
            R.d['brake'] = ai_brake
            R.d['gear'] = ai_gear

            step_count += 1
            if step_count % PRINT_EVERY == 0:
                print(f"v={speedX:5.1f} | tp={trackPos:+5.2f} | dT={delta_track:+6.1f} | "
                      f"RAW st={raw_steer:+.2f} a={raw_accel:.2f} b={raw_brake:.2f} | "
                      f"OUT st={ai_steer:+.2f} a={ai_accel:.2f} b={ai_brake:.2f} | g={ai_gear}")

            so.sendto(repr(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\n[STOP] Motore spento. Uscita.")
    finally:
        so.close()


if __name__ == "__main__":
    run_ai()