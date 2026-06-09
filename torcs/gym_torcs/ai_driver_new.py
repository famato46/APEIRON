import numpy as np
import joblib
import socket
import os

HOST = 'localhost'
PORT = 3001
SID  = 'SCR'
DATA_SIZE = 2**17

MODEL_PATH  = "model_bc.joblib"
SCALER_PATH = "out_bc/scaler.joblib"

MOVING_V = 2.0

STEER_ALPHA_STRAIGHT    = 0.40
STEER_ALPHA_CURVE       = 0.85
STEER_CURVE_THRESH      = 0.15
STEER_GAIN_CURVE        = 1.40   # era 1.8 -> testacoda, torno a 1.4
STEER_GAIN_STRAIGHT     = 1.00
STEER_DEADBAND_STRAIGHT = 0.08
STEER_DEADBAND_CURVE    = 0.02

# Heading: corregge l'angolo dell'auto rispetto alla pista
# corr(angle, target_steer) = -0.40: angle negativo -> steer positivo
# Applicato solo in curva per non introdurre zigzag su rettilineo
HEADING_GAIN_CURVE = 0.30

BRAKE_IGNORE_V = 80.0

V_TARGET = 130.0
KP_ACCEL = 0.013
AC_BASE  = 0.35

SAFETY_TP  = 0.90
SAFETY_BRK = 0.50

PRINT_EVERY = 30


class ServerState():
    def __init__(self): self.d = {}
    def parse_server_str(self, s):
        for i in s.strip()[:-1].strip().lstrip('(').rstrip(')').split(')('):
            w = i.split(' '); self.d[w[0]] = self._de(w[1:])
    def _de(self, s):
        if not s: return s
        if isinstance(s, str):
            try: return float(s)
            except: return s
        if len(s) < 2: return self._de(s[0])
        return [self._de(i) for i in s]


class DriverAction():
    def __init__(self):
        self.d = {'accel':0,'brake':0,'clutch':0,'gear':1,'steer':0,
                  'focus':[-90,-45,0,45,90],'meta':0}
    def __repr__(self):
        return ''.join(
            f"({k} {'%.3f'%v if not isinstance(v,list) else ' '.join(str(x) for x in v)})"
            for k,v in self.d.items())


def smart_gear(v):
    if v < -2:  return -1
    if v < 35:  return 1
    if v < 75:  return 2
    if v < 115: return 3
    if v < 160: return 4
    if v < 215: return 5
    return 6


def run_ai():
    print("\n================================================")
    print("   TORCS AI - PILOTA AUTOMATICO")
    print("================================================")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"[ERRORE] Mancano file modello/scaler."); return

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[OK] Modello caricato.\n")

    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1.0)
    initmsg = f"{SID}(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)"

    print("In attesa di TORCS...")
    while True:
        try:
            so.sendto(initmsg.encode(), (HOST, PORT))
            d, _ = so.recvfrom(DATA_SIZE)
            if '***identified***' in d.decode():
                print(">>> [OK] Connesso!\n"); break
        except: pass

    S = ServerState(); R = DriverAction()
    prev_st = 0.0
    step = 0

    try:
        while True:
            try:
                raw, _ = so.recvfrom(DATA_SIZE)
                msg = raw.decode()
                if '***restart***' in msg:
                    print("\n[RESET]"); R.d['meta'] = 0; step = 0; prev_st = 0.0
                    while True:
                        so.sendto(initmsg.encode(), (HOST, PORT))
                        try:
                            so.settimeout(0.5); r, _ = so.recvfrom(DATA_SIZE)
                            if '***identified***' in r.decode(): break
                        except: pass
                    so.settimeout(1.0); continue
                S.parse_server_str(msg)
            except socket.timeout: continue

            v  = S.d.get('speedX',   0)
            ag = S.d.get('angle',    0)
            tp = S.d.get('trackPos', 0)
            tr = S.d.get('track', [0]*19)
            t0=tr[0]; t4=tr[4]; t9=tr[9]; t14=tr[14]; t18=tr[18]
            dt = t18 - t0

            Xs = scaler.transform(np.array([[v, ag, tp, t0, t4, t9, t14, t18, dt]]))
            p  = model.predict(Xs)[0]
            model_st = float(p[0])
            model_br = float(np.clip(p[2], 0.0, 1.0))

            # ── STERZO ───────────────────────────────────────────────
            if v <= MOVING_V:
                st = 0.0; prev_st = 0.0
            else:
                in_curve = abs(model_st) >= STEER_CURVE_THRESH
                alpha    = STEER_ALPHA_CURVE    if in_curve else STEER_ALPHA_STRAIGHT
                deadband = STEER_DEADBAND_CURVE if in_curve else STEER_DEADBAND_STRAIGHT
                gain     = STEER_GAIN_CURVE     if in_curve else STEER_GAIN_STRAIGHT

                raw_st = float(np.clip(model_st * gain, -1.0, 1.0))

                # Heading correttivo solo in curva: angle negativo -> aggiungi sterzo positivo
                if in_curve:
                    raw_st = float(np.clip(raw_st - HEADING_GAIN_CURVE * ag, -1.0, 1.0))

                smoothed = alpha * raw_st + (1.0 - alpha) * prev_st
                prev_st  = smoothed
                st = 0.0 if abs(smoothed) < deadband else float(np.clip(smoothed, -1.0, 1.0))

            # ── GAS / FRENO ───────────────────────────────────────────
            br = 0.0
            if abs(tp) > SAFETY_TP:
                br = SAFETY_BRK; ac = 0.0
            elif v <= MOVING_V:
                ac = 1.0; br = 0.0
            elif v < BRAKE_IGNORE_V:
                br = 0.0
                delta_v = V_TARGET - v
                ac = float(np.clip(KP_ACCEL * delta_v + AC_BASE, 0.0, 1.0))
            else:
                if model_br > 0.15:
                    br = model_br; ac = 0.0
                else:
                    delta_v = V_TARGET - v
                    if delta_v > 0:
                        ac = float(np.clip(KP_ACCEL * delta_v + AC_BASE, 0.0, 1.0))
                    else:
                        ac = 0.0
                        if delta_v < -20:
                            br = float(np.clip(-0.008 * delta_v, 0.0, 0.55))

            if abs(tp) > 0.92:
                br = max(br, 0.50); ac = 0.0

            ac = float(np.clip(ac, 0.0, 1.0))
            br = float(np.clip(br, 0.0, 1.0))

            gea = int(S.d.get('gear', 1))
            if abs(st) <= 0.5:
                gea = smart_gear(v)

            R.d['steer']=st; R.d['accel']=ac; R.d['brake']=br; R.d['gear']=gea

            step += 1
            if step % PRINT_EVERY == 0:
                print(f"v={v:5.1f}|tp={tp:+.2f}|ag={ag:+.3f}|"
                      f"mST={model_st:+.2f} mBR={model_br:.2f}|"
                      f"st={st:+.2f} ac={ac:.2f} br={br:.2f}|g={gea}")

            so.sendto(repr(R).encode(), (HOST, PORT))

    except KeyboardInterrupt: print("\n[STOP]")
    finally: so.close()


if __name__ == "__main__":
    run_ai()