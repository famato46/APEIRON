import json
import time
from pathlib import Path

import joblib
import numpy as np
import snakeoil3_jm2 as snakeoil3


BASE = Path(__file__).resolve().parent

MODEL_CANDIDATES = ["models/model_bc.joblib", "model_bc.joblib"]
SCALER_CANDIDATES = ["out_bc/scaler.joblib", "scaler.joblib", "models/scaler.joblib"]
FEATURE_CANDIDATES = ["out_bc/feature_config.json", "feature_config.json", "models/feature_config.json"]

LOG_EVERY = 30

SPIN_ENTER = 1.20
SPIN_EXIT = 0.50
OFFTRACK_ENTER = 1.10
OFFTRACK_EXIT = 0.88

EDGE_WARN = 0.50
EDGE_HARD = 0.78

STEER_RATE_FAST = 0.085
STEER_RATE_MID = 0.13
STEER_RATE_SLOW = 0.22


def find_file(candidates):
    for name in candidates:
        p = BASE / name
        if p.exists():
            return p
    raise FileNotFoundError(f"File non trovato: {candidates}")


def load_features(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for key in ("feature_names", "input_features", "features"):
        if key in cfg and isinstance(cfg[key], list):
            return cfg[key]

    for v in cfg.values():
        if isinstance(v, list):
            return v

    raise ValueError("feature_config.json non valido")


MODEL_PATH = find_file(MODEL_CANDIDATES)
SCALER_PATH = find_file(SCALER_CANDIDATES)
FEATURE_PATH = find_file(FEATURE_CANDIDATES)

print("[ai_driver_newdata_v3] Caricamento...")
print(f"[ai_driver_newdata_v3] model={MODEL_PATH}")
print(f"[ai_driver_newdata_v3] scaler={SCALER_PATH}")
print(f"[ai_driver_newdata_v3] features={FEATURE_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURES = load_features(FEATURE_PATH)

print(f"[ai_driver_newdata_v3] OK. Feature richieste ({len(FEATURES)}):")
for i, f in enumerate(FEATURES):
    print(f"  [{i:02d}] {f}")


def safe_track(S):
    track = S.get("track", [200.0] * 19)

    if isinstance(track, (int, float)):
        track = [float(track)] * 19

    track = [float(x) for x in track]
    while len(track) < 19:
        track.append(200.0)

    return track[:19]


def build_state(S):
    track = safe_track(S)

    data = {
        "speedX": float(S.get("speedX", 0.0)),
        "speedY": float(S.get("speedY", 0.0)),
        "speedZ": float(S.get("speedZ", 0.0)),
        "angle": float(S.get("angle", 0.0)),
        "trackPos": float(S.get("trackPos", 0.0)),
        "rpm": float(S.get("rpm", 0.0)),
        "distFromStart": float(S.get("distFromStart", S.get("distRaced", 0.0))),
        "dist_from_start": float(S.get("distFromStart", S.get("distRaced", 0.0))),
        "distRaced": float(S.get("distRaced", 0.0)),
        "delta_track": float(track[18] - track[0]),
    }

    wheel = list(S.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0]))
    wheel += [0.0] * (4 - len(wheel))

    for i in range(4):
        data[f"wheelSpinVel_{i}"] = float(wheel[i])
        data[f"wheelSpinVel{i}"] = float(wheel[i])

    for i in range(19):
        data[f"track_{i}"] = float(track[i])
        data[f"track{i}"] = float(track[i])

    return np.array([[data.get(name, 0.0) for name in FEATURES]], dtype=np.float32)


def curvature(track):
    pairs = [(0, 18, 0.38), (1, 17, 0.30), (2, 16, 0.18), (3, 15, 0.09), (4, 14, 0.05)]
    s = 0.0

    for r, l, w in pairs:
        rv = float(track[r])
        lv = float(track[l])
        s += w * ((lv - rv) / (lv + rv + 1e-6))

    return float(s)


def aim_steer(track_pos, angle):
    return float(np.clip(-0.62 * track_pos + 1.85 * angle, -1.0, 1.0))


def gear_logic(speed, gear):
    if speed < 7:
        return 1

    gear = max(1, int(gear))

    if gear < 2 and speed > 55:
        return 2
    if gear < 3 and speed > 95:
        return 3
    if gear < 4 and speed > 138:
        return 4
    if gear < 5 and speed > 182:
        return 5
    if gear < 6 and speed > 222:
        return 6

    if gear > 5 and speed < 190:
        return 5
    if gear > 4 and speed < 155:
        return 4
    if gear > 3 and speed < 112:
        return 3
    if gear > 2 and speed < 72:
        return 2
    if gear > 1 and speed < 34:
        return 1

    return gear


def target_speed(track, angle, track_pos):
    front = float(track[9])
    c = abs(curvature(track))
    edge = abs(track_pos)
    a = abs(angle)

    if c < 0.05:
        v = 220.0
    elif c < 0.11:
        v = 185.0
    elif c < 0.20:
        v = 145.0
    elif c < 0.32:
        v = 108.0
    elif c < 0.46:
        v = 78.0
    elif c < 0.62:
        v = 56.0
    else:
        v = 42.0

    if front < 6:
        v = min(v, 28.0)
    elif front < 10:
        v = min(v, 36.0)
    elif front < 15:
        v = min(v, 46.0)
    elif front < 22:
        v = min(v, 58.0)
    elif front < 34:
        v = min(v, 76.0)
    elif front < 50:
        v = min(v, 96.0)
    elif front < 70:
        v = min(v, 118.0 if c < 0.18 else 96.0)
    elif front < 95:
        v = min(v, 150.0 if c < 0.12 else 118.0)

    if edge > 0.45:
        v = min(v, 78.0)
    if edge > 0.62:
        v = min(v, 52.0)
    if edge > 0.82:
        v = min(v, 34.0)

    if a > 0.35:
        v = min(v, 70.0)
    if a > 0.55:
        v = min(v, 48.0)
    if a > 0.90:
        v = min(v, 30.0)

    return float(np.clip(v, 28.0, 230.0))


def steer_filter(steer, prev, speed):
    if speed > 165:
        max_delta = STEER_RATE_FAST
    elif speed > 80:
        max_delta = STEER_RATE_MID
    else:
        max_delta = STEER_RATE_SLOW

    return float(prev + np.clip(steer - prev, -max_delta, max_delta))


def speed_guard(accel, brake, speed, track, angle, track_pos):
    vt = target_speed(track, angle, track_pos)
    over = speed - vt
    front = float(track[9])
    c = abs(curvature(track))
    edge = abs(track_pos)
    tag = False

    if front < 8 and speed > 30:
        brake = max(brake, 0.50)
        accel = 0.0
        tag = True
    elif front < 14 and speed > 45:
        brake = max(brake, 0.42)
        accel = 0.0
        tag = True
    elif front < 22 and speed > 58:
        brake = max(brake, 0.32)
        accel = 0.0
        tag = True
    elif front < 36 and speed > 88:
        brake = max(brake, 0.24)
        accel = 0.0
        tag = True
    elif front < 55 and speed > 122 and c > 0.06:
        brake = max(brake, 0.18)
        accel = 0.0
        tag = True

    if over > 55:
        brake = max(brake, 0.50)
        accel = 0.0
        tag = True
    elif over > 35:
        brake = max(brake, 0.34)
        accel = 0.0
        tag = True
    elif over > 18:
        brake = max(brake, 0.20)
        accel = 0.0
        tag = True
    elif over > 8:
        brake = max(brake, 0.08)
        accel = min(accel, 0.22)
        tag = True

    if edge > 0.58 and speed > 65:
        brake = max(brake, 0.20)
        accel = min(accel, 0.15)
        tag = True

    if speed < vt - 10 and brake < 0.08 and edge < 0.52 and abs(angle) < 0.42:
        brake = 0.0
        if c < 0.16:
            accel = max(accel, 0.82)
        elif c < 0.34:
            accel = max(accel, 0.55)
        else:
            accel = max(accel, 0.30)

    if speed < 8 and front > 7 and abs(angle) < 0.75 and edge < 0.85:
        brake = 0.0
        accel = max(accel, 0.75)
        tag = True

    return float(np.clip(accel, 0, 1)), float(np.clip(brake, 0, 1)), vt, tag


def steer_guard(steer, accel, brake, speed, track, angle, track_pos):
    front = float(track[9])
    c = abs(curvature(track))
    edge = abs(track_pos)
    a = abs(angle)
    tag = False

    aim = aim_steer(track_pos, angle)

    if front < 60 or c > 0.18 or a > 0.16:
        if a > 0.45 or front < 16:
            blend = 0.78
        elif speed > 100:
            blend = 0.35
        else:
            blend = 0.50

        steer = (1.0 - blend) * steer + blend * aim
        tag = True

    if edge > EDGE_WARN:
        aim = aim_steer(track_pos, angle)

        if edge > EDGE_HARD:
            blend = 0.88
        else:
            blend = 0.62

        steer = (1.0 - blend) * steer + blend * aim
        tag = True

        if speed > 85:
            accel = 0.0
            brake = max(brake, 0.25)
        elif speed > 50:
            accel = min(accel, 0.16)
            brake = max(brake, 0.10)
        else:
            accel = min(max(accel, 0.14), 0.32)
            brake = min(brake, 0.04)

    return float(np.clip(steer, -1, 1)), float(np.clip(accel, 0, 1)), float(np.clip(brake, 0, 1)), tag


def recovery_action(speed, track_pos, angle, track):
    front = float(track[9])
    steer = aim_steer(track_pos, angle)

    if speed < -2:
        return steer, 0.0, 0.85, 1

    if front < 1.5 and abs(track_pos) > 1.05:
        if speed > 15:
            return steer, 0.0, 0.35, 1
        return steer, 0.22, 0.0, 1

    if abs(angle) > 1.45:
        if speed > 18:
            return steer, 0.0, 0.50, 1
        return steer, 0.22, 0.0, 1

    if abs(track_pos) > 1.05:
        if speed > 30:
            return steer, 0.0, 0.42, 1
        return steer, 0.30, 0.0, 1

    if speed > 35:
        return steer, 0.0, 0.28, 1

    return steer, 0.30, 0.0, 1


def main():
    while True:
        client = None

        while client is None:
            try:
                client = snakeoil3.Client(p=3001, vision=False)
            except KeyboardInterrupt:
                raise
            except Exception:
                print("[ai_driver_newdata_v3] In attesa server TORCS su porta 3001...")
                time.sleep(2)

        print("[ai_driver_newdata_v3] Connesso a TORCS.")

        step = 0
        gear = 1
        steer_prev = 0.0
        in_recovery = False

        while True:
            client.get_servers_input()
            S = client.S.d

            if client.so is None:
                break

            speed = float(S.get("speedX", 0.0))
            angle = float(S.get("angle", 0.0))
            track_pos = float(S.get("trackPos", 0.0))
            track = safe_track(S)

            tag = "AI"

            enter_recovery = (
                abs(angle) > SPIN_ENTER
                or abs(track_pos) > OFFTRACK_ENTER
                or speed < -3.0
            )

            exit_recovery = (
                abs(angle) < SPIN_EXIT
                and abs(track_pos) < OFFTRACK_EXIT
                and speed > -1.0
            )

            if enter_recovery:
                in_recovery = True

            if in_recovery and exit_recovery:
                in_recovery = False

            if in_recovery:
                steer, accel, brake, gear = recovery_action(speed, track_pos, angle, track)
                tag = "REC"
            else:
                raw = build_state(S)
                scaled = scaler.transform(raw)
                out = model.predict(scaled)[0]

                steer = float(np.clip(out[0], -1.0, 1.0))
                accel = float(np.clip(out[1], 0.0, 1.0))
                brake = float(np.clip(out[2], 0.0, 1.0))

                if not np.isfinite(steer):
                    steer = 0.0
                if not np.isfinite(accel):
                    accel = 0.0
                if not np.isfinite(brake):
                    brake = 0.0

                gear = gear_logic(speed, gear)

                accel, brake, vt, gov = speed_guard(accel, brake, speed, track, angle, track_pos)
                steer, accel, brake, sg = steer_guard(steer, accel, brake, speed, track, angle, track_pos)
                steer = steer_filter(steer, steer_prev, speed)

                if gov:
                    tag = "GOV"
                if sg:
                    tag = "STEER"

                if brake > 0.14:
                    accel = 0.0

            steer_prev = steer

            R = client.R.d
            R["steer"] = float(np.clip(steer, -1.0, 1.0))
            R["accel"] = float(np.clip(accel, 0.0, 1.0))
            R["brake"] = float(np.clip(brake, 0.0, 1.0))
            R["gear"] = int(max(1, min(6, gear)))
            R["clutch"] = 0.0
            R["meta"] = 0

            client.respond_to_server()

            if step % LOG_EVERY == 0:
                vt_log = target_speed(track, angle, track_pos)
                print(
                    f"step={step:05d} | "
                    f"v={speed:+6.1f} vt={vt_log:5.1f} "
                    f"tp={track_pos:+.2f} ang={angle:+.2f} tr9={track[9]:5.1f} "
                    f"str={R['steer']:+.2f} acc={R['accel']:.2f} brk={R['brake']:.2f} "
                    f"gr={R['gear']} | {tag}"
                )

            step += 1


if __name__ == "__main__":
    main()

