# ============================================================
# FILE 1: torcs_env.py — Training definitivo da zero
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import snakeoil3_gym as snakeoil


class TorcsEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, port=3001):
        super(TorcsEnv, self).__init__()
        self.port = port
        self.client = None
        self.prev_damage = 0
        self.step_count = 0

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([-50.0, -np.pi, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -200.0], dtype=np.float32),
            high=np.array([350.0, np.pi, 3.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float32),
            dtype=np.float32
        )

    def _init_client(self):
        client = snakeoil.Client(p=self.port)
        client.maxSteps = 10 ** 6
        return client

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.client is not None:
            try:
                self.client.R.d['meta'] = 1
                self.client.respond_to_server()
            except Exception:
                pass
            try:
                self.client.shutdown()
            except Exception:
                pass

        self.client = self._init_client()
        self.prev_damage = 0
        self.step_count = 0

        self.client.get_servers_input()
        obs = self._make_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1

        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1],  0.0, 1.0))
        brake = float(np.clip(action[2],  0.0, 1.0))

        self.client.R.d['steer'] = steer
        self.client.R.d['accel'] = accel
        self.client.R.d['brake'] = brake
        self.client.R.d['meta'] = 0

        speedX = self.client.S.d.get('speedX', 0)
        gear = 1
        if speedX > 50:  gear = 2
        if speedX > 90:  gear = 3
        if speedX > 150: gear = 4
        if speedX > 200: gear = 5
        if speedX > 280: gear = 6
        self.client.R.d['gear'] = gear

        self.client.respond_to_server()
        self.client.get_servers_input()

        obs = self._make_obs()

        speedX   = self.client.S.d.get('speedX', 0)
        trackPos = self.client.S.d.get('trackPos', 0)
        angle    = self.client.S.d.get('angle', 0)
        damage   = self.client.S.d.get('damage', 0)
        track    = self.client.S.d.get('track', [200] * 19)
        if len(track) < 19:
            track = list(track) + [200] * (19 - len(track))

        delta_damage = damage - self.prev_damage
        self.prev_damage = damage

        # --- Fase 1: primi 200 step — impara ad accelerare ---
        if self.step_count < 200:
            reward = accel * 5.0
        else:
            # --- Fase 2: reward principale ---
            reward = speedX * np.cos(angle)
            reward -= abs(speedX * np.sin(angle))

            # FIX: penalità curva integrata fin dall'inizio, stessa scala della reward
            # front_sensor corto = curva → penalizza velocità eccessiva proporzionalmente
            front_sensor = track[9]
            if front_sensor < 30.0 and speedX > 60.0:
                reward -= (speedX - 60.0) * (30.0 - front_sensor) * 0.005

        reward -= abs(trackPos) * 2.0
        reward -= abs(steer) * 0.1
        if speedX < 10:
            reward -= brake * 2.0

        terminated = False

        trackpos_limit = 1.8 if self.step_count < 500 else 1.5
        if abs(trackPos) > trackpos_limit:
            reward = -100.0
            terminated = True

        if delta_damage > 100:
            reward = -100.0
            terminated = True

        if self.step_count > 300 and speedX < 1.0:
            reward = -50.0
            terminated = True

        return obs, float(reward), terminated, False, {}

    def _make_obs(self):
        track = self.client.S.d.get('track', [0] * 19)
        if len(track) < 19:
            track = list(track) + [0] * (19 - len(track))
        delta_track = track[18] - track[0]

        obs = np.array([
            self.client.S.d.get('speedX', 0),
            self.client.S.d.get('angle', 0),
            self.client.S.d.get('trackPos', 0),
            track[0], track[4], track[9], track[14], track[18],
            delta_track
        ], dtype=np.float32)

        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def close(self):
        if self.client is not None:
            try:
                self.client.R.d['meta'] = 1
                self.client.respond_to_server()
            except Exception:
                pass
            try:
                self.client.shutdown()
            except Exception:
                pass
            self.client = None