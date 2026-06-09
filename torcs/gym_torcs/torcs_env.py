# ============================================================
# FILE 1: torcs_env.py
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import snakeoil3_gym as snakeoil

class TorcsEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, port=3001):
        super(TorcsEnv, self).__init__()
        self.port = port  # FIX: porta parametrica per supportare env multipli (train + eval)
        self.client = None
        self.prev_damage = 0  # FIX: traccia danno precedente per calcolo incrementale

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
        """FIX: factory method per creare/ricreare il client in modo pulito"""
        client = snakeoil.Client(p=self.port)
        client.maxSteps = 10**6
        return client

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # FIX: se esiste già un client, invia meta=1 (restart) prima di ricreare
        if self.client is not None:
            try:
                self.client.R.d['meta'] = 1
                self.client.respond_to_server()
            except Exception:
                pass  # FIX: ignora errori di socket se la connessione è già morta
            try:
                self.client.shutdown()
            except Exception:
                pass

        self.client = self._init_client()
        self.prev_damage = 0  # FIX: reset danno accumulato

        self.client.get_servers_input()
        obs = self._make_obs()
        return obs, {}

    def step(self, action):
        # FIX: clipping esplicito per sicurezza numerica (SB3 non garantisce bounds)
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1],  0.0, 1.0))
        brake = float(np.clip(action[2],  0.0, 1.0))

        self.client.R.d['steer'] = steer
        self.client.R.d['accel'] = accel
        self.client.R.d['brake'] = brake
        self.client.R.d['meta'] = 0  # FIX: assicura che meta=0 durante step normali

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

        # --- Lettura stato aggiornato ---
        trackPos = self.client.S.d.get('trackPos', 0)
        angle    = self.client.S.d.get('angle', 0)
        damage   = self.client.S.d.get('damage', 0)
        speedX   = self.client.S.d.get('speedX', 0)  # FIX: rileggi speedX post-step

        # FIX: usa danno INCREMENTALE per non penalizzare danni già scontati
        delta_damage = damage - self.prev_damage
        self.prev_damage = damage

        # --- Reward ---
        reward = (speedX * np.cos(angle)) - abs(speedX * np.sin(angle))
        reward -= abs(trackPos) * 5.0

        # FIX: penalizza steer aggressivo per ridurre oscillazioni
        reward -= abs(steer) * 0.5

        terminated = False

        if abs(trackPos) > 1.3:
            reward = -200.0
            terminated = True

        # FIX: condizione su danno incrementale, non accumulato
        if delta_damage > 0:
            reward = -200.0
            terminated = True

        return obs, float(reward), terminated, False, {}

    def _make_obs(self):
        track = self.client.S.d.get('track', [0] * 19)
        # FIX: gestisce lista track più corta del previsto (es. primo frame)
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

        # FIX: clip osservazioni ai bounds dichiarati per evitare warning SB3
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