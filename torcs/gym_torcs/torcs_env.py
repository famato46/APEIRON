import gymnasium as gym
from gymnasium import spaces
import numpy as np
import snakeoil3_gym as snakeoil  # IMPORTIAMO IL TUO FILE!

class TorcsEnv(gym.Env):
    """Ambiente OpenAI Gymnasium per TORCS che usa Snakeoil per la comunicazione"""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(TorcsEnv, self).__init__()
        
        # Inizializza il client Snakeoil (gestisce lui host, port e socket!)
        self.client = snakeoil.Client()
        self.client.maxSteps = 100000  # Evita che il client si chiuda da solo
        
        # SPAZIO AZIONI: [steer, accel, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # SPAZIO OSSERVAZIONI: Le 9 feature del tuo MLP
        # [speedX, angle, trackPos, track_0, track_4, track_9, track_14, track_18, delta_track]
        self.observation_space = spaces.Box(
            low=np.array([-50.0, -np.pi, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -200.0], dtype=np.float32), 
            high=np.array([350.0, np.pi, 3.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float32), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Diciamo a Snakeoil di inviare il comando di riavvio (meta 1)
        self.client.R.d['meta'] = 1
        self.client.respond_to_server()
        
        # Riavvia la connessione UDP pulita
        self.client = snakeoil.Client()
        
        # Ricevi il primo frame di dati dopo il riavvio
        self.client.get_servers_input()
        obs = self._make_obs()
        
        return obs, {}

    def step(self, action):
        # 1. TRADUCI L'AZIONE GYM NEL DIZIONARIO DI SNAKEOIL
        self.client.R.d['steer'] = float(action[0])
        self.client.R.d['accel'] = float(action[1])
        self.client.R.d['brake'] = float(action[2])
        
        # Gestione del cambio basata sulla velocità
        speedX = self.client.S.d.get('speedX', 0)
        target_gear = 1
        if speedX > 50: target_gear = 2
        if speedX > 90: target_gear = 3
        if speedX > 150: target_gear = 4
        if speedX > 200: target_gear = 5
        if speedX > 280: target_gear = 6
        self.client.R.d['gear'] = target_gear
        
        # Invia i comandi tramite Snakeoil
        self.client.respond_to_server()

        # 2. RICEVI IL NUOVO STATO
        self.client.get_servers_input()
        obs = self._make_obs()
        
        # 3. CALCOLO RICOMPENSA (Reward)
        trackPos = self.client.S.d.get('trackPos', 0)
        angle = self.client.S.d.get('angle', 0)
        damage = self.client.S.d.get('damage', 0)
        
        # Premia la velocità parallela alla pista, penalizza i movimenti a zig-zag
        reward = (speedX * np.cos(angle)) - (np.abs(speedX * np.sin(angle)))
        # Penalizza l'uscita dal centro
        reward -= np.abs(trackPos) * 5.0
        
        # 4. CONDIZIONI DI FINE GARA
        terminated = False
        if np.abs(trackPos) > 1.3: # Uscito di pista nella sabbia
            reward = -200.0
            terminated = True
        
        if damage > 0: # Ha sbattuto il muso a muro
            reward = -200.0
            terminated = True

        return obs, float(reward), terminated, False, {}

    def _make_obs(self):
        """Estrae i dati dal dizionario di Snakeoil e crea il vettore per la rete neurale"""
        track = self.client.S.d.get('track', [0]*19)
        delta_track = track[18] - track[0]
        
        obs = np.array([
            self.client.S.d.get('speedX', 0),
            self.client.S.d.get('angle', 0),
            self.client.S.d.get('trackPos', 0),
            track[0], track[4], track[9], track[14], track[18],
            delta_track
        ], dtype=np.float32)
        return obs

    def close(self):
        self.client.shutdown()