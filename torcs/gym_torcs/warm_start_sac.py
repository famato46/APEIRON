# ============================================================
# FILE: warm_start_sac.py
# Prepopola il replay buffer di SAC con traiettorie del MLP BC
# Prima di lanciare train_sac.py, lancia questo script UNA VOLTA
# ============================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import joblib
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from torcs_env import TorcsEnv

# ---- Percorsi ----
MLP_PATH    = "models/model_bc.joblib"
SCALER_PATH = "out_bc/scaler.joblib"
OUTPUT_MODEL = "models/sac_warm_start"

# Quante transizioni raccogliere con il MLP prima di passare a SAC
N_WARMUP_STEPS = 5000


def make_obs_for_mlp(obs, dist_from_start=0.0):
    """
    Il MLP ha 10 feature: aggiunge dist_from_start (non presente nell'env)
    in posizione 3 (dopo speedX, angle, trackPos).
    obs shape: (9,) → output: (10,)
    """
    return np.insert(obs, 3, dist_from_start).reshape(1, -1)


def main():
    print("\n==================================================")
    print("   TORCS AI - WARM START con MLP Behavioral Cloning")
    print("==================================================")

    os.makedirs("models", exist_ok=True)

    # Carica MLP e scaler
    if not os.path.exists(MLP_PATH):
        print(f"[ERRORE] {MLP_PATH} non trovato.")
        return
    mlp    = joblib.load(MLP_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] MLP caricato: arch={mlp.hidden_layer_sizes}")

    # Crea env e modello SAC vuoto
    env = TorcsEnv(port=3001)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        target_entropy='auto',
        train_freq=1,
        gradient_steps=1,
        learning_starts=500,
        policy_kwargs=dict(
            net_arch=[128, 128],
            activation_fn=torch.nn.ReLU,
        ),
        verbose=0,
        device=device,
        tensorboard_log="./logs/sac_torcs/"
    )

    print(f"\n[>>] Raccolta {N_WARMUP_STEPS} transizioni con MLP BC...")
    obs, _ = env.reset()
    dist_from_start = 0.0
    steps_collected = 0
    episodes = 0

    while steps_collected < N_WARMUP_STEPS:
        # Prepara input per MLP (10 feature con dist_from_start)
        obs_mlp = make_obs_for_mlp(obs, dist_from_start)
        obs_scaled = scaler.transform(obs_mlp)

        # Predici azione con MLP e clippa nei bounds dell'action_space
        action_raw = mlp.predict(obs_scaled)[0]
        action = np.array([
            float(np.clip(action_raw[0], -1.0, 1.0)),   # steer
            float(np.clip(action_raw[1],  0.0, 1.0)),   # accel  FIX: clip negativo
            float(np.clip(action_raw[2],  0.0, 1.0)),   # brake  FIX: clip negativo
        ], dtype=np.float32)

        # Step nell'env
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Inserisci nel replay buffer di SAC
        model.replay_buffer.add(
            obs=obs.reshape(1, -1),
            next_obs=next_obs.reshape(1, -1),
            action=action.reshape(1, -1),
            reward=np.array([reward]),
            done=np.array([done]),
            infos=[info]
        )

        steps_collected += 1
        dist_from_start += obs[0] * 0.02  # stima distanza percorsa

        if done:
            obs, _ = env.reset()
            dist_from_start = 0.0
            episodes += 1
            if episodes % 5 == 0:
                print(f"  Episodio {episodes} | Step raccolti: {steps_collected}/{N_WARMUP_STEPS}")
        else:
            obs = next_obs

    print(f"\n[OK] Raccolte {steps_collected} transizioni in {episodes} episodi.")
    print(f"     Buffer SAC: {model.replay_buffer.size()} transizioni")

    # Salva il modello SAC con buffer prepopolato
    model.save(OUTPUT_MODEL)
    # FIX: salva anche il replay buffer separatamente
    model.save_replay_buffer(OUTPUT_MODEL + "_buffer")
    print(f"[OK] Modello salvato: {OUTPUT_MODEL}.zip")
    print(f"[OK] Buffer salvato:  {OUTPUT_MODEL}_buffer.pkl")

    env.close()
    print("\n[>>] Ora lancia: python train_sac_from_warmstart.py")


if __name__ == "__main__":
    main()
