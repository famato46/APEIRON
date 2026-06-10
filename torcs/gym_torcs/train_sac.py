# ============================================================
# FILE 2: train_sac.py — Training definitivo da zero, 400k step
# ============================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from torcs_env import TorcsEnv


def main():
    print("\n==================================================")
    print("   TORCS AI - TRAINING DEFINITIVO (SAC - SB3)")
    print("==================================================")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    print("Connessione all'ambiente TORCS in corso...")
    env = TorcsEnv(port=3001)

    print("Validazione ambiente...")
    check_env(env, warn=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
        verbose=1,
        device=device,
        tensorboard_log="./logs/sac_torcs/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,            # checkpoint ogni 50k su 400k totali
        save_path="./models/checkpoints/",
        name_prefix="sac_torcs"
    )

    print("\n>>> Avvio Training Definitivo — 400.000 step.")
    print("Lascia girare il simulatore. Premi Ctrl+C per interrompere.")

    try:
        model.learn(
            total_timesteps=400_000,
            callback=[checkpoint_callback],
            reset_num_timesteps=True,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Interrotto manualmente. Salvo...")
    finally:
        model.save("models/sac_torcs_final")
        print("\n[OK] Modello salvato in 'models/sac_torcs_final'.")
        env.close()


if __name__ == "__main__":
    main()