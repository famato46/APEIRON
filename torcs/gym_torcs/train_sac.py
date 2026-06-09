# ============================================================
# FILE 2: train_sac.py
# ============================================================
import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from torcs_env import TorcsEnv


def main():
    print("\n==================================================")
    print("   TORCS AI - ADDESTRAMENTO NOTTURNO (SAC - SB3)")
    print("==================================================")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Connessione all'ambiente TORCS in corso...")

    env      = TorcsEnv(port=3001)
    eval_env = TorcsEnv(port=3002)

    print("Validazione ambiente...")
    check_env(env, warn=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50_000,          # FIX: ridotto da 100k → meno RAM su CPU
        batch_size=128,              # FIX: ridotto da 256 → step più veloci su CPU
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        target_entropy='auto',
        train_freq=1,
        gradient_steps=1,
        learning_starts=500,         # FIX: ridotto da 1000 → inizia ad imparare prima
        policy_kwargs=dict(
            net_arch=[128, 128],     # FIX: ridotto da [256,256] → CPU non soffoca
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        device=device,
        tensorboard_log="./logs/sac_torcs/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,            # FIX: ridotto da 50k → più granulare su 200k step totali
        save_path="./models/checkpoints/",
        name_prefix="sac_torcs"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_sac/',
        log_path='./logs/eval/',
        eval_freq=5_000,             # FIX: ridotto da 10k → valutazioni più frequenti su 200k step
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )

    print("\n>>> Avvio Addestramento SAC.")
    print("Lascia girare il simulatore. Premi Ctrl+C per interrompere.")

    try:
        model.learn(
            total_timesteps=200_000, # FIX: ridotto da 500k → ~4/5h realistiche su CPU
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=True,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Interrotto manualmente. Salvo la versione corrente...")
    finally:
        model.save("models/sac_torcs_final")
        print("\n[OK] Modello salvato in 'models/sac_torcs_final'.")
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()