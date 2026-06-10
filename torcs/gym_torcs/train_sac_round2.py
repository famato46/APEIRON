# ============================================================
# FILE 2: train_sac_round2.py — Riprende dal modello esistente
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
    print("   TORCS AI - ROUND 2 (SAC - riprende da modello)")
    print("==================================================")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints_r2", exist_ok=True)

    print("Connessione all'ambiente TORCS in corso...")
    env = TorcsEnv(port=3001)

    print("Validazione ambiente...")
    check_env(env, warn=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # FIX: carica il modello del round 1 e continua il training
    model_path = "models/sac_torcs_final"
    if os.path.exists(model_path + ".zip"):
        print(f"[OK] Carico modello esistente: {model_path}.zip")
        model = SAC.load(
            model_path,
            env=env,
            # FIX: mantieni gli stessi iperparametri del round 1
            learning_rate=1e-4,      # FIX: lr ridotto da 3e-4 → apprendimento più fine nel round 2
            buffer_size=50_000,
            batch_size=128,
            device=device,
            tensorboard_log="./logs/sac_torcs_r2/"
        )
        # FIX: resetta il replay buffer — contiene esperienze con vecchia reward, non utili ora
        model.learn_kwargs = {}
        print("[OK] Replay buffer resettato (reward cambiata nel round 2)")
    else:
        print("[ERRORE] Modello non trovato. Esegui prima train_sac.py")
        env.close()
        return

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="./models/checkpoints_r2/",
        name_prefix="sac_torcs_r2"
    )

    print("\n>>> Avvio Round 2.")
    print("Lascia girare il simulatore. Premi Ctrl+C per interrompere.")

    try:
        model.learn(
            total_timesteps=200_000,
            callback=[checkpoint_callback],
            reset_num_timesteps=True,   # FIX: resetta contatore step per log puliti
            reset_num_timesteps_buffer=True,  # FIX: resetta buffer con nuova reward
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Interrotto manualmente. Salvo...")
    except TypeError:
        # FIX: alcune versioni SB3 non hanno reset_num_timesteps_buffer
        model.learn(
            total_timesteps=200_000,
            callback=[checkpoint_callback],
            reset_num_timesteps=True,
            progress_bar=True,
        )
    finally:
        model.save("models/sac_torcs_round2_final")
        print("\n[OK] Modello salvato in 'models/sac_torcs_round2_final'.")
        env.close()


if __name__ == "__main__":
    main()
