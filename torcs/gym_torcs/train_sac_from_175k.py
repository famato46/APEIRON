# ============================================================
# FILE: train_sac_from_175k.py
# Riprende da sac_torcs_final (175k step) + buffer warm start MLP
# ============================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from torcs_env import TorcsEnv

BASE_MODEL       = "models/sac_torcs_final"
WARMSTART_BUFFER = "models/sac_warm_start_buffer"


def main():
    print("\n==================================================")
    print("   TORCS AI - RIPRESA DA 175k + WARM START MLP")
    print("==================================================")

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints_175k", exist_ok=True)

    env = TorcsEnv(port=3001)
    check_env(env, warn=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not os.path.exists(BASE_MODEL + ".zip"):
        print(f"[ERRORE] {BASE_MODEL}.zip non trovato.")
        env.close()
        return

    print(f"[OK] Carico modello base: {BASE_MODEL}.zip")
    model = SAC.load(
        BASE_MODEL,
        env=env,
        device=device,
        # FIX: stessa reward, stessi iperparametri — nessun cambiamento
        learning_rate=3e-4,
        tensorboard_log="./logs/sac_torcs_175k/"
    )

    # FIX: carica buffer MLP se disponibile, altrimenti continua senza
    if os.path.exists(WARMSTART_BUFFER + ".pkl"):
        model.load_replay_buffer(WARMSTART_BUFFER)
        print(f"[OK] Buffer MLP caricato: {model.replay_buffer.size()} transizioni")
    else:
        print("[WARN] Buffer MLP non trovato — lancia prima warm_start_sac.py")
        print("       Continuo comunque dal modello a 175k step...")

    model.learning_starts = 0  # FIX: buffer già popolato, inizia subito

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,           # checkpoint frequenti — stiamo raffinando
        save_path="./models/checkpoints_175k/",
        name_prefix="sac_175k"
    )

    print("\n>>> Avvio Training da 175k step — altri 225.000 step.")
    print("Lascia girare il simulatore. Premi Ctrl+C per interrompere.")

    try:
        model.learn(
            total_timesteps=225_000,    # FIX: 175k già fatti + 225k = 400k totali
            callback=[checkpoint_callback],
            reset_num_timesteps=False,  # FIX: continua il conteggio da 175k
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Interrotto. Salvo...")
    finally:
        model.save("models/sac_175k_final")
        print("\n[OK] Modello salvato in 'models/sac_175k_final'.")
        env.close()


if __name__ == "__main__":
    main()
