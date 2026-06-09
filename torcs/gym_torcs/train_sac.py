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

    # FIX: porta separata per train ed eval (TORCS deve girare su due porte distinte)
    env      = TorcsEnv(port=3001)
    eval_env = TorcsEnv(port=3002)

    # FIX: valida l'env prima di passarlo a SB3 (rileva errori silenti)
    print("Validazione ambiente...")
    check_env(env, warn=True)

    # FIX: usa GPU se disponibile
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,                   # FIX: soft update esplicito (default SB3, ma documentato)
        gamma=0.99,                  # FIX: discount factor esplicito
        ent_coef='auto',
        target_entropy='auto',       # FIX: entropia target automatica sull'action_space
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000,        # FIX: accumula esperienza prima di iniziare il training
        policy_kwargs=dict(
            net_arch=[256, 256],     # FIX: architettura MLP esplicita (default SB3 è [64,64])
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        device=device,
        tensorboard_log="./logs/sac_torcs/"
    )

    # FIX: salva checkpoint ogni 50k step (recovery in caso di crash notturno)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/checkpoints/",
        name_prefix="sac_torcs"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_sac/',
        log_path='./logs/eval/',
        eval_freq=10_000,
        n_eval_episodes=3,           # FIX: media su 3 episodi per robustezza della valutazione
        deterministic=True,
        render=False
    )

    print("\n>>> Avvio Addestramento SAC.")
    print("Lascia girare il simulatore. Premi Ctrl+C per interrompere.")

    try:
        model.learn(
            total_timesteps=500_000,
            callback=[eval_callback, checkpoint_callback],  # FIX: lista di callback
            reset_num_timesteps=True,
            progress_bar=True,       # FIX: barra di avanzamento (richiede rich/tqdm)
        )
    except KeyboardInterrupt:
        print("\n[!] Interrotto manualmente. Salvo la versione corrente...")
    finally:
        # FIX: finally garantisce il salvataggio anche in caso di eccezione non-KeyboardInterrupt
        model.save("models/sac_torcs_final")
        print("\n[OK] Modello salvato in 'models/sac_torcs_final'.")
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()