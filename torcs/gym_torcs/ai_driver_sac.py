# ============================================================
# FILE: test_sac.py — Testa l'agente SAC addestrato su TORCS
# ============================================================
import os
import numpy as np
from stable_baselines3 import SAC
from torcs_env import TorcsEnv


def main():
    print("\n==================================================")
    print("   TORCS AI - TEST AGENTE SAC")
    print("==================================================")

    # Cerca automaticamente il modello migliore, altrimenti usa il finale
    best_model_path = "models/best_sac/best_model.zip"
    final_model_path = "models/sac_torcs_final.zip"

    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"[OK] Carico modello MIGLIORE: {best_model_path}")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        print(f"[OK] Carico modello FINALE: {final_model_path}")
    else:
        # Cerca qualsiasi .zip nella cartella models/
        zips = []
        for root, dirs, files in os.walk("models"):
            for f in files:
                if f.endswith(".zip"):
                    zips.append(os.path.join(root, f))
        if not zips:
            print("[ERRORE] Nessun modello trovato nella cartella 'models/'.")
            return
        model_path = zips[-1]  # prende l'ultimo trovato
        print(f"[OK] Carico modello trovato: {model_path}")

    # Carica modello e ambiente
    env = TorcsEnv(port=3001)
    model = SAC.load(model_path, env=env)

    N_EPISODES = 3  # numero di episodi di test

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_speed = 0.0

        print(f"\n--- Episodio {ep}/{N_EPISODES} ---")

        while not done:
            # deterministic=True: niente esplorazione, agente al massimo delle sue capacità
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            speed = obs[0]  # speedX è il primo elemento dell'observation
            if speed > max_speed:
                max_speed = speed

            # Log ogni 100 step
            if steps % 100 == 0:
                print(f"  Step {steps:5d} | Speed: {speed:6.1f} km/h | "
                      f"TrackPos: {obs[2]:+.3f} | Reward cumulativo: {total_reward:8.1f}")

        print(f"\n  [Episodio {ep} terminato]")
        print(f"  Step totali  : {steps}")
        print(f"  Reward totale: {total_reward:.1f}")
        print(f"  Velocità max : {max_speed:.1f} km/h")

    env.close()
    print("\n[OK] Test completato.")


if __name__ == "__main__":
    main()