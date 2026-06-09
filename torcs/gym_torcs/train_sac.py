import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from torcs_env import TorcsEnv

def main():
    print("\n==================================================")
    print("   TORCS AI - ADDESTRAMENTO NOTTURNO (SAC - SB3)")
    print("==================================================")

    # Creazione cartelle di salvataggio
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Connessione all'ambiente TORCS in corso...")
    # Ambiente per l'addestramento principale
    env = TorcsEnv()

    # Ambiente parallelo usato solo per i test ogni tot secondi
    eval_env = TorcsEnv() 

    # Configurazione dell'Agente SAC
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,          # Memoria delle esperienze passate
        batch_size=256,
        ent_coef='auto',             # Regolazione automatica dell'esplorazione
        train_freq=1,
        gradient_steps=1,
        verbose=1,                   # Mostra i log sul terminale
        tensorboard_log="./logs/sac_torcs/" 
    )

    # Il "Salvavita Notturno": valuta l'agente ogni 10.000 step
    # Se il tempo sul giro/reward migliora, salva una copia blindata.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_sac/',
        log_path='./logs/eval/',
        eval_freq=10000,
        deterministic=True, # Spegne l'esplorazione durante l'esame
        render=False
    )

    print("\n>>> Avvio Addestramento SAC.")
    print("Lascia girare il simulatore. Premi Ctrl+C se vuoi interrompere prima del termine.")
    
    try:
        # 500.000 step = circa 3/4 ore di simulazione ininterrotta
        model.learn(total_timesteps=500000, callback=eval_callback)
    except KeyboardInterrupt:
        print("\n[!] Addestramento interrotto manualmente. Salvo la versione corrente...")

    # Salvataggio di fine ciclo
    model.save("models/sac_torcs_final")
    print("\n[OK] Addestramento completato. Controlla la cartella 'models/best_sac'!")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()