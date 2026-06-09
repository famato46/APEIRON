import pandas as pd
import numpy as np

def inject_start_data(input_csv, output_csv):
    print(f"Leggo {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Creiamo 100 righe di "ripartenza da fermo"
    # Questi valori forzano l'MLP ad imparare che:
    # A bassa velocità (speedX < 5), accel=1 e brake=0.
    start_data = []
    for i in range(100):
        # Simuliamo una partenza: velocità crescente da 0 a 5 km/h
        row = {col: 0.0 for col in df.columns}
        row['speedX'] = np.random.uniform(0.0, 5.0)
        row['accel'] = 1.0   # Tavoletta!
        row['brake'] = 0.0   # Niente freno
        row['steer'] = np.random.uniform(-0.1, 0.1) # Quasi dritto
        row['is_clean'] = 1
        start_data.append(row)
    
    df_new = pd.concat([df, pd.DataFrame(start_data)], ignore_index=True)
    
    df_new.to_csv(output_csv, index=False)
    print(f"Salvataggio completato: {output_csv}")
    print(f"Nuovo totale righe: {len(df_new)}")

if __name__ == "__main__":
    # Sostituisci col nome del tuo file reale
    inject_start_data('dataset_track_1781031802.csv', 'dataset_track_fixed.csv')