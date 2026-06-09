import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Analisi Statistica Avanzata per TORCS")
    parser.add_argument('input_csv', help='Percorso del dataset CSV da analizzare')
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"ERRORE: File {args.input_csv} non trovato.")
        return

    print(f"Caricamento dataset da {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    # Assicuriamoci di prendere solo le colonne numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Rimuoviamo colonne inutili per l'analisi dei sensori
    cols_to_drop = ['timestamp', 'cur_lap_time', 'last_lap_time', 'dist_from_start', 'meta']
    features = [c for c in numeric_cols if c not in cols_to_drop]
    df_clean = df[features]

    # Target specifici per il terzo grafico
    targets = ['steer', 'accel', 'brake']
    targets_present = [t for t in targets if t in df.columns]

    print("Generazione dei grafici in corso. Chiudi una finestra per vedere la successiva...")

    # =================================================================
    # GRAFICO 1: VARIANZA DEI SENSORI
    # =================================================================
    # Usiamo una scala logaritmica perché la velocità ha una varianza enorme
    # rispetto allo sterzo o all'angolo che hanno varianze < 1.
    plt.figure(figsize=(14, 6))
    variances = df_clean.var().sort_values(ascending=False)
    
    sns.barplot(x=variances.index, y=variances.values, palette="viridis")
    plt.yscale('log') # Scala logaritmica fondamentale!
    plt.xticks(rotation=45, ha='right')
    plt.title('Varianza dei Sensori e dei Target (Scala Logaritmica)', fontsize=14, fontweight='bold')
    plt.ylabel('Varianza (Log)')
    plt.tight_layout()
    plt.show()

    # =================================================================
    # GRAFICO 2: MATRICE DI CORRELAZIONE E MULTICOLLINEARITÀ
    # =================================================================
    # Mostra come i sensori comunicano tra loro e con i comandi
    plt.figure(figsize=(12, 10))
    corr_matrix = df_clean.corr()
    
    # Creiamo una maschera per coprire la metà superiore del triangolo (che è speculare)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot=False, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Matrice di Correlazione (Multicollinearità)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # =================================================================
    # GRAFICO 3: DISTRIBUZIONE DELLE AZIONI (TARGET)
    # =================================================================
    if targets_present:
        fig, axes = plt.subplots(1, len(targets_present), figsize=(16, 5))
        if len(targets_present) == 1:
            axes = [axes] # Gestione caso singolo target
            
        for i, target in enumerate(targets_present):
            sns.histplot(df[target], bins=50, kde=True, ax=axes[i], color='royalblue')
            axes[i].set_title(f'Distribuzione: {target.upper()}', fontweight='bold')
            axes[i].set_ylabel('Frequenza (N. Frame)')
            axes[i].set_xlabel(f'Valore {target}')
        
        plt.suptitle('Analisi della Distribuzione delle Azioni del Pilota', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print("Nessuna colonna target (steer, accel, brake) trovata per il terzo grafico.")

if __name__ == "__main__":
    main()