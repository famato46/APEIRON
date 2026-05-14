import pandas as pd
import matplotlib.pyplot as plt

# Carica il tuo dataset
df = pd.read_csv('manual_log.csv')

# Prendi solo uno spezzone (es. 500 step, circa 10 secondi di guida)
df_subset = df['steer'].head(500)

plt.figure(figsize=(12, 6))
plt.plot(df_subset, label='Originale (Tastiera)', color='lightgray', alpha=0.7)

# Prova tre finestre diverse
plt.plot(df_subset.rolling(window=5).mean(), label='Finestra = 5', color='green')
plt.plot(df_subset.rolling(window=10).mean(), label='Finestra = 10', color='blue')
plt.plot(df_subset.rolling(window=20).mean(), label='Finestra = 20', color='red')

plt.title("Confronto Medie Mobili sullo Sterzo")
plt.legend()
plt.show()