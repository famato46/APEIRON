# <table>
  <tr>
    <td>
      <img width="195,8" height="100" alt="LogoAPEIRON copia" src="https://github.com/user-attachments/assets/7ba8eb35-5b2c-40e1-8f15-865ffcd38d04" />
    </td>
    <td>
      <img width="334" height="59" alt="LogoAPEIRON copia 2" src="https://github.com/user-attachments/assets/df21fd67-f9d2-4b1f-9bcd-98d949915cdf" />
    </td>
    <td>
      <img width="178" height="100" alt="IBM_2025_Ferrari_1101_PressReleaseHeaderImage_Social" src="https://github.com/user-attachments/assets/912d6d2f-7eba-4fb8-aca9-4c27ecc77216" />
    </td>
  </tr>
</table>

<h1>🏎️ APEIRON - AI Autonomous Racing Team</h1>

Repository ufficiale del team APEIRON (Gruppo 15), partecipante alla IBM AI Racing League — Università degli Studi di Salerno, A.A. 2025/2026.

<h3>Team:</h3>

- Francesca Gaia Amato
- Giovanni Guercia
- Bruno Oliva
- Carmine Fonzo
- Simone De Riggi

---

## 🎯 Approccio: Imitation Learning

L'agente è sviluppato tramite **Behavioral Cloning (BC)**: un bot deterministico guida il circuito Corkscrew raccogliendo dati di alta qualità, che vengono poi usati per addestrare una rete neurale MLP a imitarne il comportamento.

---

## 🛠️ Pipeline (6 fasi)

| Fase | Descrizione | Script |
|------|-------------|--------|
| 1 | Sviluppo bot esperto | `torcs_jm_par_modulare.py` |
| 2 | Raccolta dati (CSV grezzi) | `torcs_jm_par_modulare.py` |
| 3 | Filtraggio e bilanciamento | `filter_dataset.py`, `balance_dataset.py` |
| 4 | EDA, feature engineering, split, normalizzazione | `build_dataset.py` |
| 5 | Training MLP multi-output | `train_mlp.py` |
| 6 | Agente AI con safety net | `ai_driver.py` |

---

## 📂 Struttura del Repository

```
torcs/gym_torcs/
├── torcs_jm_par_modulare.py   # Bot esperto + raccolta dati
├── filter_dataset.py          # Filtraggio giri per qualità
├── balance_dataset.py         # Bilanciamento sterzo multi-file
├── build_dataset.py           # Feature engineering, split, normalizzazione
├── train_mlp.py               # Training MLPRegressor
├── ai_driver.py               # Agente finale con safety net
├── out_bc/                    # Artefatti: dataset_bc.npz, scaler.joblib, feature_config.json
└── models/                    # Modello addestrato: model_bc.joblib
```

---

## ⚙️ Installazione e Setup

### 1. Prerequisiti

- [Visual Studio Code](https://code.visualstudio.com/)
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.com/) — necessario per scaricare i file grandi (modelli, dataset)
- Python 3.x
- TORCS installato e configurato tramite **Simula Studio**

### 2. Installa Git LFS

> ⚠️ Questo passaggio va fatto **prima** di clonare il repository, altrimenti i file grandi non vengono scaricati correttamente.

```bash
# Installa Git LFS sul tuo sistema
git lfs install
```

### 3. Clona il repository

```bash
git clone https://github.com/famato46/APEIRON.git
cd APEIRON
```

### 4. Installa le dipendenze Python

```bash
pip install numpy pandas scikit-learn joblib pynput
```

---

## 🚀 Come Usare il Progetto

Tutti i comandi vanno eseguiti dal terminale di Visual Studio Code, dalla cartella `torcs/gym_torcs/`.

```bash
cd torcs/gym_torcs
```

### Fase 2 — Raccolta dati con il bot esperto

Avvia prima TORCS tramite Simula Studio, poi lancia il bot:

```bash
python torcs_jm_par_modulare.py
```

I CSV vengono salvati automaticamente in `torcs/gym_torcs/` con il nome `dataset_track_<timestamp>.csv`.
Usa le frecce della tastiera per intervenire manualmente se necessario — le manovre manuali non vengono registrate nel CSV.

### Fase 3 — Filtraggio dei giri

```bash
python filter_dataset.py dataset_track_*.csv -o dataset_filtered.csv
```

Aggiunge `--only-good` per usare solo i giri migliori:

```bash
python filter_dataset.py dataset_track_*.csv -o dataset_filtered.csv --only-good
```

### Fase 3 — Bilanciamento del dataset

```bash
python balance_dataset.py dataset_filtered.csv -o dataset_balanced.csv
```

### Fase 4 — Feature engineering, split e normalizzazione

```bash
python build_dataset.py dataset_balanced.csv -o ./out_bc
```

### Fase 5 — Training del modello MLP

```bash
python train_mlp.py --data ./out_bc --out ./models
```

Per un test rapido con meno combinazioni:

```bash
python train_mlp.py --data ./out_bc --out ./models --quick
```

### Fase 6 — Lancio dell'agente AI

Avvia prima TORCS tramite Simula Studio, poi lancia l'agente:

```bash
python ai_driver.py
```

---

## 📊 Risultati

| Agente | Tempo sul Giro |
|--------|---------------|
| Bot Esperto Deterministico | 1:36 – 1:42 |
| MLP Behavioral Cloning | ~1:36 |

---

## 🔮 Sviluppi Futuri

È stato tentato un fine-tuning tramite Reinforcement Learning (Soft Actor-Critic) per superare il tetto prestazionale del BC, ma i tentativi non hanno prodotto risultati utilizzabili a causa di instabilità durante l'addestramento. Rimane il principale sviluppo futuro del progetto.
