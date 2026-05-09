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

Welcome to the official repository of team APEIRON, one of the UNISA (University of Salerno) teams participating in the IBM AI Racing Competition.
This repository contains the driver code for developing an autonomous driving agent within the TORCS simulator, with parameters and features optimized using __IBM Granite__.

<h2>IBM AI Racing Competition Project (team 15) --> APEIRON (UNISA)</h2>

<h3>Our Team:</h3>

- Francesca Gaia Amato 
- Giovanni Guercia 
- Bruno Oliva
- Carmine Fonzo
- Simone De Riggi

<h2> 🎯 Strategia di Sviluppo: BC + RL </h2>

Per massimizzare l'efficienza dell'addestramento ed evitare che l'agente parta da una conoscenza nulla (comportamento casuale), abbiamo scelto una pipeline in due fasi:

1. **Fase 1: Behavioral Cloning (BC):** Inizializzazione della policy tramite apprendimento supervisionato su un dataset di dimostrazioni esperte.
2. **Fase 2: Reinforcement Learning (RL) Fine-tuning:** Ottimizzazione della policy pre-addestrata tramite interazione diretta con l'ambiente per superare le prestazioni dell'esperto.

<h2> 🛠️ Stato Attuale della Fase 1: Creazione dell'Esperto </h2>

Attualmente, il lavoro è focalizzato sulla **costruzione di un "Expert Bot"** solido. Prima di poter addestrare una rete neurale a imitare un comportamento, è necessario che il comportamento sorgente sia di alta qualità.

### Attività in corso:

* **Modifica di `modular_control.py`:** Stiamo rifinando la logica di controllo all'interno dell'ambiente `gym_torcs`.
* **Sviluppo con IBM Granite:** Per lo sviluppo e l'ottimizzazione di questa fase, l'utilizzo di **IBM Granite** è stato essenziale, permettendo di strutturare in modo efficiente la logica modulare e risolvere le criticità nel controllo dei sensori.
* **Feature Engineering:** Selezione dei sensori critici (raggi `track`, `trackPos`, `angle`, `speedX`) per fornire alla rete neurale un input pulito e normalizzato.
* **Data Collection:** Una volta rifinito il bot modulare, verrà utilizzato per generare il dataset di coppie (Stato, Azione) necessario per il Behavioral Cloning.

## 🚀 Prossimi Passi

### Fase 2: Reinforcement Learning

Una volta ottenuta una policy che riesce a completare i giri di pista seguendo l'esempio del bot, passeremo alla fase di ottimizzazione tramite RL.

* **Algoritmo:** [TBD - In fase di valutazione tra DDPG, PPO, SAC o TD3].
* **Reward Function:** Definizione di una funzione di ricompensa che premi la velocità media e la stabilità.
  
## 🙏 Ringraziamenti

Un ringraziamento speciale va a **IBM Granite** per il supporto fondamentale fornito nella generazione e nel raffinamento della logica algoritmica di questa fase iniziale del progetto.

## 📦 Requisiti e Setup

* **Ambiente:** `gym_torcs` / `pyTORCS`
* **Linguaggio:** Python 3.x
* **Librerie principali:** NumPy, TensorFlow/PyTorch, Stable-Baselines3 (previsto)
