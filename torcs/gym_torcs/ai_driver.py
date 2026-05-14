import socket
import sys
import numpy as np
import joblib
import json

# ==========================================
# 1. CARICAMENTO DELL'INTELLIGENZA ARTIFICIALE E FEATURE
# ==========================================
print("Caricamento del cervello AI...")
try:
    model = joblib.load('model_bc.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # NOVITÀ: Leggiamo quali sono le 10 feature che la rete neurale vuole!
    with open('dataset_balanced.csv/feature_config.json', 'r') as f:
        feat_config = json.load(f)
        # Cerchiamo la lista delle feature nel file json
        if 'features' in feat_config:
            EXPECTED_FEATURES = feat_config['features']
        else:
            # Fallback intelligente: prende la prima lista che trova
            EXPECTED_FEATURES = next(v for v in feat_config.values() if isinstance(v, list))
            
    print(f"Modello caricato! Richiede ESATTAMENTE queste {len(EXPECTED_FEATURES)} feature: {EXPECTED_FEATURES}")
    
except Exception as e:
    print(f"Errore nel caricamento dei modelli AI: {e}")
    sys.exit()

# Parametri di connessione a TORCS
host = 'localhost'
port = 3001
data_size = 2**17

def setup_connection():
    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1)
    initmsg = 'SCR(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)'
    
    while True:
        try:
            so.sendto(initmsg.encode(), (host, port))
            sockdata, _ = so.recvfrom(data_size)
            if '***identified***' in sockdata.decode('utf-8'):
                print("Connesso a TORCS! Si parte...")
                return so
        except socket.error:
            print("In attesa che TORCS si avvii sulla porta 3001...")
            pass

def parse_server_str(server_string):
    d = dict()
    servstr = server_string.strip()[:-1]
    sslisted = servstr.strip().lstrip('(').rstrip(')').split(')(')
    for i in sslisted:
        w = i.split(' ')
        if len(w) > 1:
            try: d[w[0]] = [float(x) for x in w[1:]] if len(w[1:]) > 1 else float(w[1])
            except ValueError: d[w[0]] = w[1:]
    return d

# ==========================================
# 2. IL LOOP DI GUIDA AI
# ==========================================
def run_ai():
    so = setup_connection()
    
    while True:
        try:
            sockdata, _ = so.recvfrom(data_size)
            sockdata = sockdata.decode('utf-8')
            
            if '***shutdown***' in sockdata or '***restart***' in sockdata:
                print("Gara terminata. Spengo il bot.")
                break
                
            if not sockdata: continue
            
            S = parse_server_str(sockdata)
            
            # 1. RACCOLTA DI TUTTI I SENSORI
            track_sensors = S.get('track', [200.0] * 19)
            speedX = S.get('speedX', 0.0)
            
            # Creiamo un "cesto" con tutti i dati possibili
            stato_corrente = {
                'speedX': speedX,
                'speedY': S.get('speedY', 0.0),
                'speedZ': S.get('speedZ', 0.0),
                'angle': S.get('angle', 0.0),
                'trackPos': S.get('trackPos', 0.0),
                'rpm': S.get('rpm', 0.0),
                # FEATURE ENGINEERING: Aggiungiamo il delta_track che avevi calcolato in fase di addestramento!
                'delta_track': track_sensors[18] - track_sensors[0]
            }
            
            # Aggiungiamo i 19 sensori della pista al cesto
            for i in range(19):
                stato_corrente[f'track_{i}'] = track_sensors[i]
                
            # 2. SELEZIONE CHIRURGICA DELLE 10 FEATURE
            # Pesca dal cesto solo le 10 feature che la rete neurale sta aspettando, nell'ordine giusto
            x_raw = []
            for feat in EXPECTED_FEATURES:
                x_raw.append(stato_corrente.get(feat, 0.0))
                
            x_array = np.array([x_raw])
            
            # 3. NORMALIZZAZIONE E PREDIZIONE
            x_scaled = scaler.transform(x_array)
            y_pred = model.predict(x_scaled)[0]
            
            steer = np.clip(y_pred[0], -1.0, 1.0)
            accel = np.clip(y_pred[1], 0.0, 1.0)
            brake = np.clip(y_pred[2], 0.0, 1.0)
            
            gear = 1
            if speedX > 60: gear = 2
            if speedX > 105: gear = 3
            if speedX > 150: gear = 4
            if speedX > 195: gear = 5
            if speedX > 235: gear = 6
            if speedX < -2: gear = -1

            out_msg = f"(accel {accel:.3f})(brake {brake:.3f})(gear {gear})(steer {steer:.3f})(clutch 0)(focus -90 -45 0 45 90)(meta 0)"
            so.sendto(out_msg.encode(), (host, port))
            
        except Exception as e:
            # Se c'è ancora un errore, lo stampa senza fermare il gioco
            print(f"ERRORE NEL LOOP: {e}")

if __name__ == "__main__":
    run_ai()