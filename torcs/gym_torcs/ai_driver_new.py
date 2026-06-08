import numpy as np
import joblib
import socket
import sys
import os
import time

# --- CONFIGURAZIONE TORCS ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17

# --- CONFIGURAZIONE AI ---
MODEL_PATH = "model_bc.joblib"
SCALER_PATH = "out_bc/scaler.joblib"

class ServerState():
    def __init__(self):
        self.d = dict()
    def parse_server_str(self, server_string):
        servstr = server_string.strip()[:-1]
        sslisted = servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = self.destringify(w[1:])
    def destringify(self, s):
        if not s: return s
        if type(s) is str:
            try: return float(s)
            except ValueError: return s
        elif type(s) is list:
            if len(s) < 2: return self.destringify(s[0])
            else: return [self.destringify(i) for i in s]

class DriverAction():
    def __init__(self):
        self.d = {'accel': 0, 'brake': 0, 'clutch': 0, 'gear': 1, 'steer': 0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0}
    def __repr__(self):
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not isinstance(v, list): out += '%.3f' % v
            else: out += ' '.join([str(x) for x in v])
            out += ')'
        return out

def run_ai():
    print("\n==================================================")
    print("   TORCS AI - PILOTA AUTOMATICO (INFERENZA)")
    print("==================================================")

    # 1. CARICAMENTO MODELLO E SCALER
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"\n[ERRORE] File mancanti!")
        print(f"Cerco il modello in: {MODEL_PATH}")
        print(f"Cerco lo scaler in:  {SCALER_PATH}")
        return

    print("Caricamento Rete Neurale e Scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[OK] Cervello AI pronto!\n")

    # 2. CONNESSIONE A TORCS
    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1.0)
    initmsg = f"{SID}(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)"
    
    print("In attesa di TORCS (Avvia la gara in Practice mode)...")
    while True:
        try:
            so.sendto(initmsg.encode(), (HOST, PORT))
            sockdata, _ = so.recvfrom(DATA_SIZE)
            if '***identified***' in sockdata.decode():
                print(">>> [OK] Connesso al server TORCS! La macchina è guidata dall'AI.")
                break
        except:
            pass

    S = ServerState()
    R = DriverAction()
    
    prev_steer = 0.0  # Variabile per il filtro anti-zigzag
    step_count = 0

    try:
        while True:
            # Ricevi dati dal server
            try:
                sockdata, _ = so.recvfrom(DATA_SIZE)
                sockstr = sockdata.decode()
                
                # Riavvio automatico se la gara ricomincia
                if '***restart***' in sockstr:
                    print("\n[RESET] Gara riavviata.")
                    R.d['meta'] = 0
                    while True:
                        so.sendto(initmsg.encode(), (HOST, PORT))
                        try:
                            so.settimeout(0.5)
                            resp, _ = so.recvfrom(DATA_SIZE)
                            if '***identified***' in resp.decode(): break
                        except: pass
                    so.settimeout(1.0)
                    continue
                S.parse_server_str(sockstr)
            except socket.timeout:
                continue

            # --- 1. ESTRAZIONE SENSORI ---
            speedX = S.d.get('speedX', 0)
            angle = S.d.get('angle', 0)
            trackPos = S.d.get('trackPos', 0)
            track = S.d.get('track', [0]*19)
            
            # Sensori laser specifici e calcolo del delta_track
            track_0 = track[0]
            track_4 = track[4]
            track_9 = track[9]
            track_14 = track[14]
            track_18 = track[18]
            delta_track = track_18 - track_0

            # Vettore di input: ESATTAMENTE l'ordine delle 9 feature del training
            X_raw = np.array([[speedX, angle, trackPos, track_0, track_4, track_9, track_14, track_18, delta_track]])
            
            # --- 2. NORMALIZZAZIONE E PREDIZIONE ---
            # Riduciamo i dati alla scala 0-1 usando lo scaler generato dal training
            X_scaled = scaler.transform(X_raw)
            
            # Chiediamo alla rete neurale cosa fare
            pred = model.predict(X_scaled)[0]

            ai_steer = float(pred[0])
            ai_accel = float(pred[1])
            ai_brake = float(pred[2])

            # --- 3. REGOLE HARDCODED (POST-PROCESSING) ---
            
            # Filtro Anti-ZigZag: ammorbidisce le reazioni brusche della rete
            ai_steer = 0.7 * ai_steer + 0.3 * prev_steer
            prev_steer = ai_steer

            # Mutua Esclusione: se stai frenando forte, togli il piede dal gas
            if ai_brake > 0.1:
                ai_accel = 0.0

            # Launch Control: la rete non sa partire da zero, la forziamo noi
            if speedX < 10:
                ai_accel = 1.0
                ai_brake = 0.0
                ai_steer = 0.0

            # ESP (Controllo Stabilità): taglia potenza in curve veloci
            if speedX > 130 and abs(ai_steer) > 0.25:
                ai_accel *= 0.5 
                
            # Clamping valori per sicurezza assoluta (non sforare -1.0 o 1.0)
            ai_steer = max(-1.0, min(1.0, ai_steer))
            ai_accel = max(0.0, min(1.0, ai_accel))
            ai_brake = max(0.0, min(1.0, ai_brake))

            # --- 4. CAMBIO MARCE AUTOMATICO ---
            target_gear = 1
            if speedX > 50:  target_gear = 2
            if speedX > 90:  target_gear = 3
            if speedX > 150: target_gear = 4
            if speedX > 200: target_gear = 5
            if speedX > 280: target_gear = 6
            if speedX < -2:  target_gear = -1
            
            # Non cambiare marcia (shift shock) se stiamo curvando forte
            current_server_gear = S.d.get('gear', 1)
            ai_gear = current_server_gear if abs(ai_steer) > 0.4 else target_gear

            # --- 5. INVIO COMANDI A TORCS ---
            R.d['steer'] = ai_steer
            R.d['accel'] = ai_accel
            R.d['brake'] = ai_brake
            R.d['gear'] = ai_gear

            # Mostra a schermo cosa sta pensando l'AI (1 volta su 30 frame)
            step_count += 1
            if step_count % 30 == 0:
                print(f"Vel: {speedX:5.1f} | Sterzo: {ai_steer:5.2f} | Gas: {ai_accel:5.2f} | Freno: {ai_brake:5.2f} | Marcia: {ai_gear}", end='\r')

            so.sendto(repr(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\n[STOP] Motore spento. Uscita.")
    finally:
        so.close()

if __name__ == "__main__":
    run_ai()