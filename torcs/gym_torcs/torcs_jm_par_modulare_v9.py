"""
TORCS bot di guida — VERSIONE 2 (TELEMETRY-MATCHED).

Pilota la macchina sul circuito Corkscrew e logga in un CSV, per ogni step, lo
stato del simulatore e l'azione decisa.

#############################################################################
# AGGIORNAMENTO v2 — clone dello stile umano da 6 dataset registrati.
#
# Tempo di riferimento del bot attuale: ~118 s/giro
# Tempo di riferimento dell'umano (media 6 giri): 67.86 s   (best 67.11 s)
# Target competizione IBM: 71.00 s
#
# Modifiche cardinali (tutti i numeri derivati dai dataset_amico_1..6.csv):
#  1) SPEED_MAP riscritta sui percentili p70-p75 di speedX in funzione della
#     distanza vista davanti dai sensori track. A 100 m di vista il bot
#     vecchio puntava a 195 km/h, l'umano fa 265 km/h.
#  2) Cambio marce ai regimi alti che usa l'umano:
#        - shift up a ~18000 rpm (era 9000)
#        - shift down sotto 14000 rpm (era 6500)
#     L'umano sta in 4a marcia a 17000 rpm di mediana, il bot vecchio a 8300.
#  3) Frenata: gain piu' alto (decel media umano 1.6 g, picchi 1.97 g) e
#     deadband ridotta da 8 a 5 km/h. Brake peak target = 0.70 (era 0.93).
#  4) Riapertura gas: l'umano riporta accel > 0.9 a soli 25-36 m DOPO l'apex.
#     Resa piu' aggressiva la logica is_unwinding e abbassato il lift-off.
#  5) Racing line out-in-out: aggiunto target_trackpos basato su curvatura
#     anticipata. L'umano usa |trackPos| fino a 0.95 (bot vecchio fermo a 0.74,
#     mediana 0.08). Riduzione del K_E del cross-track (Stanley) per non
#     "richiamare" il bot al centro pista.
#  6) Override Corkscrew (2300 m - 2500 m): tabella spline dist->v_target
#     estratta direttamente dal profilo medio umano dei 6 dataset.
#     L'umano entra a 292 km/h, frena a brake 0.65 per 80 m, apex a 76 km/h
#     a 2440-2455 m, ri-apre il gas gia' all'apex.
#  7) Smoothing dello sterzo: l'umano ha |Δsteer| per step due volte piu'
#     basso del bot vecchio. Slew rate piu' restrittivo per evitare strattoni.
#
#############################################################################
# PATCH v2.1 (TRAIL-BRAKE FIX) — fix curve in sequenza
#
# Sintomo: la prima curva (veloce, ~200m) era perfetta, la seconda (lenta,
# ~480m apex) faceva volare il bot fuori pista. Analisi del CSV di crash:
#   - dfs=380m: brake sale da 0.32 a 0.93
#   - dfs=410m: brake=0.93 E steer=+0.43 (entrambi alti)
#   - dfs=420m: brake=0.93 E steer=+1.00 SATURO  → ruote anteriori bloccate
#   - dfs=427m: trackPos=-1.03 (fuori pista) e angle=+0.58
#   - in totale 17 step su 59 con (|steer|>=0.5 AND brake>=0.7).
#
# Causa: bloccaggio dei freni anteriori durante la rotazione = no grip
# laterale = understeer = fuori pista. L'umano modula: brake max 0.73 in
# rettilineo e SCENDE sotto 0.5 in apex (trail-braking classico).
#
# Modifiche minimi, NON depotenzianti:
#  A) TRAIL BRAKING: cap dinamico del brake in funzione dello sterzo.
#     - |steer|<=0.20         → brake_cap = BRAKE_MAX = 0.93 (frenata piena)
#     - |steer|=0.50          → brake_cap = 0.79*BRAKE_MAX = 0.73 (come umano)
#     - |steer|=1.00          → brake_cap = 0.45*BRAKE_MAX = 0.42 (sblocca)
#     La frenata in rettilineo non e' toccata. L'aggressivita' resta.
#  B) ANTICIPO TARGET_TRACKPOS: EMA della curvatura piu' reattiva
#     (alpha 0.30 → 0.45) e max_dtp 0.04 → 0.06. Aiuta il bot a spostarsi
#     a OUT prima della seconda curva quando il rettilineo e' breve.
#############################################################################
"""

import socket
import sys
import getopt
import os
import time
import csv
import math
import threading
from pynput.keyboard import Key, Listener

PI = 3.14159265359
data_size = 2**17

ophelp  = 'Options:\n'
ophelp += ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp += ' --port, -p <port>    TORCS port. [3001]\n'
ophelp += ' --id, -i <id>        ID for server. [SCR]\n'
ophelp += ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp += ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp += ' --track, -t <track>  Your name for this track. [unknown]\n'
ophelp += ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp += ' --debug, -d          Output full telemetry.\n'
ophelp += ' --help, -h           Show this help.\n'
ophelp += ' --version, -v        Show current version.'
usage   = 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage   = usage + ophelp
version = "il-dataset-2.3-aggressive-corkscrew"


def clip(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v


# =====================================================================
# NETWORKING / PROTOCOLLO SCR
# =====================================================================

class Client():
    def __init__(self, H=None, p=None, i=None, e=None, t=None, s=None, d=None, vision=False):
        self.vision = vision
        self.host = 'localhost'
        self.port = 3001
        self.sid = 'SCR'
        self.maxEpisodes = 1
        self.trackname = 'unknown'
        self.stage = 3
        self.debug = False
        self.maxSteps = 100000
        self.parse_the_command_line()
        if H: self.host = H
        if p: self.port = p
        if i: self.sid = i
        if e: self.maxEpisodes = e
        if t: self.trackname = t
        if s: self.stage = s
        if d: self.debug = d
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, a)
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except (socket.error, ConnectionResetError):
                print("In attesa che TORCS avvii la gara sulla porta %d..." % self.port)
                if n_fail < 0:
                    print("Riavvio TORCS (Windows mode)...")
                    os.system('taskkill /IM torcs.exe /F')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('start torcs -nofuel -nodamage -nolaptime')
                    else:
                        os.system('start torcs -nofuel -nodamage -nolaptime -vision')
                    time.sleep(1.0)
                    n_fail = 5
                n_fail -= 1
            if '***identified***' in sockdata:
                print("Client connected on %d." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(
                sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                ['host=', 'port=', 'id=', 'steps=',
                 'episodes=', 'track=', 'stage=',
                 'debug', 'help', 'version'])
        except getopt.error:
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] in ('-h', '--help'):
                    sys.exit(0)
                if opt[0] in ('-d', '--debug'):
                    self.debug = True
                if opt[0] in ('-p', '--port'):
                    self.port = int(opt[1])
                if opt[0] in ('-m', '--steps'):
                    self.maxSteps = int(opt[1])
                if opt[0] in ('-t', '--track'):
                    self.trackname = opt[1]
        except ValueError:
            sys.exit(-1)

    def get_servers_input(self):
        if not self.so: return
        sockdata = str()
        while True:
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error:
                pass
            if '***identified***' in sockdata:
                continue
            elif '***shutdown***' in sockdata:
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                self.shutdown()
                return
            elif not sockdata:
                continue
            else:
                self.S.parse_server_str(sockdata)
                break

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error:
            sys.exit(-1)

    def shutdown(self):
        if not self.so: return
        self.so.close()
        self.so = None


class ServerState():
    def __init__(self):
        self.d = dict()
    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])


class DriverAction():
    def __init__(self):
        self.d = {'accel': 0.2, 'brake': 0, 'clutch': 0, 'gear': 1,
                  'steer': 0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0}
    def clip_to_limits(self):
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'], 0, 1)
        self.d['accel'] = clip(self.d['accel'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not type(v) is list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out


def destringify(s):
    if not s: return s
    if type(s) is str:
        try: return float(s)
        except ValueError: return s
    elif type(s) is list:
        if len(s) < 2: return destringify(s[0])
        return [destringify(i) for i in s]


# =====================================================================
# OVERRIDE MANUALE DA TASTIERA
# I sample raccolti durante l'override NON vengono salvati nel CSV.
# =====================================================================

manual_steer = 0.0
manual_accel_override = None
manual_brake_override = None
is_manual_override = False

def on_press(key):
    global manual_steer, manual_accel_override, manual_brake_override, is_manual_override
    if key == Key.left:
        manual_steer = 0.6
        is_manual_override = True
    elif key == Key.right:
        manual_steer = -0.6
        is_manual_override = True
    elif key == Key.up:
        manual_accel_override = 1.0
        manual_brake_override = 0.0
        is_manual_override = True
    elif key == Key.down:
        manual_brake_override = 1.0
        manual_accel_override = 0.0
        is_manual_override = True

def on_release(key):
    global manual_steer, manual_accel_override, manual_brake_override, is_manual_override
    if key in (Key.left, Key.right):
        manual_steer = 0.0
    if key in (Key.up, Key.down):
        manual_accel_override = None
        manual_brake_override = None
    if manual_steer == 0.0 and manual_accel_override is None and manual_brake_override is None:
        is_manual_override = False

listener = Listener(on_press=on_press, on_release=on_release)
listener.start()


# =====================================================================
# LOGICA DI GUIDA
# =====================================================================

# ----------------------------------------------------------------------
# PARAMETRI STERZO
# ----------------------------------------------------------------------
# v1 vs v2 (telemetry-matched dai 6 dataset umani):
#   - STEER_K_E ridotto da 1.0 -> 0.65: il bot vecchio si "incollava" alla
#     mezzeria (|trackPos| mediano = 0.08). L'umano usa |trackPos| fino a 0.95
#     (mediana 0.49). Riducendo il guadagno del cross-track il bot accetta
#     di stare sui bordi durante la racing line.
#   - STEER_K_HEADING leggermente aumentato (1.3 -> 1.45) per tenere l'angolo
#     piu' rigorosamente: l'umano ha |angle| mediano basso (~0.07).
#   - STEER_K_LOOKAHEAD aumentato (0.55 -> 0.75) per iniettare piu' anticipo
#     in curva, evitando di arrivare in ritardo sull'inserimento.
# ----------------------------------------------------------------------
STEER_K_E              = 0.65   # era 1.0 — meno richiamo al centro pista
STEER_K_SOFT           = 5.0
STEER_K_HEADING        = 1.45   # era 1.3
STEER_K_LOOKAHEAD      = 0.75   # era 0.55 — piu' feedforward su curve viste
STEER_K_LOOKAHEAD_VSCALE = 0.006
ANGLE_FILTER_ALPHA     = 0.4
STEER_MAX_DELTA        = 0.10   # era 0.13 — l'umano ha |Δsteer| meta' del bot
STEER_RAD_TO_CMD       = 1.0

# Coefficiente del target trackPos out-in-out:
# target_trackpos = -TRACKPOS_OUTIN_GAIN * curvatura_anticipata
# Cap a +/- 0.65 perche' l'umano in entrata curva sta a |tp| ~ 0.7-0.8.
# Segno: curv > 0 (curva sinistra vista davanti) -> target_tp < 0 (auto sul
# lato destro della pista in entrata, OUT). Specchio per curva destra.
TRACKPOS_OUTIN_GAIN    = 1.8    # NUOVO (v2)
TRACKPOS_OUTIN_CAP     = 0.65   # NUOVO (v2)

# --- Debug ---
DEBUG_STEERING       = True
DEBUG_PRINT_EVERY    = 25

# ----------------------------------------------------------------------
# SPEED_MAP — riscritta sui dati umani.
# Coppie (front_distance_m, v_target_kmh).
# Estratte dai 6 dataset umani come p75 di speedX a parita' di
# max(track_3, track_4, track_9, track_14, track_15).
#
# Confronto v1 -> v2 a parita' di distanza vista:
#   200 m:  310  -> 295 km/h (era irrealistico, max raggiunto 296)
#   150 m:  250  -> 270 km/h
#   100 m:  195  -> 260 km/h  (+65 km/h: qui il bot perdeva TANTO tempo)
#    70 m:  155  -> 240 km/h  (+85 km/h !!)
#    50 m:  -    -> 215 km/h
#    45 m:  118  -> -        (sostituito da nodi piu' fini)
#    35 m:  -    -> 145 km/h
#    25 m:   85  -> 105 km/h
#    15 m:  -    ->  88 km/h
#     0 m:   55  ->  55 km/h  (= velocita' di emergenza, invariata)
# ----------------------------------------------------------------------
SPEED_MAP = [
    (200.0, 295.0),   # rettilineo lungo: v_max della macchina (umano max 296)
    (160.0, 280.0),   # quasi rettilineo (curva a vista molto larga)
    (130.0, 268.0),   # uscita curva veloce / entrata curva morbida
    (100.0, 260.0),   # vista media-lunga (es. curva 5, 'curvone')
    ( 80.0, 245.0),   # vista media (umano p75 ~ 252 in questa fascia)
    ( 60.0, 225.0),   # curva medio-veloce
    ( 50.0, 215.0),   # umano p75 = 217 a fascia (45,55]
    ( 40.0, 175.0),   # curva media (umano p75 = 181)
    ( 30.0, 128.0),   # curva stretta (umano p75 = 128)
    ( 20.0,  96.0),   # ingresso/uscita curva stretta
    ( 12.0,  85.0),   # apice curva (umano p50 ~ 78-84 in questa fascia)
    (  0.0,  55.0),   # emergenza / muro davanti
]

# Riduzione per curvatura.
# v1: threshold=0.08 era troppo bassa, riduceva persino in curvoni veloci.
# v2: threshold alzata a 0.13 (curvone veloce non viene piu' rallentato),
#     full_cut alzato a 0.55 per gestire curve strette (corkscrew),
#     riduzione massima alzata a 0.42 perche' l'umano in apex Corkscrew
#     scende a 76 km/h da 250+ km/h di rettilineo.
CURV_THRESHOLD     = 0.13       # era 0.08
CURV_FULL_CUT      = 0.55       # era 0.45
CURV_MAX_REDUCTION = 0.42       # era 0.35

# ----------------------------------------------------------------------
# OVERRIDE CORKSCREW (zona dist_from_start 2300-2500 m).
# Spline (distanza_m, v_target_kmh) estratta direttamente dal profilo medio
# umano dei 6 dataset, ogni 25 m. Quando dist_from_start cade nell'intervallo,
# si usa come TETTO SUPERIORE: target_speed = min(target_speed, override).
# In questo modo non si forza un'accelerazione se il bot va gia' piano,
# ma si garantisce che NON sbandi in discesa.
# ----------------------------------------------------------------------
CORKSCREW_SPEED_OVERRIDE = [
    (2280, 292.0),  # pre-frenata: in pieno gas
    (2305, 290.0),  # inizio lift-off umano
    (2330, 265.0),  # brake = 0.65, gia' molto in frenata
    (2355, 218.0),  # ancora in piena staccata
    (2380, 175.0),  # rilascio progressivo del freno
    (2405, 122.0),  # ingresso curva stretta
    (2430,  85.0),  # FIX v2.3: era 95 — piu' lento all'ingresso S
    (2455,  65.0),  # FIX v2.3: era 77 — apex piu' lento per evitare testacoda
    (2480,  70.0),  # FIX v2.3: era 85 — uscita piu' cauta
    (2505, 115.0),  # in pieno gas, marcia su
    (2530, 141.0),
    (2555, 165.0),
    (2580, 187.0),
    (2600, 210.0),  # uscita Corkscrew, rettilineo successivo
]
CORKSCREW_START = 2280.0
CORKSCREW_END   = 2600.0

# ----------------------------------------------------------------------
# CAMBIO MARCE — completamente riprogettato sui regimi umani.
#
# DATI estratti dai 6 dataset umani:
#   - mediana up-shift:  3->4 a 18435 rpm, 4->5 a 18190 rpm, 5->6 a 18656 rpm
#   - mediana down-shift: 4->3 a 15615 rpm, 5->4 a 17120 rpm, 6->5 a 17534 rpm
#   - p50 rpm dentro la marcia: Gear 3=15651, Gear 4=17017, Gear 5=17667, Gear 6=17687
#   - p5 di speedX per marcia (= velocita' minima sostenibile):
#       gear2=64, gear3=118, gear4=179, gear5=205, gear6=219
#
# Il bot vecchio cambiava a 9000 rpm: rimaneva in 6a marcia con rpm 14600 di
# mediana (mentre l'umano sta a 17600). Risultato: meno coppia, meno gas in
# uscita, meno velocita' di punta.
# ----------------------------------------------------------------------
RPM_UP   = 18000  # era 9000  — l'umano cambia a ~18500 rpm (poco sotto rev limit)
RPM_DOWN = 14000  # era 6500  — sotto questa soglia si scala
# GEAR_MIN_SPEED[g] = velocita' minima per restare in marcia g
# Valori derivati dal p5 di speedX per marcia (un po' sotto, per evitare scalate troppo nervose).
GEAR_MIN_SPEED = [0, 0, 50, 100, 150, 180, 200]   # era [0, 0, 35, 65, 100, 130, 175]

# ----------------------------------------------------------------------
# THROTTLE/BRAKE — parametri di trail braking e gas back.
# Tutti regolati per matchare il piu' possibile lo stile umano.
# ----------------------------------------------------------------------
# Deadband: l'umano modula con minore tolleranza, lo riduco da 8 a 5 km/h
BRAKE_DEADBAND_KMH     = 5.0    # era 8.0 (hardcoded nel codice vecchio)
# Gain di frenata su overspeed: vecchio 0.032, nuovo 0.045 -> brake 0.7 con
# overspeed 16 km/h (l'umano picco a 0.73 nelle curve dure).
BRAKE_GAIN_PER_KMH     = 0.045  # era 0.032
BRAKE_MAX              = 0.93   # cap massimo (invariato)

# --- TRAIL BRAKING (FIX v2.1 — curve in sequenza) ---------------------------
# Analisi crash sulla curva 480m con CSV dataset_track_1780842396:
#   17 step su 59 con |steer|>=0.5 E brake>=0.7 simultaneamente -> ruote
#   anteriori bloccate, niente grip laterale, understeer, fuori pista.
# L'umano modula: brake max 0.73 e cala sotto 0.5 appena entra in apex.
# Soluzione: cap dinamico del brake in funzione dello sterzo CORRENTE.
#   steer=0.20 -> brake_cap = 0.93 (zero penalita')
#   steer=0.50 -> brake_cap = 0.79 * 0.93 = 0.73 (come umano)
#   steer=0.80 -> brake_cap = 0.59 * 0.93 = 0.55
#   steer=1.00 -> brake_cap = 0.45 * 0.93 = 0.42 (sblocca le ruote)
# NB: la frenata "rettilinea" (sterzo basso) resta intatta al 93%.
TRAIL_BRAKE_STEER_DEAD = 0.20       # sotto questo sterzo nessun cap
TRAIL_BRAKE_K          = 0.55       # pendenza riduzione cap vs sterzo
TRAIL_BRAKE_MIN_CAP    = 0.45       # cap minimo assoluto
# Soglia di unwinding piu' bassa: l'umano riapre gas gia' all'apex.
UNWIND_EMA_THRESHOLD   = 0.15   # era 0.20
UNWIND_DROP_THRESHOLD  = 0.06   # era 0.08
UNWIND_ACCEL_TARGET    = 1.00   # era 0.95 — pieno gas in unwinding
# Lift-off in curva: meno aggressivo, l'umano non taglia tanto gas
LIFTOFF_STEER_THRESH   = 0.40   # era 0.35
LIFTOFF_MAX_REDUCTION  = 0.18   # era 0.25

# --- Quality gate ---
QUALITY_MAX_TRACKPOS = 0.92     # alzato (era 0.85): l'umano sta a 0.95 in entrata
QUALITY_MAX_ANGLE    = 0.35
QUALITY_MIN_SPEED    = 5.0
WARMUP_STEPS         = 50

_state = {
    'prev_steer': 0.0,
    'steer_ema_slow': 0.0,
    'filtered_angle': 0.0,
    'prev_gear': 1,
    'gear_change_cooldown': 0,
    'debug_step': 0,
    'last_target_tp': 0.0,           # NUOVO: per smoothing del target trackpos
    'curv_anticip_ema': 0.0,         # NUOVO: EMA della curvatura anticipata
}


def effective_front_distance(track):
    """
    Distanza utile davanti per decidere la velocita'.
    Uso il MAX tra i sensori a piccolo angolo per evitare che un singolo
    sensore "pescando" sul muro laterale strangoli il target di velocita'.
    """
    candidates = []
    for idx in (3, 4, 9, 14, 15):
        v = track[idx]
        if v >= 0:
            candidates.append(v)
    if not candidates:
        return 50.0
    return max(candidates)


def corkscrew_override_speed(dist_from_start):
    """
    Restituisce il TETTO di velocita' per la zona Corkscrew (2280-2600 m).
    Spline lineare estratta dal profilo medio umano (6 dataset).
    Fuori range: ritorna None (= nessun override, usa SPEED_MAP normale).

    Nota: questo override viene applicato come MIN con la velocita' target,
    quindi non puo' MAI rendere il bot piu' veloce della logica base; serve
    solo a evitare sbandate in discesa quando i sensori track potrebbero
    dare letture ottimistiche.
    """
    if dist_from_start < CORKSCREW_START or dist_from_start > CORKSCREW_END:
        return None
    pts = CORKSCREW_SPEED_OVERRIDE
    # interpolazione lineare
    if dist_from_start <= pts[0][0]:
        return pts[0][1]
    if dist_from_start >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        d_lo, v_lo = pts[i]
        d_hi, v_hi = pts[i + 1]
        if d_lo <= dist_from_start <= d_hi:
            t = (dist_from_start - d_lo) / (d_hi - d_lo)
            return v_lo + t * (v_hi - v_lo)
    return None


def lookup_target_speed(track, S=None):
    front_dist = effective_front_distance(track)
    pts = SPEED_MAP

    if front_dist >= pts[0][0]:
        base_speed = pts[0][1]
    elif front_dist <= pts[-1][0]:
        base_speed = pts[-1][1]
    else:
        base_speed = pts[-1][1]
        for i in range(len(pts) - 1):
            d_hi, v_hi = pts[i]
            d_lo, v_lo = pts[i + 1]
            if d_lo <= front_dist <= d_hi:
                t = (front_dist - d_lo) / (d_hi - d_lo)
                base_speed = v_lo + t * (v_hi - v_lo)
                break

    curv = abs(estimate_curvature(track))
    if curv > CURV_THRESHOLD:
        t = clip((curv - CURV_THRESHOLD) / (CURV_FULL_CUT - CURV_THRESHOLD), 0.0, 1.0)
        reduction = CURV_MAX_REDUCTION * t
        base_speed *= (1.0 - reduction)

    # === FIX v2.3 (1) — FRENATA RITARDATA per recuperare aggressivita' ======
    # Decel teorica 5.0 -> 7.5 m/s^2 (auto frena meglio di cosi') e buffer
    # 30 -> 15 m. Il bot stacca molto piu' tardi sui rettilinei.
    if S is not None:
        speedX_kmh = S.get('speedX', 0.0)
        speed_ms = max(speedX_kmh / 3.6, 0.1)
        spazio_frenata = (speed_ms ** 2) / (2 * 7.5)   # era /(2*5.0)
        if front_dist <= spazio_frenata + 15.0:        # era +30.0
            v_apex_stimata = pts[-1][1]
            margine = spazio_frenata + 15.0
            if margine > 1.0:
                t = clip(front_dist / margine, 0.0, 1.0)
                v_cap = v_apex_stimata + t * (base_speed - v_apex_stimata)
                base_speed = min(base_speed, v_cap)
    # =========================================================================

    if S is not None:
        dist_from_start = S.get('distFromStart', None)
        if dist_from_start is not None:
            cs = corkscrew_override_speed(dist_from_start)
            if cs is not None:
                base_speed = min(base_speed, cs)

    return base_speed


def estimate_curvature(track):
    """
    Stima la curvatura imminente confrontando i sensori track laterali lontani.
    Sensori SCR (default angles): -45 -19 -12 -7 -4 -2.5 -1.7 -1 -0.5 0 0.5 1 1.7 2.5 4 7 12 19 45
    indici:                          0   1   2  3  4   5    6   7   8  9 10 11  12  13 14 15 16 17 18
    Convenzione: ritorno > 0 => sterzo POSITIVO richiesto (sinistra).
    """
    pairs = [
        ( 0, 18, 0.40),
        ( 1, 17, 0.35),
        ( 2, 16, 0.15),
        ( 3, 15, 0.07),
        ( 4, 14, 0.03),
    ]
    total = 0.0
    weight_sum = 0.0
    for i_left, i_right, w in pairs:
        l = track[i_left]
        r = track[i_right]
        if l < 0 or r < 0:
            continue
        if l > 195 and r > 195:
            continue
        denom = max(l + r, 1.0)
        total += w * (l - r) / denom
        weight_sum += w

    if weight_sum < 1e-6:
        return 0.0
    return total / weight_sum


def compute_target_trackpos(curvature_anticip, speed_kmh):
    """
    Calcola il target_trackpos out-in-out basato sulla curvatura ANTICIPATA.

    Logica (derivata dall'analisi delle racing line umane):
      - curv > 0 (vista pista che gira a sinistra) -> in entrata serve stare
        OUT, cioe' sul lato DESTRO della pista. trackPos > 0 = auto sulla
        sinistra della pista, quindi target_tp < 0.
      - curv < 0 (curva a destra) -> sto sul lato sinistro -> target_tp > 0.
      - In rettilineo (curv ~ 0) -> target_tp = 0 (mezzeria).

    A velocita' basse (< 90 km/h, tipico apex stretto) riduco l'ampiezza:
    in apice l'umano sta gia' all'interno e non serve forzare l'out.
    """
    # Smoothing EMA della curvatura per evitare oscillazioni rapide.
    # FIX v2.1: alpha portato da 0.30 a 0.45 (EMA piu' veloce). Sul rettilineo
    # tra curva 200m e curva 480m il vecchio EMA restava ancorato allo stato
    # post-curva precedente e non vedeva la nuova in tempo per spostare il bot
    # sul lato esterno. Reattivita' aumentata di ~50% senza diventare nervoso.
    ema = 0.55 * _state['curv_anticip_ema'] + 0.45 * curvature_anticip
    _state['curv_anticip_ema'] = ema

    # Scala l'ampiezza dell'OUT in base alla velocita': a bassa velocita'
    # siamo gia' dentro la curva, l'umano non insiste sull'OUT.
    speed_scale = clip((speed_kmh - 80.0) / 120.0, 0.0, 1.0)  # 0 a 80km/h, 1 a 200+
    target = -TRACKPOS_OUTIN_GAIN * ema * speed_scale
    target = clip(target, -TRACKPOS_OUTIN_CAP, TRACKPOS_OUTIN_CAP)

    # Slew rate sul target stesso (evita salti netti del setpoint).
    # FIX v2.1: max_dtp 0.04 -> 0.06: a 50Hz consente di spostare il target
    # di 3.0 al secondo (era 2.0). Sul rettilineo di 280m a 240 km/h il bot
    # ha 4.2s di tempo, quindi puo' raggiungere +0.6 in <0.4s. Margine ampio.
    prev = _state['last_target_tp']
    max_dtp = 0.06   # era 0.04
    if target > prev + max_dtp:
        target = prev + max_dtp
    elif target < prev - max_dtp:
        target = prev - max_dtp
    _state['last_target_tp'] = target
    return target


def calculate_steering(S):
    """
    Controller di sterzo a quattro componenti:
      - heading feedback (filtrato)
      - cross-track Stanley su (track_pos - target_track_pos)   <-- NUOVO v2
      - lookahead feedforward su curvatura
      - slew rate limit dipendente dalla velocita'

    In modalita' recovery (|angle|>0.5 rad o |trackPos|>0.9) disabilita
    il feedforward e azzera il target_trackpos (priorita' al rientro).
    """
    angle      = S.get('angle', 0.0)
    track_pos  = S.get('trackPos', 0.0)
    track      = S.get('track', [200.0] * 19)
    speedX_kmh = S.get('speedX', 0.0)

    # === FIX v2.2 (3) — SOGLIE ANOMALIA RIALZATE ===
    # Solo testacoda veri (>~70°) o uscita quasi totale di pista attivano
    # il recovery, NON le curve strette regolari.
    is_anomalous = abs(angle) > 1.2 or abs(track_pos) > 0.99

    af = (1.0 - ANGLE_FILTER_ALPHA) * angle + ANGLE_FILTER_ALPHA * _state['filtered_angle']
    _state['filtered_angle'] = af

    k_h = STEER_K_HEADING * (1.8 if is_anomalous else 1.0)
    heading_term = k_h * af

    if is_anomalous:
        curvature = 0.0
        target_tp = 0.0
    else:
        curvature = estimate_curvature(track)
        # Target trackpos out-in-out (NUOVO v2)
        target_tp = compute_target_trackpos(curvature, speedX_kmh)

    # Stanley cross-track: ora rispetto a (track_pos - target_tp)
    # Cosi' il bot accetta di stare OUT prima di curva, in apex, ecc.
    cross_err = track_pos - target_tp
    speed_ms = max(speedX_kmh / 3.6, 0.1)
    cross_track_term = -math.atan2(STEER_K_E * cross_err, STEER_K_SOFT + speed_ms)

    k_lookahead_eff = STEER_K_LOOKAHEAD + STEER_K_LOOKAHEAD_VSCALE * max(0.0, speedX_kmh)
    lookahead_term = k_lookahead_eff * curvature

    raw_steer = heading_term + cross_track_term + lookahead_term
    target_steer = clip(raw_steer * STEER_RAD_TO_CMD, -1.0, 1.0)

    # Slew rate limit scalato sulla velocita' — piu' restrittivo del v1
    # (umano |Δsteer| per step e' meta' di quello del bot vecchio).
    if speedX_kmh < 60:
        max_delta = 0.15            # era 0.18
    elif speedX_kmh < 100:
        max_delta = 0.10            # era 0.12
    elif speedX_kmh < 140:
        max_delta = 0.07            # era 0.08
    else:
        max_delta = 0.04            # era 0.05
    delta = target_steer - _state['prev_steer']
    if delta >  max_delta: delta =  max_delta
    if delta < -max_delta: delta = -max_delta
    final_steer = _state['prev_steer'] + delta

    # Cap dello sterzo ad alta velocita': umano in apex usa |steer| ~ 0.4-0.6
    if speedX_kmh > 180.0:
        final_steer = clip(final_steer, -0.55, 0.55)   # era -0.6, 0.6

    final_steer = clip(final_steer, -1.0, 1.0)
    _state['steer_ema_slow'] = 0.90 * _state['steer_ema_slow'] + 0.10 * final_steer
    _state['prev_steer'] = final_steer

    if DEBUG_STEERING:
        _state['debug_step'] += 1
        if _state['debug_step'] % DEBUG_PRINT_EVERY == 0:
            front_eff = effective_front_distance(track)
            tgt_v = lookup_target_speed(track, S)
            dfs = S.get('distFromStart', 0.0)
            print(
                f"[ctrl] v={speedX_kmh:6.1f} tgt={tgt_v:6.1f} d={tgt_v-speedX_kmh:+5.1f} "
                f"rpm={S.get('rpm',0):5.0f} gear={int(S.get('gear',0))} "
                f"front={front_eff:5.1f} ang={angle:+.2f} tp={track_pos:+.2f} "
                f"tp_tgt={target_tp:+.2f} dfs={dfs:6.0f} "
                f"curv={curvature:+.2f} steer={final_steer:+.2f}"
            )

    return final_steer


def calculate_throttle_and_brake(S, target_speed):
    """
    Throttle/brake con deadband ridotta, trail braking piu' deciso e
    riaccelerazione fortemente anticipata (stile umano).

    Modifiche v2 vs v1:
      - BRAKE_DEADBAND ridotto da 8 a 5 km/h (l'umano modula prima).
      - Brake gain alzato da 0.032 a 0.045 (umano picco a 0.7 con overspeed
        modesto; bot v1 non arrivava mai a brake forti perche' il gain era
        basso).
      - Soglia di unwinding ridotta (0.20->0.15, 0.08->0.06): il gas torna
        pieno appena lo sterzo si sta riaprendo, non prima.
      - Lift-off in curva ridotto: massimo 18% di riduzione (era 25%).
      - In modalita' anomala uso brake piu' deciso (0.20 era 0.15) per
        recovery piu' rapido.
    """
    speedX = S.get('speedX', 0.0)
    angle = S.get('angle', 0.0)
    track_pos = S.get('trackPos', 0.0)

    # === FIX v2.2 (3) — SOGLIE ANOMALIA RIALZATE =============================
    # Le vecchie soglie (|angle|>0.5 o |tp|>0.9) facevano scattare il "panico"
    # ogni volta che il bot affrontava una curva stretta. Ora solo cambi di
    # assetto veri (testacoda, fuoripista quasi totale) attivano il recovery.
    # In modalita' anomala uso brake LEGGERO (0.15) per non bloccare le ruote.
    is_anomalous = abs(angle) > 1.2 or abs(track_pos) > 0.99
    if is_anomalous:
        if speedX > 30:
            return (0.0, 0.15)   # FIX: era 0.20 — niente freno forte in panico
        return (0.0, 0.0)
    # =========================================================================

    delta = target_speed - speedX
    BRAKE_DEADBAND = BRAKE_DEADBAND_KMH

    # Unwinding: sterzo attuale piu' vicino a zero della EMA lenta
    prev = _state['prev_steer']
    ema = _state['steer_ema_slow']
    is_unwinding = (abs(ema) > UNWIND_EMA_THRESHOLD) and \
                   (abs(prev) < abs(ema) - UNWIND_DROP_THRESHOLD)

    if delta > -BRAKE_DEADBAND:
        if delta > 0:
            accel = clip(0.75 + 0.07 * delta, 0.0, 1.0)
        else:
            if is_unwinding:
                accel = UNWIND_ACCEL_TARGET
            else:
                accel = clip(0.75 + 0.09 * delta, 0.35, 0.75)
        brake = 0.0
    else:
        overspeed = -delta - BRAKE_DEADBAND
        # === FIX v2.2 (2) — BRAKE CLIFF PROGRESSIVO ===========================
        # Curva di frenata smussata: salita ripida a brake_max per overspeed
        # alti (>30 km/h) ma RILASCIO PROGRESSIVO per overspeed bassi, per
        # mantenere il trail-braking in inserimento curva senza crolli a 0.
        # tanh ha la forma giusta: ~lineare per piccoli x, satura a 1 per
        # grandi x. La pendenza la imposta BRAKE_GAIN_PER_KMH.
        if overspeed > 30.0:
            brake = BRAKE_MAX
        else:
            # transizione fluida: a overspeed=30 -> ~BRAKE_MAX, a 0 -> 0
            brake = BRAKE_MAX * math.tanh(BRAKE_GAIN_PER_KMH * overspeed * 1.2)
            brake = clip(brake, 0.0, BRAKE_MAX)
        # Trail braking: gas piccolo solo se freno e' davvero leggero
        if brake < 0.3:
            accel = 0.22 * (1.0 - brake / 0.3)
        else:
            accel = 0.0
        # =====================================================================

    # === FIX v2.1: TRAIL BRAKING ANTI-LOCK ===================================
    # Il bot v2 frenava al 93% anche con sterzo saturo, bloccando le anteriori.
    # Qui leggo lo sterzo CORRENTE (calculate_steering ha gia' aggiornato
    # _state['prev_steer'] prima di me — vedi ordine in drive(c)) e riduco
    # il cap del brake. Questo NON disattiva la frenata: cala solo quando
    # lo sterzo e' >0.20, e ha un floor a 0.45*BRAKE_MAX=0.42 quindi
    # l'aggressivita' resta intatta. La frenata in rettilineo non e' toccata.
    if brake > 0.0:
        steer_for_trail = abs(_state['prev_steer'])
        if steer_for_trail > TRAIL_BRAKE_STEER_DEAD:
            excess = (steer_for_trail - TRAIL_BRAKE_STEER_DEAD) / (1.0 - TRAIL_BRAKE_STEER_DEAD)
            brake_cap_ratio = max(TRAIL_BRAKE_MIN_CAP, 1.0 - TRAIL_BRAKE_K * excess)
            brake = min(brake, BRAKE_MAX * brake_cap_ratio)
    # =========================================================================

    # Lift-off in curva: solo a sterzo > 0.40, max -18%
    steer_abs = abs(prev)
    if steer_abs > LIFTOFF_STEER_THRESH and accel > 0:
        # interpolazione: 0 a steer=0.40, LIFTOFF_MAX_REDUCTION a steer=1.0
        liftoff_t = (steer_abs - LIFTOFF_STEER_THRESH) / (1.0 - LIFTOFF_STEER_THRESH)
        accel *= (1.0 - LIFTOFF_MAX_REDUCTION * clip(liftoff_t, 0.0, 1.0))
        accel = clip(accel, 0.0, 1.0)

    # === FIX v2.3 (2) — CORKSCREW THROTTLE CAP (anti-testacoda) ============
    # Nel dislivello cieco tra 2440-2485m dare gas causa testacoda perche'
    # la macchina cambia direzione e perde aderenza. Cap a 0.10.
    dfs = S.get('distFromStart', 0.0)
    if 2440.0 <= dfs <= 2485.0:
        accel = min(accel, 0.10)
    # =======================================================================

    return accel, brake


def traction_control(S, accel):
    """Riduce gas se le ruote posteriori girano molto piu' delle anteriori."""
    wsv = S.get('wheelSpinVel', None)
    if not wsv or len(wsv) != 4:
        return accel
    front = wsv[0] + wsv[1]
    rear  = wsv[2] + wsv[3]
    spin = rear - front
    if spin > 5.0:
        accel = clip(accel - 0.20, 0.0, 1.0)
    elif spin > 3.0:
        accel = clip(accel - 0.10, 0.0, 1.0)
    return accel


def shift_gears(S):
    """
    Cambio basato su RPM, con guard sulla velocita' minima per marcia
    e cooldown.

    Aggiornamento v2:
      - RPM_UP/DOWN spostati ai regimi umani (18000/14000 rpm)
      - GEAR_MIN_SPEED riallineata: l'umano sta in 6a marcia gia' a 219 km/h.
      - Cooldown 6 step (= ~0.12 s a 50 Hz) invariato.
    """
    speedX = S.get('speedX', 0.0)
    rpm    = S.get('rpm', 0.0)

    if _state['gear_change_cooldown'] > 0:
        _state['gear_change_cooldown'] -= 1
        return _state['prev_gear']

    if speedX < -2:
        gear = -1
    else:
        gear = _state['prev_gear']
        if gear < 1:
            gear = 1
        # Up-shift: rpm alti e velocita' sufficiente per la marcia superiore.
        # Tolleranza di 10 km/h sotto la GEAR_MIN_SPEED della marcia target
        # per anticipare un po' il cambio in fase di accelerazione forte.
        if gear < 6 and rpm > RPM_UP and speedX > GEAR_MIN_SPEED[gear + 1] - 10:
            gear += 1
            _state['gear_change_cooldown'] = 6
        # Down-shift: rpm sotto la soglia OPPURE velocita' sotto il minimo
        # della marcia attuale meno una piccola tolleranza.
        elif gear > 1 and (rpm < RPM_DOWN or speedX < GEAR_MIN_SPEED[gear] - 15):
            gear -= 1
            _state['gear_change_cooldown'] = 6

    _state['prev_gear'] = gear
    return gear


def drive(c):
    """Funzione principale di guida. Modifica c.R.d in-place."""
    S = c.S.d
    R = c.R.d

    track = S.get('track', [200.0] * 19)
    # Passo S a lookup_target_speed cosi' usa distFromStart per l'override Corkscrew
    target_speed = lookup_target_speed(track, S)

    R['steer'] = calculate_steering(S)
    accel, brake = calculate_throttle_and_brake(S, target_speed)
    accel = traction_control(S, accel)
    R['accel'] = accel
    R['brake'] = brake
    R['gear']  = shift_gears(S)

    if is_manual_override:
        if manual_steer != 0.0:
            R['steer'] = manual_steer
        if manual_accel_override is not None:
            R['accel'] = manual_accel_override
        if manual_brake_override is not None:
            R['brake'] = manual_brake_override


# =====================================================================
# QUALITY GATE: decide se il sample va salvato nel CSV.
# =====================================================================

def is_sample_clean(S, step_index):
    """Ritorna True se il sample e' adatto al training."""
    if step_index < WARMUP_STEPS:
        return False
    if is_manual_override:
        return False

    speedX    = S.get('speedX', 0.0)
    track_pos = S.get('trackPos', 0.0)
    angle     = S.get('angle', 0.0)
    track     = S.get('track', [200.0] * 19)

    if speedX < QUALITY_MIN_SPEED:
        return False
    if abs(track_pos) > QUALITY_MAX_TRACKPOS:
        return False
    if abs(angle) > QUALITY_MAX_ANGLE:
        return False
    if min(track) < 0:
        return False
    return True


# =====================================================================
# MAIN LOOP
# =====================================================================

if __name__ == "__main__":
    C = Client(p=3001)
    track_name = C.trackname if C.trackname != 'unknown' else 'track'
    csv_filename = f'dataset_{track_name}_{int(time.time())}.csv'

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)

        headers = [
            'step', 'cur_lap_time', 'dist_from_start', 'dist_raced',
            'speedX', 'speedY', 'speedZ',
            'angle', 'trackPos',
            'rpm', 'gear_in',
            'steer', 'accel', 'brake', 'gear_out',
            'is_clean',
        ]
        for i in range(19):
            headers.append(f'track_{i}')
        for i in range(4):
            headers.append(f'wheelSpinVel_{i}')
        for i in range(36):
            headers.append(f'opponents_{i}')
        writer.writerow(headers)

        print("=" * 60)
        print(" TORCS bot v2.1 — TRAIL-BRAKE FIX (target 1:11)")
        print(f"   RPM_UP={RPM_UP}, RPM_DOWN={RPM_DOWN}")
        print(f"   GEAR_MIN_SPEED={GEAR_MIN_SPEED}")
        print(f"   SPEED_MAP top={SPEED_MAP[0][1]:.0f} km/h, low={SPEED_MAP[-1][1]:.0f} km/h")
        print(f"   CURV thresh={CURV_THRESHOLD}, full_cut={CURV_FULL_CUT}, max_red={CURV_MAX_REDUCTION}")
        print(f"   STEER K_E={STEER_K_E}, K_HEAD={STEER_K_HEADING}, K_LOOK={STEER_K_LOOKAHEAD}")
        print(f"   TRACKPOS_OUTIN_GAIN={TRACKPOS_OUTIN_GAIN}, CAP={TRACKPOS_OUTIN_CAP}")
        print(f"   BRAKE deadband={BRAKE_DEADBAND_KMH} km/h, gain={BRAKE_GAIN_PER_KMH}, max={BRAKE_MAX}")
        print(f"   TRAIL_BRAKE: dead={TRAIL_BRAKE_STEER_DEAD}, K={TRAIL_BRAKE_K}, min_cap={TRAIL_BRAKE_MIN_CAP}")
        print(f"   CORKSCREW override [{CORKSCREW_START:.0f}-{CORKSCREW_END:.0f}] m attivo")
        print(f" Output: {csv_filename}")
        print(" Tasti: frecce sx/dx = nudge sterzo, su/giu = gas/freno")
        print(" (i sample durante l'override manuale NON vengono salvati)")
        print("=" * 60)

        clean_count = 0
        total_count = 0

        for step in range(C.maxSteps, 0, -1):
            C.get_servers_input()
            drive(C)

            S = C.S.d
            R = C.R.d
            step_index = C.maxSteps - step

            clean = is_sample_clean(S, step_index)
            total_count += 1
            if clean:
                clean_count += 1

            track_sensors = S.get('track', [0.0] * 19)
            wsv = S.get('wheelSpinVel', [0.0] * 4)
            opp = S.get('opponents', [200.0] * 36)
            if len(track_sensors) < 19: track_sensors = list(track_sensors) + [0.0] * (19 - len(track_sensors))
            if len(wsv) < 4:            wsv = list(wsv) + [0.0] * (4 - len(wsv))
            if len(opp) < 36:           opp = list(opp) + [200.0] * (36 - len(opp))

            row = [
                step_index,
                S.get('curLapTime', 0.0),
                S.get('distFromStart', 0.0),
                S.get('distRaced', 0.0),
                S.get('speedX', 0.0),
                S.get('speedY', 0.0),
                S.get('speedZ', 0.0),
                S.get('angle', 0.0),
                S.get('trackPos', 0.0),
                S.get('rpm', 0.0),
                S.get('gear', 0),
                R['steer'],
                R['accel'],
                R['brake'],
                R['gear'],
                int(clean),
            ]
            row.extend(track_sensors[:19])
            row.extend(wsv[:4])
            row.extend(opp[:36])
            writer.writerow(row)

            C.respond_to_server()

            if step_index % 1000 == 0 and step_index > 0:
                ratio = clean_count / max(1, total_count) * 100
                print(f"[step {step_index}] sample puliti: {clean_count}/{total_count} ({ratio:.1f}%)")

        print(f"\nFatto. Sample totali: {total_count}, puliti: {clean_count}")
        print(f"In addestramento usa SOLO le righe con is_clean=1.")

    C.shutdown()