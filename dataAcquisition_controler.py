"""
This script implements the data acquisition layer used in TMS-EEG experiment.
"""
import random
import time
from collections import deque
import keyboard
import magicpy as mp
import pandas as pd
import pyautogui
import serial
from Components.update_navigation import UpdateNavigationInfo
from Components import constants as consts
from modules.dataAcquisition_functions import zero_decision, sequence_generator, send_trigger_to_esp32

####
# EXPERIMENT SETTINGS
####
mode = "intensity"  # protocol

# history of last 10 target_status values
target_status_history = deque([False] * 10, maxlen=10)

####
# Volunteer's parameters
####
rmt_intensity = 36  # Resting motor threshold of the subject

# For intensity mode:
intensity_0 = int(0.8 * rmt_intensity)
intensity_1 = int(rmt_intensity)
intensity_2 = int(1.2 * rmt_intensity)

intensities = {
    0: "intensity_0",  # 80% rMT
    1: "intensity_1",  # 100% rMT
    2: "intensity_2",  # 120% rMT
}

# -----------------------------
# SEQUENCE GENERATION
# -----------------------------
values = [0, 1, 2]
sequence = [random.choice(values)]
number_of_stimuli = 400  # ajuste se quiser outro número

# inicialização idêntica ao script original
if sequence[0] == 0:
    zero_decision()
elif sequence[0] == 1:
    sequence.append(1)
elif sequence[0] == 2:
    sequence.append(1)

for _ in range(number_of_stimuli - 2):
    sequence_generator()

print("Sequência gerada:", sequence)

# opcional: checar distribuição
contagem = pd.Series(sequence).value_counts().sort_index()
print("Contagem por tipo:", contagem)
# se quiser ainda salvar:
# np.savetxt(
#     'tree_sequence_' + str(number_of_stimuli) + '_LMS.txt',
#     sequence,
#     delimiter=',',
#     fmt='%d'
# )

'''
CONNECTION TO NAVIGATION UPDATES
'''
updator = UpdateNavigationInfo(1)
updator.connect('127.0.0.1', [5000])

'''
CONNECTION WITH MAGVENTURE
'''
mp.list_serial_ports()
input(
    "Portas listadas. Certifique-se de que está conectando na porta correta. "
    "Pressione Enter para continuar..."
)
stimulator = mp.MagVenture("COM1")
stimulator.connect()
stimulator.set_page('Main', get_response=False)

'''
CONNECTION WITH ARDUINO / ESP32
'''
PORTA_SERIAL = 'COM6'      # ajuste se necessário
BAUD_RATE = 115200
ser = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=1)
time.sleep(2)  # aguarda ESP32 reiniciar

'''
PRESS S TO START EXPERIMENT WHEN EVERYTHING IS SET
'''
print("Aperte a tecla s para iniciar os pulsos...")
while True:
    if keyboard.is_pressed('s'):
        break

print("start sequence")
pulse_index = 0
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

paused = False  # flag de pausa

while True:
    # controla pausa/retomada
    if keyboard.is_pressed('b'):
        paused = True
        print("Sequência pausada (tecla B).")
    if keyboard.is_pressed('g'):
        if paused:
            paused = False
            print("Sequência retomada (tecla G).")

    # se estiver pausado, não avança pulsos
    if paused:
        time.sleep(0.05)
        continue

    # atualiza o histórico com o estado atual do alvo
    current_status = updator.target_status  # bool vindo do rastreador
    target_status_history.append(current_status)

    # só dispara se TODOS os últimos 10 estados forem True
    if all(target_status_history):
        try:
            # ---- intensity mode only ----
            estimulo = sequence[pulse_index]          # 0, 1 ou 2
            intensidade = intensities[estimulo]
            print(intensidade)

            stimulator.set_mode(
                mode='Standard',
                current_dir='Normal',
                n_pulses_per_burst=2,
                ipi=5,
                baratio=80,
            )
            time.sleep(1)
            stimulator.arm(get_response=False)
            time.sleep(1)

            if intensidade == "intensity_0":
                stimulator.set_amplitude(intensity_0, b_amp=None)
            elif intensidade == "intensity_1":
                stimulator.set_amplitude(intensity_1, b_amp=None)
            elif intensidade == "intensity_2":
                stimulator.set_amplitude(intensity_2, b_amp=None)
            time.sleep(1)

            with updator.status_lock:
                # 1) dispara o TMS
                stimulator.fire()

                # 2) dispara trigger para ESP32 (0->1, 1->2, 2->3)
                trigger_value = estimulo + 1
                send_trigger_to_esp32(trigger_value)

                # 3) opcional: trigger para navegação
                if consts.create_navigation_marker:
                    updator.send_trigger_to_navigation()

            print("disparando")
            time.sleep(random.uniform(consts.ITI[0], consts.ITI[1]))

            print("Index do pulso", pulse_index + 1)
            pulse_index += 1  # Só incrementa se tudo deu certo

        except Exception as e:
            print(f"\n⚠️ ERRO no pulso {pulse_index}: {e}")
            print("Tentando novamente o mesmo pulso...\n")
            time.sleep(2)  # pequena pausa antes de tentar de novo

    # fim definitivo da sequência (por tamanho da sequência)
    if pulse_index >= len(sequence):
        print("Encerrando sequência (fim da sequência de pulsos).")
        break

    time.sleep(0.01)

# fecha a serial ao final do experimento
try:
    ser.close()
except Exception:
    pass
