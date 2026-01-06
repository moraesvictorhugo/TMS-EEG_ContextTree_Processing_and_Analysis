# """
# This module implements the data acquisition layer used in TMS-EEG experiment.
# """
# import pyautogui
# import keyboard
# import numpy as np
# import random
# import time
# import magicpy as mp
# from Components.arduino_connection import ArduinoConnection
# from Components.update_navigation import UpdateNavigationInfo
# from Components import constants as consts, txt_configs as txt

# ####
# # EXPERIMENT SETTINGS
# ####
# mode = "intensity" #either pp or intensity
# target_status = True
# debug_arduino_ppTMS = False # True if you want just to check arduino connection

# 'For intensity mode:'
# intensity_0 = int(0.8 * rmt_intensity)
# intensity_1 = int(rmt_intensity)
# intensity_2 = int(1.2 * rmt_intensity)

# intensities = {0: "intensity_0", # 80% rMT
#                1: "intensity_1", # 100% rMT
#                2: "intensity_2"} # 120% rMT

# ####
# # Volunteer's parameters
# ####
# rmt_intensity = 36 # Resting motor treshold of the subject

# # 'For pp mode:'
# # ISI_inib = 2.5  # Inter Stimuli Interval [ms]. Ex: ISI=10 refers to 10 ms.
# # ISI_exci = 8

# # pp_mode = {0: "inibitorio",
# #            1: "single",
# #            2: "excitatorio"}

# # ''' LOAD PULSE TREE CONTEXT SEQUENCE '''
# # sequence = np.loadtxt('random_sequence_LMS.txt', delimiter=',', dtype='int')
# # print(sequence)

# ''' CONNECTION TO NAVIGATION UPDATES '''
# updator = UpdateNavigationInfo(1)
# updator.connect('127.0.0.1', [5000])

# ''' CONNECTION WITH MAGVENTURE'''
# mp.list_serial_ports()                                              # imprime uma lista de portas disponíveis
# input("Portas listadas. Certifique-se de que está conectando na porta correta. Pressione Enter para continuar...")  # Aguarda o usuário clicar Enter para continuar
# stimulator = mp.MagVenture("COM1")                                  # Inicializa um objeto que se relaciona ao estimulador
# stimulator.connect()
# stimulator.set_page('Main', get_response=False)

# ''' PRESS S TO START EXPERIMENT WHEN EVERYTHING IS SET'''
# print("Aperte a tecla s para iniciar os pulsos...")
# while True:
#     if keyboard.is_pressed('s'):
#         break

# print("start sequence")
# pulse_index = 0
# pyautogui.PAUSE = 0
# pyautogui.FAILSAFE = False

# while True:
#     if updator.target_status:
#         try:
#             if mode == "pp":
#                 start = time.time()
#                 tipo_estimulo = pp_mode[sequence[pulse_index]]
#                 print(tipo_estimulo)

#                 if tipo_estimulo == "single":
#                     stimulator.set_mode(mode='Standard', current_dir='Normal', n_pulses_per_burst=2, ipi=5, baratio=80)
#                     time.sleep(1)
#                     stimulator.arm(get_response=False)
#                     time.sleep(1)
#                     stimulator.set_amplitude(int(1.2 * rmt_intensity), b_amp=None)
#                     time.sleep(1)

#                 elif tipo_estimulo == "inibitorio":
#                     stimulator.set_mode(mode='Dual', current_dir='Normal', n_pulses_per_burst=2, ipi=ISI_inib, baratio=80)
#                     time.sleep(1)
#                     stimulator.arm(get_response=False)
#                     time.sleep(1)
#                     stimulator.set_amplitude(int(0.8 * rmt_intensity), b_amp=int(1.2 * rmt_intensity), get_response=False)
#                     time.sleep(1)

#                 elif tipo_estimulo == "excitatorio":
#                     stimulator.set_mode(mode='Dual', current_dir='Normal', n_pulses_per_burst=2, ipi=ISI_exci, baratio=80)
#                     time.sleep(1)
#                     stimulator.arm(get_response=False)
#                     time.sleep(1)
#                     stimulator.set_amplitude(int(0.8 * rmt_intensity), b_amp=int(1.2 * rmt_intensity), get_response=False)
#                     time.sleep(1)

#                 while not updator.target_status[0]:
#                     time.sleep(0.01)

#                 with updator.status_lock:
#                     stimulator.fire()
#                     if consts.create_navigation_marker:
#                         updator.send_trigger_to_navigation()

#             elif mode == "intensity":
#                 intensidade = intensities[sequence[pulse_index]]
#                 print(intensidade)

#                 stimulator.set_mode(mode='Standard', current_dir='Normal', n_pulses_per_burst=2, ipi=5, baratio=80)
#                 time.sleep(1)
#                 stimulator.arm(get_response=False)
#                 time.sleep(1)

#                 if intensidade == "intensity_0":
#                     stimulator.set_amplitude(intensity_0, b_amp=None)
#                 elif intensidade == "intensity_1":
#                     stimulator.set_amplitude(intensity_1, b_amp=None)
#                 elif intensidade == "intensity_2":
#                     stimulator.set_amplitude(intensity_2, b_amp=None)
#                 time.sleep(1)

#                 with updator.status_lock:
#                     stimulator.fire()
#                     if consts.create_navigation_marker:
#                         updator.send_trigger_to_navigation()

#             print("disparando")
#             time.sleep(random.uniform(consts.ITI[0], consts.ITI[1]))

#             print("Index do pulso", pulse_index+1)
#             pulse_index += 1  # Só incrementa se tudo deu certo

#         except Exception as e:
#             print(f"\n⚠️ ERRO no pulso {pulse_index}: {e}")
#             print("Tentando novamente o mesmo pulso...\n")
#             time.sleep(2)  # pequena pausa antes de tentar de novo

#     # Tecla B ou fim da sequência
#     if keyboard.is_pressed('b') or pulse_index >= len(sequence):
#         print("Encerrando sequência.")
#         break

#     time.sleep(0.01)

######################################### Version 2 #########################################
"""
This module implements the data acquisition layer used in TMS-EEG experiment.
"""
import pyautogui
import keyboard
import numpy as np
import random
import time
import magicpy as mp
from collections import deque
import pandas as pd  # se quiser manter a checagem de contagem; senão pode remover

from Components.arduino_connection import ArduinoConnection
from Components.update_navigation import UpdateNavigationInfo
from Components import constants as consts, txt_configs as txt

####
# EXPERIMENT SETTINGS
####
mode = "intensity"  # protocol

# history of last 10 target_status values
target_status_history = deque([False] * 10, maxlen=10)

debug_arduino_ppTMS = False  # True if you want just to check arduino connection

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
    2: "intensity_2"   # 120% rMT
}

# -----------------------------
# SEQUENCE GENERATION (mesma lógica do outro script)
# -----------------------------
values = [0, 1, 2]
sequence = [random.choice(values)]
number_of_stimuli = 300  # ajuste se quiser outro número

def zero_decision():
    next_prob = random.randrange(100)
    if next_prob <= 30:  # 30% chance de ser 1
        sequence.append(1)
    else:                # 70% chance de ser 2
        sequence.append(2)

def sequence_generator():
    sequence_past = sequence[-2:].copy()
    if sequence_past[-1] == 2:
        sequence.append(1)
    elif sequence_past[-1] == 0:
        zero_decision()
    elif sequence_past[-2] == 0 and sequence_past[-1] == 1:
        sequence.append(1)
    elif sequence_past[-2] == 1 and sequence_past[-1] == 1:
        sequence.append(0)
    elif sequence_past[-2] == 2 and sequence_past[-1] == 1:
        sequence.append(0)
    else:
        print("bug")

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
# np.savetxt('tree_sequence_' + str(number_of_stimuli) + '_LMS.txt', sequence, delimiter=',', fmt='%d')

''' CONNECTION TO NAVIGATION UPDATES '''
updator = UpdateNavigationInfo(1)
updator.connect('127.0.0.1', [5000])


''' CONNECTION WITH MAGVENTURE '''
mp.list_serial_ports()
input("Portas listadas. Certifique-se de que está conectando na porta correta. Pressione Enter para continuar...")
stimulator = mp.MagVenture("COM1")
stimulator.connect()
stimulator.set_page('Main', get_response=False)


''' PRESS S TO START EXPERIMENT WHEN EVERYTHING IS SET '''
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
    current_status = updator.target_status      # bool vindo do rastreador
    target_status_history.append(current_status)

    # só dispara se TODOS os últimos 10 estados forem True
    if all(target_status_history):
        try:
            # ---- intensity mode only ----
            intensidade = intensities[sequence[pulse_index]]
            print(intensidade)

            stimulator.set_mode(
                mode='Standard',
                current_dir='Normal',
                n_pulses_per_burst=2,
                ipi=5,
                baratio=80
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
                stimulator.fire()
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
        print("Encerrando sequência (fim da lista de pulsos).")
        break

    time.sleep(0.01)
