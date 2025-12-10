import pyautogui
import keyboard
import magicpy as mp

''' CONNECTION WITH MAGVENTURE'''
mp.list_serial_ports()                                              # imprime uma lista de portas disponíveis
input("Portas listadas. Certifique-se de que está conectando na porta correta. Pressione Enter para continuar...")         # Aguarda o usuário clicar Enter para continuar
stimulator = mp.MagVenture("COM11")                                  # Inicializa um objeto que se relaciona ao estimulador
stimulator.connect()
stimulator.set_page('Main', get_response=False)

''' PRESS S TO START EXPERIMENT WHEN EVERYTHING IS SET'''
print("Aperte a tecla s para iniciar os pulsos...")
while True:
    if keyboard.is_pressed('s'):
        break