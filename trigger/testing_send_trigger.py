import serial
import time

# ================= CONFIGURAÇÃO =================
PORTA_SERIAL = 'COM6'
BAUD_RATE = 115200
DELAY_BETWEEN_TRIGGERS = 3  # segundos

# ================= CONEXÃO =================
ser = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=1)
time.sleep(2)  # aguarda ESP32 reiniciar

# ================= ENVIO DE TRIGGERS =================
for trigger in range(1, 16):  # triggers de 1 a 15
    print(f'Enviando trigger {trigger}')
    ser.reset_input_buffer()  # limpa buffer antes de enviar
    ser.write(f'{trigger}\n'.encode('ascii'))

    # Lê todas as linhas que o ESP32 enviar para esse trigger
    start_time = time.time()
    while time.time() - start_time < 1:  # 1 segundo de espera para resposta
        if ser.in_waiting > 0:
            resposta = ser.readline().decode('ascii', errors='ignore').strip()
            if resposta:
                print(f'ESP32 respondeu: {resposta}')

    time.sleep(DELAY_BETWEEN_TRIGGERS)

# ================= FINALIZA =================
ser.close()