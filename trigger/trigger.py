import serial
import time

# ================= CONFIGURAÇÃO =================

# Ajuste a porta conforme o seu sistema
# Linux: '/dev/ttyACM0' ou '/dev/ttyUSB0'
# Windows: 'COM3', 'COM4', etc.
PORTA_SERIAL = '/dev/ttyUSB0' 
BAUD_RATE = 115200

# ================= CONEXÃO =================

ser = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=1)

# IMPORTANTE: o Arduino reinicia quando a porta abre
time.sleep(2)

# ================= ENVIO DE TRIGGERS =================

# Exemplo: envia triggers de 1 a 7
for trigger in range(1, 8):
    print(f'Enviando trigger {trigger}')
    ser.write(f'{trigger}\n'.encode('ascii'))
    time.sleep(3)  # espera 1 segundo entre triggers

# ================= FINALIZA =================

ser.close()

