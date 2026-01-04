/*
  Arduino trigger generator for NeurOne EEG
  - Receives an integer (1–7) via Serial
  - Outputs a 3-bit binary code on digital pins
  - Pulse duration: 5 ms
*/

// ================= CONFIGURAÇÃO =================

// 3 pinos digitais do Arduino (2, 3 e 4) como bits binários para os pinos IN1, 2 e 3 do NeurOne
const int BIT0_PIN = 2;  // -> IN1
const int BIT1_PIN = 3;  // -> IN2
const int BIT2_PIN = 4;  // -> IN3

// Duração do pulso em milissegundos
const unsigned long PULSE_DURATION_MS = 5;

// ================= SETUP =================
void setup() {
  // Configura os pinos digitais do arduino como saída
  pinMode(BIT0_PIN, OUTPUT);
  pinMode(BIT1_PIN, OUTPUT);
  pinMode(BIT2_PIN, OUTPUT);

  // Garante que tudo começa em LOW
  clearBits();

  // Inicializa comunicação serial a uma taxa de transmissão (baud rate) em bits/seg
  Serial.begin(115200); // velocidade padrão -> setar no script py
}

// ================= LOOP PRINCIPAL =================
void loop() {
  // Verifica se chegou algum dado pela serial
  if (Serial.available() > 0) {

    // Lê um número inteiro (ex: "3\n")
    int triggerValue = Serial.parseInt();

    // Limpa buffer serial
    Serial.read();

    // Validação básica
    if (triggerValue >= 1 && triggerValue <= 7) {
      sendTrigger(triggerValue);
    }
  }
}

// ================= FUNÇÕES =================

// Coloca todos os bits em LOW
void clearBits() {
  digitalWrite(BIT0_PIN, LOW);
  digitalWrite(BIT1_PIN, LOW);
  digitalWrite(BIT2_PIN, LOW);
}

// Envia trigger binário
void sendTrigger(int value) {

  // === Monta os bits (simultaneamente) ===
  digitalWrite(BIT0_PIN, value & 0b001 ? HIGH : LOW);
  digitalWrite(BIT1_PIN, value & 0b010 ? HIGH : LOW);
  digitalWrite(BIT2_PIN, value & 0b100 ? HIGH : LOW);

  // Mantém o estado por 5 ms
  delay(PULSE_DURATION_MS);

  // Retorna tudo para LOW
  clearBits();
}
