def zero_decision():
    """
    Generate the next element in the sequence based on a probabilistic decision.

    There is a 30% chance of the next element being 1, and a 70% chance of it being 2.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    next_prob = random.randrange(100)
    if next_prob <= 30:  # 30% chance de ser 1
        sequence.append(1)
    else:                # 70% chance de ser 2
        sequence.append(2)

def sequence_generator():
    """
    This function generates the next element in the sequence based on the last two elements.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    
    Notes
    -----
    The sequence is generated based on a set of rules:
    - If the last element is 2, the next element is 1.
    - If the last element is 0, the next element is determined by the zero_decision function.
    - If the last element is 1 and the second to last element is 0, the next element is 1.
    - If the last element is 1 and the second to last element is 1, the next element is 0.
    - If the last element is 1 and the second to last element is 2, the next element is 0.
    - If none of the above rules apply, print "bug".
    """
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

def send_trigger_to_esp32(trigger: int):
    """
    Envia um trigger para o ESP32.

    Parameters
    ----------
    trigger : int
        Valor do trigger a ser enviado para o ESP32 (0, 1, 2, 3, ...).

    Raises
    ------
    Exception
        Se houver um erro ao enviar o trigger para o ESP32.

    Returns
    -------
    None
    """
    try:
        ser.write(f"{trigger}\n".encode("ascii"))
    except Exception as e:
        print(f"Erro ao enviar trigger {trigger} para ESP32: {e}")