import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    h = np.array(initial_hidden_state)
    for x_t in input_sequence:
        x_t = np.array(x_t)
        h = np.tanh(Wx @ x_t + Wh @ h + b)
    return h.tolist()