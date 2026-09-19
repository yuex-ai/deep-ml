import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.
        """
        h = initial_hidden_state.copy()
        c = initial_cell_state.copy()
        hidden_states = []
        for t in range(len(x)):
            x_t = x[t].reshape(-1, 1)
            concat = np.vstack([h, x_t])

            f_t = sigmoid(self.Wf @ concat + self.bf)
            i_t = sigmoid(self.Wi @ concat + self.bi)
            c_tilde = np.tanh(self.Wc @ concat + self.bc)

            c = f_t * c + i_t * c_tilde

            o_t = sigmoid(self.Wo @ concat + self.bo)
            h = o_t * np.tanh(c)

            hidden_states.append(h.copy())

        return hidden_states, h, c