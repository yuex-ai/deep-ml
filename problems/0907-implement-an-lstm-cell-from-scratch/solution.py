import math
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # TODO: create W_ih, W_hh, b_ih, b_hh with the shapes specified above
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.W_ih = nn.Parameter(torch.empty(4*self.hidden_size,input_size))
        self.W_hh=nn.Parameter(torch.empty(4*self.hidden_size,hidden_size))
        self.b_ih=nn.Parameter(torch.empty(4*self.hidden_size,))
        self.b_hh=nn.Parameter(torch.empty(4*self.hidden_size,))


    def forward(self, x, state):
        h_prev, c_prev = state
        # TODO: compute the four gates and return (h_new, c_new)
        g=x@self.W_ih.T+self.b_ih+h_prev@self.W_hh.T+self.b_hh
        i,f,c_tilde,o=torch.Tensor.chunk(g,4,1)
        i=torch.sigmoid(i)
        f=torch.sigmoid(f)
        c_tilde=torch.tanh(c_tilde)
        o=torch.sigmoid(o)
        c_new=f*c_prev+i*c_tilde
        h_new=o*torch.tanh(c_new)
        return h_new,c_new





