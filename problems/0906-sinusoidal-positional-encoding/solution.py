import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    position=torch.arange(seq_len,dtype=torch.float32).unsqueeze(-1)
    div_term=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(10000.0)/d_model))
    angle=position*div_term
    pe=torch.zeros(seq_len,d_model)
    pe[:,0::2]=torch.sin(angle)
    pe[:,1::2]=torch.cos(angle)
    return pe