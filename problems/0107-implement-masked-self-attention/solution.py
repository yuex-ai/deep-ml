import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # Your code here
    seq_len,d_k=K.shape
    score=torch.matmul(Q,K.T)/torch.sqrt(torch.tensor(d_k,dtype=K.dtype))
    mask_score=score+mask
    row_max=torch.max(mask_score,dim=1,keepdim=True).values
    softmaxScore=torch.exp(mask_score-row_max)/torch.sum(torch.exp(mask_score-row_max),dim=1,keepdim=True)
    ouput=torch.matmul(softmaxScore, V)
    return ouput

