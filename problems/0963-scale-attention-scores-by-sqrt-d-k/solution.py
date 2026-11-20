import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    n,d=K.shape
    matrix=(Q@K.T)/np.sqrt(d)
    scores=matrix-matrix.max(axis=-1,keepdims=True)
    scores_e=np.exp(scores)
    softmax_scores=scores_e/scores_e.sum(axis=-1,keepdims=True)
    return softmax_scores
