import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    X = np.asarray(X, dtype=np.float32)              # (T, d)
    scores = X @ X.T                                  # (T, T)

    # 数值稳定的 softmax，沿最后一维（按行）归一化
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)   # 每行和为 1

    context = attn @ X                                # (T, d)
    return context.tolist()                           # np.float32 -> Python float