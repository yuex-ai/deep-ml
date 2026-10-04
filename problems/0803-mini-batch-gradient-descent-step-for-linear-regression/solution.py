import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    n = len(batch_indices)
    w_grad = np.zeros_like(weights)
    b_grad = 0.0
    loss=0.0
    for i in batch_indices:
        x=X[i]
        y_true=y[i]
        y_pred=np.dot(x,weights)+bias
        err=(y_pred-y_true)
        w_grad+=2*(1/n)*x*(err)
        b_grad+=2*(1/n)*err
        w=weights-lr*w_grad
        b=bias-lr*b_grad
    return np.append(w,b)

        
