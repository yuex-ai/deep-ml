import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_sample,_=X.shape
    if n_iter==0:
        return weights.tolist()
    for i in range(n_iter):
        idx=i%n_sample
        x=X[idx]
        y_true=y[idx]
        y_pred=x.T@weights
        loss=(y_pred-y_true)**2
        grad=2*(y_pred-y_true)*x
        weights=weights-learning_rate*grad
    return weights.tolist()

