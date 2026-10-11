import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    num_iterations=500
    n_samples,n_features=X.shape
    lr=0.01
    eps=1e-15
    beta1=beta2=0.9
    
    for _ in range(num_iterations):
        y_pred=X@W+b
        error=y_pred-y
        grad_w=(2/n_samples)*(X.T@error)
        grad_b=(2/n_samples)*sum(error)
        W-=lr*grad_w
        b-=lr*grad_b
    return (W,b)


