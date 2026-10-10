import torch

def train_logreg(X: torch.Tensor, y: torch.Tensor, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    """
    Train logistic regression using gradient descent with BCE loss.
    Args:
        X: Feature matrix of shape (n_samples, n_features) as Tensor
        y: Binary labels of shape (n_samples,) as Tensor
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations
    Returns:
        Tuple of (coefficients, losses) where:
        - coefficients: List of learned weights (bias first, then feature weights), rounded to 4 decimals
        - losses: List of BCE loss values at each iteration, rounded to 4 decimals
    Notes:
        - Initialize all coefficients to zero
        - Add bias column as FIRST column of X
        - Use sum-based BCE loss: -sum(y*log(p) + (1-y)*log(1-p))
    """
    # Your code here
    n_samples,n_features=X.shape
    #weights=torch.zeros_like(X)
    #bias=torch.zeros(n_samples)
    weights = torch.zeros(n_features, dtype=X.dtype)
    bias = torch.zeros(1, dtype=X.dtype)
    eps = 1e-12
    Loss=[]
    for _ in range(iterations):
        y_pred=torch.matmul(X,weights)+bias
        p=torch.sigmoid(y_pred)
        loss=-sum(y*torch.log(p+eps)+(1-y)*torch.log(1-p+eps))
        grad=p-y
        grad_weights=X.T@grad
        grad_bias = torch.sum(grad) 
        weights=weights-learning_rate*grad_weights
        bias=bias-learning_rate*grad_bias
        Loss.append(loss.item())
    coefficients = [round(bias.item(), 4)] + [round(w, 4) for w in weights.tolist()]
    return (coefficients,Loss)


