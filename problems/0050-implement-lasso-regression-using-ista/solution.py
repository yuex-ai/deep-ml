import torch

def soft_threshold(w: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input tensor
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded tensor where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    # Your code here
    return torch.sign(w) * torch.clamp(torch.abs(w) - threshold, min=0.0)

def l1_regularization_gradient_descent(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix tensor of shape (n_samples, n_features)
        y: Target vector tensor of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias) as torch.Tensors
    
    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    weights = torch.zeros(n_features, dtype=torch.float32)
    bias = torch.tensor(0.0, dtype=torch.float32)
    
    # Your code here
    for _ in range(max_iter):
        y_pred=X@weights+bias
        residual=y_pred-y
    #loss=(0.5*n_samples)*sum(error**2)+alpha*sum(torch.abs(weights))
        grad_w = (1.0 / n_samples) * (X.T @ residual)
        grad_b = torch.mean(residual)
        w_temp=weights-learning_rate*grad_w
        w_new=soft_threshold(w_temp,learning_rate*alpha)
        bias-=learning_rate*grad_b
        delta = torch.linalg.norm(w_new - weights)

        weights = w_new
        if delta < tol:
            break
        
    return (weights,bias)

