import torch

def elastic_net_gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    n_sample,n_features=X.shape
    weights=torch.zeros(n_features,dtype=torch.float32)
    bias=torch.zeros(1,dtype=torch.float32)
    #y_pred=X@weights+bias
    #residual=y_pred-y
    #loss=(0.5*n)*sum((y_pred-y)**2)+alpha1*sum(torch.abs(weights))+alpha2*sum((weights**2))
    for _ in range(max_iter):
        y_pred=X@weights+bias
        residual=y_pred-y

        grad_w=(1/n_sample)*(X.T@residual)+alpha1*torch.sign(weights)+2*alpha2*weights
        grad_b=torch.mean(residual)
        weights=weights-learning_rate*grad_w
        bias-=learning_rate*grad_b
        w1_norm=sum(torch.abs(grad_w))
        if w1_norm<tol:
            break
    return (weights,bias)

        
        


