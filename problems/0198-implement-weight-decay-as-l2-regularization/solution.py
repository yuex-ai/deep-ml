import torch

def apply_weight_decay(parameters: list[torch.Tensor], gradients: list[torch.Tensor], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[torch.Tensor]:
    """
    Apply weight decay (L2 regularization) to parameters.
    Args:
        parameters: List of parameter tensors
        gradients: List of gradient tensors
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight decay
    Returns:
        Updated parameters
    """
    # Your code here
    updated = []
    for p, g, use_wd in zip(parameters, gradients, apply_to_all):
        if use_wd:
            new_p = p - lr * g - lr * weight_decay * p
        else:
            new_p = p - lr * g
        updated.append(new_p)
    return updated




