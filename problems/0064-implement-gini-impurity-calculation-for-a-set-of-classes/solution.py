import torch

def gini_impurity(y: torch.Tensor) -> float:
    """
    Gini impurity for a 1D tensor of class labels (any number of classes).
    :return: 1 - Σ p_k², rounded to three decimal places
    """
    y = y.flatten()
    n = y.numel()
    if n == 0:
        raise ValueError("y must not be empty")

    _, counts = torch.unique(y, return_counts=True)
    p = counts.float() / n
    gini = 1.0 - (p ** 2).sum()
    return round(gini.item(), 3)