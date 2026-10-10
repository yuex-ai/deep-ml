import torch

def confusion_matrix(data: list) -> torch.Tensor:
    """
    Generate a 2x2 confusion matrix for binary classification.

    Args:
        data: A list of [y_true, y_pred] pairs for binary labels (0 or 1)

    Returns:
        A 2x2 torch.Tensor confusion matrix arranged as [[TP, FN], [FP, TN]]
    """
    data=torch.tensor(data,dtype=torch.float32)
    tp = fp = tn = fn = 0

    for yt, yp in data:
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1

    return torch.tensor([[tp, fn], [fp, tn]], dtype=torch.float32)
        
