import torch
from typing import Union

def accuracy_score(y_true: Union[torch.Tensor, list, "np.ndarray"],
                   y_pred: Union[torch.Tensor, list, "np.ndarray"]) -> float:
    """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """
    # Your implementation here
    m=len(y_true)
    correct=0
    for i in range(m):
        if y_true[i]==y_pred[i]:
            correct+=1
    acc=correct/m
    return acc
