import torch
from typing import Tuple
def find_best_split(X: torch.Tensor, y: torch.Tensor) -> Tuple[int, float]:
    n,d=X.shape
    y=y.float()
    best_gini=float("inf")
    best_feature=-1
    best_threshold=0.0

    for f in range(d):
        values=torch.unique(X[:,f])
        if values.numel()<2:
            continue
        for i in range(values.numel()-1):
            threshold=values[i]
            left_mask=X[:,f]<=threshold
            right_mask=~left_mask

            n_left=int(left_mask.sum().item())
            n_right=n-n_left
            if n_left==0 or n_right==0:
                continue

            p_left=y[left_mask].mean()
            p_right=y[right_mask].mean()
            gini_left=2.0*p_left*(1.0-p_left)
            gini_right=2.0*p_right*(1.0-p_right)

            gini_split=(n_left/n)*gini_left+(n_right/n)*gini_right


            if gini_split<best_gini:
                best_gini=gini_split
                best_feature=f
                best_threshold=threshold.item()

    return best_feature,best_threshold
