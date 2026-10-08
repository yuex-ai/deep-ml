import torch

def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the R-squared (R²) coefficient of determination using PyTorch.

    Args:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    total=torch.mean(y_pred)
    if y_pred.size()==y_true.size():
        residuals=torch.sum((y_true-y_pred)**2)
        sst=torch.sum((y_true-total)**2)
    output=1-(residuals/sst)
    return output.item()
