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
    y_true = y_true.float()
    y_pred = y_pred.float()
    total=torch.mean(y_true)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    y_mean = torch.mean(y_true)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    if ss_tot == 0:
    # 真实值无方差：若预测完美则 R²=1，否则 R²=0
        return 1.0 if ss_res == 0 else 0.0
    output = 1 - ss_res / ss_tot
    return output.item()
