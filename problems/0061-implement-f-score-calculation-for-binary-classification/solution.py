import torch

def f_score(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float) -> float:
    """
    Calculate F-Score for a binary classification task.

    :param y_true: torch.Tensor of true labels (binary)
    :param y_pred: torch.Tensor of predicted labels (binary)
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    y_true = y_true.int()
    y_pred = y_pred.int()

    # 向量化计算 TP, FP, FN
    tp = torch.sum((y_true == 1) & (y_pred == 1)).item()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).item()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).item()

    # 计算 Precision
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # 计算 Recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # 计算 F-Score
    beta2 = beta ** 2
    denominator = beta2 * precision + recall
    if denominator == 0:
        f = 0.0
    else:
        f = (1 + beta2) * (precision * recall) / denominator

    return round(f, 3)
