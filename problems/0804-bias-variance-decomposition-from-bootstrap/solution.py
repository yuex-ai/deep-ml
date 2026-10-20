import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.asarray(predictions, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    B, M = predictions.shape

    # 每个测试点上的平均预测，形状 (M,)
    mean_pred = np.mean(predictions, axis=0)

    # 逐点偏差平方，再对 M 个测试点平均 -> 标量
    bias_squared = np.mean((mean_pred - y_true) ** 2)

    # 每个预测相对平均预测的平方偏离，对 B 和 M 平均 -> 标量
    variance = np.mean((predictions - mean_pred) ** 2)

    # 每个预测相对真实目标的平方误差，对 B 和 M 平均 -> 标量
    mse = np.mean((predictions - y_true) ** 2)

    return {
        'bias_squared': float(bias_squared),
        'variance': float(variance),
        'mse': float(mse),
    }

