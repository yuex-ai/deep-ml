import numpy as np


def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree,
                   bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Args:
        X_train: shape (n, 1) training inputs
        y_train: 1D training targets
        X_val:   shape (m, 1) validation inputs
        y_val:   1D validation targets
        train_sizes: list of ints, prefix lengths to fit on
        degree: non-negative int, polynomial degree
        bias_threshold: threshold for high-bias diagnosis
        variance_threshold: threshold for high-variance diagnosis

    Returns:
        dict with 'train_errors', 'val_errors', 'diagnosis'
    """
    # 统一转成浮点一维数组，避免二维 vander 报错与整数幂溢出
    x_tr = np.asarray(X_train, dtype=float).ravel()
    y_tr = np.asarray(y_train, dtype=float).ravel()
    x_val = np.asarray(X_val, dtype=float).ravel()
    y_val = np.asarray(y_val, dtype=float).ravel()

    # 验证集特征矩阵与循环无关，构造一次即可复用
    Phi_val = np.vander(x_val, degree + 1, increasing=True)

    train_err = []
    val_err = []

    for n in train_sizes:
        # 只用前 n 个训练样本构造设计矩阵
        Phi_tr = np.vander(x_tr[:n], degree + 1, increasing=True)

        # Moore–Penrose 伪逆求最小二乘权重
        w = np.linalg.pinv(Phi_tr) @ y_tr[:n]

        # 训练 MSE：在当前前缀的 n 个训练样本上计算
        y_tr_pred = Phi_tr @ w
        train_mse = np.mean((y_tr_pred - y_tr[:n]) ** 2)

        # 验证 MSE：在全部验证样本上计算
        y_val_pred = Phi_val @ w
        val_mse = np.mean((y_val_pred - y_val) ** 2)

        train_err.append(float(train_mse))
        val_err.append(float(val_mse))

    # 诊断只基于最后一项（最大训练规模处）
    last_train = train_err[-1]
    last_val = val_err[-1]

    if last_train > bias_threshold:
        diagnosis = 'high_bias'
    elif last_val - last_train > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'

    return {
        'train_errors': train_err,
        'val_errors': val_err,
        'diagnosis': diagnosis,
    }