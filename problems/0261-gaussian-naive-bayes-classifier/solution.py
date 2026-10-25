import numpy as np
from math import pi


def gaussian_naive_bayes(X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_test: np.ndarray) -> np.ndarray:
    """
    Gaussian Naive Bayes 分类器
    Args:
        X_train: (N_train, D)
        y_train: (N_train,)
        X_test : (N_test, D)
    Returns:
        (N_test,) 预测标签
    """
    epsilon = 1e-9
    N_train, D = X_train.shape
    N_test = X_test.shape[0]
    classes = np.unique(y_train)
    K = len(classes)
	
    # log 后验：每类一行
    log_posterior = np.zeros((K, N_test))
    for k, c in enumerate(classes):
        X_c = X_train[y_train == c]            # (N_c, D)
        n_c = X_c.shape[0]
        # 1) 先验
        log_prior = np.log(n_c / N_train)
        # 2) 每类每个特征的均值与方差
        X_mean = np.mean(X_c, axis=0)          # (D,)
        X_var  = np.var(X_c,  axis=0)          # (D,)
        X_var  = np.maximum(X_var, epsilon)    # 防止除零
        # 3) 在 X_test 上计算 log 高斯似然
        #    log N(x|μ,σ²) = -0.5*log(2πσ²) - 0.5*(x-μ)²/σ²
        log_lik = -0.5 * np.log(2 * pi * X_var) \
                  - 0.5 * (X_test - X_mean) ** 2 / X_var   # (N_test, D)
        log_lik = np.sum(log_lik, axis=1)                  # (N_test,)
        # 4) log 后验
        log_posterior[k] = log_prior + log_lik
    # 5) 在每个测试样本上取类别 argmax，再映射回真实标签
    pred_idx = np.argmax(log_posterior, axis=0)
    return classes[pred_idx]