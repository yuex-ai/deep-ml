import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier using logistic regression with gradient descent.
    """
    n_samples, n_features = X_train.shape
    
    # 初始化参数
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    
    # 超参数
    lr = 0.1
    iterations = 500
    
    for _ in range(iterations):
        # 前向传播
        z = X_train @ weights + bias
        p = 1.0 / (1.0 + np.exp(-z))  # sigmoid
        
        # 计算梯度
        error = p - y_train
        grad_w = X_train.T @ error / n_samples
        grad_b = np.sum(error) / n_samples
        
        # 更新参数
        weights -= lr * grad_w
        bias -= lr * grad_b
    
    # 返回预测函数
    def predict(X):
        prob = 1.0 / (1.0 + np.exp(-(X @ weights + bias)))
        return (prob > 0.5).astype(int)
    
    return predict