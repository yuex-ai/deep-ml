import numpy as np

def pairwise_cosine_similarity(X):
    # 1. 转成浮点 NumPy 数组
    X = np.asarray(X, dtype=np.float64)
    
    # 2. 计算每一行的 L2 范数，形状 (n, 1)
    norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    
    # 3. 找出零范数行，形状 (n,)
    zero_mask = (norms == 0).squeeze(axis=1)
    
    # 4. 避免除零：零范数先替换为 1
    safe_norms = norms.copy()
    safe_norms[zero_mask] = 1.0
    
    # 5. 按行归一化
    X_norm = X / safe_norms
    
    # 6. 零行归一化后应为全零
    X_norm[zero_mask] = 0.0
    
    # 7. 计算余弦相似度矩阵
    S = X_norm @ X_norm.T
    
    # 8. 设置对角线：非零行 1.0，零行 0.0
    diag = np.where(zero_mask, 0.0, 1.0)
    np.fill_diagonal(S, diag)
    
    # 9. 四舍五入并转成嵌套列表
    return np.round(S, 4).tolist()