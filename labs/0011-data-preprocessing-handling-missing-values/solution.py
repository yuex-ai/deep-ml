import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    # TODO: Fill in NaN values
    n_samples,n_features=X_clean.shape
    mask=np.isnan(X_clean)
    col_mean = np.nanmean(X_clean, axis=0)


    global_mean = np.nanmean(X_clean)
    if np.isnan(global_mean):
        global_mean = 0.0

    col_mean = np.where(np.isnan(col_mean), global_mean, col_mean)

    # 向量化填充
    rows, cols = np.where(mask)
    X_clean[rows, cols] = col_mean[cols]

    return X_clean

    return X_clean
