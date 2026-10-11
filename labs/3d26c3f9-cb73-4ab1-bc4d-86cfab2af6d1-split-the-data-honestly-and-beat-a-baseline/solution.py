import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    rng = np.random.default_rng(seed)
    n_samples,n_featurs=X.shape
    n=len(y)
    idx = rng.permutation(n)
    train_num=int(n*train_frac)
    val_num=int(n*val_frac)
    test_num=int(n*test_frac)
    X_shuffled=X[idx]
    y_shuffled=y[idx]
    train_idx=idx[:train_num]
    val_idx=idx[train_num:train_num+val_num]
    test_idx=idx[train_num+val_num:]
    x_train,y_train=X_shuffled[train_idx],y_shuffled[train_idx]
    x_val,y_val=X_shuffled[val_idx],y_shuffled[val_idx]
    x_test,y_test=X_shuffled[test_idx],y_shuffled[val_idx]
    train_mean = y_train.mean()
    test_pred = np.full_like(y_test, train_mean,dtype=float)
    return (test_pred,train_idx,val_idx,test_idx)







