
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
    y = np.asarray(y).ravel()
    n = y.size
    if n == 0:
        raise ValueError("y must not be empty")

    _, counts = np.unique(y, return_counts=True)
    p = counts / n
    val = 1 - (p ** 2).sum()
    return round(float(val), 3)