import numpy as np
def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.
    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree
    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n=len(x)
    x_vanderm=np.zeros((n,degree+1),dtype=float)
    for i in range(n):
        for j in range(degree+1):
            x_vanderm[i][j]=x[i]**j
    #X = np.vander(x, N=degree + 1, increasing=True)
    c=np.linalg.solve((x_vanderm.T@x_vanderm),(x_vanderm.T@y))
    return [float(v) for v in c]


