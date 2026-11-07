import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    A=np.asarray(A,dtype=float)
    if A.ndim!=2:
        raise ValueError("error")
    work=A.copy()
    m,n=work.shape
    rank=0

    for col in range(n):
        if rank>=m:
            break
        pivot_offset=int(np.argmax(np.abs(work[rank:,col])))
        pivot_row=rank+pivot_offset

        if abs(work[pivot_row,col])<=tol:
            continue

        if pivot_row!=rank:
            work[[rank,pivot_row],:]=work[[pivot_row,rank],:]

        pivot_val=work[rank,col]

        for i in range(rank+1,m):
            factor=work[i,col]/pivot_val
            if factor!=0.0:
                work[i,col+1:]-=factor*work[rank,col+1:]
            work[i,col]=0.0
        rank+=1

    return int(rank)

