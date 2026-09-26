import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    if norm_type=='l1':
        l1=torch.sum(torch.abs(arr))
        return l1.item()
    elif norm_type=='l2':
        l2=torch.sqrt(torch.sum(arr**2))
        return l2.item()
    elif norm_type=='linf':
        linf=torch.max(torch.abs(arr))
        return linf.item()
    elif norm_type=='frobenius':
        if arr.dim()!=2:
            raise ValueError("Frobenius norm requires a 2D tensor")
        fro=torch.sqrt(torch.sum(torch.abs(arr)**2))
        return fro.item()
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
