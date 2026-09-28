import torch

def matrix_determinant_and_trace(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) as a torch.Tensor
    
    Returns:
        Tuple of (determinant, trace) as torch.Tensors
    """
    # Your code here
    if matrix.ndim!=2 or matrix.size(0)!=matrix.size(1):
        raise ValueError("ValueError")
    trace = torch.zeros((), dtype=matrix.dtype, device=matrix.device)
    for i in range(matrix.size(0)):
                trace+=matrix[i,i]
    determinant=torch.linalg.det(matrix)
    return (determinant,trace)