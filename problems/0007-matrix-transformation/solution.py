import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    det_T=torch.linalg.det(T_t)
    det_S=torch.linalg.det(S_t)
    if det_T.item()==0 or det_S.item()==0:
        return torch.tensor(-1.)
    T_t_inv=torch.linalg.inv(T_t)
    output=T_t_inv@(A_t@S_t)
    return output
