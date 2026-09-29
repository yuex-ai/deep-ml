import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    m = torch.as_tensor(matrix, dtype=torch.float)
    # 检查行列式是否接近 0
    det = torch.linalg.det(m)
    if torch.abs(det) < 1e-12:
        return None
    return torch.linalg.inv(m)