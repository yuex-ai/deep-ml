import torch

def k_nearest_neighbors(points: torch.Tensor, query_point: torch.Tensor, k: int) -> torch.Tensor:
    """
    Find k nearest neighbors to a query point
    Args:
        points: Tensor of shape (n, d) representing n points in d-dimensional space
        query_point: Tensor of shape (d,) representing the query point
        k: Number of nearest neighbors to return
    Returns:
        Tensor of shape (k, d) containing the k nearest neighbor points
        When distances are tied, points appearing earlier in the input tensor come first.
    """
    distances = torch.cdist(points, query_point.unsqueeze(0)).squeeze(1)
    # 2. 稳定排序，保证平局时保持原始顺序
    idx = torch.argsort(distances, stable=True)
    # 3. 取前 k 个索引对应的原始点
    return points[idx[:k]]
    