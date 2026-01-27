"""
Loss functions for unsupervised GNN clustering.

No labels are needed - clustering quality is measured by graph structure:
    - MinCut: Minimize edges between clusters (maximize within-cluster connectivity)
    - Orthogonality: Ensure balanced, non-overlapping clusters
"""

import torch


def mincut_loss(z, adj, deg):
    """Normalized MinCut loss for graph clustering."""
    num = torch.trace(z.T @ adj @ z)
    denom = torch.trace(z.T @ deg @ z)
    return -num / (denom + 1e-8)


def orthogonality_loss(z):
    """Encourage balanced, non-overlapping clusters."""
    n, k = z.shape
    identity = torch.eye(k, device=z.device) / k
    cluster_sim = (z.T @ z) / n
    return torch.norm(cluster_sim - identity, p='fro') ** 2
