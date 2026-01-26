"""
Loss functions for unsupervised GNN clustering.

No labels are needed - clustering quality is measured by graph structure:
    - MinCut: Minimize edges between clusters (maximize within-cluster connectivity)
    - Orthogonality: Ensure balanced, non-overlapping clusters
    - Entropy: Encourage confident (non-uniform) assignments
"""

import torch
from typing import Optional


def mincut_loss(
    z: torch.Tensor,
    adj: torch.Tensor,
    deg: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalized MinCut loss.

    L_cut = -Tr(Z^T A Z) / Tr(Z^T D Z)

    Minimizing this loss maximizes within-cluster edges (good clustering
    keeps connected nodes together).

    Args:
        z: (N, K) soft cluster assignment matrix
        adj: (N, N) adjacency matrix
        deg: (N, N) degree matrix (diagonal)
        eps: Small constant for numerical stability

    Returns:
        loss: Scalar MinCut loss (negative, to be minimized)
    """
    # Z^T A Z measures within-cluster edges
    # (Z^T A Z)_kk = sum over edges where both endpoints assigned to cluster k
    ztaz = torch.mm(torch.mm(z.T, adj), z)
    numerator = torch.trace(ztaz)

    # Z^T D Z normalizes by cluster sizes
    ztdz = torch.mm(torch.mm(z.T, deg), z)
    denominator = torch.trace(ztdz)

    # Negative because we want to maximize within-cluster edges
    loss = -numerator / (denominator + eps)

    return loss


def orthogonality_loss(
    z: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Orthogonality regularization loss.

    L_ortho = || Z^T Z / N  -  I_K / K ||_F^2

    Prevents trivial solution where all nodes go to one cluster.
    Encourages balanced cluster sizes.

    Args:
        z: (N, K) soft cluster assignment matrix
        eps: Small constant for stability

    Returns:
        loss: Scalar orthogonality loss
    """
    n, k = z.shape

    # Z^T Z / N: measures cluster overlap, should be diagonal
    ztz = torch.mm(z.T, z) / n

    # Target: I_K / K (identity scaled so trace = 1)
    identity = torch.eye(k, device=z.device, dtype=z.dtype) / k

    # Frobenius norm squared of difference
    loss = torch.norm(ztz - identity, p='fro') ** 2

    return loss


def entropy_loss(
    z: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Entropy regularization loss.

    L_entropy = -(1/N) * sum_i sum_k z_ik * log(z_ik)

    Low entropy means confident predictions (z_ik close to 0 or 1).
    Can be used to encourage sharper cluster assignments.

    Args:
        z: (N, K) soft cluster assignment matrix
        eps: Small constant to avoid log(0)

    Returns:
        loss: Scalar mean entropy (negative for confident assignments)
    """
    n = z.shape[0]

    # Shannon entropy: -sum z * log(z)
    # Add eps for numerical stability with log
    entropy = -torch.sum(z * torch.log(z + eps))

    # Average over nodes
    loss = entropy / n

    return loss


def cluster_size_loss(
    z: torch.Tensor,
    target_size: Optional[int] = None
) -> torch.Tensor:
    """
    Cluster size regularization.

    Penalizes deviation from uniform cluster sizes.
    Optional: can target specific cluster sizes.

    Args:
        z: (N, K) soft cluster assignment matrix
        target_size: Target elements per cluster (default: N/K)

    Returns:
        loss: Scalar size variance loss
    """
    n, k = z.shape

    # Soft cluster sizes: sum of probabilities per cluster
    sizes = z.sum(dim=0)  # (K,)

    # Target size
    if target_size is None:
        target_size = n / k

    # Variance from target size
    loss = torch.var(sizes - target_size)

    return loss


def total_loss(
    z: torch.Tensor,
    adj: torch.Tensor,
    deg: torch.Tensor,
    lambda_ortho: float = 1.0,
    lambda_entropy: float = 0.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Combined loss function.

    L_total = L_cut + lambda_ortho * L_ortho + lambda_entropy * L_entropy

    Args:
        z: (N, K) soft cluster assignments
        adj: (N, N) adjacency matrix
        deg: (N, N) degree matrix
        lambda_ortho: Weight for orthogonality term
        lambda_entropy: Weight for entropy term (optional)
        eps: Numerical stability constant

    Returns:
        loss: Total scalar loss
    """
    loss = mincut_loss(z, adj, deg, eps)
    loss = loss + lambda_ortho * orthogonality_loss(z, eps)

    if lambda_entropy > 0:
        loss = loss + lambda_entropy * entropy_loss(z, eps)

    return loss


class ClusteringLoss(torch.nn.Module):
    """
    Module wrapper for clustering loss computation.

    Convenient for use in training loops with configurable weights.
    """

    def __init__(
        self,
        lambda_ortho: float = 1.0,
        lambda_entropy: float = 0.0
    ):
        """
        Args:
            lambda_ortho: Weight for orthogonality loss
            lambda_entropy: Weight for entropy loss
        """
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_entropy = lambda_entropy

    def forward(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        deg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total loss.

        Args:
            z: Soft cluster assignments
            adj: Adjacency matrix
            deg: Degree matrix

        Returns:
            Total loss scalar
        """
        return total_loss(
            z, adj, deg,
            self.lambda_ortho,
            self.lambda_entropy
        )
