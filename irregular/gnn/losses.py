"""
Loss functions for unsupervised GNN clustering.
"""

import torch
import torch.nn.functional as F


def coupling_mincut_loss(z, W):
    """
    MinCut loss using coupling-weighted adjacency.

    Minimizes inter-cluster coupling: groups mutually coupled
    antenna elements into the same cluster.
    """
    D = torch.diag(W.sum(dim=1))
    num = torch.trace(z.T @ W @ z)
    denom = torch.trace(z.T @ D @ z) + 1e-8
    return -num / denom


def clustering_factor_loss(z, target_cf):
    """
    Target a specific clustering factor (avg elements per cluster).

    CF = N / n_active_clusters.  The loss penalizes deviation of
    the number of active clusters from N / target_cf.
    """
    n, k = z.shape
    cluster_sizes = z.sum(dim=0)  # (K,) soft sizes

    # Soft count of active clusters
    active = torch.sigmoid(5.0 * (cluster_sizes - 0.5))
    n_active = active.sum()

    # Target number of active clusters
    target_n = n / target_cf

    # Relative squared error on cluster count
    return ((n_active - target_n) / target_n) ** 2


def entropy_loss(z):
    """
    Encourage confident (non-uniform) assignments.

    High entropy = uncertain assignments (bad).
    Low entropy = confident assignments (good).
    """
    eps = 1e-8
    entropy = -(z * torch.log(z + eps)).sum(dim=1)  # (N,)
    return entropy.mean()


def balance_loss(z, target_cf):
    """
    Penalize imbalanced cluster sizes.

    Each active cluster should have approximately *target_cf* elements.
    The loss is the normalised variance of the active soft-cluster sizes.
    """
    cluster_sizes = z.sum(dim=0)  # (K,) soft sizes

    # Consider a cluster "active" when its soft size exceeds 0.5
    active_mask = cluster_sizes > 0.5
    active_sizes = cluster_sizes[active_mask]

    if active_sizes.numel() <= 1:
        return torch.tensor(0.0, device=z.device)

    # Penalize deviation of each active cluster size from target_cf
    return ((active_sizes - target_cf) ** 2).mean() / (target_cf ** 2)


def contiguity_loss(z, positions):
    """
    Penalize fragmented clusters by encouraging spatial compactness.
    """
    n, k = z.shape

    cluster_sizes = z.sum(dim=0, keepdim=True).T + 1e-8  # (K, 1)
    centroids = (z.T @ positions) / cluster_sizes  # (K, 2)

    # Weighted average distance from each node to its cluster centroid
    # Vectorised: dist[i,k] = ||pos_i - centroid_k||^2
    # loss = sum_k  (z[:,k] . dist[:,k]) / size_k
    diff = positions.unsqueeze(1) - centroids.unsqueeze(0)  # (N, K, 2)
    dist_sq = (diff ** 2).sum(dim=2)  # (N, K)
    weighted = (z * dist_sq).sum(dim=0)  # (K,)
    per_cluster = weighted / cluster_sizes.squeeze()  # (K,)

    return per_cluster.mean()


def total_loss(z, W, positions, target_cf=3.0,
               lambda_cf=10.0, lambda_entropy=0.5,
               lambda_contiguity=0.5, lambda_balance=5.0):
    """
    Combined loss for physics-informed clustering with target CF.

    Components:
      - coupling MinCut : group coupled elements together
      - clustering factor: target the desired avg cluster size
      - balance          : penalize imbalanced cluster sizes
      - entropy          : encourage confident hard assignments
      - contiguity       : spatial compactness
    """
    loss_cut = coupling_mincut_loss(z, W)
    loss_cf = clustering_factor_loss(z, target_cf)
    loss_bal = balance_loss(z, target_cf)
    loss_ent = entropy_loss(z)
    loss_cont = contiguity_loss(z, positions)

    total = (loss_cut
             + lambda_cf * loss_cf
             + lambda_balance * loss_bal
             + lambda_entropy * loss_ent
             + lambda_contiguity * loss_cont)

    return total, {
        'cut': loss_cut.item(),
        'cf': loss_cf.item(),
        'balance': loss_bal.item(),
        'entropy': loss_ent.item(),
        'contiguity': loss_cont.item(),
    }
