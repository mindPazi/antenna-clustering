"""
Loss functions for unsupervised GNN clustering.

ROBUST implementation that prevents cluster collapse.
"""

import torch
import torch.nn.functional as F


def mincut_loss(z, adj, deg):
    """
    Normalized MinCut loss for graph clustering.
    """
    num = torch.trace(z.T @ adj @ z)
    denom = torch.trace(z.T @ deg @ z) + 1e-8
    return -num / denom


def orthogonality_loss(z):
    """Alias for balance_loss."""
    return balance_loss(z)


def balance_loss(z):
    """
    Encourage balanced cluster sizes.
    
    STRONG version that really forces equal sizes.
    """
    n, k = z.shape
    # Target: each cluster should have n/k elements
    cluster_sizes = z.sum(dim=0)  # (K,)
    target_size = n / k
    
    # L2 penalty on deviation from target
    size_diff = cluster_sizes - target_size
    return (size_diff ** 2).sum() / (n ** 2)


def entropy_loss(z):
    """
    Encourage confident (non-uniform) assignments.
    
    High entropy = uncertain assignments (bad)
    Low entropy = confident assignments (good)
    """
    # Per-node entropy
    eps = 1e-8
    entropy = -(z * torch.log(z + eps)).sum(dim=1)  # (N,)
    return entropy.mean()


def anti_collapse_loss(z):
    """
    CRITICAL: Prevent all nodes going to one cluster.
    
    Penalizes if any cluster has < 10% or > 40% of nodes.
    """
    n, k = z.shape
    cluster_sizes = z.sum(dim=0)  # (K,)
    
    # Penalize clusters that are too small
    min_size = 0.1 * n  # At least 10% of nodes
    too_small = F.relu(min_size - cluster_sizes)
    
    # Penalize clusters that are too large  
    max_size = 0.4 * n  # At most 40% of nodes
    too_large = F.relu(cluster_sizes - max_size)
    
    return (too_small.sum() + too_large.sum()) / n


def coupling_mincut_loss(z, W):
    """
    MinCut loss using coupling-weighted adjacency.
    
    NOTE: This is kept for compatibility but balance_loss is more important.
    """
    D = torch.diag(W.sum(dim=1))
    num = torch.trace(z.T @ W @ z)
    denom = torch.trace(z.T @ D @ z) + 1e-8
    return -num / denom


def contiguity_loss(z, positions):
    """
    Penalize fragmented clusters.
    
    Simplified version that's more stable.
    """
    n, k = z.shape
    
    # Compute cluster centroids
    cluster_sizes = z.sum(dim=0, keepdim=True).T + 1e-8  # (K, 1)
    centroids = (z.T @ positions) / cluster_sizes  # (K, 2)
    
    # Average distance from each node to its cluster centroid
    loss = 0.0
    for cluster in range(k):
        z_k = z[:, cluster]  # (N,)
        centroid = centroids[cluster]  # (2,)
        
        # Distance to centroid
        dist_to_centroid = ((positions - centroid) ** 2).sum(dim=1)  # (N,)
        
        # Weighted by assignment probability
        loss += (z_k * dist_to_centroid).sum() / (z_k.sum() + 1e-8)
    
    return loss / k


def total_loss(z, W, positions, lambda_balance=10.0, lambda_entropy=1.0, 
               lambda_anti_collapse=5.0, lambda_contiguity=0.1):
    """
    Combined loss function with STRONG regularization.
    
    Key insight: balance and anti-collapse losses must be STRONG
    to prevent the GNN from putting everything in one cluster.
    """
    # MinCut: keep coupled elements together
    loss_cut = coupling_mincut_loss(z, W)
    
    # Balance: STRONG - force equal cluster sizes
    loss_bal = balance_loss(z)
    
    # Entropy: encourage confident assignments
    loss_ent = entropy_loss(z)
    
    # Anti-collapse: CRITICAL - prevent degenerate solutions
    loss_collapse = anti_collapse_loss(z)
    
    # Contiguity: keep clusters spatially compact (mild)
    loss_cont = contiguity_loss(z, positions)
    
    total = (loss_cut + 
             lambda_balance * loss_bal + 
             lambda_entropy * loss_ent +
             lambda_anti_collapse * loss_collapse +
             lambda_contiguity * loss_cont)
    
    return total, {
        'cut': loss_cut.item(),
        'balance': loss_bal.item(),
        'entropy': loss_ent.item(),
        'anti_collapse': loss_collapse.item(),
        'contiguity': loss_cont.item(),
    }