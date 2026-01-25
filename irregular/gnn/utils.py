"""
Utility functions for GNN-based antenna clustering.
"""

import torch
import numpy as np
from typing import Union, Optional, List


def normalize_positions(
    positions: torch.Tensor,
    method: str = "standard"
) -> torch.Tensor:
    """
    Normalize antenna positions for stable training.

    Args:
        positions: (N, 2) tensor of (x, y) coordinates
        method: Normalization method
            - "standard": Zero mean, unit variance (z-score)
            - "minmax": Scale to [0, 1] range

    Returns:
        Normalized positions (N, 2)
    """
    if method == "standard":
        mean = positions.mean(dim=0, keepdim=True)
        std = positions.std(dim=0, keepdim=True)
        return (positions - mean) / (std + 1e-8)

    elif method == "minmax":
        min_val = positions.min(dim=0, keepdim=True).values
        max_val = positions.max(dim=0, keepdim=True).values
        return (positions - min_val) / (max_val - min_val + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_hard_assignments(z: torch.Tensor) -> torch.Tensor:
    """
    Convert soft cluster probabilities to hard assignments.

    c_i = argmax_k z_ik

    Args:
        z: (N, K) soft assignment matrix

    Returns:
        c: (N,) hard cluster labels in {0, ..., K-1}
    """
    return z.argmax(dim=-1)


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get torch device from string specification.

    Args:
        device_str: "auto", "cuda", or "cpu"

    Returns:
        torch.device object
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def cluster_sizes(assignments: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Count elements in each cluster.

    Args:
        assignments: (N,) cluster labels

    Returns:
        sizes: (K,) count per cluster
    """
    if isinstance(assignments, torch.Tensor):
        assignments = assignments.cpu().numpy()
    return np.bincount(assignments)


def cluster_to_list(
    assignments: Union[torch.Tensor, np.ndarray],
    num_clusters: Optional[int] = None
) -> List[np.ndarray]:
    """
    Convert flat assignments to list of index arrays per cluster.

    Useful for interfacing with antenna_physics.py which expects
    List[np.ndarray] format for clusters.

    Args:
        assignments: (N,) cluster labels
        num_clusters: K (inferred from data if None)

    Returns:
        List of K arrays, each containing indices of elements in that cluster
    """
    if isinstance(assignments, torch.Tensor):
        assignments = assignments.cpu().numpy()

    if num_clusters is None:
        num_clusters = assignments.max() + 1

    clusters = []
    for k in range(num_clusters):
        indices = np.where(assignments == k)[0]
        clusters.append(indices)

    return clusters


def assignments_to_antenna_format(
    assignments: Union[torch.Tensor, np.ndarray],
    grid_shape: tuple = (16, 16)
) -> List[np.ndarray]:
    """
    Convert flat cluster assignments to antenna array format.

    Compatible with AntennaArray.index_to_position_cluster() method.

    Args:
        assignments: (N,) cluster labels for flattened array
        grid_shape: (Nz, Ny) shape of the antenna grid

    Returns:
        List of K arrays with shape (L_k, 2) containing [col, row] indices
    """
    if isinstance(assignments, torch.Tensor):
        assignments = assignments.cpu().numpy()

    Nz, Ny = grid_shape
    num_clusters = assignments.max() + 1

    clusters = []
    for k in range(num_clusters):
        flat_indices = np.where(assignments == k)[0]

        # Convert flat index to 2D grid indices
        # Assuming row-major (C) ordering: flat_idx = row * Ny + col
        rows = flat_indices // Ny
        cols = flat_indices % Ny

        # Format as [col, row] to match antenna_physics convention
        cluster_coords = np.stack([cols, rows], axis=1)
        clusters.append(cluster_coords)

    return clusters


def compute_clustering_metrics(
    assignments: np.ndarray,
    positions: np.ndarray
) -> dict:
    """
    Compute basic clustering quality metrics.

    Args:
        assignments: (N,) cluster labels
        positions: (N, 2) antenna positions

    Returns:
        Dictionary with metrics:
            - num_clusters: Actual number of non-empty clusters
            - cluster_sizes: Elements per cluster
            - size_variance: Variance in cluster sizes
            - mean_intra_distance: Average within-cluster distance
    """
    # TODO: Implement clustering metrics
    raise NotImplementedError


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
