"""
Utility functions for GNN-based antenna clustering.
"""

import numpy as np


def assignments_to_antenna_format(assignments, grid_shape=(16, 16)):
    """
    Convert flat cluster assignments to antenna array format.

    Compatible with AntennaArray.evaluate_clustering() method.

    Args:
        assignments: (N,) cluster labels for flattened array
        grid_shape: (Nz, Ny) shape of the antenna grid

    Returns:
        List of K arrays with shape (L_k, 2) containing [col, row] indices
    """
    if not isinstance(assignments, np.ndarray):
        assignments = np.asarray(assignments)

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
