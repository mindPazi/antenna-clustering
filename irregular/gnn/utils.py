"""
Utility functions for GNN-based antenna clustering.
"""

import numpy as np


def _lattice_indices(n: int) -> np.ndarray:
    """
    Reproduce the 1D lattice index convention used in antenna_physics.py.
    """
    if n % 2 == 1:
        return np.arange(-(n - 1) / 2, (n - 1) / 2 + 1)
    return np.arange(-n / 2 + 1, n / 2 + 1)


def assignments_to_antenna_format(assignments, grid_shape=(16, 16)):
    """
    Convert flat cluster assignments to antenna array format.

    Compatible with AntennaArray.evaluate_clustering() method.

    Args:
        assignments: (N,) cluster labels for flattened array
        grid_shape: (Nz, Ny) shape of the antenna grid

    Returns:
        List of K arrays with shape (L_k, 2) containing [NN, MM] indices
        compatible with AntennaArray.index_to_position_cluster().
    """
    if not isinstance(assignments, np.ndarray):
        assignments = np.asarray(assignments)

    Nz, Ny = grid_shape
    num_clusters = assignments.max() + 1
    expected_size = Nz * Ny
    if assignments.size != expected_size:
        raise ValueError(
            f"assignments has size {assignments.size}, expected {expected_size} "
            f"for grid_shape={grid_shape}"
        )

    # Map 0-based row/col grid indices to the antenna lattice NN/MM indices.
    # This must match AntennaArray._generate_lattice().
    N_vals = _lattice_indices(Ny)
    M_vals = _lattice_indices(Nz)

    clusters = []
    for k in range(num_clusters):
        flat_indices = np.where(assignments == k)[0]

        # Skip empty clusters to avoid division by zero in evaluate_clustering
        if len(flat_indices) == 0:
            continue

        # Convert flat index to 2D grid indices
        # Assuming row-major (C) ordering: flat_idx = row * Ny + col
        rows = flat_indices // Ny
        cols = flat_indices % Ny

        # Convert to lattice coordinates (NN, MM) expected by antenna_physics.
        nn = N_vals[cols].astype(int)
        mm = M_vals[rows].astype(int)
        cluster_coords = np.stack([nn, mm], axis=1)
        clusters.append(cluster_coords)

    return clusters
