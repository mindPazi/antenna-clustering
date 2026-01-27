"""
Graph construction for Uniform Rectangular Array.

Creates graph from URA with fixed spacing using grid connectivity.
"""

import torch
import numpy as np


def create_ura_graph(config, connectivity='8-connected'):
    """
    Create graph from URA with fixed spacing.

    Args:
        config: URAConfig object
        connectivity: '4-connected' or '8-connected'

    Returns:
        positions: (N, 2) tensor of physical positions
        edge_index: (2, E) tensor of edges
        edge_attr: (E, num_features) tensor of edge features
    """
    rows, cols = config.rows, config.cols

    # Generate grid positions (fixed spacing)
    m_indices = torch.arange(cols).float()
    n_indices = torch.arange(rows).float()
    mm, nn = torch.meshgrid(m_indices, n_indices, indexing='xy')

    # Physical positions
    positions = torch.stack([
        mm.flatten() * config.dx,  # x in wavelengths
        nn.flatten() * config.dy   # y in wavelengths
    ], dim=1)  # Shape: (N, 2)

    # Build edges based on grid connectivity
    edges = []
    edge_features = []

    for m in range(cols):
        for n in range(rows):
            idx = n * cols + m

            # Define neighbor offsets
            if connectivity == '4-connected':
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            else:  # 8-connected
                offsets = [(dm, dn) for dm in [-1, 0, 1]
                           for dn in [-1, 0, 1] if (dm, dn) != (0, 0)]

            for dm, dn in offsets:
                m2, n2 = m + dm, n + dn
                if 0 <= m2 < cols and 0 <= n2 < rows:
                    idx2 = n2 * cols + m2
                    edges.append([idx, idx2])

                    # Edge features: [dm, dn, normalized_distance]
                    dist = np.sqrt(dm**2 + dn**2)
                    edge_features.append([dm, dn, dist])

    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return positions, edge_index, edge_attr
