"""
Graph construction for Uniform Rectangular Array.

Creates graph from URA with fixed spacing using grid connectivity
and mutual coupling as edge weights.
"""

import torch
import numpy as np


def compute_mutual_coupling(config):
    """
    Compute mutual coupling matrix for URA.
    
    Uses simplified analytical model. For realistic results,
    replace with full-wave simulation data (HFSS, CST).
    
    Args:
        config: URAConfig object
        
    Returns:
        M: (N, N) complex mutual coupling matrix
        positions: (N, 2) element positions in wavelengths
    """
    N = config.N
    rows, cols = config.rows, config.cols
    dx, dy = config.dx, config.dy
    
    # Element positions in wavelengths
    positions = np.zeros((N, 2))
    for n in range(rows):
        for m in range(cols):
            idx = n * cols + m
            positions[idx] = [m * dx, n * dy]
    
    # Compute coupling matrix
    M = np.zeros((N, N), dtype=complex)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i, j] = 1.0  # Self-coupling normalized to 1
            else:
                # Distance in wavelengths
                r = np.linalg.norm(positions[i] - positions[j])
                
                # Simplified coupling model: M ~ exp(-jkr) / r
                phase = 2 * np.pi * r
                M[i, j] = np.exp(-1j * phase) / (r + 0.1)
                
                # Edge effect: corner/edge elements couple differently
                mi, ni = i % cols, i // cols
                mj, nj = j % cols, j // cols
                
                # Reduce coupling for edge elements (realistic effect)
                edge_factor = 1.0
                is_edge_i = (mi == 0 or mi == cols-1 or ni == 0 or ni == rows-1)
                is_edge_j = (mj == 0 or mj == cols-1 or nj == 0 or nj == rows-1)
                
                if is_edge_i:
                    edge_factor *= 0.85
                if is_edge_j:
                    edge_factor *= 0.85
                    
                # Corner elements have even more different coupling
                is_corner_i = (mi in [0, cols-1]) and (ni in [0, rows-1])
                is_corner_j = (mj in [0, cols-1]) and (nj in [0, rows-1])
                
                if is_corner_i:
                    edge_factor *= 0.9
                if is_corner_j:
                    edge_factor *= 0.9
                
                M[i, j] *= edge_factor
    
    return M, positions


def create_ura_graph(config, connectivity='8-connected', use_coupling=True):
    """
    Create graph from URA with fixed spacing.

    Args:
        config: URAConfig object
        connectivity: '4-connected' or '8-connected'
        use_coupling: if True, include coupling magnitude in edge features
                      and return coupling-weighted adjacency matrix

    Returns:
        positions: (N, 2) tensor of physical positions
        edge_index: (2, E) tensor of edges
        edge_attr: (E, num_features) tensor of edge features
        W: (N, N) coupling-weighted adjacency matrix (if use_coupling=True)
           or None (if use_coupling=False)
    """
    rows, cols = config.rows, config.cols
    dx, dy = config.dx, config.dy

    # Compute mutual coupling if requested
    if use_coupling:
        M, pos_np = compute_mutual_coupling(config)
        coupling_magnitude = np.abs(M)
        positions = torch.tensor(pos_np, dtype=torch.float32)
    else:
        # Generate grid positions (fixed spacing)
        m_indices = torch.arange(cols).float()
        n_indices = torch.arange(rows).float()
        mm, nn = torch.meshgrid(m_indices, n_indices, indexing='xy')
        positions = torch.stack([
            mm.flatten() * dx,
            nn.flatten() * dy
        ], dim=1)
        coupling_magnitude = None

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

                    # Physical offsets
                    delta_x = dm * dx
                    delta_y = dn * dy
                    dist = np.sqrt(delta_x**2 + delta_y**2)
                    
                    if use_coupling:
                        coupling = coupling_magnitude[idx, idx2]
                        edge_features.append([delta_x, delta_y, dist, coupling])
                    else:
                        edge_features.append([delta_x, delta_y, dist])

    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    # Return coupling-weighted adjacency if requested
    if use_coupling:
        W = torch.tensor(coupling_magnitude, dtype=torch.float32)
        return positions, edge_index, edge_attr, W
    else:
        return positions, edge_index, edge_attr, None