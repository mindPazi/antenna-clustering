"""
Graph construction utilities for antenna array clustering.

Converts antenna positions into graph structures (nodes, edges, adjacency matrix).
Supports multiple edge creation strategies: k-NN, radius-based, mutual coupling.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
from scipy.spatial import cKDTree
from .config import GraphConfig


class GraphBuilder:
    """
    Builds graph representations from antenna positions.

    The antenna array is represented as:
        - Nodes: Individual antenna elements
        - Edges: Connections based on proximity or coupling
        - Node features: Positions (and optionally other attributes)
        - Edge features: Distance, mutual coupling magnitude/phase
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()

    def build_knn_edges(
        self,
        positions: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Build edge index using k-nearest neighbors.

        Args:
            positions: (N, 2) tensor of antenna positions
            k: Number of neighbors (uses config default if None)

        Returns:
            edge_index: (2, E) tensor of edge indices in COO format
        """
        k = k or self.config.k_neighbors
        n = positions.shape[0]

        # Convert to numpy for scipy KDTree
        pos_np = positions.detach().cpu().numpy()

        # Build KD-tree for efficient neighbor search
        tree = cKDTree(pos_np)

        # Query k+1 neighbors (includes self)
        _, indices = tree.query(pos_np, k=min(k + 1, n))

        # Build edge list (excluding self-loops here, added later if needed)
        source = []
        target = []
        for i in range(n):
            for j in indices[i]:
                if i != j:  # Exclude self-loops
                    source.append(i)
                    target.append(j)

        edge_index = torch.tensor([source, target], dtype=torch.long,
                                   device=positions.device)
        return edge_index

    def build_radius_edges(
        self,
        positions: torch.Tensor,
        radius: Optional[float] = None
    ) -> torch.Tensor:
        """
        Build edge index using radius-based connectivity (epsilon-ball).

        Connects all nodes within distance `radius` of each other.

        Args:
            positions: (N, 2) tensor of antenna positions
            radius: Connection radius (uses config default if None)

        Returns:
            edge_index: (2, E) tensor of edge indices
        """
        radius = radius or self.config.radius

        # Convert to numpy for scipy KDTree
        pos_np = positions.detach().cpu().numpy()

        # Build KD-tree
        tree = cKDTree(pos_np)

        # Find all pairs within radius
        pairs = tree.query_pairs(radius, output_type='ndarray')

        # Create bidirectional edges
        if len(pairs) > 0:
            source = np.concatenate([pairs[:, 0], pairs[:, 1]])
            target = np.concatenate([pairs[:, 1], pairs[:, 0]])
            edge_index = torch.tensor([source, target], dtype=torch.long,
                                       device=positions.device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long,
                                      device=positions.device)

        return edge_index

    def build_coupling_edges(
        self,
        coupling_matrix: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Build edge index based on mutual coupling magnitude.

        Connects antennas with coupling above threshold.

        Args:
            coupling_matrix: (N, N) complex mutual coupling matrix M
            threshold: Minimum |M_ij| to create edge

        Returns:
            edge_index: (2, E) tensor of edge indices
        """
        threshold = threshold or self.config.coupling_threshold

        # Compute magnitude of coupling
        coupling_mag = torch.abs(coupling_matrix)

        # Find edges where coupling exceeds threshold (excluding diagonal)
        n = coupling_matrix.shape[0]
        mask = (coupling_mag > threshold) & ~torch.eye(n, dtype=torch.bool,
                                                        device=coupling_matrix.device)

        # Get edge indices
        edge_index = torch.nonzero(mask, as_tuple=False).T

        return edge_index

    def compute_adjacency_matrix(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        symmetric: bool = True
    ) -> torch.Tensor:
        """
        Convert edge index to dense adjacency matrix A.

        Args:
            edge_index: (2, E) COO format edges
            num_nodes: N, number of nodes
            symmetric: Symmetrize the adjacency matrix

        Returns:
            adj: (N, N) adjacency matrix
        """
        device = edge_index.device
        adj = torch.zeros((num_nodes, num_nodes), device=device)

        if edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = 1.0

        if symmetric:
            adj = (adj + adj.T) / 2
            # Ensure binary values
            adj = (adj > 0).float()

        return adj

    def compute_degree_matrix(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute degree matrix D from adjacency matrix.

        D_ii = sum_j A_ij (number of neighbors)

        Args:
            adj: (N, N) adjacency matrix

        Returns:
            deg: (N, N) diagonal degree matrix
        """
        degrees = adj.sum(dim=1)
        deg = torch.diag(degrees)
        return deg

    def compute_edge_features(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        coupling_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics-informed edge features.

        Features per edge (i, j):
            - Euclidean distance d_ij
            - Mutual coupling magnitude |M_ij| (if provided)
            - Mutual coupling phase angle(M_ij) (if provided)

        Args:
            positions: (N, 2) node positions
            edge_index: (2, E) edges
            coupling_matrix: (N, N) optional mutual coupling matrix

        Returns:
            edge_attr: (E, F) edge feature matrix
        """
        if edge_index.shape[1] == 0:
            # No edges
            num_features = 1 if coupling_matrix is None else 3
            return torch.zeros((0, num_features), device=positions.device)

        src, dst = edge_index[0], edge_index[1]

        # Euclidean distance
        pos_src = positions[src]
        pos_dst = positions[dst]
        distances = torch.norm(pos_src - pos_dst, dim=1, keepdim=True)

        if coupling_matrix is None:
            return distances

        # Mutual coupling features
        coupling_vals = coupling_matrix[src, dst]
        coupling_mag = torch.abs(coupling_vals).unsqueeze(1)
        coupling_phase = torch.angle(coupling_vals).unsqueeze(1)

        # Concatenate all features
        edge_attr = torch.cat([distances, coupling_mag, coupling_phase], dim=1)

        return edge_attr

    def build_graph(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        coupling_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build complete graph representation from antenna positions.

        This is the main entry point for graph construction.

        Args:
            positions: (N, 2) antenna positions
            coupling_matrix: (N, N) optional mutual coupling matrix

        Returns:
            Tuple of:
                - edge_index: (2, E) edge indices
                - adj: (N, N) adjacency matrix
                - deg: (N, N) degree matrix
                - edge_attr: (E, F) edge features (or None)
        """
        # Convert numpy to torch if needed
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()

        if coupling_matrix is not None and isinstance(coupling_matrix, np.ndarray):
            coupling_matrix = torch.from_numpy(coupling_matrix)

        n = positions.shape[0]
        device = positions.device

        # Build edges based on connection type
        if self.config.connection_type == "knn":
            edge_index = self.build_knn_edges(positions)
        elif self.config.connection_type == "radius":
            edge_index = self.build_radius_edges(positions)
        elif self.config.connection_type == "coupling":
            if coupling_matrix is None:
                raise ValueError("Coupling matrix required for coupling-based edges")
            edge_index = self.build_coupling_edges(coupling_matrix)
        else:
            raise ValueError(f"Unknown connection type: {self.config.connection_type}")

        # Compute adjacency matrix
        adj = self.compute_adjacency_matrix(edge_index, n, symmetric=True)

        # Add self-loops if configured
        if self.config.add_self_loops:
            adj = adj + torch.eye(n, device=device)

        # Compute degree matrix
        deg = self.compute_degree_matrix(adj)

        # Compute edge features
        edge_attr = self.compute_edge_features(positions, edge_index, coupling_matrix)

        return edge_index, adj, deg, edge_attr


def normalized_adjacency(adj: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized adjacency: D^{-1/2} A D^{-1/2}

    Used in GCN to prevent exploding/vanishing gradients during message passing.

    Args:
        adj: (N, N) adjacency matrix (with or without self-loops)
        deg: (N, N) degree matrix

    Returns:
        norm_adj: (N, N) normalized adjacency matrix
    """
    # Get diagonal degrees
    d = torch.diag(deg)

    # Compute D^{-1/2}, handling zero degrees
    d_inv_sqrt = torch.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = torch.pow(d[mask], -0.5)

    # Create diagonal matrix
    d_inv_sqrt_mat = torch.diag(d_inv_sqrt)

    # Compute normalized adjacency: D^{-1/2} A D^{-1/2}
    norm_adj = d_inv_sqrt_mat @ adj @ d_inv_sqrt_mat

    return norm_adj
