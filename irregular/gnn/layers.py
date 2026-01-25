"""
GNN layer implementations for antenna clustering.

Implements message passing layers:
    - GCN (Graph Convolutional Network): Simple aggregation with learned weights
    - GAT (Graph Attention Network): Attention-weighted aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.

    Implements: H^{l+1} = sigma(D^{-1/2} A D^{-1/2} H^{l} W^{l})

    Each node aggregates normalized neighbor features, applies linear transform,
    then non-linearity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        """
        Args:
            in_features: Input dimension per node
            out_features: Output dimension per node
            bias: Include learnable bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix W
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights using Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (N, in_features) node features
            adj_norm: (N, N) normalized adjacency matrix

        Returns:
            out: (N, out_features) updated node features
        """
        # TODO: Implement GCN forward pass
        # 1. Linear transform: XW
        # 2. Neighborhood aggregation: A_norm @ (XW)
        # 3. Add bias
        raise NotImplementedError


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.

    Learns attention weights to focus on relevant neighbors during aggregation.
    Supports multi-head attention for richer representations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2
    ):
        """
        Args:
            in_features: Input dimension per node
            out_features: Output dimension per head
            heads: Number of attention heads
            concat: Concatenate heads (True) or average (False)
            dropout: Dropout on attention weights
            negative_slope: LeakyReLU negative slope
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Linear transform for node features: W
        self.W = nn.Parameter(torch.empty(heads, in_features, out_features))

        # Attention mechanism parameters: a = [a_l || a_r]
        self.a_l = nn.Parameter(torch.empty(heads, out_features, 1))
        self.a_r = nn.Parameter(torch.empty(heads, out_features, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize using Xavier/Glorot."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with attention mechanism.

        Args:
            x: (N, in_features) node features
            edge_index: (2, E) edge indices in COO format
            edge_attr: (E, edge_dim) optional edge features

        Returns:
            out: (N, heads * out_features) if concat else (N, out_features)
        """
        # TODO: Implement GAT forward pass
        # 1. Linear transform: Wh_i for all nodes
        # 2. Compute attention coefficients: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # 3. Normalize with softmax over neighbors: alpha_ij
        # 4. Aggregate: h'_i = sum_j alpha_ij * Wh_j
        # 5. Concatenate or average heads
        raise NotImplementedError

    def _compute_attention(
        self,
        h_l: torch.Tensor,
        h_r: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention coefficients for each edge.

        Args:
            h_l: (N, heads, out_features) source node representations
            h_r: (N, heads, out_features) target node representations
            edge_index: (2, E) edges

        Returns:
            alpha: (E, heads) attention weights
        """
        # TODO: Implement attention computation
        raise NotImplementedError


class EdgeConvLayer(nn.Module):
    """
    Edge-conditioned convolution layer.

    Incorporates edge features (distance, coupling) into message passing.
    Useful for physics-informed learning.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int
    ):
        """
        Args:
            in_features: Node feature dimension
            out_features: Output dimension
            edge_features: Edge feature dimension
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features

        # MLP for combining node and edge features
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_features + edge_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with edge features.

        Args:
            x: (N, in_features) node features
            edge_index: (2, E) edges
            edge_attr: (E, edge_features) edge features

        Returns:
            out: (N, out_features) updated features
        """
        # TODO: Implement edge-conditioned message passing
        # h'_i = sum_j MLP([h_i || h_j || e_ij])
        raise NotImplementedError
