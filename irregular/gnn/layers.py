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


def scatter_add_native(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None
) -> torch.Tensor:
    """
    Native PyTorch implementation of scatter_add.

    Sums values from src at indices specified by index along dimension dim.

    Args:
        src: Source tensor with values to scatter
        index: Index tensor (same size as src along dim)
        dim: Dimension along which to scatter
        dim_size: Size of output along dim (default: max(index) + 1)

    Returns:
        Output tensor with scattered values
    """
    if dim_size is None:
        dim_size = index.max().item() + 1 if index.numel() > 0 else 0

    # Create output shape
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)

    # Expand index to match src shape
    if src.dim() > 1:
        # For multi-dimensional src, expand index
        index_expanded = index.view(-1, *([1] * (src.dim() - 1)))
        index_expanded = index_expanded.expand_as(src)
    else:
        index_expanded = index

    out.scatter_add_(dim, index_expanded, src)

    return out


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
        # 1. Linear transform: XW
        support = torch.mm(x, self.weight)

        # 2. Neighborhood aggregation: A_norm @ (XW)
        out = torch.mm(adj_norm, support)

        # 3. Add bias
        if self.bias is not None:
            out = out + self.bias

        return out


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
        N = x.shape[0]
        H = self.heads
        E = edge_index.shape[1]

        # Handle edge case of no edges
        if E == 0:
            # No message passing possible, just return transformed features
            # Shape: (N, in_features) @ (H, in_features, out_features)
            # Broadcast: (N, 1, in_features) @ (H, in_features, out_features) -> (N, H, out_features)
            h = torch.einsum('ni,hio->nho', x, self.W)
            if self.concat:
                return h.view(N, H * self.out_features)
            else:
                return h.mean(dim=1)

        # 1. Linear transform: Wh_i for all nodes
        # x: (N, in_features), W: (H, in_features, out_features)
        # h: (N, H, out_features)
        h = torch.einsum('ni,hio->nho', x, self.W)

        # 2. Compute attention coefficients
        # Source and target attention scores
        # a_l: (H, out_features, 1), h: (N, H, out_features)
        # attn_l: (N, H)
        attn_l = torch.einsum('nho,hol->nh', h, self.a_l).squeeze(-1)
        attn_r = torch.einsum('nho,hol->nh', h, self.a_r).squeeze(-1)

        # Get source and target indices
        src, dst = edge_index[0], edge_index[1]

        # Compute e_ij = LeakyReLU(a_l^T Wh_i + a_r^T Wh_j)
        # e: (E, H)
        e = attn_l[src] + attn_r[dst]
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # 3. Normalize with softmax over neighbors
        # Compute softmax per destination node
        alpha = self._sparse_softmax(e, dst, N)

        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 4. Aggregate: h'_i = sum_j alpha_ij * Wh_j
        # h[src]: (E, H, out_features), alpha: (E, H)
        # Weighted messages
        msg = h[src] * alpha.unsqueeze(-1)  # (E, H, out_features)

        # Aggregate at destination nodes
        out = scatter_add_native(msg, dst, dim=0, dim_size=N)  # (N, H, out_features)

        # 5. Concatenate or average heads
        if self.concat:
            out = out.view(N, H * self.out_features)
        else:
            out = out.mean(dim=1)

        return out

    def _sparse_softmax(
        self,
        e: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute softmax over edges grouped by destination node.

        Args:
            e: (E, H) attention logits
            index: (E,) destination node indices
            num_nodes: N, total number of nodes

        Returns:
            alpha: (E, H) normalized attention weights
        """
        # Subtract global max for numerical stability
        e_exp = torch.exp(e - e.max())

        # Sum exp per destination
        e_sum = scatter_add_native(e_exp, index, dim=0, dim_size=num_nodes)
        e_sum = e_sum[index]

        # Normalize
        alpha = e_exp / (e_sum + 1e-8)

        return alpha


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
        N = x.shape[0]
        E = edge_index.shape[1]

        if E == 0:
            # No edges, return zero output
            return torch.zeros(N, self.out_features, device=x.device)

        src, dst = edge_index[0], edge_index[1]

        # Get source and destination node features
        x_src = x[src]  # (E, in_features)
        x_dst = x[dst]  # (E, in_features)

        # Concatenate node features with edge features
        # [h_i || h_j || e_ij]
        msg_input = torch.cat([x_src, x_dst, edge_attr], dim=1)  # (E, 2*in + edge)

        # Apply MLP to compute messages
        msg = self.mlp(msg_input)  # (E, out_features)

        # Aggregate messages at destination nodes
        out = scatter_add_native(msg, dst, dim=0, dim_size=N)  # (N, out_features)

        return out
