"""
Main GNN model for antenna array clustering.

Architecture:
    Input(N, 2) -> GAT/GCN layers -> Embeddings(N, d) -> Linear+Softmax -> Z(N, K)

The model outputs soft cluster assignments Z where Z_ik = P(node i in cluster k).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import GNNConfig
from .layers import GCNLayer, GATLayer, EdgeConvLayer


class AntennaClusteringGNN(nn.Module):
    """
    GNN model for unsupervised clustering of irregular antenna arrays.

    Pipeline:
        1. Stack of GAT/GCN layers to learn node embeddings
        2. Final linear layer to map embeddings to K cluster logits
        3. Softmax to get soft cluster probabilities
    """

    def __init__(self, config: Optional[GNNConfig] = None):
        """
        Args:
            config: GNN architecture configuration
        """
        super().__init__()
        self.config = config or GNNConfig()

        self._build_layers()

    def _build_layers(self):
        """Construct GNN layers based on configuration."""
        cfg = self.config

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input dimension for first layer
        in_dim = cfg.in_dim

        # Build GNN layers
        for i in range(cfg.num_layers):
            # Last layer: single head, no concat
            is_last = (i == cfg.num_layers - 1)

            if cfg.layer_type == "gat":
                heads = 1 if is_last else cfg.heads
                out_dim = cfg.hidden_dim
                layer = GATLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    heads=heads,
                    concat=not is_last,
                    dropout=cfg.dropout
                )
                # Update input dim for next layer
                in_dim = out_dim if is_last else out_dim * heads
            else:  # gcn
                out_dim = cfg.hidden_dim
                layer = GCNLayer(
                    in_features=in_dim,
                    out_features=out_dim
                )
                in_dim = out_dim

            self.layers.append(layer)

            # Optional: layer normalization
            if not is_last:
                self.norms.append(nn.LayerNorm(in_dim))

        # Final embedding dimension
        self.embed_dim = in_dim

        # Output layer: embeddings -> cluster logits
        self.classifier = nn.Linear(self.embed_dim, cfg.num_clusters)

        # Dropout
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj_norm: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (N, in_dim) input node features (positions)
            edge_index: (2, E) edges in COO format
            adj_norm: (N, N) normalized adjacency (for GCN)
            edge_attr: (E, edge_dim) edge features (optional)
            return_embeddings: Also return intermediate embeddings

        Returns:
            z: (N, K) soft cluster assignment probabilities
            h: (N, embed_dim) node embeddings (if return_embeddings=True)
        """
        h = x

        # Pass through GNN layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GATLayer):
                h = layer(h, edge_index, edge_attr)
            else:  # GCN
                h = layer(h, adj_norm)

            # Activation + dropout (except last layer)
            if i < len(self.layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
                if i < len(self.norms):
                    h = self.norms[i](h)

        # Final activation
        h = F.elu(h)

        # Cluster probabilities via softmax
        logits = self.classifier(h)
        z = F.softmax(logits, dim=-1)

        if return_embeddings:
            return z, h
        return z, None

    def get_hard_assignments(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert soft probabilities to hard cluster assignments.

        c_i = argmax_k z_ik

        Args:
            z: (N, K) soft assignment probabilities

        Returns:
            c: (N,) hard cluster labels in {0, ..., K-1}
        """
        return z.argmax(dim=-1)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AntennaClusteringGNNWithEdgeFeatures(AntennaClusteringGNN):
    """
    Extended model that incorporates physics-informed edge features.

    Edge features can include:
        - Euclidean distance between antennas
        - Mutual coupling magnitude
        - Mutual coupling phase
    """

    def __init__(self, config: Optional[GNNConfig] = None):
        # Ensure edge features are enabled
        if config is None:
            config = GNNConfig(use_edge_features=True)
        else:
            config.use_edge_features = True

        super().__init__(config)

    def _build_layers(self):
        """Build layers with edge feature support."""
        cfg = self.config

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(cfg.edge_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )

        # Rest of the architecture
        super()._build_layers()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj_norm: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with edge feature processing."""
        # Encode edge features
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Standard forward pass
        return super().forward(
            x, edge_index, adj_norm, edge_attr, return_embeddings
        )
