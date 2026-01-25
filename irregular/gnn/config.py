"""
Configuration dataclasses for GNN-based antenna clustering.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GNNConfig:
    """
    GNN architecture configuration.

    Attributes:
        in_dim: Input feature dimension (2 for x,y positions)
        hidden_dim: Hidden layer dimension
        num_clusters: K, number of output clusters
        num_layers: Number of GNN layers
        heads: Number of attention heads (GAT only)
        dropout: Dropout probability for regularization
        layer_type: Type of GNN layer ('gat' or 'gcn')
        use_edge_features: Whether to incorporate edge features (distance, coupling)
    """
    in_dim: int = 2
    hidden_dim: int = 64
    num_clusters: int = 4
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    layer_type: Literal["gat", "gcn"] = "gat"
    use_edge_features: bool = False
    edge_dim: int = 1  # Dimension of edge features if used


@dataclass
class GraphConfig:
    """
    Graph construction configuration.

    Attributes:
        k_neighbors: Number of neighbors for k-NN graph
        connection_type: Strategy for edge creation ('knn', 'radius', 'coupling')
        radius: Connection radius for radius-based graphs (in normalized units)
        coupling_threshold: Threshold for mutual coupling-based edges
        add_self_loops: Whether to add self-loops to the adjacency matrix
    """
    k_neighbors: int = 8
    connection_type: Literal["knn", "radius", "coupling"] = "knn"
    radius: float = 0.5
    coupling_threshold: float = 0.1
    add_self_loops: bool = True


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.

    Attributes:
        epochs: Number of training iterations
        lr: Learning rate for Adam optimizer
        weight_decay: L2 regularization coefficient
        lambda_ortho: Weight for orthogonality loss
        lambda_entropy: Weight for entropy regularization
        device: Compute device ('cuda' or 'cpu')
        verbose: Print training progress every N epochs (0 = silent)
    """
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 5e-4
    lambda_ortho: float = 1.0
    lambda_entropy: float = 0.0
    device: str = "auto"  # 'auto', 'cuda', or 'cpu'
    verbose: int = 50  # Print every N epochs


@dataclass
class ClusteringConfig:
    """
    Complete configuration combining all sub-configs.
    """
    gnn: GNNConfig = field(default_factory=GNNConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
