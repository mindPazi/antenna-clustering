"""
GNN-based Antenna Array Clustering Module

This module implements Graph Neural Networks for clustering irregular antenna arrays
using unsupervised learning with MinCut optimization.

Architecture overview:
    1. Graph Construction: Convert antenna positions to k-NN graph
    2. GNN Layers: GAT/GCN for learning node embeddings
    3. Clustering Head: Soft assignment via softmax
    4. Loss: MinCut + Orthogonality (no labels needed)
"""

from .config import GNNConfig, GraphConfig, TrainingConfig, ClusteringConfig
from .graph import GraphBuilder, normalized_adjacency
from .layers import GCNLayer, GATLayer, EdgeConvLayer
from .model import AntennaClusteringGNN, AntennaClusteringGNNWithEdgeFeatures
from .losses import (
    mincut_loss,
    orthogonality_loss,
    entropy_loss,
    cluster_size_loss,
    total_loss,
    ClusteringLoss
)
from .train import Trainer, TrainingResult, train_clustering
from .utils import (
    normalize_positions,
    get_hard_assignments,
    get_device,
    cluster_sizes,
    cluster_to_list,
    assignments_to_antenna_format,
    compute_clustering_metrics,
    set_seed
)

__all__ = [
    # Config
    "GNNConfig",
    "GraphConfig",
    "TrainingConfig",
    "ClusteringConfig",
    # Graph
    "GraphBuilder",
    "normalized_adjacency",
    # Layers
    "GCNLayer",
    "GATLayer",
    "EdgeConvLayer",
    # Model
    "AntennaClusteringGNN",
    "AntennaClusteringGNNWithEdgeFeatures",
    # Losses
    "mincut_loss",
    "orthogonality_loss",
    "entropy_loss",
    "cluster_size_loss",
    "total_loss",
    "ClusteringLoss",
    # Training
    "Trainer",
    "TrainingResult",
    "train_clustering",
    # Utils
    "normalize_positions",
    "get_hard_assignments",
    "get_device",
    "cluster_sizes",
    "cluster_to_list",
    "assignments_to_antenna_format",
    "compute_clustering_metrics",
    "set_seed",
]
