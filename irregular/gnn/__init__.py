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

from .config import GNNConfig, TrainingConfig
from .graph import GraphBuilder
from .layers import GCNLayer, GATLayer
from .model import AntennaClusteringGNN
from .losses import mincut_loss, orthogonality_loss, entropy_loss, total_loss
from .train import Trainer
from .utils import normalize_positions, get_hard_assignments

__all__ = [
    # Config
    "GNNConfig",
    "TrainingConfig",
    # Graph
    "GraphBuilder",
    # Layers
    "GCNLayer",
    "GATLayer",
    # Model
    "AntennaClusteringGNN",
    # Losses
    "mincut_loss",
    "orthogonality_loss",
    "entropy_loss",
    "total_loss",
    # Training
    "Trainer",
    # Utils
    "normalize_positions",
    "get_hard_assignments",
]
