"""
GNN-based Antenna Array Clustering Module

This module implements Graph Neural Networks for clustering Uniform Rectangular
Arrays (URAs) using unsupervised learning with MinCut optimization.

Architecture overview:
    1. Graph Construction: Convert URA grid to 8-connected graph
    2. GNN Layers: GAT with edge features (PyTorch Geometric)
    3. Clustering Head: Soft assignment via softmax
    4. Loss: MinCut + Orthogonality (no labels needed)
"""

from .config import URAConfig
from .graph import create_ura_graph
from .model import URAClusteringGNN
from .losses import mincut_loss, orthogonality_loss
from .train import train_ura_clustering
from .utils import assignments_to_antenna_format

__all__ = [
    "URAConfig",
    "create_ura_graph",
    "URAClusteringGNN",
    "mincut_loss",
    "orthogonality_loss",
    "train_ura_clustering",
    "assignments_to_antenna_format",
]
