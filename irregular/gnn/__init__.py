"""
GNN-based Antenna Array Clustering Module

Physics-informed clustering with target clustering factor.
"""

from .config import URAConfig
from .graph import create_ura_graph, compute_mutual_coupling
from .model import URAClusteringGNN
from .losses import (
    coupling_mincut_loss,
    clustering_factor_loss,
    contiguity_loss,
    entropy_loss,
    total_loss,
)
from .train import train_ura_clustering
from .utils import assignments_to_antenna_format

__all__ = [
    "URAConfig",
    "create_ura_graph",
    "compute_mutual_coupling",
    "URAClusteringGNN",
    "coupling_mincut_loss",
    "clustering_factor_loss",
    "contiguity_loss",
    "entropy_loss",
    "total_loss",
    "train_ura_clustering",
    "assignments_to_antenna_format",
]
