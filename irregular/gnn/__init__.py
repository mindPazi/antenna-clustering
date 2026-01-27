"""
GNN-based Antenna Array Clustering Module

Two training modes:
    - train_ura_clustering(): Full physics-informed (may need tuning)
    - train_ura_clustering_simple(): Simpler, more stable
"""

from .config import URAConfig
from .graph import create_ura_graph, compute_mutual_coupling
from .model import URAClusteringGNN
from .losses import (
    mincut_loss,
    orthogonality_loss,
    coupling_mincut_loss,
    balance_loss,
    contiguity_loss,
    entropy_loss,
    anti_collapse_loss,
    total_loss,
)
from .train import train_ura_clustering, train_ura_clustering_simple
from .utils import assignments_to_antenna_format

__all__ = [
    "URAConfig",
    "create_ura_graph",
    "compute_mutual_coupling",
    "URAClusteringGNN",
    "mincut_loss",
    "orthogonality_loss",
    "coupling_mincut_loss",
    "balance_loss",
    "contiguity_loss",
    "entropy_loss",
    "anti_collapse_loss",
    "total_loss",
    "train_ura_clustering",
    "train_ura_clustering_simple",
    "assignments_to_antenna_format",
]