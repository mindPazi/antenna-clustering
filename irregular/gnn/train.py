"""
Training utilities for GNN-based antenna clustering.

Provides a Trainer class that handles:
    - Model initialization
    - Graph construction from positions
    - Training loop with loss computation
    - Cluster extraction
"""

import torch
import numpy as np
from typing import Optional, Union, List
from dataclasses import dataclass

from .config import GNNConfig, GraphConfig, TrainingConfig
from .model import AntennaClusteringGNN
from .graph import GraphBuilder, normalized_adjacency
from .losses import ClusteringLoss
from .utils import normalize_positions, get_device


@dataclass
class TrainingResult:
    """Container for training results."""
    cluster_assignments: np.ndarray  # (N,) hard cluster labels
    soft_assignments: np.ndarray     # (N, K) cluster probabilities
    loss_history: List[float]        # Loss per epoch
    final_loss: float


class Trainer:
    """
    Trainer for GNN-based antenna array clustering.

    Usage:
        trainer = Trainer(num_clusters=4)
        result = trainer.fit(positions)
        clusters = result.cluster_assignments
    """

    def __init__(
        self,
        num_clusters: int = 4,
        gnn_config: Optional[GNNConfig] = None,
        graph_config: Optional[GraphConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Args:
            num_clusters: Number of clusters K
            gnn_config: GNN architecture config
            graph_config: Graph construction config
            training_config: Training hyperparameters
        """
        self.gnn_config = gnn_config or GNNConfig(num_clusters=num_clusters)
        self.gnn_config.num_clusters = num_clusters

        self.graph_config = graph_config or GraphConfig()
        self.training_config = training_config or TrainingConfig()

        self.model: Optional[AntennaClusteringGNN] = None
        self.graph_builder = GraphBuilder(self.graph_config)

        # Cached graph data
        self._edge_index: Optional[torch.Tensor] = None
        self._adj: Optional[torch.Tensor] = None
        self._deg: Optional[torch.Tensor] = None

    def fit(
        self,
        positions: Union[np.ndarray, torch.Tensor],
        coupling_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> TrainingResult:
        """
        Train the GNN model on antenna positions.

        Args:
            positions: (N, 2) array of antenna (x, y) positions
            coupling_matrix: (N, N) optional mutual coupling matrix

        Returns:
            TrainingResult with cluster assignments and training info
        """
        # Setup device
        device = get_device(self.training_config.device)

        # Prepare data
        positions = self._prepare_positions(positions, device)

        # Build graph
        self._build_graph(positions, coupling_matrix, device)

        # Initialize model
        self.model = AntennaClusteringGNN(self.gnn_config).to(device)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay
        )
        criterion = ClusteringLoss(
            lambda_ortho=self.training_config.lambda_ortho,
            lambda_entropy=self.training_config.lambda_entropy
        )

        # Training loop
        loss_history = self._train_loop(
            positions, optimizer, criterion, device
        )

        # Extract results
        self.model.eval()
        with torch.no_grad():
            z, _ = self.model(positions, self._edge_index, self._adj_norm)
            clusters = z.argmax(dim=-1).cpu().numpy()
            soft_assignments = z.cpu().numpy()

        return TrainingResult(
            cluster_assignments=clusters,
            soft_assignments=soft_assignments,
            loss_history=loss_history,
            final_loss=loss_history[-1] if loss_history else float('inf')
        )

    def _prepare_positions(
        self,
        positions: Union[np.ndarray, torch.Tensor],
        device: torch.device
    ) -> torch.Tensor:
        """Convert and normalize positions."""
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()

        positions = normalize_positions(positions)
        return positions.to(device)

    def _build_graph(
        self,
        positions: torch.Tensor,
        coupling_matrix: Optional[Union[np.ndarray, torch.Tensor]],
        device: torch.device
    ):
        """Construct graph from positions using GraphBuilder."""
        # Build graph using GraphBuilder
        edge_index, adj, deg, edge_attr = self.graph_builder.build_graph(
            positions, coupling_matrix
        )

        # Move to device
        self._edge_index = edge_index.to(device)
        self._adj = adj.to(device)
        self._deg = deg.to(device)
        self._edge_attr = edge_attr.to(device) if edge_attr is not None else None

        # Compute normalized adjacency for GCN layers
        self._adj_norm = normalized_adjacency(self._adj, self._deg)

    def _train_loop(
        self,
        positions: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: ClusteringLoss,
        device: torch.device
    ) -> List[float]:
        """Execute training loop."""
        self.model.train()
        loss_history = []

        for epoch in range(self.training_config.epochs):
            optimizer.zero_grad()

            # Forward pass
            z, _ = self.model(positions, self._edge_index, self._adj_norm)

            # Compute loss
            loss = criterion(z, self._adj, self._deg)

            # Backward pass
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            # Logging
            if self.training_config.verbose > 0:
                if (epoch + 1) % self.training_config.verbose == 0:
                    print(f"Epoch {epoch + 1}/{self.training_config.epochs}: "
                          f"Loss = {loss_val:.4f}")

        return loss_history

    def predict(
        self,
        positions: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Get cluster assignments for new positions (after training).

        Args:
            positions: (N, 2) antenna positions

        Returns:
            clusters: (N,) cluster labels
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        device = next(self.model.parameters()).device
        positions = self._prepare_positions(positions, device)

        self.model.eval()
        with torch.no_grad():
            z, _ = self.model(positions, self._edge_index, self._adj_norm)
            return z.argmax(dim=-1).cpu().numpy()


def train_clustering(
    positions: Union[np.ndarray, torch.Tensor],
    num_clusters: int = 4,
    k_neighbors: int = 8,
    epochs: int = 500,
    lr: float = 1e-3,
    verbose: int = 50
) -> np.ndarray:
    """
    Convenience function for quick clustering.

    Args:
        positions: (N, 2) antenna positions
        num_clusters: Number of clusters K
        k_neighbors: Neighbors for graph construction
        epochs: Training iterations
        lr: Learning rate
        verbose: Print progress every N epochs

    Returns:
        clusters: (N,) array of cluster labels
    """
    trainer = Trainer(
        num_clusters=num_clusters,
        graph_config=GraphConfig(k_neighbors=k_neighbors),
        training_config=TrainingConfig(epochs=epochs, lr=lr, verbose=verbose)
    )
    result = trainer.fit(positions)
    return result.cluster_assignments
