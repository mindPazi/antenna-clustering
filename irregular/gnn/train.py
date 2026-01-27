"""
Training for GNN-based URA clustering.
"""

import torch
import numpy as np

from .config import URAConfig
from .graph import create_ura_graph
from .model import URAClusteringGNN
from .losses import mincut_loss, orthogonality_loss


def train_ura_clustering(config, num_clusters=4, epochs=500, lr=0.001):
    """
    Train GNN for URA clustering.

    Args:
        config: URAConfig with spacing parameters
        num_clusters: K clusters
        epochs: training iterations
        lr: learning rate

    Returns:
        cluster_assignments: (N,) array of cluster labels
    """
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')

    # Create graph from URA configuration
    positions, edge_index, edge_attr = create_ura_graph(
        config, connectivity='8-connected'
    )

    # Normalize positions to [0, 1]
    positions = (positions - positions.min(0)[0]) / \
                (positions.max(0)[0] - positions.min(0)[0])

    # Move to device
    positions = positions.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    # Build adjacency matrix for loss computation
    n = config.N
    adj = torch.zeros(n, n, device=device)
    adj[edge_index[0], edge_index[1]] = 1
    deg = torch.diag(adj.sum(dim=1))

    # Initialize model
    model = URAClusteringGNN(
        in_dim=2,
        hidden_dim=64,
        num_clusters=num_clusters,
        heads=4,
        edge_dim=edge_attr.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=5e-4)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        z = model(positions, edge_index, edge_attr)

        loss_cut = mincut_loss(z, adj, deg)
        loss_ortho = orthogonality_loss(z)
        loss = loss_cut + 1.0 * loss_ortho

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

    # Extract clusters
    model.eval()
    with torch.no_grad():
        z = model(positions, edge_index, edge_attr)
        clusters = z.argmax(dim=1).cpu().numpy()

    return clusters
