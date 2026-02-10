"""
Training for GNN-based URA clustering.

Physics-informed mode with target clustering factor.
The GNN discovers the optimal number of clusters automatically.
"""

import torch
import numpy as np

from .config import URAConfig
from .graph import create_ura_graph
from .model import URAClusteringGNN
from .losses import total_loss


def train_ura_clustering(config, target_cf=3.0, num_clusters_max=None,
                         epochs=500, lr=0.001,
                         lambda_cf=10.0, lambda_contiguity=0.5,
                         lambda_entropy=0.5, lambda_balance=5.0,
                         verbose=True):
    """
    Train GNN for URA clustering with target clustering factor.

    The model uses coupling-weighted edges and discovers the optimal
    number of clusters to achieve the target clustering factor
    (average elements per cluster).  A balance loss penalizes
    imbalanced cluster sizes.

    Args:
        config: URAConfig with array parameters
        target_cf: target clustering factor (avg elements per cluster)
        num_clusters_max: upper bound on number of clusters (auto if None)
        epochs: training iterations
        lr: learning rate
        lambda_cf: weight for clustering-factor loss
        lambda_contiguity: weight for contiguity loss
        lambda_entropy: weight for entropy loss
        lambda_balance: weight for balance loss (cluster size variance)
        verbose: print training progress

    Returns:
        cluster_assignments: (N,) array of cluster labels (consecutive ints)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Upper bound on number of clusters
    if num_clusters_max is None:
        num_clusters_max = min(config.N, int(config.N / target_cf * 1.5))

    # --- graph with mutual coupling ---
    positions, edge_index, edge_attr, W = create_ura_graph(
        config, connectivity='8-connected', use_coupling=True
    )

    # Normalise positions to [0, 1]
    pos_min = positions.min(0)[0]
    pos_max = positions.max(0)[0]
    positions_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)

    positions_norm = positions_norm.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    W = W.to(device)

    # --- model ---
    model = URAClusteringGNN(
        in_dim=2,
        hidden_dim=64,
        num_clusters=num_clusters_max,
        heads=4,
        edge_dim=edge_attr.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- training loop ---
    model.train()
    best_loss = float('inf')
    best_clusters = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        z = model(positions_norm, edge_index, edge_attr)

        if torch.isnan(z).any():
            if verbose:
                print(f"WARNING: NaN at epoch {epoch+1}, reinitialising...")
            model = URAClusteringGNN(
                in_dim=2, hidden_dim=64, num_clusters=num_clusters_max,
                heads=4, edge_dim=edge_attr.shape[1],
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch)
            continue

        loss, loss_dict = total_loss(
            z, W, positions_norm,
            target_cf=target_cf,
            lambda_cf=lambda_cf,
            lambda_entropy=lambda_entropy,
            lambda_contiguity=lambda_contiguity,
            lambda_balance=lambda_balance,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # --- monitoring ---
        with torch.no_grad():
            hard = z.argmax(dim=1)
            sizes = torch.bincount(hard, minlength=num_clusters_max)
            active_sizes = sizes[sizes > 0]
            n_active = len(active_sizes)
            current_cf = config.N / n_active if n_active > 0 else 0.0

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_clusters = hard.cpu().numpy().copy()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f} | "
                  f"clusters={n_active}, CF={current_cf:.2f}, "
                  f"sizes: min={active_sizes.min().item()}, "
                  f"max={active_sizes.max().item()}, "
                  f"mean={active_sizes.float().mean().item():.1f} | "
                  f"cut={loss_dict['cut']:.3f}, "
                  f"cf={loss_dict['cf']:.3f}, "
                  f"bal={loss_dict['balance']:.3f}")

    # --- extract final clusters ---
    model.eval()
    with torch.no_grad():
        z = model(positions_norm, edge_index, edge_attr)
        clusters = z.argmax(dim=1).cpu().numpy()

    # Prefer best-seen if final is degenerate (single cluster)
    if len(np.unique(clusters)) <= 1 and best_clusters is not None:
        if verbose:
            print("Final result degenerate, using best saved clusters.")
        clusters = best_clusters

    # Relabel to consecutive integers (remove gaps from empty clusters)
    unique_labels = np.unique(clusters)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    clusters = np.array([label_map[c] for c in clusters])

    if verbose:
        n_final = len(unique_labels)
        print(f"\nFinal: {n_final} clusters, CF={config.N / n_final:.2f}")

    return clusters
