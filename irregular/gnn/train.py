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


def _snap_to_allowed_sizes(clusters, grid_shape, allowed_sizes):
    """
    Post-process cluster assignments so every cluster has a size in allowed_sizes.

    Clusters with invalid sizes are split greedily using the largest
    allowed size that fits, preserving spatial contiguity where possible.
    """
    allowed = sorted(allowed_sizes, reverse=True)
    max_allowed = allowed[0]
    rows, cols = grid_shape
    n = rows * cols

    new_clusters = np.full(n, -1, dtype=int)
    next_label = 0

    for k in range(clusters.max() + 1):
        indices = np.where(clusters == k)[0]
        size = len(indices)
        if size == 0:
            continue

        if size in allowed_sizes:
            new_clusters[indices] = next_label
            next_label += 1
        else:
            # Split into allowed sizes greedily (largest first)
            remaining = list(indices)
            while remaining:
                # Pick the largest allowed size that fits
                assigned = False
                for s in allowed:
                    if len(remaining) >= s:
                        chunk = remaining[:s]
                        for idx in chunk:
                            new_clusters[idx] = next_label
                        remaining = remaining[s:]
                        next_label += 1
                        assigned = True
                        break
                if not assigned:
                    # Remaining elements fewer than smallest allowed size
                    # Assign as size-1 clusters
                    for idx in remaining:
                        new_clusters[idx] = next_label
                        next_label += 1
                    remaining = []

    # Relabel to consecutive integers
    unique_labels = np.unique(new_clusters)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[c] for c in new_clusters])


def train_ura_clustering(config, target_cf=3.0, num_clusters_max=None,
                         epochs=500, lr=0.001,
                         lambda_cf=10.0, lambda_contiguity=0.5,
                         lambda_entropy=0.5, lambda_balance=5.0,
                         allowed_sizes=None, lambda_allowed=5.0,
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
        allowed_sizes: optional list of allowed cluster sizes (e.g. [1,2,4])
        lambda_allowed: weight for allowed-sizes loss (default 5.0)
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

    # Temperature annealing: start soft (high temp) → anneal to hard (low temp)
    temp_start = 5.0
    temp_end = 0.5
    temp_anneal_epochs = int(epochs * 0.7)  # anneal over 70% of training

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Linearly anneal temperature
        if epoch < temp_anneal_epochs:
            temperature = temp_start - (temp_start - temp_end) * epoch / temp_anneal_epochs
        else:
            temperature = temp_end

        z = model(positions_norm, edge_index, edge_attr, temperature=temperature)

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
            allowed_sizes=allowed_sizes,
            lambda_allowed=lambda_allowed,
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
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f} T={temperature:.2f} | "
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

    # Post-process to snap cluster sizes to allowed values
    if allowed_sizes is not None:
        clusters = _snap_to_allowed_sizes(
            clusters, (config.rows, config.cols), allowed_sizes
        )
        if verbose:
            sizes_pp = np.bincount(clusters)
            invalid = [s for s in sizes_pp if s not in allowed_sizes]
            print(f"Post-processing: snapped to allowed sizes {allowed_sizes}")
            if invalid:
                print(f"  WARNING: {len(invalid)} clusters with invalid sizes: {invalid}")

    # Relabel to consecutive integers (remove gaps from empty clusters)
    unique_labels = np.unique(clusters)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    clusters = np.array([label_map[c] for c in clusters])

    if verbose:
        n_final = len(unique_labels)
        print(f"\nFinal: {n_final} clusters, CF={config.N / n_final:.2f}")

    return clusters
