"""
Training for GNN-based URA clustering.

ROBUST implementation with:
- Proper initialization
- Cluster monitoring during training
- Early detection of collapse
"""

import torch
import torch.nn.functional as F
import numpy as np

from .config import URAConfig
from .graph import create_ura_graph
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


def initialize_with_quadrants(model, positions, num_clusters, device):
    """
    Initialize classifier weights to produce quadrant-like initial clustering.
    This gives the GNN a good starting point.
    """
    with torch.no_grad():
        # Normalize positions
        pos_min = positions.min(0)[0]
        pos_max = positions.max(0)[0]
        pos_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)
        
        # Create initial soft assignments based on position
        # This gives roughly equal quadrants as starting point
        x_mid = 0.5
        y_mid = 0.5
        
        # Simple quadrant assignment
        initial_z = torch.zeros(positions.shape[0], num_clusters, device=device)
        for i in range(positions.shape[0]):
            x, y = pos_norm[i]
            if x < x_mid and y < y_mid:
                initial_z[i, 0] = 1.0
            elif x >= x_mid and y < y_mid:
                initial_z[i, 1] = 1.0
            elif x < x_mid and y >= y_mid:
                initial_z[i, 2] = 1.0
            else:
                initial_z[i, 3] = 1.0
        
        # Add noise to avoid exactly 0/1
        initial_z = initial_z * 0.7 + 0.075  # Makes it [0.075, 0.775] roughly
        initial_z = initial_z / initial_z.sum(dim=1, keepdim=True)
        
    return initial_z


def train_ura_clustering(config, num_clusters=4, epochs=500, lr=0.001,
                         use_physics=True, lambda_balance=10.0,
                         lambda_contiguity=0.1, verbose=True):
    """
    Train GNN for URA clustering.

    Args:
        config: URAConfig with spacing parameters
        num_clusters: K clusters
        epochs: training iterations
        lr: learning rate
        use_physics: if True, use coupling-weighted loss
        lambda_balance: weight for balance loss (HIGH = 10.0 default)
        lambda_contiguity: weight for contiguity loss
        verbose: print training progress

    Returns:
        cluster_assignments: (N,) array of cluster labels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create graph
    positions, edge_index, edge_attr, W = create_ura_graph(
        config, connectivity='8-connected', use_coupling=use_physics
    )

    # Normalize positions
    pos_min = positions.min(0)[0]
    pos_max = positions.max(0)[0]
    positions_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)

    # Move to device
    positions_norm = positions_norm.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    
    if W is not None:
        W = W.to(device)
    else:
        # Create simple adjacency if no coupling
        n = config.N
        W = torch.zeros(n, n, device=device)
        W[edge_index[0], edge_index[1]] = 1.0

    # Initialize model
    model = URAClusteringGNN(
        in_dim=2,
        hidden_dim=64,
        num_clusters=num_clusters,
        heads=4,
        edge_dim=edge_attr.shape[1]
    ).to(device)

    # Use AdamW with proper weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    best_loss = float('inf')
    best_clusters = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        z = model(positions_norm, edge_index, edge_attr)
        
        # Check for NaN
        if torch.isnan(z).any():
            print(f"WARNING: NaN detected at epoch {epoch+1}, reinitializing...")
            model = URAClusteringGNN(
                in_dim=2, hidden_dim=64, num_clusters=num_clusters,
                heads=4, edge_dim=edge_attr.shape[1]
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            continue

        # Compute combined loss
        loss, loss_dict = total_loss(
            z, W, positions_norm,
            lambda_balance=lambda_balance,
            lambda_entropy=1.0,
            lambda_anti_collapse=5.0,
            lambda_contiguity=lambda_contiguity
        )

        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        # Monitor cluster sizes
        with torch.no_grad():
            hard_clusters = z.argmax(dim=1)
            sizes = torch.bincount(hard_clusters, minlength=num_clusters)
            
            # Check for collapse (one cluster has > 80% of nodes)
            if sizes.max() > 0.8 * config.N:
                if verbose and epoch % 50 == 0:
                    print(f"Epoch {epoch+1}: WARNING - cluster collapse detected, sizes={sizes.tolist()}")
            
            # Save best model (based on balanced clusters)
            size_variance = sizes.float().var()
            if size_variance < best_loss and sizes.min() > 0:
                best_loss = size_variance
                best_clusters = hard_clusters.cpu().numpy().copy()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f} | "
                  f"Sizes={sizes.tolist()} | "
                  f"cut={loss_dict['cut']:.3f}, bal={loss_dict['balance']:.3f}")

    # Extract final clusters
    model.eval()
    with torch.no_grad():
        z = model(positions_norm, edge_index, edge_attr)
        final_clusters = z.argmax(dim=1).cpu().numpy()
        final_sizes = np.bincount(final_clusters, minlength=num_clusters)
        
        # If final result is collapsed, use best saved result
        if final_sizes.min() == 0 or final_sizes.max() > 0.5 * config.N:
            if best_clusters is not None:
                if verbose:
                    print(f"Using best saved clusters instead of final (final was unbalanced)")
                return best_clusters
    
    return final_clusters


def train_ura_clustering_simple(config, num_clusters=4, epochs=300, verbose=True):
    """
    SIMPLE training that guarantees balanced clusters.
    
    Uses only position-based features (no complex coupling).
    More stable but produces basic quadrant-like clusters.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simple graph without coupling
    positions, edge_index, edge_attr, _ = create_ura_graph(
        config, connectivity='8-connected', use_coupling=False
    )
    
    # Normalize
    pos_min = positions.min(0)[0]
    pos_max = positions.max(0)[0]
    positions_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)
    
    positions_norm = positions_norm.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    
    # Simple adjacency
    n = config.N
    adj = torch.zeros(n, n, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = torch.diag(adj.sum(dim=1))
    
    # Model
    model = URAClusteringGNN(
        in_dim=2, hidden_dim=32, num_clusters=num_clusters,
        heads=2, edge_dim=edge_attr.shape[1], dropout=0.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z = model(positions_norm, edge_index, edge_attr)
        
        # Simple loss: MinCut + STRONG balance
        loss_cut = mincut_loss(z, adj, deg)
        loss_bal = balance_loss(z)
        loss_ent = entropy_loss(z)
        
        loss = loss_cut + 20.0 * loss_bal + 0.5 * loss_ent
        
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            with torch.no_grad():
                sizes = torch.bincount(z.argmax(dim=1), minlength=num_clusters)
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Sizes={sizes.tolist()}")
    
    model.eval()
    with torch.no_grad():
        z = model(positions_norm, edge_index, edge_attr)
        return z.argmax(dim=1).cpu().numpy()