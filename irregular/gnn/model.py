"""
GNN model for URA clustering using PyTorch Geometric.

Architecture:
    Input(N, 2) -> InputProj -> GAT layers -> Linear+Softmax -> Z(N, K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class URAClusteringGNN(nn.Module):
    """
    Graph Attention Network for antenna array clustering.
    
    Args:
        in_dim: input feature dimension (2 for positions)
        hidden_dim: hidden layer dimension
        num_clusters: number of clusters K
        heads: number of attention heads
        dropout: dropout probability (default 0.0 for stability)
        edge_dim: edge feature dimension
    """
    def __init__(self, in_dim=2, hidden_dim=64,
                 num_clusters=4, heads=4, dropout=0.0,
                 edge_dim=3):
        super().__init__()

        self.num_clusters = num_clusters
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # GAT layers with edge features
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             dropout=dropout, edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                             dropout=dropout, edge_dim=edge_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim * heads)
        
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                             dropout=dropout, edge_dim=edge_dim)

        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_clusters)
        self.dropout = dropout
        
        # Initialize classifier with small weights for balanced start
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, edge_index, edge_attr, temperature=1.0):
        """
        Forward pass.
        
        Args:
            x: (N, in_dim) node features (positions)
            edge_index: (2, E) edge connectivity
            edge_attr: (E, edge_dim) edge features
            temperature: softmax temperature (higher = softer assignments)
            
        Returns:
            z: (N, K) soft cluster assignments (probabilities)
        """
        # Input projection with batch norm
        h = self.input_proj(x)
        h = self.bn1(h)
        h = F.elu(h)

        # GNN layers
        h = self.conv1(h, edge_index, edge_attr)
        h = self.bn2(h)
        h = F.elu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_attr)
        h = self.bn3(h)
        h = F.elu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index, edge_attr)
        h = F.elu(h)

        # Classifier with temperature scaling
        logits = self.classifier(h)
        z = F.softmax(logits / temperature, dim=-1)
        
        return z