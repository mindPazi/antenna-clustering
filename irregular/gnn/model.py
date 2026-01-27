"""
GNN model for URA clustering using PyTorch Geometric.

Architecture:
    Input(N, 2) -> InputProj -> GAT layers with edge features -> Linear+Softmax -> Z(N, K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class URAClusteringGNN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64,
                 num_clusters=4, heads=4, dropout=0.1,
                 edge_dim=3):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GAT layers with edge features
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             dropout=dropout, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                             dropout=dropout, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                             dropout=dropout, edge_dim=edge_dim)

        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_clusters)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        # Project input features
        h = F.elu(self.input_proj(x))

        # GNN layers
        h = F.elu(self.conv1(h, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.elu(self.conv2(h, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index, edge_attr)

        # Soft cluster assignments
        z = F.softmax(self.classifier(h), dim=-1)
        return z
