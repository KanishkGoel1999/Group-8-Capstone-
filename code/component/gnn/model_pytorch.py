import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class HeteroGNN(torch.nn.Module):
    def __init__(self, in_size_dict, hidden_size, out_size, n_layers, etypes, target_node):
        """
        Defines a Heterogeneous Graph Neural Network using PyTorch Geometric.
        """
        super().__init__()
        self.target_node = target_node

        # Input projection layers to ensure all node types have hidden_size embeddings
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_size_dict[ntype], hidden_size) for ntype in in_size_dict
        })

        # Define HeteroConv layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HeteroConv({
                etype: SAGEConv((-1, -1), hidden_size) for etype in etypes
            }, aggr='sum'))

        # Final linear layer for classification
        self.lin = Linear(hidden_size, out_size)

    def forward(self, data):
        """
        Forward pass of the Heterogeneous GNN model.
        """
        # Apply input projection to make all node features of size hidden_size
        x_dict = {ntype: self.input_proj[ntype](data.x_dict[ntype]) for ntype in data.x_dict}

        # Apply HeteroConv layers
        for i, layer in enumerate(self.layers):
            x_dict = layer(x_dict, data.edge_index_dict)
            if i != len(self.layers) - 1:  # Apply activation except for last layer
                x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}

        return self.lin(x_dict[self.target_node])  # Predict target node embeddings
