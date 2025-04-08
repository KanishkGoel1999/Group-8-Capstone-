import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv  # Import GATConv
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.constants import EDGE_TYPES

# Define the path to the config file (same folder as model.py)
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

# Load the YAML configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
gnn_config = config["gnn"]["sets"][1]  # change index for different configurations

class Models:
    """
    A class to define and retrieve different machine learning models.
    """
    
    @staticmethod
    def get_gnn_model(in_channels_dict, hidden_channels, out_channels):
        """
        Returns an instance of the Heterogeneous GNN model with Graph Attention Networks.
        
        Parameters:
            in_channels_dict (dict): Dictionary of input channels for each node type.
            hidden_channels (int): Number of hidden units in GNN layers.
            out_channels (int): Number of output classes.
        
        Returns:
            HeteroGNNWithAttention: An instance of the GNN model with attention.
        """
        class HeteroGNNWithAttention(torch.nn.Module):
            def __init__(self, in_channels_dict, hidden_channels, out_channels):
                super().__init__()
                # Using multi-head attention
                self.conv1 = HeteroConv({
                    EDGE_TYPES['ASKS']: GATConv(in_channels_dict['user'], hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['REV_ASKS']: GATConv(in_channels_dict['question'], hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['HAS']: GATConv(in_channels_dict['question'], hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['REV_HAS']: GATConv(in_channels_dict['answer'], hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['ANSWERS']: GATConv(in_channels_dict['user'], hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['REV_ANSWERS']: GATConv(in_channels_dict['answer'], hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['ACCEPTED_ANSWER']: GATConv(in_channels_dict['question'], hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['REV_ACCEPTED']: GATConv(in_channels_dict['answer'], hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['SELF_LOOP']: GATConv(in_channels_dict['user'], hidden_channels, add_self_loops=False, heads=8),
                }, aggr=gnn_config['aggregation'])

                self.conv2 = HeteroConv({
                    EDGE_TYPES['ASKS']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),  # Multiply by 8 for multi-head attention
                    EDGE_TYPES['REV_ASKS']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['HAS']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['REV_HAS']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['ANSWERS']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['REV_ANSWERS']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['ACCEPTED_ANSWER']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['REV_ACCEPTED']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),
                    EDGE_TYPES['SELF_LOOP']: GATConv(hidden_channels * 8, hidden_channels, add_self_loops=False, heads=8),
                }, aggr=gnn_config['aggregation'])

                self.dropout = torch.nn.Dropout(p=0.5)
                self.user_lin = torch.nn.Linear(hidden_channels * 8, out_channels)

            def forward(self, x_dict, edge_index_dict):
                x_dict = self.conv1(x_dict, edge_index_dict)
                for node_type in x_dict:
                    x_dict[node_type] = F.relu(x_dict[node_type])

                x_dict = self.conv2(x_dict, edge_index_dict)
                for node_type in x_dict:
                    x_dict[node_type] = F.relu(x_dict[node_type])

                # Apply dropout to avoid overfitting
                x_dict = self.dropout(x_dict['user'])

                # Final classification for user influence
                out_user = self.user_lin(x_dict)
                return out_user

        return HeteroGNNWithAttention(in_channels_dict, hidden_channels, out_channels)

