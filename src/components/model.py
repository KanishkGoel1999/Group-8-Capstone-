from xgboost import XGBClassifier
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
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
xgb_config = config["xgboost"]["sets"][0]  # change index for different configurations

class Models:
    """
    A class to define and retrieve different machine learning models.
    """
    
    @staticmethod
    def get_xgboost_model(random_state=42):
        """
        Returns an instance of the XGBoost classifier with default parameters.
        
        Parameters:
            random_state (int): Random state for reproducibility.
        
        Returns:
            XGBClassifier: An instance of XGBoost classifier.
        """
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    
    @staticmethod
    def get_gnn_model(in_channels_dict, hidden_channels, out_channels):
        """
        Returns an instance of the Heterogeneous GNN model.
        
        Parameters:
            in_channels_dict (dict): Dictionary of input channels for each node type.
            hidden_channels (int): Number of hidden units in GNN layers.
            out_channels (int): Number of output classes.
        
        Returns:
            HeteroGNNWithReverse: An instance of the defined GNN model.
        """
        class HeteroGNNWithReverse(torch.nn.Module):
            def __init__(self, in_channels_dict, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = HeteroConv({
                    EDGE_TYPES['ASKS']: SAGEConv(in_channels_dict['user'], hidden_channels),
                    EDGE_TYPES['REV_ASKS']: SAGEConv(in_channels_dict['question'], hidden_channels),
                    EDGE_TYPES['HAS']: SAGEConv(in_channels_dict['question'], hidden_channels),
                    EDGE_TYPES['REV_HAS']: SAGEConv(in_channels_dict['answer'], hidden_channels),
                    EDGE_TYPES['ANSWERS']: SAGEConv(in_channels_dict['user'], hidden_channels),
                    EDGE_TYPES['REV_ANSWERS']: SAGEConv(in_channels_dict['answer'], hidden_channels),
                    EDGE_TYPES['ACCEPTED_ANSWER']: SAGEConv(in_channels_dict['question'], hidden_channels),
                    EDGE_TYPES['REV_ACCEPTED']: SAGEConv(in_channels_dict['answer'], hidden_channels),
                    EDGE_TYPES['SELF_LOOP']: SAGEConv(in_channels_dict['user'], hidden_channels),
                }, aggr=gnn_config['aggregation'])

                self.conv2 = HeteroConv({
                    EDGE_TYPES['ASKS']: SAGEConv(hidden_channels, hidden_channels),
                    EDGE_TYPES['REV_ASKS']: SAGEConv(hidden_channels, hidden_channels),
                    EDGE_TYPES['HAS']: SAGEConv(hidden_channels, hidden_channels),
                    EDGE_TYPES['REV_HAS']: SAGEConv(hidden_channels, hidden_channels),
                    EDGE_TYPES['ANSWERS']: SAGEConv(hidden_channels, hidden_channels),
                    EDGE_TYPES['REV_ANSWERS']: SAGEConv(hidden_channels, hidden_channels),
                    EDGE_TYPES['ACCEPTED_ANSWER']: SAGEConv(hidden_channels, hidden_channels),
                    EDGE_TYPES['REV_ACCEPTED']: SAGEConv(hidden_channels, hidden_channels),
                    EDGE_TYPES['SELF_LOOP']: SAGEConv(hidden_channels, hidden_channels),
                }, aggr=gnn_config['aggregation'])

                self.user_lin = torch.nn.Linear(hidden_channels, out_channels)

            def forward(self, x_dict, edge_index_dict):
                # Apply first conv layer with ReLU activation.
                x_dict = self.conv1(x_dict, edge_index_dict)
                for node_type in x_dict:
                    x_dict[node_type] = F.relu(x_dict[node_type])
                # Apply second conv layer.
                x_dict = self.conv2(x_dict, edge_index_dict)
                for node_type in x_dict:
                    x_dict[node_type] = F.relu(x_dict[node_type])
                # Classify only the user nodes.
                out_user = self.user_lin(x_dict['user'])
                return out_user

        return HeteroGNNWithReverse(in_channels_dict, hidden_channels, out_channels)