from xgboost import XGBClassifier
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, to_hetero
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.constants import EDGE_TYPES, NODE_TYPES, DATASET_TYPE_1

# Define the path to the config file (same folder as model.py)
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

# Load the YAML configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
gnn_config = config["gnn"]["sets"][1]  # change index for different configurations
xgb_config = config["xgboost"]["sets"][0]  # change index for different configurations

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x

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
    def get_hetero_model(dataset_name, in_channels_dict, hidden_channels, out_channels):
        base_model = SAGE(
            in_channels=next(iter(in_channels_dict.values())),  # get any one in_channels
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )

        node_types = NODE_TYPES[dataset_name]
        edge_types = EDGE_TYPES[dataset_name]

        hetero_model = to_hetero(base_model, metadata=(node_types, edge_types))
        return hetero_model
