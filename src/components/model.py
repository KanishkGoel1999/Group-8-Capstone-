from xgboost import XGBClassifier
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

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
    def get_gnn_model(in_channels_dict, hidden_channels=32, out_channels=2):
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
                    ('user', 'asks', 'question'): SAGEConv(in_channels_dict['user'], hidden_channels),
                    ('question', 'rev_asks', 'user'): SAGEConv(in_channels_dict['question'], hidden_channels),
                    ('question', 'has', 'answer'): SAGEConv(in_channels_dict['question'], hidden_channels),
                    ('answer', 'rev_has', 'question'): SAGEConv(in_channels_dict['answer'], hidden_channels),
                    ('user', 'answers', 'answer'): SAGEConv(in_channels_dict['user'], hidden_channels),
                    ('answer', 'rev_answers', 'user'): SAGEConv(in_channels_dict['answer'], hidden_channels),
                    ('question', 'accepted_answer', 'answer'): SAGEConv(in_channels_dict['question'], hidden_channels),
                    ('answer', 'rev_accepted', 'question'): SAGEConv(in_channels_dict['answer'], hidden_channels),
                    ('user', 'self_loop', 'user'): SAGEConv(in_channels_dict['user'], hidden_channels),
                }, aggr='sum')

                self.conv2 = HeteroConv({
                    ('user', 'asks', 'question'): SAGEConv(hidden_channels, hidden_channels),
                    ('question', 'rev_asks', 'user'): SAGEConv(hidden_channels, hidden_channels),
                    ('question', 'has', 'answer'): SAGEConv(hidden_channels, hidden_channels),
                    ('answer', 'rev_has', 'question'): SAGEConv(hidden_channels, hidden_channels),
                    ('user', 'answers', 'answer'): SAGEConv(hidden_channels, hidden_channels),
                    ('answer', 'rev_answers', 'user'): SAGEConv(hidden_channels, hidden_channels),
                    ('question', 'accepted_answer', 'answer'): SAGEConv(hidden_channels, hidden_channels),
                    ('answer', 'rev_accepted', 'question'): SAGEConv(hidden_channels, hidden_channels),
                    ('user', 'self_loop', 'user'): SAGEConv(hidden_channels, hidden_channels),
                }, aggr='sum')

                self.user_lin = torch.nn.Linear(hidden_channels, out_channels)

            def forward(self, x_dict, edge_index_dict):
                x_dict = self.conv1(x_dict, edge_index_dict)
                for node_type in x_dict:
                    x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict = self.conv2(x_dict, edge_index_dict)
                for node_type in x_dict:
                    x_dict[node_type] = F.relu(x_dict[node_type])
                out_user = self.user_lin(x_dict['user'])
                return out_user

        return HeteroGNNWithReverse(in_channels_dict, hidden_channels, out_channels)