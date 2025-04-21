import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.constants import EDGE_TYPES, DATASET_TYPE_1
from components.metric import Metrics
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on a specified dataset and config set")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., STACK_OVERFLOW or ASK_REDDIT)")
    parser.add_argument("config_index", type=int, help="Index of the GNN configuration set from YAML")
    return parser.parse_args()

# function for checking missing values #TODO
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(df, columns_to_remove):
    """
    Removes specified columns from the dataset.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        columns_to_remove (list): List of column names to drop.
    
    Returns:
        pd.DataFrame: Processed dataframe.
    """
    return df.drop(columns=columns_to_remove, errors='ignore')

# def get_edge_index_dict(batch):
#     """
#     Constructs the edge index dictionary for a given batch dynamically.
    
#     Parameters:
#         batch (dict): A batch of data from the NeighborLoader containing edge indices.
    
#     Returns:
#         dict: Edge index dictionary mapping edge types to their respective edge indices.
#     """
#     return {
#         EDGE_TYPES['ASKS']: batch[EDGE_TYPES['ASKS']].edge_index,
#         EDGE_TYPES['REV_ASKS']: batch[EDGE_TYPES['REV_ASKS']].edge_index,
#         EDGE_TYPES['HAS']: batch[EDGE_TYPES['HAS']].edge_index,
#         EDGE_TYPES['REV_HAS']: batch[EDGE_TYPES['REV_HAS']].edge_index,
#         EDGE_TYPES['ANSWERS']: batch[EDGE_TYPES['ANSWERS']].edge_index,
#         EDGE_TYPES['REV_ANSWERS']: batch[EDGE_TYPES['REV_ANSWERS']].edge_index,
#         EDGE_TYPES['ACCEPTED_ANSWER']: batch[EDGE_TYPES['ACCEPTED_ANSWER']].edge_index,
#         EDGE_TYPES['REV_ACCEPTED']: batch[EDGE_TYPES['REV_ACCEPTED']].edge_index,
#         EDGE_TYPES['SELF_LOOP']: batch[EDGE_TYPES['SELF_LOOP']].edge_index,
#     }

def get_edge_index_dict(batch):
    """
    Constructs the edge index dictionary for any heterogeneous batch dynamically.

    Parameters:
        batch (HeteroData): A mini-batch from NeighborLoader.

    Returns:
        dict: Edge index dictionary {edge_type: edge_index}.
    """
    edge_index_dict = {}
    for edge_type in batch.edge_types:
        edge_index_dict[edge_type] = batch[edge_type].edge_index
    
    return edge_index_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on a specified dataset")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., STACKOVER_FLOW or ASK_REDDIT)")
    return parser.parse_args()

def read_yaml(set_num):
    # Load YAML configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "components", "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load GNN hyperparameters from config
    gnn_config = config["gnn"]["sets"][set_num]  # change index for different configurations
    return gnn_config

def get_in_channels(dataset_type, data):
    if dataset_type == DATASET_TYPE_1:
        # Define input channels for GNN model
        in_channels_dict = {
            'user': data['user'].x.size(-1),
            'question': data['question'].x.size(-1),
            'answer': data['answer'].x.size(-1)
        }
    else:
        # raise ValueError('Invalid dataset type')
        in_channels_dict = {
            'author': data['author'].x.size(-1),
            'post': data['post'].x.size(-1),
            'comment': data['comment'].x.size(-1)
        }
 
    return in_channels_dict

def train_mini_batch(model, train_loader, optimizer, dataset_name):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    # Determine node type dynamically
    target_node = "user" if dataset_name == DATASET_TYPE_1 else "author"

    for batch in train_loader:
        optimizer.zero_grad()

        # Dynamically construct x_dict
        x_dict = {
            key: batch[key].x
            for key in batch.node_types
        }

        edge_index_dict = get_edge_index_dict(batch)
        out = model(x_dict, edge_index_dict)

        # Get labels and compute loss
        y_true = batch[target_node].y
        y_pred = out[target_node]

        loss = F.cross_entropy(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Accuracy
        preds = y_pred.argmax(dim=-1)
        correct += (preds == y_true).sum().item()
        total_samples += y_true.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total_samples

    return avg_loss, accuracy

@torch.no_grad()
def test_mini_batch(model, test_loader, dataset_name):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Dynamically determine the node type
    target_node = "user" if dataset_name == DATASET_TYPE_1 else "author"

    for batch in test_loader:
        # Construct x_dict dynamically
        x_dict = {
            key: batch[key].x for key in batch.node_types
        }

        edge_index_dict = get_edge_index_dict(batch)
        out = model(x_dict, edge_index_dict)

        # Extract predictions and labels for target node
        node_out = out[target_node]
        node_labels = batch[target_node].y

        loss = F.cross_entropy(node_out, node_labels)
        total_loss += loss.item()

        preds = node_out.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(node_labels.cpu())

    avg_loss = total_loss / len(test_loader)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    metrics = Metrics.compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss

    return metrics




