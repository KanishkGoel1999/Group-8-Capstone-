import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.constants import EDGE_TYPES
from components.metric import Metrics

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

def get_edge_index_dict(batch):
    """
    Constructs the edge index dictionary for a given batch dynamically.
    
    Parameters:
        batch (dict): A batch of data from the NeighborLoader containing edge indices.
    
    Returns:
        dict: Edge index dictionary mapping edge types to their respective edge indices.
    """
    return {
        EDGE_TYPES['ASKS']: batch[EDGE_TYPES['ASKS']].edge_index,
        EDGE_TYPES['REV_ASKS']: batch[EDGE_TYPES['REV_ASKS']].edge_index,
        EDGE_TYPES['HAS']: batch[EDGE_TYPES['HAS']].edge_index,
        EDGE_TYPES['REV_HAS']: batch[EDGE_TYPES['REV_HAS']].edge_index,
        EDGE_TYPES['ANSWERS']: batch[EDGE_TYPES['ANSWERS']].edge_index,
        EDGE_TYPES['REV_ANSWERS']: batch[EDGE_TYPES['REV_ANSWERS']].edge_index,
        EDGE_TYPES['ACCEPTED_ANSWER']: batch[EDGE_TYPES['ACCEPTED_ANSWER']].edge_index,
        EDGE_TYPES['REV_ACCEPTED']: batch[EDGE_TYPES['REV_ACCEPTED']].edge_index,
        EDGE_TYPES['SELF_LOOP']: batch[EDGE_TYPES['SELF_LOOP']].edge_index,
    }

def train_mini_batch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        # Extract node features from the mini-batch for each node type.
        x_dict = {
            'user': batch['user'].x,
            'question': batch['question'].x,
            'answer': batch['answer'].x
        }
        # Build the edge index dictionary from the subgraph in the batch.
        edge_index_dict = get_edge_index_dict(batch)
        out = model(x_dict, edge_index_dict)
        # Compute loss on the current mini-batch of user nodes.
        loss = F.cross_entropy(out, batch['user'].y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def train_full_batch(model, optimizer, data, train_mask):
    model.train()
    optimizer.zero_grad()
    
    x_dict = {
        'user': data['user'].x,
        'question': data['question'].x,
        'answer': data['answer'].x
    }
    
    edge_index_dict = get_edge_index_dict(data)
    out = model(x_dict, edge_index_dict)

    loss = F.cross_entropy(out[train_mask], data['user'].y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def test_mini_batch(model, test_loader):
    model.eval()
    total_correct = 0
    total_examples = 0
    all_preds = []
    all_labels = []
    for batch in test_loader:
        x_dict = {
            'user': batch['user'].x,
            'question': batch['question'].x,
            'answer': batch['answer'].x
        }
        edge_index_dict = get_edge_index_dict(batch)
        out = model(x_dict, edge_index_dict)
        preds = out.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(batch['user'].y.cpu())
        total_correct += (preds == batch['user'].y).sum().item()
        total_examples += batch['user'].y.size(0)
    # accuracy = total_correct / total_examples
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    metrics = Metrics.compute_metrics(all_labels, all_preds)
    
    return metrics

@torch.no_grad()
def test_full_batch(model, data, test_mask):
    model.eval()

    x_dict = {
        'user': data['user'].x,
        'question': data['question'].x,
        'answer': data['answer'].x
    }
    
    edge_index_dict = get_edge_index_dict(data)
    out = model(x_dict, edge_index_dict)

    preds = out[test_mask].argmax(dim=-1)
    labels = data['user'].y[test_mask]

    all_preds = preds.cpu().numpy()
    all_labels = labels.cpu().numpy()
    
    metrics = Metrics.compute_metrics(all_labels, all_preds)
    
    return metrics

