import yaml
from sklearn.model_selection import train_test_split
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import dgl
import prettytable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from component.preprocess import *
from component.gnn.utils import *
from component.gnn.model_dgl import *

import argparse
import warnings
warnings.filterwarnings("ignore")


# Set device as cpu
device = torch.device('cpu')

m_name = "GNN"

def load_config(config_path="/home/ubuntu/code/component/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = load_config()

    data_path = config["data"]["train_data_path1"]
    data_path = os.path.join(os.path.expanduser("~"), data_path)

    df = pd.read_csv(data_path)
    
    df = preprocess_data_1(df)

    transaction_feats = df[
        ['transaction_id', 'amt', 'is_weekend', 'Month_Sin', 'Month_Cos', 'hour_sin', 'hour_cos', 'day_sin',
         'day_cos']].drop_duplicates(
        subset=['transaction_id'])
    user_feats = df[
        ['card_number', 'age', 'gender', 'lat', 'long']].drop_duplicates(subset=['card_number'])
    merchant_feats = df[['merchant_id', 'fraud_merchant_pct', 'merch_lat', 'merch_long']].drop_duplicates(
        subset=['merchant_id'])

    g, transaction_feats, user_feats, merchant_feats, classified_idx = create_graph_dgl(df, transaction_feats, merchant_feats, user_feats)

    #train_idx, test_idx = train_test_split(classified_idx.values, random_state=42, test_size=0.2, stratify=df['is_fraud'])
    train_idx, valid_idx = train_test_split(classified_idx.values, random_state=42, test_size=0.2, stratify=df['is_fraud'])

    print_graph_info_dgl(g)

    # Dictionary of node types
    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
    print(ntype_dict.keys)
    labels = g.nodes['transaction'].data['y'].long()
    print("Labels dim: {}".format(labels.dim()))

    # Dictionary mapping node types to their input feature dimensions
    in_size_dict = {
        'transaction': transaction_feats.shape[1],
        'user': user_feats.shape[1],
        'merchant': merchant_feats.shape[1]
    }

    #Debug
    print("Num Transactions:", len(transaction_feats))
    print("Num Users:", len(user_feats))
    print("Num Merchants:", len(merchant_feats))

    features_dict = {
        'transaction': transaction_feats,  # Tensor of shape (num_transactions, feature_dim)
        'merchant': merchant_feats,        # Tensor of shape (num_merchants, feature_dim)
        'user': user_feats                 # Tensor of shape (num_users, feature_dim)
    }

    for ntype, tensor in features_dict.items():
        print(f"Node Type: {ntype}, Feature Shape: {tensor.shape}")

    # Define model parameters
    hidden_size = config['gnn']['hidden_size']
    out_size = config['gnn']['out_size']
    n_layers = config['gnn']['n_layers']
    epochs = config['gnn']['num_epochs']
    target_node = config['gnn']['target_node']
    etypes = g.canonical_etypes
    # Initialize model
    model = HeteroRGCN(ntype_dict, etypes, in_size_dict, hidden_size, out_size, n_layers, target_node)

    lr = config['gnn']['learning_rate']
    weight_decay = config['gnn']['weight_decay']
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss()



    trainer = GNN_Trainer(model)
    trainer.train_val(g, features_dict, epochs, train_idx, valid_idx, optimizer, criterion, -np.inf, m_name, labels, target_node)

    visualize_sampled_hetero_graph(g, sample_size=50)


if __name__ == "__main__":
    main()





