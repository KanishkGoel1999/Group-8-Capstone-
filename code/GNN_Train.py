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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve, auc

import prettytable

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from component.preprocess import *
from component.gnn.utils import *
from component.gnn.model import *

import argparse
import warnings
warnings.filterwarnings("ignore")


data_path = '/Users/kanishkgoel/Downloads/GNN_for _Finance/processed_train.csv'

# Set device as cpu 
device = torch.device('cpu')

m_name = "GNN"

def main():

    df = pd.read_csv(data_path)
    
    df = preprocess_data(df)

    g, transaction_feats, user_feats, merchant_feats, classified_idx = create_graph(df)

    train_idx, test_idx = train_test_split(classified_idx.values, random_state=42, test_size=0.2, stratify=df['is_fraud'])
    train_idx, valid_idx = train_test_split(train_idx, random_state=42, test_size=0.2)

    print_graph_info(g)

    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
    labels = g.nodes['transaction'].data['y'].long()
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
    hidden_size = 64
    out_size = 2
    n_layers = 1
    target_node = 'transaction'
    etypes = g.canonical_etypes
    # Initialize model
    model = HeteroRGCN(ntype_dict, etypes, in_size_dict, hidden_size, out_size, n_layers, target_node)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay = 5e-4)
    criterion = nn.CrossEntropyLoss()



    trainer = GNN_Trainer(model)
    trainer.train_val(g, features_dict, 1, train_idx, valid_idx, optimizer, criterion, -np.inf, m_name, labels, "transaction")

    visualize_sampled_hetero_graph(g, sample_size=50)


if __name__ == "__main__":
    main()





