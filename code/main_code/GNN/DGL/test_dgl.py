import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import uuid

import dgl
import torch
from torch import nn
import torch.optim as optim
import prettytable
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from component.preprocess import *
from component.gnn.utils import *
from component.gnn.model_dgl import *

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cpu')

m_name = "GNN"

model_path = "models/"+f'model_{m_name}.pt'




def load_config(config_path="/home/ubuntu/code/component/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()

    data_path = config["data"]["test_data_path1"]
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

    test_g, transaction_feats_test, user_feats_test, merchant_feats_test, classified_idx = create_graph_dgl(df, transaction_feats,
                                                                                        merchant_feats, user_feats)


    ntype_dict = {n_type: test_g.number_of_nodes(n_type) for n_type in test_g.ntypes}
    labels_test = test_g.nodes['transaction'].data['y'].long()
    in_size_dict_test = {
        'transaction': transaction_feats_test.shape[1],
        'user': user_feats_test.shape[1],
        'merchant': merchant_feats_test.shape[1]
    }

    # Define model parameters
    hidden_size = config['gnn']['hidden_size']
    out_size = config['gnn']['out_size']
    n_layers = config['gnn']['n_layers']
    target_node = config['gnn']['target_node']
    etypes = test_g.canonical_etypes

    features_dict_test = {
        'transaction': transaction_feats_test,  # Tensor of shape (num_transactions, feature_dim)
        'merchant': merchant_feats_test,  # Tensor of shape (num_merchants, feature_dim)
        'user': user_feats_test  # Tensor of shape (num_users, feature_dim)
    }

    result = test_dgl_model(model_path, test_g, features_dict_test, labels_test, target_node, ntype_dict, etypes,
                            in_size_dict_test, hidden_size, out_size, 5, device)

    return result

if __name__ == "__main__":
    result = main()
    print(result)





