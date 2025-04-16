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
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import yaml
import prettytable

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from component.preprocess import *
from component.gnn.utils import *
from component.gnn.model_pytorch import *

import argparse
import warnings

warnings.filterwarnings("ignore")

# Using 2 datasets
#data_path = '/home/ubuntu/fraudTrain.csv'

# Set device as cpu
#device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m_name = "GNN"

def load_config(config_path="/home/ubuntu/code/component/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()

    # Load data
    data_path = config["data"]["data_path2"]
    df = pd.read_csv(data_path)


    # df = preprocess_data_cat(df, 'trans_date_trans_time', 'merchant', 'trans_num')
    # transaction_feats = df[
    #     ['transaction_id', 'amt', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']].drop_duplicates(subset=['transaction_id'])
    # user_feats = df[['card_number', 'city_encoded', 'state_encoded', 'zip_encoded', 'age_encoded', 'gender_encoded', 'lat', 'long']].drop_duplicates(subset=['card_number'])
    # merchant_feats = df[['merchant_id', 'fraud_merchant_pct', 'merch_lat', 'merch_long']].drop_duplicates(
    #     subset=['merchant_id'])

    # df = preprocess_data2(df, 'trans_date_trans_time', 'merchant', 'trans_num')
    # transaction_feats = df[
    #     ['transaction_id', 'amt', 'is_weekend', 'Month_Sin', 'Month_Cos', 'hour_sin', 'hour_cos', 'day_sin',
    #      'day_cos']].drop_duplicates(
    #     subset=['transaction_id'])
    # user_feats = df[
    #     ['card_number', 'age', 'gender', 'lat', 'long']].drop_duplicates(subset=['card_number'])
    # merchant_feats = df[['merchant_id', 'fraud_merchant_pct', 'merch_lat', 'merch_long']].drop_duplicates(
    #     subset=['merchant_id'])

    df, test = preprocessDS2(df)
    # transaction_feats = df[['transaction_id', 'amount', 'weekend_transaction', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'currency_AUD', 'currency_BRL', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_JPY', 'currency_MXN', 'currency_NGN', 'currency_RUB', 'currency_SGD', 'currency_USD', 'device_Android App', 'device_Chip Reader', 'device_Chrome', 'device_Edge', 'device_Firefox', 'device_Magnetic Stripe', 'device_NFC Payment', 'device_Safari', 'device_iOS App']]
    # user_feats = df[['card_number', 'country_Australia', 'country_Brazil', 'country_Canada', 'country_France', 'country_Germany', 'country_Japan', 'country_Mexico', 'country_Nigeria', 'country_Russia', 'country_Singapore', 'country_UK', 'country_USA']].drop_duplicates(subset=['card_number'])
    # merchant_feats = df[['merchant_id', 'merchant_category_Education', 'merchant_category_Entertainment', 'merchant_category_Gas', 'merchant_category_Grocery', 'merchant_category_Healthcare', 'merchant_category_Restaurant', 'merchant_category_Retail', 'merchant_category_Travel']].drop_duplicates(
    #     subset=['merchant_id'])

    transaction_feats = df[
        ['transaction_id', 'amount', 'weekend_transaction', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
         'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_JPY', 'currency_RUB', 'currency_SGD', 'currency_USD', 'device_Android App', 'device_Chip Reader', 'device_Magnetic Stripe', 'device_NFC Payment',
         'device_Safari', 'device_iOS App']]
    user_feats = df[
        ['card_number', 'country_Canada', 'country_France',
         'country_Germany', 'country_Japan', 'country_Russia', 'country_Singapore', 'country_UK',
         'country_USA']].drop_duplicates(subset=['card_number'])
    merchant_feats = df[
        ['merchant_id', 'merchant_category_Education', 'merchant_category_Entertainment', 'merchant_category_Gas', 'merchant_category_Grocery']].drop_duplicates(
        subset=['merchant_id'])

    classified_idx = torch.tensor(transaction_feats.index.values, dtype=torch.long)
    print(df.info())
    # Create PyG graph
    data = create_pyg_graph(df, transaction_feats, merchant_feats, user_feats)

    random_state = config["data"]["random_state"]
    # Train-validation split
    train_idx, valid_idx = train_test_split(classified_idx.numpy(), random_state=42, test_size=0.2, stratify=df['is_fraud'])

    # Print graph info
    print_pyg_graph_info(data)

    # Extract labels
    labels = data['transaction'].y.long()
    print(f"Labels dim: {labels.dim()}")

    # Dictionary mapping node types to their input feature dimensions
    in_size_dict = {
        'transaction': data['transaction'].x.shape[1],
        'user': data['user'].x.shape[1],
        'merchant': data['merchant'].x.shape[1]
    }

    # Define model
    etypes = data.edge_types
    NUM_EPOCHS = config["gnn"]["num_epochs"]
    batch_size = config["gnn"]["batch_size"]
    hidden_size = config["gnn"]["hidden_size"]
    n_layers = config["gnn"]["n_layers"]
    lr = config["gnn"]["learning_rate"]
    weight_decay = float(config["gnn"]["weight_decay"])
    patience = config["gnn"]["patience"]
    out_size = config["gnn"]["out_size"]
    target_node = config["gnn"]["target_node"]
    min_epochs = config["gnn"]["min_epochs"]

    model = HeteroGNN(in_size_dict, hidden_size, out_size, n_layers, etypes, target_node)

    # Train model
    f1_scores, losses = train_pyg_model_ES(model, data, train_idx, valid_idx, NUM_EPOCHS, lr, weight_decay, batch_size,
                    m_name="best_pyg_model_DS2.pth", patience=patience, min_epochs=min_epochs)
    #train_pyg_model_without_dataloader(model, data, train_idx, valid_idx, num_epochs=300, m_name="best_pyg_model.pth")

    #visualize_loss(NUM_EPOCHS, f1_scores, losses)
    #f1_scores.to_csv('f1_scores.csv')
    #losses.to_csv('losses.csv')




if __name__ == "__main__":
    main()
