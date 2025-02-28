import numpy as np
import pandas as pd

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
from component.gnn.model import *

import warnings
warnings.filterwarnings("ignore")

data_path = os.path.join('/Users/kanishkgoel/Downloads/GNN_for _Finance/processed_test.csv')

device = torch.device('cpu')

m_name = "GNN"

model_path = "models/"+f'model_{m_name}.pt'


def generate_aucpr_plot(y_test, y_prob_dict):
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in y_prob_dict.items():
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]  
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AUC={average_precision_score(y_test, y_prob):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('plots/pr_curve.png')
    plt.show()

def generate_aucroc_plot(y_test, y_prob_dict):
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in y_prob_dict.items():
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]  
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_score(y_test, y_prob):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('plots/roc_curve.png')
    plt.show()
def test_model(model_path, test_g, test_features_dict, test_labels, target_node, ntype_dict, etypes, in_size_dict, hidden_size, out_size, n_layers):
    """
    Loads a trained model and evaluates it on the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model (match architecture used during training)
    model = HeteroRGCN(ntype_dict, etypes, in_size_dict, hidden_size, out_size, n_layers, target_node)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(test_g, test_features_dict, target_node)
        predictions = torch.argmax(logits, dim=1)

    # Evaluate performance
    accuracy = (predictions == test_labels).sum().item() / len(test_labels)
    precision = torch.tensor(precision_score(test_labels.cpu(), predictions.cpu(), average='binary'))
    recall = torch.tensor(recall_score(test_labels.cpu(), predictions.cpu(), average='binary'))
    f1 = torch.tensor(f1_score(test_labels.cpu(), predictions.cpu(), average='binary'))
    roc_auc = torch.tensor(roc_auc_score(test_labels.cpu(), predictions.cpu()))

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    return predictions


def main():
    df = pd.read_csv(data_path)

    df = preprocess_data(df)

    test_g, transaction_feats_test, user_feats_test, merchant_feats_test, classified_idx = create_graph(df)

    ntype_dict = {n_type: test_g.number_of_nodes(n_type) for n_type in test_g.ntypes}
    labels_test = test_g.nodes['transaction'].data['y'].long()
    in_size_dict_test = {
        'transaction': transaction_feats_test.shape[1],
        'user': user_feats_test.shape[1],
        'merchant': merchant_feats_test.shape[1]
    }

    # Define model parameters
    hidden_size = 64
    out_size = 2
    n_layers = 2
    target_node = 'transaction'
    etypes = test_g.canonical_etypes

    features_dict_test = {
        'transaction': transaction_feats_test,  # Tensor of shape (num_transactions, feature_dim)
        'merchant': merchant_feats_test,  # Tensor of shape (num_merchants, feature_dim)
        'user': user_feats_test  # Tensor of shape (num_users, feature_dim)
    }

    result = test_model(model_path, test_g, features_dict_test, labels_test, target_node, ntype_dict, etypes,
                        in_size_dict_test, hidden_size, out_size, n_layers)

    return result

if __name__ == "__main__":
    result = main()
    print(result)





