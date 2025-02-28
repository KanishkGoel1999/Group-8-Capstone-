import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
import dgl
import torch
from dgl.nn import RelGraphConv
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import networkx as nx
import prettytable


def create_graph(df):
    # Ensure DGL is using CPU
    torch.set_default_device(torch.device("cpu"))

    # For tensors
    df = df.astype(np.float32)

    # Get features
    transaction_feats = df[['trans_id', 'amt', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'Month_Sin',
                            'Month_Cos']].drop_duplicates(subset=['trans_id'])
    user_feats = df[['cc_num', 'age', 'gender', 'lat', 'long']].drop_duplicates(subset=['cc_num'])
    merchant_feats = df[['merch_id', 'fraud_merchant_pct', 'merch_lat', 'merch_long']].drop_duplicates(
        subset=['merch_id'])

    classified_idx = transaction_feats.index

    # Normalize features and convert to tensors
    scaler = StandardScaler()
    transaction_feats = torch.tensor(scaler.fit_transform(transaction_feats.drop(columns=['trans_id'])),
                                     dtype=torch.float)
    user_feats = torch.tensor(scaler.fit_transform(user_feats.drop(columns=['cc_num'])), dtype=torch.float)
    merchant_feats = torch.tensor(scaler.fit_transform(merchant_feats.drop(columns=['merch_id'])), dtype=torch.float)

    # Create node indexes for transactions, users, and merchants
    transaction_nodes = df['trans_id'].unique()
    user_nodes = df['cc_num'].unique()
    merchant_nodes = df['merch_id'].unique()

    # Creates mapping dictionaries to convert raw IDs into sequential numerical indices
    transaction_map = {j: i for i, j in enumerate(transaction_nodes)}
    user_map = {j: i for i, j in enumerate(user_nodes)}
    merchant_map = {j: i for i, j in enumerate(merchant_nodes)}

    # Create edge indexes for different relationships
    transaction_to_user = df[['trans_id', 'cc_num']].astype(int)
    transaction_to_merchant = df[['trans_id', 'merch_id']].astype(int)
    user_to_merchant = df[['cc_num', 'merch_id']].drop_duplicates().astype(int).reset_index(drop=True)

    transaction_to_user['trans_id'] = transaction_to_user['trans_id'].map(transaction_map)
    transaction_to_user['cc_num'] = transaction_to_user['cc_num'].map(user_map)

    transaction_to_merchant['trans_id'] = transaction_to_merchant['trans_id'].map(transaction_map)
    transaction_to_merchant['merch_id'] = transaction_to_merchant['merch_id'].map(merchant_map)

    user_to_merchant['cc_num'] = user_to_merchant['cc_num'].map(user_map)
    user_to_merchant['merch_id'] = user_to_merchant['merch_id'].map(merchant_map)

    # Construct graph
    graph_data = {
        ('user', 'user<>transaction', 'transaction'): (transaction_to_user['cc_num'], transaction_to_user['trans_id']),
        ('merchant', 'merchant<>transaction', 'transaction'): (
            transaction_to_merchant['merch_id'], transaction_to_merchant['trans_id']),
        ('transaction', 'transaction<>user', 'user'): (transaction_to_user['trans_id'], transaction_to_user['cc_num']),
        ('transaction', 'transaction<>merchant', 'merchant'): (
            transaction_to_merchant['trans_id'], transaction_to_merchant['merch_id']),
        ('transaction', 'self_relation_transaction', 'transaction'): (
            transaction_to_user['trans_id'], transaction_to_user['trans_id']),
        ('user', 'user<>merchant', 'merchant'): (user_to_merchant['cc_num'], user_to_merchant['merch_id']),
        ('merchant', 'merchant<>user', 'user'): (user_to_merchant['merch_id'], user_to_merchant['cc_num']),
    }

    print("User-to-Transaction Edges:", len(transaction_to_user))
    print("Transaction-to-Merchant Edges:", len(transaction_to_merchant))
    print("User-to-Merchant Edges:", len(user_to_merchant))

    g = dgl.heterograph(graph_data)

    # Assign fraud labels ('is_fraud') to transaction nodes
    g.nodes['transaction'].data['y'] = torch.tensor(df['is_fraud'].values, dtype=torch.float)

    return g, transaction_feats, user_feats, merchant_feats, classified_idx


def print_graph_info(g):
    """
    Prints graph properties and information
    """
    print(g)
    print("Graph properties: ")
    print("Total number of nodes in graph: ", g.num_nodes())
    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
    print("Number of nodes for each node type:", ntype_dict)
    print("Edge Dictionary for different edge types: ")
    print("User_To_Transaction Edges: ", g.edges(etype='user<>transaction'))
    print("Merchant_To_Transaction Edges: ", g.edges(etype='merchant<>transaction'))
    print("Transaction_Self_Loop: ", g.edges(etype='self_relation_transaction'))
    print("Transaction_To_User Edges: ", g.edges(etype='transaction<>user'))
    print("Transaction_To_Merchant Edges: ", g.edges(etype='transaction<>merchant'))
    print("User_To_Merchant Edges: ", g.edges(etype='user<>merchant'))
    print("Merchant_To_User Edges: ", g.edges(etype='merchant<>user'))


def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    aucpr = auc(recall_curve, precision_curve)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc, aucpr, cm


def print_metrics(accuracy, precision, recall, f1, roc_auc, aucpr, m_name):
    results = prettytable.PrettyTable(title=f'{m_name} Results')
    results.field_names = ["Metric", "Value"]
    results.add_row(["Accuracy", accuracy])
    results.add_row(["Precision", precision])
    results.add_row(["Recall", recall])
    results.add_row(["F1 Score", f1])
    results.add_row(["ROC AUC", roc_auc])
    results.add_row(["AUCPR", aucpr])
    print(results)


class GNN_Trainer:
    def __init__(self, model):
        self.model = model

    def save_model(self, m_name, save_dir="models"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'model_{m_name}.pt')
        torch.save(self.model.state_dict(), save_path)

    def train_val(self, g, features_dict, num_epochs, train_idx, val_idx, optimizer, criterion, best_val_f1, m_name,
                  labels, target_node):
        total_loss = 0
        for epoch in range(num_epochs):
            start_time = time.time()
            self.model.train()
            optimizer.zero_grad()
            out = self.model(g, features_dict, target_node)
            pred_c = out.argmax(1)
            loss = criterion(out[train_idx], labels[train_idx])
            pred_scores = pred_c[train_idx]
            pred = pred_scores > 0.5
            accuracy, precision, recall, f1, _, _, _ = get_metrics(labels[train_idx], pred)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            duration = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f} - Accuracy Train: {accuracy:.4f} - F1 Score Train: {f1:.2f} - Duration: {duration:.2f}s")

            self.model.eval()
            with torch.no_grad():
                pred_scores = pred_c[val_idx]
                pred = pred_scores > 0.5
                accuracy, precision, recall, f1, _, _, _ = get_metrics(labels[val_idx], pred)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    self.save_model(m_name)
                    print('Model Saved !!')
            print(f"Validation - Loss: {loss.item():.4f} - Accuracy Val: {accuracy:.4f} - F1 Score Val: {f1:.2f}")

    def predict(self, g, features_dict, test_idx, model_path, m_name, target_node, labels):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            out = self.model(g, features_dict, target_node)
            pred_c = out.argmax(1)
            pred_scores = pred_c[test_idx]
            pred = pred_scores > 0.5
            targets = labels[test_idx].cpu().numpy()
            outputs = pred.cpu().numpy()
            probs = F.softmax(out[test_idx], dim=1).cpu().numpy()
            accuracy, precision, recall, f1, roc_auc, auc_pr, _ = get_metrics(targets, outputs)
            print_metrics(accuracy, precision, recall, f1, roc_auc, auc_pr, m_name)
            return accuracy, precision, recall, f1, roc_auc, auc_pr, targets, outputs, probs


def visualize_sampled_hetero_graph(dgl_graph, sample_size=500):
    """
    Visualizes a sampled subgraph of a DGL heterogeneous graph using networkx.

    :param dgl_graph: The DGL heterograph.
    :param sample_size: Number of nodes to sample for visualization.
    """
    # Randomly sample nodes per node type
    sampled_nodes = {}
    for ntype in dgl_graph.ntypes:
        num_nodes = dgl_graph.num_nodes(ntype)
        sampled_nodes[ntype] = torch.randperm(num_nodes)[:min(sample_size, num_nodes)]

    # Extract the sampled subgraph
    sampled_graph = dgl.node_subgraph(dgl_graph, sampled_nodes)

    # Convert subgraph to networkx
    nx_graph = nx.Graph()

    # Map node types to colors
    color_map = {
        'transaction': 'blue',
        'user': 'green',
        'merchant': 'red'
    }

    # Add sampled edges to networkx
    node_type_mapping = {}
    node_index_offset = 0

    for ntype in sampled_graph.ntypes:
        num_nodes = sampled_graph.num_nodes(ntype)
        for i in range(num_nodes):
            node_type_mapping[node_index_offset + i] = color_map.get(ntype, 'black')
        node_index_offset += num_nodes

    for edge_type in sampled_graph.canonical_etypes:
        src, etype, dst = edge_type
        edges = sampled_graph.edges(etype=edge_type)
        nx_graph.add_edges_from(zip(edges[0].numpy(), edges[1].numpy()), label=etype)

    # Assign colors correctly
    node_colors = [node_type_mapping.get(n, 'black') for n in nx_graph.nodes]

    # Draw the sampled graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(nx_graph, seed=42)  # Layout for positioning
    nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=100)
    plt.title(f"Sampled Heterogeneous Graph Visualization ({sample_size} nodes)")
    plt.show()
