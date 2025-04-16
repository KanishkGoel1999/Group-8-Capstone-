import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
#import dgl
#from dgl.nn import RelGraphConv
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import networkx as nx
import prettytable
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import NeighborLoader

def create_pyg_graph(df, transaction_feats, merchant_feats, user_feats):
    # Convert dataframe values to float32 for tensor compatibility
    df = df.astype(np.float32)

    # Normalize and convert features to tensors
    scaler = StandardScaler()
    transaction_feats_tensor = torch.tensor(scaler.fit_transform(transaction_feats.drop(columns=['transaction_id'])),
                                            dtype=torch.float)
    user_feats_tensor = torch.tensor(scaler.fit_transform(user_feats.drop(columns=['card_number'])), dtype=torch.float)
    merchant_feats_tensor = torch.tensor(scaler.fit_transform(merchant_feats.drop(columns=['merchant_id'])),
                                         dtype=torch.float)

    # Create mappings from unique identifiers to index values
    transaction_map = {j: i for i, j in enumerate(df['transaction_id'].unique())}
    user_map = {j: i for i, j in enumerate(df['card_number'].unique())}
    merchant_map = {j: i for i, j in enumerate(df['merchant_id'].unique())}

    # Create edge index tensors for different relationships
    transaction_to_user = df[['transaction_id', 'card_number']].astype(int)
    transaction_to_merchant = df[['transaction_id', 'merchant_id']].astype(int)
    user_to_merchant = df[['card_number', 'merchant_id']].drop_duplicates().astype(int).reset_index(drop=True)

    transaction_to_user['transaction_id'] = transaction_to_user['transaction_id'].map(transaction_map)
    transaction_to_user['card_number'] = transaction_to_user['card_number'].map(user_map)

    transaction_to_merchant['transaction_id'] = transaction_to_merchant['transaction_id'].map(transaction_map)
    transaction_to_merchant['merchant_id'] = transaction_to_merchant['merchant_id'].map(merchant_map)

    user_to_merchant['card_number'] = user_to_merchant['card_number'].map(user_map)
    user_to_merchant['merchant_id'] = user_to_merchant['merchant_id'].map(merchant_map)

    # Convert edge lists to tensors
    edge_index_transaction_user = torch.tensor(
        [transaction_to_user['transaction_id'].values, transaction_to_user['card_number'].values], dtype=torch.long)
    edge_index_transaction_merchant = torch.tensor(
        [transaction_to_merchant['transaction_id'].values, transaction_to_merchant['merchant_id'].values], dtype=torch.long)
    edge_index_user_merchant = torch.tensor([user_to_merchant['card_number'].values, user_to_merchant['merchant_id'].values],
                                            dtype=torch.long)

    # Construct the Heterogeneous Graph
    data = HeteroData()

    # Assign node features
    data['transaction'].x = transaction_feats_tensor
    data['user'].x = user_feats_tensor
    data['merchant'].x = merchant_feats_tensor

    # Assign edges (bi-directional relations)
    data['transaction', 'transaction_to_user', 'user'].edge_index = edge_index_transaction_user
    data['user', 'user_to_transaction', 'transaction'].edge_index = edge_index_transaction_user.flip(0)

    data['transaction', 'transaction_to_merchant', 'merchant'].edge_index = edge_index_transaction_merchant
    data['merchant', 'merchant_to_transaction', 'transaction'].edge_index = edge_index_transaction_merchant.flip(0)

    data['user', 'user_to_merchant', 'merchant'].edge_index = edge_index_user_merchant
    data['merchant', 'merchant_to_user', 'user'].edge_index = edge_index_user_merchant.flip(0)

    # Self-relation for transactions
    data['transaction', 'self_relation_transaction', 'transaction'].edge_index = torch.vstack(
        [edge_index_transaction_user[0], edge_index_transaction_user[0]])

    # Assign fraud labels ('is_fraud') to transaction nodes
    data['transaction'].y = torch.tensor(df['is_fraud'].values, dtype=torch.float)

    return data


# DGL
def create_graph_dgl(df, transaction_feats, merchant_feats, user_feats):
    # Ensure DGL is using CPU
    torch.set_default_device(torch.device("cpu"))

    # For tensors
    df = df.astype(np.float32)

    # Saves the indexes of the transaction nodes.
    classified_idx = transaction_feats.index

    # Normalize features and convert to tensors
    scaler = StandardScaler()
    transaction_feats = torch.tensor(scaler.fit_transform(transaction_feats.drop(columns=['transaction_id'])),
                                     dtype=torch.float)
    user_feats = torch.tensor(scaler.fit_transform(user_feats.drop(columns=['card_number'])), dtype=torch.float)
    merchant_feats = torch.tensor(scaler.fit_transform(merchant_feats.drop(columns=['merchant_id'])), dtype=torch.float)

    # Create node indexes for transactions, users, and merchants
    transaction_nodes = df['transaction_id'].unique()
    user_nodes = df['card_number'].unique()
    merchant_nodes = df['merchant_id'].unique()

    # Creates mappings from original IDs to numerical indices
    transaction_map = {j: i for i, j in enumerate(transaction_nodes)}
    user_map = {j: i for i, j in enumerate(user_nodes)}
    merchant_map = {j: i for i, j in enumerate(merchant_nodes)}

    # Create edge indexes for different relationships
    transaction_to_user = df[['transaction_id', 'card_number']].astype(int)
    transaction_to_merchant = df[['transaction_id', 'merchant_id']].astype(int)
    user_to_merchant = df[['card_number', 'merchant_id']].drop_duplicates().astype(int).reset_index(drop=True)

    transaction_to_user['transaction_id'] = transaction_to_user['transaction_id'].map(transaction_map)
    transaction_to_user['card_number'] = transaction_to_user['card_number'].map(user_map)

    transaction_to_merchant['transaction_id'] = transaction_to_merchant['transaction_id'].map(transaction_map)
    transaction_to_merchant['merchant_id'] = transaction_to_merchant['merchant_id'].map(merchant_map)

    user_to_merchant['card_number'] = user_to_merchant['card_number'].map(user_map)
    user_to_merchant['merchant_id'] = user_to_merchant['merchant_id'].map(merchant_map)

    # Construct graph
    graph_data = {
        ('user', 'user<>transaction', 'transaction'): (transaction_to_user['card_number'], transaction_to_user['transaction_id']),
        ('merchant', 'merchant<>transaction', 'transaction'): (
            transaction_to_merchant['merchant_id'], transaction_to_merchant['transaction_id']),
        ('transaction', 'transaction<>user', 'user'): (transaction_to_user['transaction_id'], transaction_to_user['card_number']),
        ('transaction', 'transaction<>merchant', 'merchant'): (
            transaction_to_merchant['transaction_id'], transaction_to_merchant['merchant_id']),
        ('transaction', 'self_relation_transaction', 'transaction'): (
            transaction_to_user['transaction_id'], transaction_to_user['transaction_id']),
        ('user', 'user<>merchant', 'merchant'): (user_to_merchant['card_number'], user_to_merchant['merchant_id']),
        ('merchant', 'merchant<>user', 'user'): (user_to_merchant['merchant_id'], user_to_merchant['card_number']),
    }

    g = dgl.heterograph(graph_data)

    # Assign fraud labels ('is_fraud') to transaction nodes
    g.nodes['transaction'].data['y'] = torch.tensor(df['is_fraud'].values, dtype=torch.float)

    return g, transaction_feats, user_feats, merchant_feats, classified_idx


def print_graph_info_dgl(g):
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


def print_pyg_graph_info(data):
    """
    Prints PyG graph properties and information
    """
    print("Heterogeneous Graph Summary:")
    print(data)

    # Print total number of nodes for each node type
    print("\nGraph Properties:")
    for node_type in data.node_types:
        print(f"Total number of '{node_type}' nodes: {data[node_type].num_nodes}")

    # Print edge index details
    print("\nEdge Dictionary for different edge types:")
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        edge_index = data[edge_type].edge_index
        print(f"Edge Type: {src} -> {rel} -> {dst}")
        print(f"  Number of edges: {edge_index.shape[1]}")
        print(
            f"  Edge indices (first 5 shown):\n  {edge_index[:, :5].tolist() if edge_index.numel() > 0 else 'No edges'}")


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


def train_pyg_model(model, data, train_idx, valid_idx, num_epochs=20, lr=0.001, weight_decay=5e-4, batch_size=128,
                    m_name="best_pyg_model.pth"):
    """
    Trains the PyG model and saves the best one based on F1-score.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # DataLoader for efficient training
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [5, 5] for key in data.edge_types},
        input_nodes=('transaction', torch.tensor(train_idx, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = NeighborLoader(
        data,
        num_neighbors={key: [5, 5] for key in data.edge_types},
        input_nodes=('transaction', torch.tensor(valid_idx, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False
    )

    best_f1 = -np.inf  # Track best F1-score
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", m_name)

    losses = []
    f1_scores = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(batch)

            # Compute loss
            loss = criterion(out, batch['transaction'].y.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Get predictions
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch['transaction'].y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_train_loss = total_loss / len(train_loader)

        # Compute training metrics
        train_accuracy, train_precision, train_recall, train_f1, _, _, _= get_metrics(all_labels, all_preds)

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                out = model(batch)

                # Compute loss
                loss = criterion(out, batch['transaction'].y.long())
                total_val_loss += loss.item()

                # Get predictions
                preds = out.argmax(dim=1).cpu().numpy()
                labels = batch['transaction'].y.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_val_loss = total_val_loss / len(valid_loader)

        # Compute validation metrics
        val_accuracy, val_precision, val_recall, val_f1, _, _, _ = get_metrics(all_labels, all_preds)

        duration = time.time() - start_time

        # Print metrics
        print(f"Epoch {epoch + 1}/{num_epochs} - Time: {duration:.2f}s")
        print(f"Train - Loss: {avg_train_loss:.4f} - Acc: {train_accuracy:.4f} - F1: {train_f1:.4f}")
        print(f"Valid - Loss: {avg_val_loss:.4f} - Acc: {val_accuracy:.4f} - F1: {val_f1:.4f}")

        losses.append(avg_val_loss)
        f1_scores.append(val_f1)

        # Save best model based on F1-score
        if val_f1 > best_f1:
            best_f1 = val_f1
            print("Saving Model Keys:")
            print(model.state_dict().keys())
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1-score: {best_f1:.4f}!")

    print(f"Training complete. Best model saved at: {best_model_path}")

    return f1_scores, losses


def train_pyg_model_ES(model, data, train_idx, valid_idx, num_epochs=20, lr=0.001, weight_decay=5e-4, batch_size=128,
                    m_name="best_pyg_model.pth", patience=10, min_epochs=25):
    """
    Trains the PyG model and saves the best one based on F1-score.
    Stops early if F1-score does not improve for 'patience' epochs.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # DataLoaders
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [5, 5] for key in data.edge_types},
        input_nodes=('transaction', torch.tensor(train_idx, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = NeighborLoader(
        data,
        num_neighbors={key: [5, 5] for key in data.edge_types},
        input_nodes=('transaction', torch.tensor(valid_idx, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False
    )

    best_f1 = -np.inf
    epochs_since_improvement = 0
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", m_name)

    losses = []
    f1_scores = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch['transaction'].y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch['transaction'].y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy, _, _, train_f1, _, _, _ = get_metrics(all_labels, all_preds)

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch['transaction'].y.long())
                total_val_loss += loss.item()
                preds = out.argmax(dim=1).cpu().numpy()
                labels = batch['transaction'].y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_val_loss = total_val_loss / len(valid_loader)
        val_accuracy, _, _, val_f1, _, _, _ = get_metrics(all_labels, all_preds)

        duration = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} - Time: {duration:.2f}s")
        print(f"Train - Loss: {avg_train_loss:.4f} - Acc: {train_accuracy:.4f} - F1: {train_f1:.4f}")
        print(f"Valid - Loss: {avg_val_loss:.4f} - Acc: {val_accuracy:.4f} - F1: {val_f1:.4f}")

        losses.append(avg_val_loss)
        f1_scores.append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_since_improvement = 0
            print("Saving Model Keys:")
            print(model.state_dict().keys())
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1-score: {best_f1:.4f}!")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s).")

        # Early stopping condition
        if epoch + 1 >= min_epochs and epochs_since_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1} (no improvement for {patience} epochs).")
            break

    print(f"Training complete. Best model saved at: {best_model_path}")
    return f1_scores, losses

def train_pyg_model_without_dataloader(model, data, train_idx, valid_idx, num_epochs=20, lr=0.01, weight_decay=5e-4, m_name="best_pyg_model.pth"):
    """
    Trains the PyG model without using DataLoader, processing the full dataset in each epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", m_name)
    best_f1 = -np.inf  # Track best F1-score

    # Move data to device
    data = data.to(device)

    # Extract train and validation labels
    labels = data['transaction'].y.long()
    losses = []
    f1_scores = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        optimizer.zero_grad()

        # Forward pass (entire dataset at once)
        out = model(data)

        # Compute loss on training nodes only
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Get predictions
        preds_train = out[train_idx].argmax(dim=1).cpu().numpy()
        labels_train = labels[train_idx].cpu().numpy()

        # Compute training metrics
        train_accuracy, train_precision, train_recall, train_f1, _, _, _ = get_metrics(labels_train, preds_train)

        # Validation phase
        model.eval()
        with torch.no_grad():
            out_valid = model(data)

            # Compute loss on validation nodes
            val_loss = criterion(out_valid[valid_idx], labels[valid_idx])

            # Get predictions
            preds_valid = out_valid[valid_idx].argmax(dim=1).cpu().numpy()
            labels_valid = labels[valid_idx].cpu().numpy()

            # Compute validation metrics
            val_accuracy, val_precision, val_recall, val_f1, _, _, _ = get_metrics(labels_valid, preds_valid)

        duration = time.time() - start_time

        # Print metrics
        print(f"Epoch {epoch + 1}/{num_epochs} - Time: {duration:.2f}s")
        print(f"Train - Loss: {loss.item():.4f} - Acc: {train_accuracy:.4f} - F1: {train_f1:.4f}")
        print(f"Valid - Loss: {val_loss.item():.4f} - Acc: {val_accuracy:.4f} - F1: {val_f1:.4f}")

        losses.append(val_loss.item())
        f1_scores.append(val_f1())
        # Save best model based on F1-score
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1-score: {best_f1:.4f}!")

    print(f"Training complete. Best model saved at: {best_model_path}")


def visualize_loss(epochs, f1_scores, losses):
    # Plot Loss and F1-score
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), f1_scores, label='F1 Score', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training F1 Score')
    plt.legend()

    plt.show()
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
