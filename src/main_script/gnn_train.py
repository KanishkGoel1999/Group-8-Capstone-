import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.metric import Metrics
from components.model import Models
from components.constants import EDGE_TYPES, DATASET_TYPE_1, EDGE_TYPES_DICT
from components.utils import get_edge_index_dict, train_mini_batch, train_full_batch, test_mini_batch, test_full_batch, set_seed, get_in_channels, parse_args, read_yaml
import matplotlib.pyplot as plt
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on a specified dataset and config set")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., STACK_OVERFLOW or ASK_REDDIT)")
    parser.add_argument("config_index", type=int, help="Index of the GNN configuration set from YAML")
    return parser.parse_args()

def split_data(gnn_config, dataset):
    # Apply RandomNodeSplit transformation
    transform = RandomNodeSplit(split=gnn_config["split"], num_val=0.1, num_test=0.1)
    data = transform(dataset)
    data = data.to(device)

    return data

def add_reverse_and_self_loop_edges(data, dataset_name):
    # edge_types = EDGE_TYPES[dataset_name]
    # Reverse for user -> question (asks)
    if EDGE_TYPES_DICT['ASKS'] in data.edge_types:
        edge_index = data[EDGE_TYPES_DICT['ASKS']].edge_index
        data[EDGE_TYPES_DICT['REV_ASKS']].edge_index = edge_index.flip(0)

    # Reverse for user -> answer (answers)
    if EDGE_TYPES_DICT['ANSWERS'] in data.edge_types:
        edge_index = data[EDGE_TYPES_DICT['ANSWERS']].edge_index
        data[EDGE_TYPES_DICT['REV_ANSWERS']].edge_index = edge_index.flip(0)

    # Reverse for question -> answer (has)
    if EDGE_TYPES_DICT['HAS'] in data.edge_types:
        edge_index = data[EDGE_TYPES_DICT['HAS']].edge_index
        data[EDGE_TYPES_DICT['REV_HAS']].edge_index = edge_index.flip(0)

    # Reverse for question -> answer (accepted_answer)
    if EDGE_TYPES_DICT['ACCEPTED_ANSWER'] in data.edge_types:
        edge_index = data[EDGE_TYPES_DICT['ACCEPTED_ANSWER']].edge_index
        data[EDGE_TYPES_DICT['REV_ACCEPTED']].edge_index = edge_index.flip(0)

    # Add self-loop for user nodes: ensure each user node receives its own message.
    num_users = data['user'].num_nodes
    row = torch.arange(num_users, dtype=torch.long)
    self_loop_edge = torch.stack([row, row], dim=0)
    data[EDGE_TYPES_DICT['SELF_LOOP']].edge_index = self_loop_edge

    return data

# def add_reverse_and_self_loop_edges(data, dataset_name):
#     edge_types = EDGE_TYPES[dataset_name]

#     # Add reverse edges
#     for (src, rel, dst) in edge_types:
#         if rel.startswith("rev_") or rel == "self_loop":
#             continue  # Skip if already reverse or self-loop edge

#         rev_rel = f"rev_{rel}"
#         rev_edge_type = (dst, rev_rel, src)

#         if rev_edge_type not in data.edge_types:
#             edge_index = data[(src, rel, dst)].edge_index
#             if edge_index is not None and edge_index.shape[0] == 2:
#                 data[rev_edge_type].edge_index = edge_index.flip(0).clone()

#     # Add self-loops for each node type
#     for node_type in data.node_types:
#         self_loop_edge_type = (node_type, 'self_loop', node_type)
#         if self_loop_edge_type not in data.edge_types:
#             num_nodes = data[node_type].num_nodes
#             row = torch.arange(num_nodes, dtype=torch.long)
#             edge_index = torch.stack([row, row], dim=0)
#             data[self_loop_edge_type].edge_index = edge_index.clone()

#     return data



def add_reverse_edges_reddit(data):
    # Get a list of all original edge types.
    original_edge_types = list(data.edge_types)
    for (src, rel, dst) in original_edge_types:
        # Define a name for the reverse relation
        reverse_rel = rel + '__reverse'
        reverse_type = (dst, reverse_rel, src)
        # Only add reverse if it doesn't already exist.
        if reverse_type not in data.edge_types:
            edge_index = data[src, rel, dst].edge_index
            # Reverse edge_index (swap sources and targets)
            reverse_edge_index = edge_index.flip(0)
            data[dst, reverse_rel, src].edge_index = reverse_edge_index

    # Add self-loops for author nodes.
    num_authors = data['author'].num_nodes
    self_loop = torch.arange(num_authors, dtype=torch.long)
    self_loop_edge_index = torch.stack([self_loop, self_loop], dim=0)
    data['author', 'self_loop', 'author'].edge_index = self_loop_edge_index

    return data

def initialize_model(dataset_name, in_channels_dict):
    # Initialize GNN model
    model = Models.get_hetero_model(
        dataset_name,
        in_channels_dict,
        hidden_channels=int(gnn_config["hidden_channels"]),
        out_channels=int(gnn_config["out_channels"])
    ).to(device)
    return model

def initialize_optimizer(model, gnn_config):
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(gnn_config["learning_rate"]), 
        weight_decay=float(gnn_config["weight_decay"])
    )
    return optimizer

def stratified_input_batches(data, node_type='user', mask_key='train_mask', label_key='y', batch_size=64):
    torch.manual_seed(42)
    labels = data[node_type][label_key]
    mask = data[node_type][mask_key]
    node_indices = torch.arange(data[node_type].num_nodes)[mask]

    class_indices = {}
    for class_id in labels[node_indices].unique():
        cls = int(class_id)
        idx = node_indices[labels[node_indices] == cls]
        class_indices[cls] = idx[torch.randperm(len(idx))]
    
    num_classes = len(class_indices)
    samples_per_class = batch_size // num_classes
    min_batches = min(len(idxs) // samples_per_class for idxs in class_indices.values())

    batches = []
    for i in range(min_batches):
        batch = torch.cat([
            class_indices[cls][i * samples_per_class: (i + 1) * samples_per_class]
            for cls in class_indices
        ])
        batches.append(batch)
    print('Batches:', len(batches))
    return batches

def loader(batch_size, num_neighbors):
    if dataset_name == DATASET_TYPE_1:
        nodes = 'user'
    else:
        nodes = 'author'
    if batch_size != 0: # mini-batch or full-batch condition
        print('Enteredddddddddddddd')
        train_input_batches = stratified_input_batches(data, batch_size=batch_size)
        torch.manual_seed(42)
        train_loaders = [
            NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                input_nodes=(nodes, batch),
                batch_size=batch_size,
                shuffle=False,
            )
            for batch in train_input_batches
        ]
        return train_loaders
    
def train(gnn_config, model, optimizer, loader, dataset_name):
    train_losses = []
    train_accuracies = []
    for epoch in range(1, gnn_config['num_epochs'] + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_acc = 0
        torch.manual_seed(42)
        for loader in train_loaders:
            loss, acc = train_mini_batch(model, loader, optimizer, dataset_name)
            total_loss += loss
            total_acc += acc
            # total_auc += auc

        epoch_loss = total_loss / len(train_loaders)
        epoch_acc = total_acc / len(train_loaders)
        # epoch_auc = total_auc / len(train_loaders)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        # train_aucs.append(epoch_auc)

        print(f"Epoch {epoch:03d} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
    return model, train_losses, train_accuracies

# Save model
# torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "gnn_model.pth"))
    
if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset
    config_index = args.config_index

    print(f"Dataset selected: {dataset_name}")
    print(f"Config set selected: {config_index}")
    # Determine node type dynamically
    target_node = "user" if dataset_name == DATASET_TYPE_1 else "author"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_name == DATASET_TYPE_1:
        dataset = torch.load("hetero_graph.pt", weights_only=False)
    else:
        dataset = torch.load("reddit_hetero_graph_1.pt", weights_only=False)
    gnn_config = read_yaml(config_index)
    data = split_data(gnn_config, dataset)
    data = add_reverse_and_self_loop_edges(data, dataset_name) if dataset_name == DATASET_TYPE_1 else add_reverse_edges_reddit(data)
    edge_index_dict = get_edge_index_dict(data)   
    in_channels_dict = get_in_channels(dataset_name, data)
    model = initialize_model(dataset_name, in_channels_dict)
    optimizer = initialize_optimizer(model, gnn_config)
    batches = stratified_input_batches(data, target_node, mask_key='train_mask', label_key='y', batch_size=64)
    
    train_loaders = loader(gnn_config['batch_size'], gnn_config['num_neighbors'])
    print('----------------------------')
    model, train_losses, train_accuracies = train(gnn_config, model, optimizer, train_loaders, dataset_name)
    # Save metrics, model, and dataset in model_artifacts/
    artifact_dir = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(artifact_dir, f"gnn_model_{dataset_name}.pth")
    )
    torch.save(data, os.path.join(artifact_dir, f"dataset_graph_{dataset_name}.pt"))
    with open(os.path.join(artifact_dir, f"train_metrics_{dataset_name}.pkl"), "wb") as f:
        pickle.dump({
            "train_losses": train_losses,
            "train_accuracies": train_accuracies
        }, f)
    print("Model, dataset, and training metrics saved!!!")


