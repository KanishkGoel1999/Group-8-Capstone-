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
from components.constants import EDGE_TYPES, DATASET_TYPE_1
from components.utils import get_edge_index_dict, train_mini_batch, train_full_batch, test_mini_batch, test_full_batch, set_seed, get_in_channels, parse_args, read_yaml
import matplotlib.pyplot as plt
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on a specified dataset and config set")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., STACK_OVERFLOW or ASK_REDDIT)")
    parser.add_argument("config_index", type=int, help="Index of the GNN configuration set from YAML")
    return parser.parse_args()

# def read_yaml(set_num):
#     # Load YAML configuration
#     config_path = os.path.join(os.path.dirname(__file__), "..", "components", "config.yaml")
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)

#     # Load GNN hyperparameters from config
#     gnn_config = config["gnn"]["sets"][set_num]  # change index for different configurations
#     return gnn_config
    # num_epochs = gnn_config["num_epochs"]
    # batch_size = gnn_config["batch_size"]
    # num_neighbors = gnn_config["num_neighbors"]

def split_data(gnn_config, dataset):
    # Apply RandomNodeSplit transformation
    transform = RandomNodeSplit(split=gnn_config["split"], num_val=0.1, num_test=0.1)
    data = transform(dataset)
    data = data.to(device)

    return data

def add_reverse_and_self_loop_edges(data):
    # Reverse for user -> question (asks)
    if EDGE_TYPES['ASKS'] in data.edge_types:
        edge_index = data[EDGE_TYPES['ASKS']].edge_index
        data[EDGE_TYPES['REV_ASKS']].edge_index = edge_index.flip(0)

    # Reverse for user -> answer (answers)
    if EDGE_TYPES['ANSWERS'] in data.edge_types:
        edge_index = data[EDGE_TYPES['ANSWERS']].edge_index
        data[EDGE_TYPES['REV_ANSWERS']].edge_index = edge_index.flip(0)

    # Reverse for question -> answer (has)
    if EDGE_TYPES['HAS'] in data.edge_types:
        edge_index = data[EDGE_TYPES['HAS']].edge_index
        data[EDGE_TYPES['REV_HAS']].edge_index = edge_index.flip(0)

    # Reverse for question -> answer (accepted_answer)
    if EDGE_TYPES['ACCEPTED_ANSWER'] in data.edge_types:
        edge_index = data[EDGE_TYPES['ACCEPTED_ANSWER']].edge_index
        data[EDGE_TYPES['REV_ACCEPTED']].edge_index = edge_index.flip(0)

    # Add self-loop for user nodes: ensure each user node receives its own message.
    num_users = data['user'].num_nodes
    row = torch.arange(num_users, dtype=torch.long)
    self_loop_edge = torch.stack([row, row], dim=0)
    data[EDGE_TYPES['SELF_LOOP']].edge_index = self_loop_edge

    return data

def initialize_model(in_channels_dict):
    # Initialize GNN model
    model = Models.get_gnn_model(
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
    
def train(gnn_config, model, optimizer, loader):
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
            loss, acc = train_mini_batch(model, loader, optimizer)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.load("hetero_graph.pt", weights_only=False)
    gnn_config = read_yaml(config_index)
    data = split_data(gnn_config, dataset)
    data = add_reverse_and_self_loop_edges(data)
    edge_index_dict = get_edge_index_dict(data)
    in_channels_dict = get_in_channels(dataset_name, data)
    model = initialize_model(in_channels_dict)
    optimizer = initialize_optimizer(model, gnn_config)
    if dataset_name == DATASET_TYPE_1:
        batches = stratified_input_batches(data, 'user', mask_key='train_mask', label_key='y', batch_size=64)
    else:
        batches = stratified_input_batches(data, 'author', mask_key='train_mask', label_key='y', batch_size=64)
    train_loaders = loader(gnn_config['batch_size'], gnn_config['num_neighbors'])
    model, train_losses, train_accuracies = train(gnn_config, model, optimizer, train_loaders)
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


