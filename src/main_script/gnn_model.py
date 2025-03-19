# gnn_script.py
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
from components.constants import EDGE_TYPES
from components.utils import get_edge_index_dict, train_mini_batch, train_full_batch, test_mini_batch, test_full_batch

import matplotlib.pyplot as plt

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.load("hetero_graph.pt")

# Load YAML configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "components", "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load GNN hyperparameters from config
gnn_config = config["gnn"]["sets"][1]  # change index for different configurations
num_epochs = gnn_config["num_epochs"]
batch_size = gnn_config["batch_size"]
num_neighbors = gnn_config["num_neighbors"]
print(batch_size)
# Apply RandomNodeSplit transformation
transform = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
data = transform(data)

data = data.to(device)
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


data = add_reverse_and_self_loop_edges(data)
edge_index_dict = get_edge_index_dict(data)

# Define input channels for GNN model
in_channels_dict = {
    'user': data['user'].x.size(-1),
    'question': data['question'].x.size(-1),
    'answer': data['answer'].x.size(-1)
}

# Initialize GNN model
model = Models.get_gnn_model(
    in_channels_dict,
    hidden_channels=int(gnn_config["hidden_channels"]),
    out_channels=int(gnn_config["out_channels"])
).to(device)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=float(gnn_config["learning_rate"]), 
    weight_decay=float(gnn_config["weight_decay"])
)

train_mask = data['user'].train_mask
test_mask = data['user'].test_mask
if batch_size != 0: # mini-batch or full-batch condition
    print('Enteredddddddddddddd')
    # Create DataLoaders
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=('user', train_mask),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=('user', test_mask),
        batch_size=batch_size,
        shuffle=False,
    )

for epoch in range(1, num_epochs + 1):
    if batch_size != 0:
        loss, accuracy = train_mini_batch(model, train_loader, optimizer)
        test_metrics = test_mini_batch(model, test_loader)
    else:
        loss, accuracy = train_full_batch(model, optimizer, data, train_mask)
        test_metrics = test_full_batch(model, data, test_mask)

    # Collect loss and accuracy for plotting
    train_losses.append(loss)
    train_accuracies.append(accuracy)
    test_losses.append(test_metrics["loss"])
    test_accuracies.append(test_metrics["accuracy"])

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} - Train Loss: {loss:.4f}, Test Loss: {test_metrics['loss']:.4f}")
        print(f"Epoch {epoch:03d} - Train Accuracy: {accuracy:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")

print(f"Test Performance Metrics - Recall: {test_metrics['recall']:.4f}")
print(f"Test Performance Metrics - Precision: {test_metrics['precision']:.4f}")
print(f"Test Performance Metrics - F1-score: {test_metrics['f1_score']:.4f}")
print(f"Test Performance Metrics - Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test Performance Metrics - AUC: {test_metrics['auc']:.4f}")

plt.figure(figsize=(12, 5))

# Subplot 1: Train Loss vs Test Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="red")
plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("GNN Train vs Test Loss")
plt.legend()

# Subplot 2: Train Accuracy vs Test Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", color="green")
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("GNN Train vs Test Accuracy")
plt.legend()

plt.show()
