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
from components.utils import get_edge_index_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.load("hetero_graph.pt")

# Load YAML configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "components", "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load GNN hyperparameters from config
gnn_config = config["gnn"]["sets"][0]  # Select first set, change index for different configurations

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
# --- Randomly Split the User Nodes into Train and Test Sets ---
num_users = data['user'].num_nodes
indices = torch.randperm(num_users)       # Random permutation of user node indices
train_size = int(0.8 * num_users)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

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

num_epochs = gnn_config["num_epochs"]
batch_size = gnn_config["batch_size"]
num_neighbors = gnn_config["num_neighbors"]

# Create DataLoaders
train_loader = NeighborLoader(
    data,
    num_neighbors=num_neighbors,
    input_nodes=('user', data['user'].train_mask),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=num_neighbors,
    input_nodes=('user', data['user'].test_mask),
    batch_size=batch_size,
    shuffle=False,
)

# Initialize GNN model
# model = Models.get_gnn_model(in_channels_dict, hidden_channels=32, out_channels=2).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        # Extract node features from the mini-batch for each node type.
        x_dict = {
            'user': batch['user'].x,
            'question': batch['question'].x,
            'answer': batch['answer'].x
        }
        # Build the edge index dictionary from the subgraph in the batch.
        edge_index_dict = get_edge_index_dict(batch)
        out = model(x_dict, edge_index_dict)
        # Compute loss on the current mini-batch of user nodes.
        loss = F.cross_entropy(out, batch['user'].y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# --- Evaluation Function Using the DataLoader ---
@torch.no_grad()
def test():
    model.eval()
    total_correct = 0
    total_examples = 0
    all_preds = []
    all_labels = []
    for batch in test_loader:
        x_dict = {
            'user': batch['user'].x,
            'question': batch['question'].x,
            'answer': batch['answer'].x
        }
        edge_index_dict = get_edge_index_dict(batch)
        out = model(x_dict, edge_index_dict)
        preds = out.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(batch['user'].y.cpu())
        total_correct += (preds == batch['user'].y).sum().item()
        total_examples += batch['user'].y.size(0)
    # accuracy = total_correct / total_examples
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    metrics = Metrics.compute_metrics(all_labels, all_preds)
    
    return metrics

# Training loop
# num_epochs = 50
for epoch in range(1, num_epochs + 1):
    loss = train()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} - Loss: {loss:.4f}")

# Evaluate model
test_metrics = test()

print(f"Test Performance Metrics - Recall: {test_metrics['recall']:.4f}")
print(f"Test Performance Metrics - Precision: {test_metrics['precision']:.4f}")
print(f"Test Performance Metrics - F1-score: {test_metrics['f1_score']:.4f}")
print(f"Test Performance Metrics - Accuracy: {test_metrics['accuracy']:.4f}")

