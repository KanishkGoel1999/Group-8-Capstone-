# gnn_script.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.metric import Metrics
from components.model import Models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.load("hetero_graph.pt")

# Apply RandomNodeSplit transformation
transform = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
data = transform(data)

data = data.to(device)
def add_reverse_and_self_loop_edges(data):
    # Reverse for user -> question (asks)
    if ('user', 'asks', 'question') in data.edge_types:
        edge_index = data[('user', 'asks', 'question')].edge_index
        data[('question', 'rev_asks', 'user')].edge_index = edge_index.flip(0)

    # Reverse for user -> answer (answers)
    if ('user', 'answers', 'answer') in data.edge_types:
        edge_index = data[('user', 'answers', 'answer')].edge_index
        data[('answer', 'rev_answers', 'user')].edge_index = edge_index.flip(0)

    # Reverse for question -> answer (has)
    if ('question', 'has', 'answer') in data.edge_types:
        edge_index = data[('question', 'has', 'answer')].edge_index
        data[('answer', 'rev_has', 'question')].edge_index = edge_index.flip(0)

    # Reverse for question -> answer (accepted_answer)
    if ('question', 'accepted_answer', 'answer') in data.edge_types:
        edge_index = data[('question', 'accepted_answer', 'answer')].edge_index
        data[('answer', 'rev_accepted', 'question')].edge_index = edge_index.flip(0)

    # Add self-loop for user nodes: ensure each user node receives its own message.
    num_users = data['user'].num_nodes
    row = torch.arange(num_users, dtype=torch.long)
    self_loop_edge = torch.stack([row, row], dim=0)
    data[('user', 'self_loop', 'user')].edge_index = self_loop_edge

    return data


data = add_reverse_and_self_loop_edges(data)

edge_index_dict = {
    ('user', 'asks', 'question'): data[('user', 'asks', 'question')].edge_index,
    ('question', 'rev_asks', 'user'): data[('question', 'rev_asks', 'user')].edge_index,
    ('question', 'has', 'answer'): data[('question', 'has', 'answer')].edge_index,
    ('answer', 'rev_has', 'question'): data[('answer', 'rev_has', 'question')].edge_index,
    ('user', 'answers', 'answer'): data[('user', 'answers', 'answer')].edge_index,
    ('answer', 'rev_answers', 'user'): data[('answer', 'rev_answers', 'user')].edge_index,
    ('question', 'accepted_answer', 'answer'): data[('question', 'accepted_answer', 'answer')].edge_index,
    ('answer', 'rev_accepted', 'question'): data[('answer', 'rev_accepted', 'question')].edge_index,
    # Include the self-loop relation for user nodes:
    ('user', 'self_loop', 'user'): data[('user', 'self_loop', 'user')].edge_index,
}
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
model = Models.get_gnn_model(in_channels_dict, hidden_channels=32, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# Create DataLoaders
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    input_nodes=('user', data['user'].train_mask),
    batch_size=64,
    shuffle=True,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    input_nodes=('user', data['user'].test_mask),
    batch_size=64,
    shuffle=False,
)

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
        edge_index_dict = {
            ('user', 'asks', 'question'): batch[('user', 'asks', 'question')].edge_index,
            ('question', 'rev_asks', 'user'): batch[('question', 'rev_asks', 'user')].edge_index,
            ('question', 'has', 'answer'): batch[('question', 'has', 'answer')].edge_index,
            ('answer', 'rev_has', 'question'): batch[('answer', 'rev_has', 'question')].edge_index,
            ('user', 'answers', 'answer'): batch[('user', 'answers', 'answer')].edge_index,
            ('answer', 'rev_answers', 'user'): batch[('answer', 'rev_answers', 'user')].edge_index,
            ('question', 'accepted_answer', 'answer'): batch[('question', 'accepted_answer', 'answer')].edge_index,
            ('answer', 'rev_accepted', 'question'): batch[('answer', 'rev_accepted', 'question')].edge_index,
            ('user', 'self_loop', 'user'): batch[('user', 'self_loop', 'user')].edge_index,
        }
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
        edge_index_dict = {
            ('user', 'asks', 'question'): batch[('user', 'asks', 'question')].edge_index,
            ('question', 'rev_asks', 'user'): batch[('question', 'rev_asks', 'user')].edge_index,
            ('question', 'has', 'answer'): batch[('question', 'has', 'answer')].edge_index,
            ('answer', 'rev_has', 'question'): batch[('answer', 'rev_has', 'question')].edge_index,
            ('user', 'answers', 'answer'): batch[('user', 'answers', 'answer')].edge_index,
            ('answer', 'rev_answers', 'user'): batch[('answer', 'rev_answers', 'user')].edge_index,
            ('question', 'accepted_answer', 'answer'): batch[('question', 'accepted_answer', 'answer')].edge_index,
            ('answer', 'rev_accepted', 'question'): batch[('answer', 'rev_accepted', 'question')].edge_index,
            ('user', 'self_loop', 'user'): batch[('user', 'self_loop', 'user')].edge_index,
        }
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
num_epochs = 50
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

