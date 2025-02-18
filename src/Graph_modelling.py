#%%
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, RGCNConv
from torch_geometric.loader import NeighborLoader
import numpy as np

# If you prefer GPU (and have one available), uncomment:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
# Load the HeteroData object saved previously
data = torch.load("hetero_graph.pt")

print(data)  # Quick summary of the graph
# Move to GPU if available
data = data.to(device)
#%%
import torch


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

#%%
num_users = data['user'].num_nodes
all_user_indices = torch.arange(num_users)

# Shuffle user indices
perm = torch.randperm(num_users)

# 80% train, 20% test
train_size = int(0.8 * num_users)
test_size  = num_users - train_size

train_idx = perm[:train_size]
test_idx  = perm[train_size:]

# Create boolean masks for users
train_mask = torch.zeros(num_users, dtype=torch.bool)
train_mask[train_idx] = True

test_mask = torch.zeros(num_users, dtype=torch.bool)
test_mask[test_idx] = True

# Store these in data for convenience
data['user'].train_mask = train_mask
data['user'].test_mask = test_mask
#%%
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F


class HeteroGNNWithReverse(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels):
        super().__init__()
        # First layer: use all eight relations plus self-loop for user nodes.
        self.conv1 = HeteroConv({
            ('user', 'asks', 'question'): SAGEConv(in_channels_dict['user'], hidden_channels),
            ('question', 'rev_asks', 'user'): SAGEConv(in_channels_dict['question'], hidden_channels),
            ('question', 'has', 'answer'): SAGEConv(in_channels_dict['question'], hidden_channels),
            ('answer', 'rev_has', 'question'): SAGEConv(in_channels_dict['answer'], hidden_channels),
            ('user', 'answers', 'answer'): SAGEConv(in_channels_dict['user'], hidden_channels),
            ('answer', 'rev_answers', 'user'): SAGEConv(in_channels_dict['answer'], hidden_channels),
            ('question', 'accepted_answer', 'answer'): SAGEConv(in_channels_dict['question'], hidden_channels),
            ('answer', 'rev_accepted', 'question'): SAGEConv(in_channels_dict['answer'], hidden_channels),
            ('user', 'self_loop', 'user'): SAGEConv(in_channels_dict['user'], hidden_channels)
        }, aggr='sum')

        # Second layer: same set of relations (now with hidden_channels inputs)
        self.conv2 = HeteroConv({
            ('user', 'asks', 'question'): SAGEConv(hidden_channels, hidden_channels),
            ('question', 'rev_asks', 'user'): SAGEConv(hidden_channels, hidden_channels),
            ('question', 'has', 'answer'): SAGEConv(hidden_channels, hidden_channels),
            ('answer', 'rev_has', 'question'): SAGEConv(hidden_channels, hidden_channels),
            ('user', 'answers', 'answer'): SAGEConv(hidden_channels, hidden_channels),
            ('answer', 'rev_answers', 'user'): SAGEConv(hidden_channels, hidden_channels),
            ('question', 'accepted_answer', 'answer'): SAGEConv(hidden_channels, hidden_channels),
            ('answer', 'rev_accepted', 'question'): SAGEConv(hidden_channels, hidden_channels),
            ('user', 'self_loop', 'user'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='sum')

        # Final classification layer for user nodes
        self.user_lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # First HeteroConv layer with ReLU activation
        x_dict = self.conv1(x_dict, edge_index_dict)
        for node_type in x_dict:
            x_dict[node_type] = F.relu(x_dict[node_type])

        # Second HeteroConv layer with ReLU activation
        x_dict = self.conv2(x_dict, edge_index_dict)
        for node_type in x_dict:
            x_dict[node_type] = F.relu(x_dict[node_type])

        # Only update and classify user nodes
        out_user = self.user_lin(x_dict['user'])
        return out_user
#%%
from sklearn.metrics import f1_score
# --- Model and Optimizer Initialization ---
# Determine input dimensions for each node type
in_channels_dict = {
    'user': data['user'].x.size(-1),
    'question': data['question'].x.size(-1),
    'answer': data['answer'].x.size(-1)
}
# Instantiate the model
model = HeteroGNNWithReverse(in_channels_dict, hidden_channels=32, out_channels=2).to(device)
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)


# --- Training Function ---
def train():
    model.train()
    optimizer.zero_grad()

    # Prepare node features
    x_dict = {
        'user': data['user'].x,
        'question': data['question'].x,
        'answer': data['answer'].x
    }

    # Prepare edge indices including self-loop for user nodes
    edge_index_dict = {
        ('user', 'asks', 'question'): data[('user', 'asks', 'question')].edge_index,
        ('question', 'rev_asks', 'user'): data[('question', 'rev_asks', 'user')].edge_index,
        ('question', 'has', 'answer'): data[('question', 'has', 'answer')].edge_index,
        ('answer', 'rev_has', 'question'): data[('answer', 'rev_has', 'question')].edge_index,
        ('user', 'answers', 'answer'): data[('user', 'answers', 'answer')].edge_index,
        ('answer', 'rev_answers', 'user'): data[('answer', 'rev_answers', 'user')].edge_index,
        ('question', 'accepted_answer', 'answer'): data[('question', 'accepted_answer', 'answer')].edge_index,
        ('answer', 'rev_accepted', 'question'): data[('answer', 'rev_accepted', 'question')].edge_index,
        ('user', 'self_loop', 'user'): data[('user', 'self_loop', 'user')].edge_index,
    }

    # Forward pass
    out = model(x_dict, edge_index_dict)  # shape: [num_users, out_channels]

    # Compute cross-entropy loss on training user nodes
    loss = F.cross_entropy(out[data['user'].train_mask],
                           data['user'].y[data['user'].train_mask])

    loss.backward()
    optimizer.step()
    return loss.item()


# --- Testing Function (with Accuracy and F1 Score) ---
@torch.no_grad()
def test():
    model.eval()

    x_dict = {
        'user': data['user'].x,
        'question': data['question'].x,
        'answer': data['answer'].x
    }

    edge_index_dict = {
        ('user', 'asks', 'question'): data[('user', 'asks', 'question')].edge_index,
        ('question', 'rev_asks', 'user'): data[('question', 'rev_asks', 'user')].edge_index,
        ('question', 'has', 'answer'): data[('question', 'has', 'answer')].edge_index,
        ('answer', 'rev_has', 'question'): data[('answer', 'rev_has', 'question')].edge_index,
        ('user', 'answers', 'answer'): data[('user', 'answers', 'answer')].edge_index,
        ('answer', 'rev_answers', 'user'): data[('answer', 'rev_answers', 'user')].edge_index,
        ('question', 'accepted_answer', 'answer'): data[('question', 'accepted_answer', 'answer')].edge_index,
        ('answer', 'rev_accepted', 'question'): data[('answer', 'rev_accepted', 'question')].edge_index,
        ('user', 'self_loop', 'user'): data[('user', 'self_loop', 'user')].edge_index,
    }

    out = model(x_dict, edge_index_dict)
    preds = out.argmax(dim=-1)

    # Accuracy on test user nodes
    test_mask = data['user'].test_mask
    correct = (preds[test_mask] == data['user'].y[test_mask]).sum().item()
    accuracy = correct / test_mask.sum().item()

    # Compute F1 Score (macro average)
    preds_np = preds[test_mask].cpu().numpy()
    labels_np = data['user'].y[test_mask].cpu().numpy()
    f1 = f1_score(labels_np, preds_np, average='macro')

    return accuracy, f1


# --- Training Loop ---
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    loss = train()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} - Loss: {loss:.4f}")

# Evaluate the model on test data
test_accuracy, test_f1 = test()
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")