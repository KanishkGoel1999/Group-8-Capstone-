import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.metrics import f1_score
#pip install pyg-lib
#pip install torch-sparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.load("hetero_graph.pt")

print(data)  # Quick summary of the graph
# Move to GPU if available
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

# --- Create DataLoaders Using NeighborLoader ---
# This loader samples a fixed number of neighbors per layer for the specified input nodes.
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],  # Adjust neighbor sampling per layer as needed
    input_nodes=('user', train_indices),
    batch_size=64,
    shuffle=True,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    input_nodes=('user', test_indices),
    batch_size=64,
    shuffle=False,
)

# --- Define the Heterogeneous GNN Model ---
class HeteroGNNWithReverse(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels):
        super().__init__()
        # Define the first heterogeneous convolution layer.
        self.conv1 = HeteroConv({
            ('user', 'asks', 'question'): SAGEConv(in_channels_dict['user'], hidden_channels),
            ('question', 'rev_asks', 'user'): SAGEConv(in_channels_dict['question'], hidden_channels),
            ('question', 'has', 'answer'): SAGEConv(in_channels_dict['question'], hidden_channels),
            ('answer', 'rev_has', 'question'): SAGEConv(in_channels_dict['answer'], hidden_channels),
            ('user', 'answers', 'answer'): SAGEConv(in_channels_dict['user'], hidden_channels),
            ('answer', 'rev_answers', 'user'): SAGEConv(in_channels_dict['answer'], hidden_channels),
            ('question', 'accepted_answer', 'answer'): SAGEConv(in_channels_dict['question'], hidden_channels),
            ('answer', 'rev_accepted', 'question'): SAGEConv(in_channels_dict['answer'], hidden_channels),
            ('user', 'self_loop', 'user'): SAGEConv(in_channels_dict['user'], hidden_channels),
        }, aggr='sum')

        # Second heterogeneous convolution layer.
        self.conv2 = HeteroConv({
            ('user', 'asks', 'question'): SAGEConv(hidden_channels, hidden_channels),
            ('question', 'rev_asks', 'user'): SAGEConv(hidden_channels, hidden_channels),
            ('question', 'has', 'answer'): SAGEConv(hidden_channels, hidden_channels),
            ('answer', 'rev_has', 'question'): SAGEConv(hidden_channels, hidden_channels),
            ('user', 'answers', 'answer'): SAGEConv(hidden_channels, hidden_channels),
            ('answer', 'rev_answers', 'user'): SAGEConv(hidden_channels, hidden_channels),
            ('question', 'accepted_answer', 'answer'): SAGEConv(hidden_channels, hidden_channels),
            ('answer', 'rev_accepted', 'question'): SAGEConv(hidden_channels, hidden_channels),
            ('user', 'self_loop', 'user'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        # Final classification layer for user nodes.
        self.user_lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Apply first conv layer with ReLU activation.
        x_dict = self.conv1(x_dict, edge_index_dict)
        for node_type in x_dict:
            x_dict[node_type] = F.relu(x_dict[node_type])
        # Apply second conv layer.
        x_dict = self.conv2(x_dict, edge_index_dict)
        for node_type in x_dict:
            x_dict[node_type] = F.relu(x_dict[node_type])
        # Classify only the user nodes.
        out_user = self.user_lin(x_dict['user'])
        return out_user

# --- Initialize Model and Optimizer ---
in_channels_dict = {
    'user': data['user'].x.size(-1),
    'question': data['question'].x.size(-1),
    'answer': data['answer'].x.size(-1)
}
model = HeteroGNNWithReverse(in_channels_dict, hidden_channels=32, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# --- Training Function Using the DataLoader ---
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
    accuracy = total_correct / total_examples
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, f1

# --- Training Loop ---
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    loss = train()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} - Loss: {loss:.4f}")

# Evaluate the model on the test data.
test_accuracy, test_f1 = test()
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
