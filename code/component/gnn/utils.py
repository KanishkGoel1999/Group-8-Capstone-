from component.packages import *
from component.gnn.model_dgl import *

def create_graph_pyg(df, transaction_feats, merchant_feats, user_feats):
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
    # df_txn_unique = df.drop_duplicates(subset=['transaction_id'])
    # g.nodes['transaction'].data['y'] = torch.tensor(df_txn_unique['is_fraud'].values, dtype=torch.float)
    # classified_idx = df_txn_unique.index
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


def print_graph_info_pyg(data):
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

def test_dgl_model(model_path, test_g, test_features_dict, test_labels, target_node, ntype_dict, etypes, in_size_dict, hidden_size, out_size, n_layers, device):
    """
    Loads a trained model and evaluates it on the test dataset.
    """
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


def visualize_network_graph(data, sample_size=300):
    """
    Custom NetworkX visualization for PyG HeteroData graphs.
    """
    print(f"Visualizing a sample of {sample_size} nodes...")

    G = nx.Graph()

    # Add nodes with type labels
    for ntype in data.node_types:
        for i in range(min(sample_size, data[ntype].num_nodes)):
            G.add_node(f"{ntype}_{i}", node_type=ntype)

    # Add edges for selected types
    selected_edges = [
        ('transaction', 'transaction_to_user', 'user'),
        ('transaction', 'transaction_to_merchant', 'merchant')
    ]

    for src_type, rel, dst_type in selected_edges:
        edge_index = data[src_type, rel, dst_type].edge_index
        num_edges = edge_index.shape[1]

        for i in range(min(sample_size, num_edges)):
            src = f"{src_type}_{edge_index[0, i].item()}"
            dst = f"{dst_type}_{edge_index[1, i].item()}"
            G.add_edge(src, dst)

    # Color by node type
    color_map = []
    for node, data_dict in G.nodes(data=True):
        if data_dict['node_type'] == 'transaction':
            color_map.append('orange')
        elif data_dict['node_type'] == 'user':
            color_map.append('lightblue')
        elif data_dict['node_type'] == 'merchant':
            color_map.append('green')
        else:
            color_map.append('gray')

    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=color_map, with_labels=False, node_size=50, edge_color='gray', alpha=0.7)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Transaction', markerfacecolor='orange', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='User', markerfacecolor='lightblue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Merchant', markerfacecolor='green', markersize=8),
    ]
    plt.legend(handles=legend_elements)
    plt.title("Sample Heterogeneous Transaction Graph")
    plt.tight_layout()
    plt.savefig("output/Graph DS2.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()


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
    plt.savefig("output/AUCPR GNN DS2.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()
