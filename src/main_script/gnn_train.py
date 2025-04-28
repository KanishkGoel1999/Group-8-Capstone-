# gnn_train.py
import os
import sys
import torch
import pickle
import argparse
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.model import Models
from components.metric import Metrics
from components.constants import DATASET_TYPE_1
from components.utils import get_edge_index_dict, train_mini_batch, get_in_channels, read_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on a specified dataset and config set")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., STACK_OVERFLOW or ASK_REDDIT)")
    parser.add_argument("config_index", type=int, help="Index of the GNN configuration set from YAML")
    return parser.parse_args()

class Trainer:
    def __init__(self, dataset_name, config_index):
        self.dataset_name = dataset_name
        self.config_index = config_index
        self.target_node = "user" if dataset_name == DATASET_TYPE_1 else "author"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = read_yaml(config_index)
        self.artifact_dir = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

    def load_dataset(self):
        path = "StackOverflow_hetero_graph.pt" if self.dataset_name == DATASET_TYPE_1 else "reddit_hetero_graph.pt"
        self.dataset = torch.load(path, weights_only=False)

    def preprocess(self):
        transform = RandomNodeSplit(split=self.config["split"], num_val=0.1, num_test=0.1)
        self.data = transform(self.dataset).to(self.device)
        self._add_reverse_edges()
        self.in_channels_dict = get_in_channels(self.dataset_name, self.data)
        self.model = Models.get_hetero_model(
            self.dataset_name,
            self.in_channels_dict,
            hidden_channels=self.config["hidden_channels"],
            out_channels=self.config["out_channels"]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"])
        )

    def _add_reverse_edges(self):
        original_edge_types = list(self.data.edge_types)
        for (src, rel, dst) in original_edge_types:
            rev_rel = "rev_accepted" if self.dataset_name == DATASET_TYPE_1 and rel == "accepted_answer" \
                else ("rev_" + rel if self.dataset_name == DATASET_TYPE_1 else rel + "__reverse")
            rev_type = (dst, rev_rel, src)
            if rev_type not in self.data.edge_types:
                edge_index = self.data[src, rel, dst].edge_index
                self.data[rev_type].edge_index = edge_index.flip(0)

        num_nodes = self.data[self.target_node].num_nodes
        loop = torch.arange(num_nodes, dtype=torch.long)
        self_loop_edge = torch.stack([loop, loop], dim=0)
        self.data[self.target_node, "self_loop", self.target_node].edge_index = self_loop_edge

        # Safety: ensure *all* edge_index tensors are contiguous
        for et in self.data.edge_types:
            self.data[et].edge_index = self.data[et].edge_index.contiguous()

    def stratified_batches(self):
        labels = self.data[self.target_node]["y"]
        mask = self.data[self.target_node]["train_mask"]
        node_indices = torch.arange(self.data[self.target_node].num_nodes)[mask]

        class_indices = {}
        for cls in labels[node_indices].unique():
            idx = node_indices[labels[node_indices] == cls]
            class_indices[int(cls)] = idx[torch.randperm(len(idx))]

        num_classes = len(class_indices)
        samples_per_class = self.config["batch_size"] // num_classes
        min_batches = min(len(v) // samples_per_class for v in class_indices.values())

        batches = []
        for i in range(min_batches):
            batch = torch.cat([
                class_indices[cls][i * samples_per_class: (i + 1) * samples_per_class]
                for cls in class_indices
            ])
            batches.append(batch)
        print('Batches:', len(batches))
        return batches

    def create_loader(self):
        input_batches = self.stratified_batches()
        return [
            NeighborLoader(
                self.data,
                num_neighbors=self.config["num_neighbors"],
                input_nodes=(self.target_node, batch),
                batch_size=self.config["batch_size"],
                shuffle=False
            ) for batch in input_batches
        ]

    def train(self):
        train_loaders = self.create_loader()
        losses, accuracies = [], []

        for epoch in range(1, self.config["num_epochs"] + 1):
            self.model.train()
            total_loss, total_acc = 0, 0

            for loader in train_loaders:
                loss, acc = train_mini_batch(self.model, loader, self.optimizer, self.dataset_name)
                total_loss += loss
                total_acc += acc

            epoch_loss = total_loss / len(train_loaders)
            epoch_acc = total_acc / len(train_loaders)
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)

            print(f"Epoch {epoch:03d} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        return losses, accuracies

    def save_artifacts(self, losses, accuracies):
        torch.save(self.model.state_dict(), os.path.join(self.artifact_dir, f"gnn_model_{self.dataset_name}.pth"))
        torch.save(self.data, os.path.join(self.artifact_dir, f"dataset_graph_{self.dataset_name}.pt"))
        with open(os.path.join(self.artifact_dir, f"train_metrics_{self.dataset_name}.pkl"), "wb") as f:
            pickle.dump({
                "train_losses": losses,
                "train_accuracies": accuracies
            }, f)
        print("✅ Model, data, and metrics saved!")

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args.dataset, args.config_index)
    trainer.load_dataset()
    trainer.preprocess()
    print("----- Starting Training -----")
    losses, accuracies = trainer.train()
    trainer.save_artifacts(losses, accuracies)
