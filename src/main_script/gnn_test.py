import torch
import os
import pickle
import argparse
from torch_geometric.loader import NeighborLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.utils import test_mini_batch, get_edge_index_dict, get_in_channels, read_yaml
from components.model import Models
from components.constants import DATASET_TYPE_1

def parse_args():
    parser = argparse.ArgumentParser(description="Test GNN model on the specified dataset")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., STACK_OVERFLOW or ASK_REDDIT)")
    parser.add_argument("config_index", type=int, help="Configuration index from config.yaml")
    return parser.parse_args()

class GNNTester:
    def __init__(self, dataset_name, config_index):
        self.dataset_name = dataset_name
        self.config_index = config_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.artifact_dir = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")
        self.data = self._load_data()
        self.config = read_yaml(config_index)
        self.in_channels_dict = get_in_channels(dataset_name, self.data)
        self.model = self._load_model()
        self.test_loader = self._create_loader()

    def _load_data(self):
        path = os.path.join(self.artifact_dir, f"dataset_graph_{self.dataset_name}.pt")
        return torch.load(path, weights_only=False).to(self.device)

    def _load_model(self):
        model = Models.get_hetero_model(
            self.dataset_name,
            self.in_channels_dict,
            self.config["hidden_channels"],
            self.config["out_channels"]
        ).to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.artifact_dir, f"gnn_model_{self.dataset_name}.pth")))
        model.eval()
        return model

    def _create_loader(self):
        node_type = "user" if self.dataset_name == DATASET_TYPE_1 else "author"
        return NeighborLoader(
            self.data,
            num_neighbors=self.config["num_neighbors"],
            input_nodes=(node_type, self.data[node_type].test_mask),
            batch_size=self.config["batch_size"],
            shuffle=False
        )

    def test(self):
        metrics = test_mini_batch(self.model, self.test_loader, self.dataset_name)
        with open(os.path.join(self.artifact_dir, f"test_metrics_{self.dataset_name}.pkl"), "wb") as f:
            pickle.dump({
                "test_losses": [metrics["loss"]],
                "test_accuracies": [metrics["accuracy"]]
            }, f)

        print("\nTest Performance Metrics")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

if __name__ == "__main__":
    args = parse_args()
    tester = GNNTester(args.dataset, args.config_index)
    tester.test()
