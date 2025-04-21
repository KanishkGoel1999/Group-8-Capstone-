import torch
import os
import pickle
from torch_geometric.loader import NeighborLoader
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.utils import test_mini_batch
from components.model import Models
from components.constants import EDGE_TYPES, DATASET_TYPE_1
from components.utils import get_edge_index_dict, get_in_channels, read_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on a specified dataset and config set")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., STACK_OVERFLOW or ASK_REDDIT)")
    parser.add_argument("config_index", type=int, help="Index of the GNN configuration set from YAML")
    return parser.parse_args()

def loader(data):
    if dataset_name == DATASET_TYPE_1:
        nodes = "user"
    else:
        nodes = "author"
    mask = data[nodes].test_mask
    test_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        input_nodes=(nodes, mask),
        batch_size=64,
        shuffle=False
    )
    return test_loader

if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset
    config_index = args.config_index

    # Load model/data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

    data = torch.load(os.path.join(artifact_dir, f"dataset_graph_{dataset_name}.pt"), weights_only=False)
    data = data.to(device)
    gnn_config = read_yaml(config_index)
    in_channels_dict = get_in_channels(dataset_name, data)
    model = Models.get_hetero_model(dataset_name, in_channels_dict, gnn_config["hidden_channels"], gnn_config["out_channels"]).to(device)
    model.load_state_dict(torch.load(os.path.join(artifact_dir, f"gnn_model_{dataset_name}.pth")))
    model.eval()  
    # Test Loader
    test_loader = loader(data)
    # Run testing
    test_metrics = test_mini_batch(model, test_loader, dataset_name)  

    # Save test metrics for visualization
    with open(os.path.join(artifact_dir, f"test_metrics_{dataset_name}.pkl"), "wb") as f:
        pickle.dump({
            "test_losses": [test_metrics["loss"]],
            "test_accuracies": [test_metrics["accuracy"]]
        }, f)

    print("\nTest Performance Metrics")
    for k, v in test_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")


