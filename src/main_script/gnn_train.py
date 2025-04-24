# gnn_train_fixed.py
"""
End‑to‑end training script for heterogeneous GNNs on the Stack Overflow or
AskReddit graphs.

▶ Key fixes
------------
* Every newly‑created `edge_index` is made **contiguous** to satisfy
  `pyg_lib` / `NeighborSampler` kernels.
* Global post‑check that *all* edge indices are contiguous, just in case
  future transforms introduce non‑contiguous tensors.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit

# -----------------------------------------------------------------------------
# Local imports (make sure project root is on PYTHONPATH)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from components.model import Models  # noqa: E402
from components.metric import Metrics  # noqa: E402
from components.constants import DATASET_TYPE_1  # noqa: E402
from components.utils import (  # noqa: E402
    get_in_channels,
    train_mini_batch,
    read_yaml,
)

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse dataset name and YAML config index from the command line."""
    parser = argparse.ArgumentParser(
        description="Train a heterogeneous GNN on Stack Overflow or AskReddit",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name (STACK_OVERFLOW or ASK_REDDIT)",
    )
    parser.add_argument(
        "config_index",
        type=int,
        help="Index of the hyper‑parameter set inside configs.yml",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Trainer class
# -----------------------------------------------------------------------------
class Trainer:
    """Encapsulates the whole training pipeline."""

    def __init__(self, dataset_name: str, config_index: int):
        self.dataset_name = dataset_name
        self.config_index = config_index
        self.target_node = "user" if dataset_name == DATASET_TYPE_1 else "author"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = read_yaml(config_index)
        self.artifact_dir = ROOT / "model_artifacts"
        self.artifact_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading / preprocessing
    # ------------------------------------------------------------------
    def load_dataset(self) -> None:
        filename = (
            "StackOverflow_hetero_graph.pt"
            if self.dataset_name == DATASET_TYPE_1
            else "reddit_hetero_graph.pt"
        )
        path = self.artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found at {path}")
        self.dataset = torch.load(path, weights_only=False)

    def _add_reverse_edges(self) -> None:
        """Create symmetric reverse relations + self‑loops (contiguous)."""
        original_edge_types = list(self.data.edge_types)

        for (src, rel, dst) in original_edge_types:
            # —— Build reverse relation name
            if self.dataset_name == DATASET_TYPE_1:
                rev_rel = "rev_accepted" if rel == "accepted_answer" else f"rev_{rel}"
            else:
                rev_rel = f"{rel}__reverse"
            rev_type = (dst, rev_rel, src)

            if rev_type not in self.data.edge_types:
                ei = self.data[src, rel, dst].edge_index
                # flip() returns a non‑contiguous view ➜ make contiguous
                self.data[rev_type].edge_index = ei.flip(0).contiguous()

        # Add self loops for the target node type
        num_nodes = self.data[self.target_node].num_nodes
        loop = torch.arange(num_nodes, dtype=torch.long, device=self.device)
        self_loop_ei = torch.stack([loop, loop], dim=0).contiguous()
        self.data[self.target_node, "self_loop", self.target_node].edge_index = (
            self_loop_ei
        )

        # Safety: ensure *all* edge_index tensors are contiguous
        for et in self.data.edge_types:
            self.data[et].edge_index = self.data[et].edge_index.contiguous()

    def preprocess(self) -> None:
        transform = RandomNodeSplit(split=self.config["split"], num_val=0.1, num_test=0.1)
        self.data = transform(self.dataset).to(self.device)
        self._add_reverse_edges()

        self.in_channels_dict = get_in_channels(self.dataset_name, self.data)
        self.model = Models.get_hetero_model(
            self.dataset_name,
            self.in_channels_dict,
            hidden_channels=self.config["hidden_channels"],
            out_channels=self.config["out_channels"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )

    # ------------------------------------------------------------------
    # Batching helpers
    # ------------------------------------------------------------------
    def _stratified_batches(self):
        """Return balanced node‑id batches for training."""
        lbl = self.data[self.target_node]["y"]
        mask = self.data[self.target_node]["train_mask"]
        node_ids = torch.arange(self.data[self.target_node].num_nodes, device=self.device)[mask]

        class_ids = {}
        for cls in lbl[node_ids].unique():
            idx = node_ids[lbl[node_ids] == cls]
            class_ids[int(cls)] = idx[torch.randperm(len(idx))]

        n_cls = len(class_ids)
        per_cls = self.config["batch_size"] // n_cls
        n_batches = min(len(v) // per_cls for v in class_ids.values())

        batches = [
            torch.cat([
                class_ids[c][i * per_cls : (i + 1) * per_cls] for c in class_ids
            ])
            for i in range(n_batches)
        ]
        print("Batches:", len(batches))
        return batches

    def _create_loader(self):
        input_batches = self._stratified_batches()
        return [
            NeighborLoader(
                self.data,
                num_neighbors=self.config["num_neighbors"],
                input_nodes=(self.target_node, batch),
                batch_size=self.config["batch_size"],
                shuffle=False,
            )
            for batch in input_batches
        ]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        loaders = self._create_loader()
        losses, accs = [], []

        for epoch in range(1, self.config["num_epochs"] + 1):
            self.model.train()
            tot_loss = tot_acc = 0.0

            for ld in loaders:
                loss, acc = train_mini_batch(self.model, ld, self.optimizer, self.dataset_name)
                tot_loss += loss
                tot_acc += acc

            losses.append(tot_loss / len(loaders))
            accs.append(tot_acc / len(loaders))
            print(f"Epoch {epoch:03d}  loss={losses[-1]:.4f}  acc={accs[-1]:.4f}")

        return losses, accs

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, losses, accs):
        torch.save(self.model.state_dict(), self.artifact_dir / f"gnn_model_{self.dataset_name}.pth")
        torch.save(self.data, self.artifact_dir / f"dataset_graph_{self.dataset_name}.pt")
        with open(self.artifact_dir / f"train_metrics_{self.dataset_name}.pkl", "wb") as fh:
            pickle.dump({"train_losses": losses, "train_accuracies": accs}, fh)
        print("✅ Model, data, and metrics saved in", self.artifact_dir)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args.dataset, args.config_index)
    trainer.load_dataset()
    trainer.preprocess()

    print("----- Starting Training -----")
    losses, accuracies = trainer.train()
    trainer.save(losses, accuracies)
