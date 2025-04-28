"""
HeteroGraphVisualizer.py
------------------------
Display ego-graphs from a saved PyG HeteroData object
using NetworkX + Matplotlib.

Author: <your-name>
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from pathlib import Path

import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class HeteroGraphVisualizer:
    """Visualise ego-graphs in a heterogeneous graph with clear legend and edge labels."""

    def __init__(
        self,
        graph_path,
        node_type,
        seed: int = None,
        seed_count: int = 5,
        radius: int = 2,
    ):
        self.graph_path = Path(graph_path)
        self.node_type = node_type
        self.seed = seed
        self.seed_count = seed_count
        self.radius = radius

        # Load the graph
        self.data = torch.load(self.graph_path, weights_only=False)
        self.G = nx.MultiDiGraph()

        # Color map by node type
        self.color_map = {
            "author": "skyblue",
            "post": "orange",
            "comment": "lightgreen",
            "user": "salmon",
            "question": "violet",
            "answer": "lightcoral",
        }

    def build_networkx_graph(self):
        """Convert PyG HeteroData to NetworkX MultiDiGraph."""
        for ntype in self.data.node_types:
            for idx in range(self.data[ntype].num_nodes):
                self.G.add_node((ntype, idx), node_type=ntype)

        for etype in self.data.edge_types:
            src_type, relation, dst_type = etype
            src_nodes, dst_nodes = self.data[etype].edge_index
            for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
                self.G.add_edge((src_type, src), (dst_type, dst), relation=relation)

    def visualize(self):
        """Generate and display a simple ego-graph with legend and edge labels."""
        self.build_networkx_graph()

        # Select seed node(s)
        if self.seed is not None:
            seeds = [(self.node_type, self.seed)] if (self.node_type, self.seed) in self.G else []
        else:
            candidates = [n for n, attr in self.G.nodes(data=True) if attr["node_type"] == self.node_type]
            seeds = random.sample(candidates, min(len(candidates), self.seed_count))

        if not seeds:
            print(f"No seed found for node type '{self.node_type}'.")
            return

        for seed in seeds:
            subG = nx.ego_graph(self.G, seed, radius=self.radius)

            # Layout with wider spread
            pos = nx.spring_layout(subG, seed=1, k=1.0, iterations=100)

            plt.figure(figsize=(6, 6), dpi=150)

            # Draw edges
            nx.draw_networkx_edges(subG, pos, alpha=0.5)

            # Draw nodes colored by type
            handles = []
            for ntype, color in self.color_map.items():
                nodes = [n for n, attr in subG.nodes(data=True) if attr['node_type'] == ntype]
                if nodes:
                    nx.draw_networkx_nodes(
                        subG, pos,
                        nodelist=nodes,
                        node_color=color,
                        node_size=300,
                        edgecolors='black'
                    )
                    handles.append(Line2D([0], [0], marker='o', color='w', label=ntype,
                                          markerfacecolor=color, markersize=8, markeredgecolor='black'))

            # Draw edge labels (relations)
            edge_labels = nx.get_edge_attributes(subG, 'relation')
            nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=6)

            # Legend placed outside plot on the right
            plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1),
                       fontsize=8, framealpha=0.9)

            plt.title(f"Sample {self.node_type} node", fontsize=12)
            plt.axis('off')
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend
            plt.show()

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent / "model_artifacts"

    # Example with a sample author node
    HeteroGraphVisualizer(
        graph_path=BASE_DIR / "reddit_hetero_graph.pt",
        node_type="author",
        seed=8966,
        radius=3
    ).visualize()

    # Example with a sample user node
    HeteroGraphVisualizer(
        graph_path=BASE_DIR / "stackoverflow_hetero_graph.pt",
        node_type="user",
        seed=21922,
        radius=3
    ).visualize()
