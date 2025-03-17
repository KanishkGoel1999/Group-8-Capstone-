import torch
import networkx as nx
import matplotlib.pyplot as plt
import random

# Load the heterogeneous graph saved as 'hetero_graph.pt'
data = torch.load('hetero_graph.pt')

# Create an empty MultiDiGraph to handle multiple edge types and directions.
G = nx.MultiDiGraph()

# Add nodes to the NetworkX graph.
for node_type in data.node_types:  # e.g. 'user', 'question', 'answer'
    num_nodes = data[node_type].num_nodes
    for i in range(num_nodes):
        # Use a tuple (node_type, index) as a unique identifier.
        G.add_node((node_type, i), node_type=node_type)

# Add edges for each relation.
for edge_type in data.edge_types:
    src_type, relation, dst_type = edge_type
    edge_index = data[edge_type].edge_index  # shape: [2, num_edges]
    for src, dst in edge_index.t().tolist():
        G.add_edge((src_type, src), (dst_type, dst), relation=relation)

# Select user nodes with at least one neighbor.
user_nodes = [n for n, attr in G.nodes(data=True)
              if attr['node_type'] == 'user' and len(list(G.neighbors(n))) > 0]
if len(user_nodes) < 10:
    raise ValueError("Not enough user nodes with neighbors found in the graph.")

# Randomly select 10 seed user nodes.
seed_nodes = random.sample(user_nodes, 10)

# Extract ego graphs with a radius of 2 for each selected seed node.
ego_graphs = []
for seed in seed_nodes:
    ego_g = nx.ego_graph(G, seed, radius=2)
    # Only include subgraphs that have more than just the seed node.
    if len(ego_g.nodes) > 1:
        ego_graphs.append((seed, ego_g))

# Plot all subgraphs in a single figure with 10 subplots (2 rows x 5 columns).
fig, axes = plt.subplots(2, 5, figsize=(20, 10))

for ax, (seed, subG) in zip(axes.flatten(), ego_graphs):
    pos = nx.spring_layout(subG, seed=42)
    nx.draw(subG, pos, with_labels=True, node_size=500, font_size=8, ax=ax)
    edge_labels = nx.get_edge_attributes(subG, 'relation')
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_title(f"Seed: {seed}")

plt.tight_layout()
plt.show()
