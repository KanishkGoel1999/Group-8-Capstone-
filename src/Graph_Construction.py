#%%
import pandas as pd
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

# -------------------------
# Step 1: Load and construct the graph
# -------------------------

# Load datasets
users_df = pd.read_csv('preprocessed_data/preprocessed_users.csv')
questions_df = pd.read_csv('preprocessed_data/preprocessed_questions.csv')
answers_df = pd.read_csv('preprocessed_data/preprocessed_answers.csv')

# Create mappings from unique IDs to node indices
user_id_to_idx = {uid: idx for idx, uid in enumerate(users_df['user_id'])}
question_id_to_idx = {qid: idx for idx, qid in enumerate(questions_df['question_id'])}
answer_id_to_idx = {aid: idx for idx, aid in enumerate(answers_df['answer_id'])}

# Initialize the heterogeneous graph
data = HeteroData()

# ---- User Nodes ----
num_users = len(users_df)
data['user'].num_nodes = num_users
# For demonstration, use a constant feature (you can add more features)
data['user'].x = torch.ones((num_users, 1))
# Influential label (target for classification)
data['user'].y = torch.tensor(users_df['influential'].values, dtype=torch.long)

# ---- Question Nodes ----
num_questions = len(questions_df)
data['question'].num_nodes = num_questions
# Use the question score as a feature (can be extended with tag embeddings, etc.)
data['question'].x = torch.tensor(questions_df['score'].values, dtype=torch.float).view(-1, 1)

# ---- Answer Nodes ----
num_answers = len(answers_df)
data['answer'].num_nodes = num_answers
# Use answer score as a feature (or other features as needed)
data['answer'].x = torch.tensor(answers_df['score'].values, dtype=torch.float).view(-1, 1)

# -------------------------
# Step 2: Create edges between nodes
# -------------------------

# (1) User -> Question (asks)
user_indices = []
question_indices = []
for i, row in questions_df.iterrows():
    uid = row['user_id']
    if uid in user_id_to_idx:
        user_indices.append(user_id_to_idx[uid])
        question_indices.append(i)  # row index is the question node index

if user_indices:
    edge_index_user_question = torch.tensor([user_indices, question_indices], dtype=torch.long)
    data['user', 'asks', 'question'].edge_index = edge_index_user_question

# (2) Question -> Answer (has)
question_indices_ans = []
answer_indices = []
for i, row in answers_df.iterrows():
    qid = row['question_id']
    if qid in question_id_to_idx:
        question_indices_ans.append(question_id_to_idx[qid])
        answer_indices.append(i)  # row index is the answer node index

if question_indices_ans:
    edge_index_question_answer = torch.tensor([question_indices_ans, answer_indices], dtype=torch.long)
    data['question', 'has', 'answer'].edge_index = edge_index_question_answer

# (3) User -> Answer (posted answer)
user_indices_ans = []
answer_indices_ans = []
for i, row in answers_df.iterrows():
    uid = row['user_id']
    if uid in user_id_to_idx:
        user_indices_ans.append(user_id_to_idx[uid])
        answer_indices_ans.append(i)

if user_indices_ans:
    edge_index_user_answer = torch.tensor([user_indices_ans, answer_indices_ans], dtype=torch.long)
    data['user', 'answers', 'answer'].edge_index = edge_index_user_answer

# (4) Optional: Question -> Accepted Answer edge
accepted_question_indices = []
accepted_answer_indices = []
for i, row in questions_df.iterrows():
    accepted_aid = row.get('accepted_answer_id')
    if pd.notna(accepted_aid):
        try:
            accepted_aid = int(accepted_aid)
        except:
            continue
        if accepted_aid in answer_id_to_idx:
            accepted_question_indices.append(i)
            accepted_answer_indices.append(answer_id_to_idx[accepted_aid])

if accepted_question_indices:
    edge_index_accepted = torch.tensor([accepted_question_indices, accepted_answer_indices], dtype=torch.long)
    data['question', 'accepted_answer', 'answer'].edge_index = edge_index_accepted

# Optional: Print a summary of the heterogeneous graph
print(data)

# -------------------------
# Step 3: Save the graph to disk
# -------------------------
torch.save(data, "hetero_graph.pt")
print("Graph saved to hetero_graph.pt")
#%%
# -------------------------------------------------------------------
# 2. Convert to homogeneous and keep track of node types
# -------------------------------------------------------------------

hom_data = data.to_homogeneous()

# The 'node_type' tensor in 'hom_data' helps us recover the type of each node
#   user = 0, question = 1, answer = 2  (the exact IDs can vary)
node_types = hom_data.node_type.tolist()

# Convert to a NetworkX graph
G = to_networkx(hom_data, to_undirected=True, node_attrs=['node_type'])

# Add a string label for node type so we can color them easily in NetworkX
type_map = {0: 'user', 1: 'question', 2: 'answer'}
for n in G.nodes():
    G.nodes[n]['type_str'] = type_map[node_types[n]]

# -------------------------------------------------------------------
# 3. Sample a subgraph for visualization
# -------------------------------------------------------------------
# We sample 50 nodes at random to keep the subgraph small
num_sample = 50
sampled_nodes = random.sample(list(G.nodes()), num_sample)
subG = G.subgraph(sampled_nodes).copy()

# -------------------------------------------------------------------
# 4. Create a color map based on node type
# -------------------------------------------------------------------
color_dict = {'user': 'red', 'question': 'blue', 'answer': 'green'}
node_colors = []
for node in subG.nodes():
    node_type_str = subG.nodes[node]['type_str']
    node_colors.append(color_dict[node_type_str])

# -------------------------------------------------------------------
# 5. Draw the subgraph using a force-directed layout
# -------------------------------------------------------------------
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(subG, k=0.35, seed=42)  # Force-directed layout

# Draw nodes
nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=300, alpha=0.8)
# Draw edges
nx.draw_networkx_edges(subG, pos, alpha=0.5)
# Optionally, draw labels (can be very busy if many nodes)
nx.draw_networkx_labels(subG, pos, font_size=8)

# Create a legend for node types
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='User',   markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Question', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Answer',  markerfacecolor='green', markersize=10),
]
plt.legend(handles=legend_elements, loc='best')

plt.title("Sampled Subgraph (Colored by Node Type)")
plt.axis('off')
plt.show()