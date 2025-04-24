import os
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from matplotlib.lines import Line2D
from torch_geometric.data import HeteroData

# ============================================================
# Directory Setup
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ============================================================
# Graph Builders
# ============================================================

class BaseGraphBuilder:
    def __init__(self, save_path="hetero_graph.pt"):
        self.data = HeteroData()
        self.save_path = save_path

    def save_graph(self):
        torch.save(self.data, self.save_path)
        print(f"Graph saved to {self.save_path}")

    def build(self):
        raise NotImplementedError("Subclasses must implement `build()` method")


class RedditGraphBuilder(BaseGraphBuilder):
    def __init__(self, posts_path, comments_path, save_path="reddit_hetero_graph.pt"):
        super().__init__(save_path)
        self.df_posts = pd.read_csv(posts_path, low_memory=False)
        self.df_comments = pd.read_csv(comments_path, low_memory=False)

    def build(self):
        print("Building Reddit Hetero Graph")
        self._prepare_data()
        self._create_nodes()
        self._create_edges()

    def _prepare_data(self):
        self.df_posts['score'] = pd.to_numeric(self.df_posts['score'], errors='coerce').fillna(0.0)
        self.df_comments['score'] = pd.to_numeric(self.df_comments['score'], errors='coerce').fillna(0.0)
        self.df_posts['active'] = pd.to_numeric(self.df_posts['active'], errors='coerce').fillna(0).astype(int)
        self.df_comments['active'] = pd.to_numeric(self.df_comments['active'], errors='coerce').fillna(0).astype(int)

        self.unique_authors = pd.concat([self.df_posts['author'], self.df_comments['author']]).unique()
        self.author2idx = {name: i for i, name in enumerate(self.unique_authors)}

        self.post2idx = {pid: i for i, pid in enumerate(self.df_posts['post_id'].unique())}
        self.df_comments['comment_id'] = range(len(self.df_comments))
        self.comment2idx = {cid: i for i, cid in enumerate(self.df_comments['comment_id'])}

    def _create_nodes(self):
        self.data['author'].num_nodes = len(self.author2idx)
        self.data['author'].x = torch.arange(len(self.author2idx)).unsqueeze(-1).float()
        author_labels = [0] * len(self.author2idx)
        grouped_active = self.df_posts.groupby("author")["active"].max()
        for author, active_value in grouped_active.items():
            idx = self.author2idx[author]
            author_labels[idx] = int(active_value)
        self.data['author'].y = torch.tensor(author_labels, dtype=torch.long)

        self.data['post'].num_nodes = len(self.post2idx)
        post_features = torch.zeros((len(self.post2idx), 1))
        for _, row in tqdm(self.df_posts.iterrows(), total=len(self.df_posts), desc="Post features"):
            post_features[self.post2idx[row['post_id']], 0] = row['score']
        self.data['post'].x = post_features

        self.data['comment'].num_nodes = len(self.comment2idx)
        comment_features = torch.zeros((len(self.comment2idx), 1))
        for _, row in tqdm(self.df_comments.iterrows(), total=len(self.df_comments), desc="Comment features"):
            comment_features[self.comment2idx[row['comment_id']], 0] = row['score']
        self.data['comment'].x = comment_features

    def _create_edges(self):
        self.data['author', 'wrote_post', 'post'].edge_index = torch.tensor([
            [self.author2idx[row['author']] for _, row in self.df_posts.iterrows()],
            [self.post2idx[row['post_id']] for _, row in self.df_posts.iterrows()]
        ], dtype=torch.long)

        self.data['author', 'wrote_comment', 'comment'].edge_index = torch.tensor([
            [self.author2idx[row['author']] for _, row in self.df_comments.iterrows()],
            [self.comment2idx[row['comment_id']] for _, row in self.df_comments.iterrows()]
        ], dtype=torch.long)

        edges = []
        for _, row in tqdm(self.df_comments.iterrows(), total=len(self.df_comments), desc="has_comment edges"):
            post_id_clean = row['post_id'].replace("t3_", "")
            if post_id_clean in self.post2idx:
                edges.append((self.post2idx[post_id_clean], self.comment2idx[row['comment_id']]))
        self.data['post', 'has_comment', 'comment'].edge_index = torch.tensor(edges, dtype=torch.long).T


class StackOverflowGraphBuilder(BaseGraphBuilder):
    def __init__(self, user_file, question_file, answer_file, save_path="StackOverflow_hetero_graph.pt"):
        super().__init__(save_path)
        self.users_df = pd.read_csv(user_file)
        self.questions_df = pd.read_csv(question_file)
        self.answers_df = pd.read_csv(answer_file)

    def build(self):
        print("Building Stackoverflow Hetero Graph")
        self._create_nodes()
        self._create_edges()

    def _create_nodes(self):
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.users_df['user_id'])}
        self.data['user'].num_nodes = len(self.user_id_to_idx)
        self.data['user'].x = torch.ones((len(self.user_id_to_idx), 1))
        self.data['user'].y = torch.tensor(self.users_df['influential'].values, dtype=torch.long)

        self.question_id_to_idx = {qid: idx for idx, qid in enumerate(self.questions_df['question_id'])}
        self.data['question'].num_nodes = len(self.question_id_to_idx)
        self.data['question'].x = torch.tensor(self.questions_df['score'].values, dtype=torch.float).view(-1, 1)

        self.answer_id_to_idx = {aid: idx for idx, aid in enumerate(self.answers_df['answer_id'])}
        self.data['answer'].num_nodes = len(self.answer_id_to_idx)
        self.data['answer'].x = torch.tensor(self.answers_df['score'].values, dtype=torch.float).view(-1, 1)

    def _create_edges(self):
        self.data['user', 'asks', 'question'].edge_index = torch.tensor([
            [
                self.user_id_to_idx[row['user_id']]
                for _, row in self.questions_df.iterrows()
                if row['user_id'] in self.user_id_to_idx
            ],
            [
                self.question_id_to_idx[row['question_id']]
                for _, row in self.questions_df.iterrows()
                if row['user_id'] in self.user_id_to_idx
            ]
        ], dtype=torch.long)

        self.data['question', 'has', 'answer'].edge_index = torch.tensor([
            [
                self.question_id_to_idx[row['question_id']]
                for _, row in self.answers_df.iterrows()
                if row['question_id'] in self.question_id_to_idx
            ],
            [
                self.answer_id_to_idx[row['answer_id']]
                for _, row in self.answers_df.iterrows()
                if row['question_id'] in self.question_id_to_idx
            ]
        ], dtype=torch.long)

        self.data['user', 'answers', 'answer'].edge_index = torch.tensor([
            [
                self.user_id_to_idx[row['user_id']]
                for _, row in self.answers_df.iterrows()
                if row['user_id'] in self.user_id_to_idx
            ],
            [
                self.answer_id_to_idx[row['answer_id']]
                for _, row in self.answers_df.iterrows()
                if row['user_id'] in self.user_id_to_idx
            ]
        ], dtype=torch.long)

        accepted_edges = []
        for _, row in self.questions_df.iterrows():
            aid = row.get('accepted_answer_id')
            if (
                    pd.notna(aid) and int(aid) in self.answer_id_to_idx
                    and row['question_id'] in self.question_id_to_idx
            ):
                q = self.question_id_to_idx[row['question_id']]
                a = self.answer_id_to_idx[int(aid)]
                accepted_edges.append((q, a))
        self.data['question', 'accepted_answer', 'answer'].edge_index = torch.tensor(accepted_edges, dtype=torch.long).T


# ============================================================
# Visualizer
# ============================================================

class HeteroGraphVisualizer:
    def __init__(self, graph_path, node_type, seed_count=5, radius=2, output_file=None):
        self.graph_path = graph_path
        self.node_type = node_type
        self.seed_count = seed_count
        self.radius = radius
        self.output_file = output_file

        self.data = torch.load(graph_path)
        self.G = nx.MultiDiGraph()
        self.seed_nodes = []
        self.ego_graphs = []

        self.color_map_by_type = {
            'author': 'skyblue', 'post': 'orange', 'comment': 'lightgreen',
            'user': 'skyblue', 'question': 'salmon', 'answer': 'lightcoral'
        }
        self.label_colors = ['lightgrey', 'green', 'red', 'blue', 'purple']

    def _get_node_color(self, attr):
        if 'label' in attr and attr['label'] is not None:
            return self.label_colors[attr['label'] % len(self.label_colors)]
        return self.color_map_by_type.get(attr['node_type'], 'gray')

    def build_networkx_graph(self):
        for node_type in self.data.node_types:
            has_labels = 'y' in self.data[node_type]
            for i in range(self.data[node_type].num_nodes):
                label = int(self.data[node_type].y[i].item()) if has_labels else None
                self.G.add_node((node_type, i), node_type=node_type, label=label)

        for edge_type in self.data.edge_types:
            src_type, rel, dst_type = edge_type
            edge_index = self.data[edge_type].edge_index
            for src, dst in edge_index.t().tolist():
                self.G.add_edge((src_type, src), (dst_type, dst), relation=rel)

    def visualize(self):
        self.build_networkx_graph()

        # Select seed nodes
        candidates = [n for n, attr in self.G.nodes(data=True)
                      if attr.get('node_type') == self.node_type and len(list(self.G.neighbors(n))) > 0]

        if len(candidates) < self.seed_count:
            raise ValueError(f"Not enough nodes of type '{self.node_type}' with neighbors.")
        self.seed_nodes = random.sample(candidates, self.seed_count)

        # Build ego subgraphs
        for seed in self.seed_nodes:
            ego = nx.ego_graph(self.G, seed, radius=self.radius)
            if len(ego.nodes) > 1:
                self.ego_graphs.append((seed, ego))

        fig, axes = plt.subplots(1, self.seed_count, figsize=(5 * self.seed_count, 5))
        if self.seed_count == 1:
            axes = [axes]

        for ax, (seed, subG) in zip(axes, self.ego_graphs):
            pos = nx.spring_layout(subG, seed=42)
            node_colors = [self._get_node_color(subG.nodes[n]) for n in subG.nodes]

            nx.draw(subG, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, ax=ax)

            G_simple = nx.DiGraph()
            for u, v, attr in subG.edges(data=True):
                if G_simple.has_edge(u, v):
                    G_simple[u][v]["relation"] += f", {attr['relation']}"
                else:
                    G_simple.add_edge(u, v, relation=attr["relation"])
            nx.draw_networkx_edge_labels(G_simple, pos, ax=ax, font_size=7)
            ax.set_title(f"Seed: {seed}")

        plt.tight_layout()
        plt.show()

# ============================================================
# Run Everything
# ============================================================

if __name__ == "__main__":
    reddit_save_path = os.path.join(ARTIFACTS_DIR, "reddit_hetero_graph.pt")
    reddit_graph = RedditGraphBuilder(
        posts_path=os.path.join(DATA_DIR, "preprocessed_posts.csv"),
        comments_path=os.path.join(DATA_DIR, "preprocessed_comments.csv"),
        save_path=reddit_save_path
    )
    reddit_graph.build()
    reddit_graph.save_graph()

    qa_save_path = os.path.join(ARTIFACTS_DIR, "StackOverflow_hetero_graph.pt")
    qa_graph = StackOverflowGraphBuilder(
        user_file=os.path.join(DATA_DIR, "preprocessed_users.csv"),
        question_file=os.path.join(DATA_DIR, "preprocessed_questions.csv"),
        answer_file=os.path.join(DATA_DIR, "preprocessed_answers.csv"),
        save_path=qa_save_path
    )
    qa_graph.build()
    qa_graph.save_graph()

    reddit_visualizer = HeteroGraphVisualizer(
        graph_path=reddit_save_path,
        node_type="author",
        seed_count=5,
        radius=2
    )
    reddit_visualizer.visualize()

    qa_visualizer = HeteroGraphVisualizer(
        graph_path=qa_save_path,
        node_type="user",
        seed_count=5,
        radius=2
    )
    qa_visualizer.visualize()
