import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

import os

# Get the parent directory of the current script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Construct paths to dataset files
user_file = os.path.join(BASE_DIR, "data", "preprocessed_users.csv")
question_file = os.path.join(BASE_DIR, "data", "preprocessed_questions.csv")
answer_file = os.path.join(BASE_DIR, "data", "preprocessed_answers.csv")

class HeterogeneousGraph:
    def __init__(self, user_file: str, question_file: str, answer_file: str):
        """
        Initialize the graph by loading data from CSV files.

        :param user_file: Path to the preprocessed users CSV file.
        :param question_file: Path to the preprocessed questions CSV file.
        :param answer_file: Path to the preprocessed answers CSV file.
        """
        self.user_file = user_file
        self.question_file = question_file
        self.answer_file = answer_file
        self.data = HeteroData()

        # Load datasets
        self.users_df = pd.read_csv(user_file)
        self.questions_df = pd.read_csv(question_file)
        self.answers_df = pd.read_csv(answer_file)

        # Create mappings from unique IDs to node indices
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.users_df['user_id'])}
        self.question_id_to_idx = {qid: idx for idx, qid in enumerate(self.questions_df['question_id'])}
        self.answer_id_to_idx = {aid: idx for idx, aid in enumerate(self.answers_df['answer_id'])}

    def create_nodes(self):
        """Creates user, question, and answer nodes in the heterogeneous graph."""
        # User Nodes
        num_users = len(self.users_df)
        self.data['user'].num_nodes = num_users
        self.data['user'].x = torch.ones((num_users, 1))  # Example feature
        self.data['user'].y = torch.tensor(self.users_df['influential'].values, dtype=torch.long)

        # Question Nodes
        num_questions = len(self.questions_df)
        self.data['question'].num_nodes = num_questions
        self.data['question'].x = torch.tensor(self.questions_df['score'].values, dtype=torch.float).view(-1, 1)

        # Answer Nodes
        num_answers = len(self.answers_df)
        self.data['answer'].num_nodes = num_answers
        self.data['answer'].x = torch.tensor(self.answers_df['score'].values, dtype=torch.float).view(-1, 1)

    def create_edges(self):
        """Creates edges between users, questions, and answers in the heterogeneous graph."""
        # User -> Question (asks)
        user_indices, question_indices = [], []
        for i, row in self.questions_df.iterrows():
            uid = row['user_id']
            if uid in self.user_id_to_idx:
                user_indices.append(self.user_id_to_idx[uid])
                question_indices.append(i)  # row index is the question node index

        if user_indices:
            self.data['user', 'asks', 'question'].edge_index = torch.tensor([user_indices, question_indices],
                                                                            dtype=torch.long)

        # Question -> Answer (has)
        question_indices_ans, answer_indices = [], []
        for i, row in self.answers_df.iterrows():
            qid = row['question_id']
            if qid in self.question_id_to_idx:
                question_indices_ans.append(self.question_id_to_idx[qid])
                answer_indices.append(i)

        if question_indices_ans:
            self.data['question', 'has', 'answer'].edge_index = torch.tensor([question_indices_ans, answer_indices],
                                                                             dtype=torch.long)

        # User -> Answer (answers)
        user_indices_ans, answer_indices_ans = [], []
        for i, row in self.answers_df.iterrows():
            uid = row['user_id']
            if uid in self.user_id_to_idx:
                user_indices_ans.append(self.user_id_to_idx[uid])
                answer_indices_ans.append(i)

        if user_indices_ans:
            self.data['user', 'answers', 'answer'].edge_index = torch.tensor([user_indices_ans, answer_indices_ans],
                                                                             dtype=torch.long)

        # Question -> Accepted Answer (accepted_answer)
        accepted_question_indices, accepted_answer_indices = [], []
        for i, row in self.questions_df.iterrows():
            accepted_aid = row.get('accepted_answer_id')
            if pd.notna(accepted_aid):
                try:
                    accepted_aid = int(accepted_aid)
                except ValueError:
                    continue
                if accepted_aid in self.answer_id_to_idx:
                    accepted_question_indices.append(i)
                    accepted_answer_indices.append(self.answer_id_to_idx[accepted_aid])

        if accepted_question_indices:
            self.data['question', 'accepted_answer', 'answer'].edge_index = torch.tensor(
                [accepted_question_indices, accepted_answer_indices], dtype=torch.long
            )

    def save_graph(self, file_path="hetero_graph.pt"):
        """Saves the heterogeneous graph to a file."""
        torch.save(self.data, file_path)
        print(f"Graph saved to {file_path}")

    def build_graph(self):
        """Builds the entire heterogeneous graph by creating nodes and edges."""
        self.create_nodes()
        self.create_edges()
        print(self.data)


# Example usage
if __name__ == "__main__":
    graph = HeterogeneousGraph(
        user_file,
        question_file,
        answer_file"
    )
    graph.build_graph()
    graph.save_graph()
