This repository contains two projects:
- GNN for Social Network
The main branch hosts the GNN for Social Network project.
The `shikha` and `harsha` branches have been merged into the main branch.
- GNN for Finance
The `kanishk` branch contains the GNN for Finance project.

# 🚧 Project In Progress: Identifying Influential Users Using Classical Models and Graph Neural Networks

> ⚠️ This project is currently under active development. Final model comparisons, visualizations, and documentation are subject to change.

---

## 🧠 Objective

To evaluate the effectiveness of graph-based learning (GNNs) over traditional tabular models (XGBoost) in classifying users based on their influence or activity, leveraging structured and relational information from Q&A and discussion forums.

## 📊 Datasets

- **Stack Overflow**: Extracted using the Stack Exchange API. Includes users, questions, and answers.
- **AskReddit (Reddit)**: Sourced from Kaggle’s “A Month of AskReddit” dataset.

Each dataset is transformed into:
- A tabular format for XGBoost.
- A heterogeneous graph format for GNNs.

## 🔍 Key Components

- **Data Preprocessing**: Cleaning, labeling, and feature extraction.
- **Graph Construction**: Building heterogeneous graphs with user-question-answer (Stack Overflow) and user-post-comment (Reddit) relationships.
- **Modeling**:
  - XGBoost for baseline performance.
  - GNN using GraphSAGE with message passing over multi-typed edges.
- **Imbalance Handling**: Custom stratified mini-batching technique for class-balanced training.

## 📈 Results Summary

| Dataset        | Model     | F1-Score | Accuracy | AUC   |
|----------------|-----------|----------|----------|--------|
| Stack Overflow | XGBoost   | 0.602    | 0.812    | 0.646 |
| Stack Overflow | GNN       | 0.619    | 0.816    | 0.667 |
| Reddit         | XGBoost   | 0.680    | 0.813    | 0.895 |
| Reddit         | GNN       | 0.677    | 0.912    | 0.954 |

## 🛠️ Technologies Used

- Ronin Virtual Machine
- Python, PyTorch Geometric
- XGBoost
- NetworkX / DGL for preprocessing
- LaTeX for documentation

## 📄 Output

The final report (PDF) includes:
- Dataset statistics
- Graph types and visual examples
- Model architectures
- Performance evaluation
- Discussion and conclusion

## 👥 Authors

- Shikha Kumari 
- Harshavardana Reddy Kolan   
- Prof. Amir Hossein Jafari

---
