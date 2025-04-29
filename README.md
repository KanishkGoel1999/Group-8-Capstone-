
# Capstone Project
## Graph Neural Network (PyG)
#### Author: Kanishk Goel
#### Advisor: Dr Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program

---

# Unmasking Fraud in Transaction Networks: Harnessing Heterogeneous Graph Neural Networks for Enhanced Detection

---

## Abstract

Credit card fraud pose a significant threat to global financial systems. They lose billions of dollars annually. Traditional machine learning techniques such as Random Forests and XGBoost have demonstrated effectiveness in fraud detection. However, they often fail to capture the complex interdependencies among entities involved in transactions. This paper explores the efficacy of Graph Neural Networks motivated by recent advances in graph-based learning. We focus on Heterogeneous Graph Convolutional Networks for enhanced credit card fraud detection.

We conduct a comparative study between classical ML models and GNNs using a synthetically generated credit card transaction dataset. Over one million records simulate interactions between customers, merchants, and transactions. The graph-based structure these entities as distinct node types connected by meaningful edges. We implement the GNN framework using PyTorch Geometric and evaluate the models using the F1-score to account for class imbalance.

Experimental results demonstrate that GNNs outperform traditional ML classifiers by leveraging relational information and neighbourhood context. They detect fraudulent patterns, especially when we have more embeddings and fewer meaningful features. This work reinforces the potential of graph-based learning as a powerful approach for high-volume and relationally rich transaction networks.

---

## Dataset

The dataset used in this project is a **synthetic credit card fraud detection dataset** provided by Sparkov.

- ~2 million transaction records
- Only 0.1% of the transactions are fraudulent

Follow Readme in code folder to get data.

---

## Models Implemented

This project evaluates the following models:

- **Graph Neural Networks:**
  - Relational Graph Convolution Network (R-GCN)
  
- **Classical Machine Learning Models:**
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - XGBoost
  

Each model is trained and evaluated using techniques and metrics tailored for fraud detection tasks (such as Precision-Recall AUC, F1 Score).

---

## Installation

To set up the project environment:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/KanishkGoel1999/Capstone_project_Kanishk_Goel.git
   cd Capstone_project_Kanishk_Goel
   ```

2. **Install Required Dependencies:**
   ```bash
   pip install -r code/requirements.txt
   ```

---

## Usage

The project provides scripts for preprocessing, training, and evaluating different models.  
Here’s how to use them:


### Train Classical Machine Learning Models
```bash
python code/main_code/classical_model/train_ML.py
```

### Train GNN Models

- **Using DGL:**
  ```bash
  python code/main_code/gnn/DGL/train_dgl.py
  ```

- **Using PyG:**
  ```bash
  python code/main_code/gnn/PyG/train_pyg.py
  ```

### Test Models (with already trained/pretrained models)

- Download pretrained models from: [Download Link Placeholder]  
- Place models into:
  - `code/main_code/gnn/DGL/models/` for DGL models
  - `code/main_code/gnn/PyG/models/` for PyG models
  
Then run:

- **Classical Models:**
  ```bash
  python code/main_code/classical_model/test_ML.py
  ```

- **GNN DGL Models:**
  ```bash
  python code/main_code/gnn/DGL/test_dgl.py
  ```

- **GNN PyG Models:**
  ```bash
  python code/main_code/gnn/PyG/test_pyg.py
  ```

---

## Files Overview

The `code/` folder contains two main subfolders:

### 1. components/

Reusable code modules:
- Utility functions for classical models (`classical_model/utils/`)
- GNN utilities and model definitions (`gnn/utils/` and `gnn/model/`)
- Configuration files (`config/`)
- Preprocessing scripts (`preprocessing.py`)
- Package installation script (`packages.py`)

### 2. main_code/

Main scripts to train, test, and evaluate models:
- **classical_model/**:  
  Train and test classical machine learning models.
- **gnn/DGL/**:  
  Train and test GNN models using the DGL framework.  
- **gnn/PyG/**:  
  Train and test GNN models using the PyTorch Geometric (PyG) framework.

Saved models are stored in respective `models/` subfolders.

---

## Notes

- Adjust hyperparameters and file paths in the training and testing scripts as needed in config file.
- Pretrained model checkpoints can be loaded automatically during testing if available.
- Visit individual directories (`components/` and `main_code/`) for detailed documentation for each model and utility.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
