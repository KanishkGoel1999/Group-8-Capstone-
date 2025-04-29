This repository contains two projects:
- GNN for Social Network
The main branch hosts the GNN for Social Network project.
The `shikha` and `harsha` branches have been merged into the main branch.
- GNN for Finance
The `kanishk` branch contains the GNN for Finance project.

# Reimagining Influence Detection in Social Networks via Graph Neural Networks

---

## Objective

To evaluate the effectiveness of graph-based learning (GNNs) over traditional tabular models (XGBoost) in classifying users based on their influence or activity, leveraging structured and relational information from Q&A and discussion forums.

## Datasets

- **Stack Overflow**: Extracted using the Stack Exchange API. Includes users, questions, and answers.
- **AskReddit (Reddit)**: Sourced from Kaggle’s “A Month of AskReddit” dataset.

Each dataset is transformed into:
- A tabular format for XGBoost.
- A heterogeneous graph format for GNNs.

## Key Components

- **Data Preprocessing**: Cleaning, labeling, and feature extraction.
- **Graph Construction**: Building heterogeneous graphs with user-question-answer (Stack Overflow) and user-post-comment (Reddit) relationships.
- **Modeling**:
  - XGBoost for baseline performance.
  - GNN using GraphSAGE with message passing over multi-typed edges.
- **Imbalance Handling**: Custom stratified mini-batching technique for class-balanced training.

Run the following command to clone the repository:
   ```bash
   git clone https://github.com/KanishkGoel1999/Group-8-Capstone-.git
   ```
Navigate to the cloned repository:
   ```bash
   cd Group-8-Capstone-
   ```

### Setting Up the Environment
Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```
Activate the virtual environment:

   ```bash
     venv\Scripts\activate
  ```
## To run this file:
1. Please make sure to have python version `3.10.12`
- For running the below commands, please ensure you are inside main_script folder.
```bash
cd main_script
```
2. Install the required packages using the command:
```bash
pip install requirements.txt
```
3. Download datasets from GCP/AWS in `data` folder and models and split to `model_artifacts` folder using the command:
```bash
python s3_downloader.py
```


4. Run the following commands to train and test the XGBoost model:
```bash 
python xgboost_model.py "ASK_REDDIT" 1
```
5. Run 
```bash
python xgboost_model.py "STACK_OVERFLOW" 1
```

- For testing GNN (for both datasets)
6. Run
```bash 
python gnn_test.py "STACK_OVERFLOW" 1
```
- For Stack Overflow data with config set `1`(same config set). - Loads the saved model and data split from `model_artifact` and gives performance metrics on test dataset.
7. Run 
```bash
python gnn_test.py "ASK_REDDIT" 1
```
 - For Stack Overflow data with config set `1`(same config set).

- Below commands are for training gnn and graph consturction(skip point 8,9 and 10 if you don't want to train gnn again)
8. Once the data is downloaded, run 
```bash
python graph_construction.py
```
 - This commands constructs graph for both the datasets.
9. Run 
```bash
python gnn_train.py "STACK_OVERFLOW" 1
```
 - For Stack Overflow data with config set `1` - Saves the splitted data, trains, and saves the model into  `model_artifact` folder.
10. Run
```bash 
python gnn_train.py "ASK_REDDIT" 1
```
 - For Stack Overflow data with config set `1`

- Once test_metrics are generated for GNN, and saved
7. Run to visualize the performance of the models:
```bash 
python performance_visualization.py
```

## Results Summary

| Dataset        | Model     | F1-Score | Accuracy | AUC   |  Precsion | Recall  |
|----------------|-----------|----------|----------|-------|-----------|---------|
| Stack Overflow | XGBoost   | 0.603    | 0.813    | 0.655 |  0.589    | 0.646   |
| Stack Overflow | GNN       | 0.612    | 0.813    | 0.672 |  0.596    | 0.661   |
| Reddit         | XGBoost   | 0.658    | 0.701    | 0.706 |  0.657    | 0.659   |
| Reddit         | GNN       | 0.616    | 0.655    | 0.792 |  0.664    | 0.792   |

## Technologies Used

- Python, PyTorch Geometric
- XGBoost
- NetworkX / DGL for preprocessing
- LaTeX for documentation

## Output

The final report (PDF) includes:
- Dataset statistics
- Graph types and visual examples
- Model architectures
- Performance evaluation
- Discussion and conclusion

## Authors

- Shikha Kumari 
- Harshavardana Reddy Kolan   
- Prof. Amir Hossein Jafari

---
