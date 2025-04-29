# Code Repository

This repository contains the full pipeline for training, evaluating, and saving models for classical machine learning and graph neural networks (GNNs) using DGL and PyTorch Geometric (PyG) frameworks. It also includes utilities for data preprocessing, configuration management, and package setup.

## Directory Structure

```
code/
│
├── components/
│   ├── classical_model/
│   │   └── utils/          # Utilities for classical models
│   ├── gnn/
│   │   ├── model/          # GNN model architectures
│   │   └── utils/          # Utilities for GNNs
│   ├── config/             # Configuration files (e.g., hyperparameters, paths)
│   ├── packages.py         # Install required packages
│   └── preprocessing.py    # Preprocessing scripts for data cleaning and feature engineering
│
├── main_code/
│   ├── classical_model/
│   │   ├── train_ML.py     # Train classical models
│   │   └── test_ML.py      # Test classical models
│   ├── gnn/
│   │   ├── DGL/
│   │   │   ├── models/     # Best saved DGL models
│   │   │   ├── train_dgl.py  # Train GNN using DGL
│   │   │   └── test_dgl.py   # Test GNN using DGL
│   │   └── PyG/
│   │       ├── models/     # Best saved PyG models
│   │       ├── train_pyg.py  # Train GNN using PyG
│   │       └── test_pyg.py   # Test GNN using PyG
```

---

## How to Use

Follow these steps to set up and run training/testing workflows.

### 1. Install Required Packages

Use the provided script to install all dependencies:

Install Python 10 and make a virtual environment. Run these commands in bash and you're good to go.
```bash
sudo wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
sudo tar xvf Python-3.10.13.tgz
python3.10 -m venv GNN_310_env
source GNN_310_env/bin/activate
pip install --upgrade pip
cd code/
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install pyg-lib -f https://data.pyg.org/whl
pip install torch-sparse -f https://data.pyg.org/whl
```

_(Note: You might need to set up CUDA-specific versions of `torch`, `torch_geometric`, or `dgl` based on your hardware.)_

---

### 2. Get data

Download the dataset before training. This script downloads both datasets, please refer comments in the train and test files on  which one to use:

```bash
cd ~ 
%% add data and model code
```

---

### 3. Train a Model

Choose the model type you want to train:

- **Classical Machine Learning Models**:
  ```bash
  cd main_code/classical_model/
  python train_ML.py
  ```

- **Graph Neural Networks (GNNs)**:

  - **Using DGL**:
    ```bash
    cd main_code/gnn/DGL/
    python train_dgl.py
    ```

  - **Using PyG**:
    ```bash
    cd main_code/gnn/PyG/
    python train_pyg.py
    ```

Trained models will be automatically saved into their respective `models/` folder upon achieving the best validation performance.

---

### 4. Load trained models
Download already trained ML models here, GNN models are already in the folders:
Place the models into the corresponding directories:

For ML models, put the files inside:
```bash
%% add wget code
cd main_code/classical_machine_learning_models/models
```


### 5. Test a Model

Evaluate the trained model on the test set:

- **Classical Machine Learning Models**:
  ```bash
  cd main_code/classical_model/
  python test_ML.py
  ```

- **Graph Neural Networks (GNNs)**:

  - **Using DGL**:
    ```bash
    cd main_code/gnn/DGL/
    python test_dgl.py
    ```

  - **Using PyG**:
    ```bash
    cd main_code/gnn/PyG/
    python test_pyg.py
    ```

---

## Key Highlights

- Modular codebase separating utilities (`components/`) and model operations (`main_code/`).
- Support for both classical machine learning models and modern graph-based deep learning models.
- Flexible configuration management through the `config/` folder.
- Best model checkpoints saved during training for reproducibility.

---

## Requirements

- Python 3.10
- Make a virtual environment

---

## Notes

- Ensure correct relative imports if running scripts manually from different folders.
- Customize hyperparameters, dataset paths, or model settings in the `config/` folder or directly inside the training scripts as needed.
- Saved models are automatically loaded during testing.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
