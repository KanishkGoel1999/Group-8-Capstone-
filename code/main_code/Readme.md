# Main Code

This repository contains the core scripts for training, testing, and saving models across classical machine learning models and Graph Neural Networks (GNNs), implemented using both **DGL** and **PyTorch Geometric (PyG)** frameworks.

## Repository Structure

```
main_code/
│
├── classical_model/
│   ├── train_ML.py    # Train classical machine learning models
│   └── test_ML.py     # Evaluate classical models on test data
│
├── gnn/
│   ├── DGL/
│   │   ├── models/    # Directory storing best saved DGL models
│   │   ├── train_dgl.py   # Train GNNs using DGL
│   │   └── test_dgl.py    # Test GNNs using DGL
│   │
│   └── PyG/
│       ├── models/    # Directory storing best saved PyG models
│       ├── train_pyg.py   # Train GNNs using PyG
│       └── test_pyg.py    # Test GNNs using PyG
```

---

## Modules Description

- **`classical_model/train_ML.py`**  
  Script to train classical machine learning models (e.g., Random Forest, XGBoost, Logistic Regression).

- **`classical_model/test_ML.py`**  
  Script to evaluate the performance of trained classical models on test datasets.

- **`gnn/DGL/train_dgl.py`**  
  Script to train GNN models using the **Deep Graph Library (DGL)** framework.

- **`gnn/DGL/test_dgl.py`**  
  Script to test DGL-trained models and generate evaluation metrics.

- **`gnn/DGL/models/`**  
  Folder to store the best-performing DGL model checkpoints (saved during training).

- **`gnn/PyG/train_pyg.py`**  
  Script to train GNN models using the **PyTorch Geometric (PyG)** framework.

- **`gnn/PyG/test_pyg.py`**  
  Script to test PyG-trained models and evaluate performance.

- **`gnn/PyG/models/`**  
  Folder to store the best-performing PyG model checkpoints.

---

## How to Use

1. **Train Classical Models**
   ```bash
   python classical_model/train_ML.py
   ```

2. **Test Classical Models**
   ```bash
   python classical_model/test_ML.py
   ```

3. **Train GNN Models**
   - Using **DGL**:
     ```bash
     python gnn/DGL/train_dgl.py
     ```
   - Using **PyG**:
     ```bash
     python gnn/PyG/train_pyg.py
     ```

4. **Test GNN Models**
   - Using **DGL**:
     ```bash
     python gnn/DGL/test_dgl.py
     ```
   - Using **PyG**:
     ```bash
     python gnn/PyG/test_pyg.py
     ```

---

## Requirements

- Python 3.8+
- Install requirements.txt
- Also, install these special libraries:
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      pip install torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
      pip install pyg-lib -f https://data.pyg.org/whl
      pip install torch-sparse -f https://data.pyg.org/whl


Make sure you have both **DGL** and **PyG** installed correctly, as they have their own specific installation steps, especially regarding CUDA support.

---

## Notes

- Models are automatically saved during training if a new best validation score is achieved.
- You can edit config file for hyperparameters.
- This structure allows easy switching between classical and graph-based machine learning experiments.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
