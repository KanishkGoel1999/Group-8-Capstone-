# Project Components

This repository contains the core modules for building, training, and evaluating classical machine learning models and graph neural networks (GNNs). It also includes utilities for preprocessing, configuration, and package management.

## Repository Structure

```
components/
│
├── classical_model/
│   └── utils/
│       - Utility functions specific to classical machine learning models
│
├── gnn/
│   ├── model/
│   │   - GNN model architectures
│   └── utils/
│       - Utility functions for GNN training and evaluation
│
├── config/
│   - Configuration files for model parameters, training settings, and environment variables
│
├── packages.py
│   - Script to manage and import required Python packages
│
├── preprocessing.py
│   - Scripts for data cleaning, feature engineering, and preprocessing
```

---

## Modules Description

- **`classical_model/utils/`**  
  Helper functions supporting classical machine learning models (e.g., training routines, evaluation metrics, etc.).

- **`gnn/model/`**  
  Custom implementations of Graph Neural Network architectures.

- **`gnn/utils/`**  
  Helper functions for GNN tasks such as data loading, batching, evaluation, and visualization.

- **`config/`**  
  YAML, JSON, or Python-based configuration files to control the hyperparameters, data paths, and experiment settings.

- **`packages.py`**  
  Script to automate environment setup by checking and importing required dependencies.

- **`preprocessing.py`**  
  Scripts to prepare datasets for training and evaluation — includes operations like missing value handling, encoding, normalization, and graph construction.

---

## How to Use
All these files are imported and used by the main_code files. No need to run them indivisially. 

---

## Notes

- Ensure that you have proper directory access and correct relative imports while running scripts.
- This project is modularized for easy extension to additional models, datasets, or utility functions.
- For GNN-related tasks, ensure that `torch_geometric` dependencies (e.g., `torch_scatter`, `torch_sparse`) are installed properly.

---

## License

This project is licensed under the [MIT License](LICENSE).
