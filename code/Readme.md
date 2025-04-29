## Directory overview

This directory contains the code for the Capstone project, which includes various machine learning and deep learning models for training and inference.

The directory is structured as follows:
```
.
├── code
│   ├── component
│   │   ├── classical_machine_learning_models
│   │   │   └── utils.py
│   │   ├── gnn
│   │   │   ├── model_dgl.py
│   │   │   ├── model_pytorch.py
│   │   │   └── utils.py
│   │   ├── config.yaml
│   │   ├── packages.py
│   │   └── preprocess.py
│   └── main_code
│       ├── classical_machine_learning_models
│       │   ├── models
│       │   ├── train_ML.py
│       │   └── test_ML.py
│       └── GNN
│           ├── DGL
│           │   ├── models
│           │   ├── test_dgl.py
│           │   └── train_dgl.py
│           └── PyG
│               ├── models
│               ├── test_pyg.py
│               └── train_pyg.py

```

## Usage

For detailed usage information and instructions for each script, please refer to the individual files in the respective directories.

The `main_code` directory contains subdirectories for different model types, including classical machine learning, and graph neural networks (GNN). Each subdirectory contains scripts for training and inference.

The `component` directory provides utility functions and helper classes that can be used across different models for tasks such as data preprocessing, loading, and other common operations.

When running the scripts, make sure to provide the required input data and configure the appropriate model parameters as specified in each script's documentation.

## Requirements

The code in this directory requires Python 3.10 and the following libraries to name a few:
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- PyTorch
- dgl (for GNN)
- Matplotlib (for plotting)

Please refer to the individual script files for specific dependencies and requirements.
