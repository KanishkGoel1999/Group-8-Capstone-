# xgboost_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.utils import (
    split_data,
    preprocess_data
)
from components.metric import Metrics
from components.model import Models
import yaml

# Load dataset
file_path = "../data/preprocessed_data.csv"
df = pd.read_csv(file_path)

# Display basic info
print(df.info())

# Preprocess data
columns_to_remove = ["user_id", "display_name"]
df = preprocess_data(df, columns_to_remove=columns_to_remove)
print(df.columns)

# Prepare features and target, then split data
X_train, X_test, y_train, y_test = split_data(df, target_column="influential")

# Check dataset sizes
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Compute scale_pos_weight for class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print("Computed scale_pos_weight:", scale_pos_weight)

# Hyperparameter tuning with RandomizedSearchCV for XGBoost
# Load hyperparameters from config.yaml
config_path = "../components/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

param_dist = config["xgboost"]["sets"][1]
param_dist['scale_pos_weight'] = [scale_pos_weight]  # using computed imbalance ratio

# Initialize XGBoost model from Models class
xgb_model = Models.get_xgboost_model()

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    xgb_model, param_distributions=param_dist, n_iter=20, scoring='f1',
    cv=3, n_jobs=-1, verbose=1, random_state=42
)

random_search.fit(X_train, y_train)

# Get the best model
best_xgb_model = random_search.best_estimator_

# Print best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Train best model on full training data
best_xgb_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = best_xgb_model.predict(X_test)

# Evaluate Model Performance
metrics = Metrics.compute_metrics(y_test, y_pred)
print("Model Performance Metrics:", metrics)

# Compute and display confusion matrix
cm = Metrics.compute_confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
