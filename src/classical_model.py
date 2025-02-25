# %% Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Import utility functions from the correct path
from utility.classical_model_utility import (
    split_data,
    preprocess_data,
    compute_metrics,
    compute_confusion_matrix
)

# %% Load dataset
file_path = "preprocessed_data.csv"
df = pd.read_csv(file_path)

# Display basic info
print(df.info())

# Preprocess data (drop unnecessary columns and handle missing values)
df = preprocess_data(df, columns_to_remove=["user_id", "display_name"])
print(df.columns)
# %% Prepare features and target, then split data
X_train, X_test, y_train, y_test = split_data(df, target_column="influential")

# Check dataset sizes
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# %% Compute scale_pos_weight for class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print("Computed scale_pos_weight:", scale_pos_weight)

# %% Hyperparameter tuning with RandomizedSearchCV for XGBoost
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'scale_pos_weight': [scale_pos_weight]  # using computed imbalance ratio
}

# Initialize XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

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

# %% Train best model on full training data
best_xgb_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = best_xgb_model.predict(X_test)

# %% Evaluate Model Performance
metrics = compute_metrics(y_test, y_pred)
print("Model Performance Metrics:", metrics)

# Compute and display confusion matrix
cm = compute_confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Interpretation:
# True Negatives (TN) = 12,580 → Correctly predicted non-influential users.
# False Positives (FP) = 2,176 → Incorrectly predicted as influential when they are not.
# False Negatives (FN) = 898 → Incorrectly predicted as non-influential when they are influential.
# True Positives (TP) = 706 → Correctly predicted influential users.


