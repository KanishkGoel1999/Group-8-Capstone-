# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


# %%
file_path = "processed_data_noadded_features.csv"
df = pd.read_csv(file_path)

# Display basic info
print(df.info())

df.head()

# %%
# Drop 'reputation' and unnecessary columns
df = df.drop(columns=["user_id", "display_name", "reputation"])

# Check for missing values
print(df.isnull().sum())

# %%
df = df.fillna(0)

# %%
X = df.drop(columns=["influential"])  # Features
y = df["influential"]  # Target variable

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check dataset sizes
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# %%

# Check class distribution
class_counts = y_train.value_counts()
print("Class Distribution in Training Data:\n", class_counts)

# Calculate scale_pos_weight for XGBoost (to handle class imbalance)
scale_pos_weight = class_counts[0] / class_counts[1]  # Majority class / Minority class
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# %%

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Step size
    'max_depth': [3, 5, 7],  # Tree depth
    'subsample': [0.7, 0.8, 1.0],  # Row sampling
    'colsample_bytree': [0.7, 0.8, 1.0],  # Feature sampling
    'scale_pos_weight': [1, scale_pos_weight]  # Class balancing
}

# Initialize XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model from tuning
best_xgb_model = grid_search.best_estimator_

# Print best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# %%

# Train the best model on full training data
best_xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = best_xgb_model.predict(X_test)

# Evaluate F1-score
f1_score_best = f1_score(y_test, y_pred)
print(f"Optimized F1-score: {f1_score_best:.4f}")

# %%
# xgb_model = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric="logloss",
#     n_estimators=50,  # Reduced number of trees
#     learning_rate=0.1,  # Learning rate
#     max_depth=4,  # Lower depth for efficiency
#     subsample=0.8,  # Row sampling
#     colsample_bytree=0.8  # Feature sampling
# )

# # Train the model
# xgb_model.fit(X_train, y_train)

# %%
# Make predictions on test set
# y_pred = xgb_model.predict(X_test)

# # Compute F1-score
# f1 = f1_score(y_test, y_pred)
# print(f"F1-score: {f1:.4f}")
# %%
