# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
file_path = "processed_data_noadded_features.csv"
df = pd.read_csv(file_path)

# Drop non-relevant columns
df_filtered = df.drop(columns=["user_id", "reputation", "display_name"])

# Compute correlation matrix
corr_matrix = df_filtered.corr()

# Plot heatmap to visualize collinearity
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Features and Target Variable")
plt.show()
# %%
# ---- Multi-Collinearity Analysis using VIF ----
# Remove the target variable before computing VIF
X = df_filtered.drop(columns=["influential"])

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF values
print("\nVariance Inflation Factor (VIF) Analysis:")
print(vif_data.sort_values(by="VIF", ascending=False))
# %%

df_refined = df_filtered.drop(columns=["total_badges"])  # Removing total_badges as it's redundant

# Compute VIF again after dropping the feature
X_drop_1 = df_refined.drop(columns=["influential"])  # Keep only independent variables
vif_data_1 = pd.DataFrame()
vif_data_1["Feature"] = X_drop_1.columns
vif_data_1["VIF"] = [variance_inflation_factor(X_drop_1.values, i) for i in range(X_drop_1.shape[1])]
print(vif_data_1.sort_values(by="VIF", ascending=False))


# %%
df_refined = df_filtered.drop(columns=["total_badges","bronze_badges"])

# Compute VIF again after dropping the feature
X_drop_2 = df_refined.drop(columns=["influential"])  # Keep only independent variables
vif_data_2 = pd.DataFrame()
vif_data_2["Feature"] = X_drop_2.columns
vif_data_2["VIF"] = [variance_inflation_factor(X_drop_2.values, i) for i in range(X_drop_2.shape[1])]
print(vif_data_2.sort_values(by="VIF", ascending=False))

# Feature       VIF
# 1     silver_badges  3.912050
# 0       gold_badges  3.844726
# 2     total_answers  3.781052
# 4  accepted_answers  3.768573
# 3       total_score  1.006400
# %%
df_refined = df_filtered.drop(columns=["total_badges","bronze_badges", "gold_badges"])

# Compute VIF again after dropping the feature
X_drop_2 = df_refined.drop(columns=["influential"])  # Keep only independent variables
vif_data_2 = pd.DataFrame()
vif_data_2["Feature"] = X_drop_2.columns
vif_data_2["VIF"] = [variance_inflation_factor(X_drop_2.values, i) for i in range(X_drop_2.shape[1])]
print(vif_data_2.sort_values(by="VIF", ascending=False))

#
# Feature       VIF
# 1     total_answers  3.780790
# 3  accepted_answers  3.765127
# 0     silver_badges  1.033300
# 2       total_score  1.006354

# %%
df_refined = df_filtered.drop(columns=["total_badges","bronze_badges", "gold_badges", "accepted_answers"])

# Compute VIF again after dropping the feature
X_drop_2 = df_refined.drop(columns=["influential"])  # Keep only independent variables
vif_data_2 = pd.DataFrame()
vif_data_2["Feature"] = X_drop_2.columns
vif_data_2["VIF"] = [variance_inflation_factor(X_drop_2.values, i) for i in range(X_drop_2.shape[1])]
print(vif_data_2.sort_values(by="VIF", ascending=False))

#
#  Feature       VIF
# 0  silver_badges  1.032886
# 1  total_answers  1.032385
# 2    total_score  1.005847
# %%
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Select final features based on VIF results
final_features = ["silver_badges", "total_answers", "total_score"]

# Define X (features) and y (target)
X = df_refined[final_features]
y = df_refined["influential"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize XGBoost model with class weighting (to handle imbalance)
scale_pos_weight = 9.0  # Approximate class imbalance ratio (90:10)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=scale_pos_weight)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Compute F1-score
f1 = f1_score(y_test, y_pred, average="binary")

# Print results
print(f"F1-Score: {f1:.4f}")

# F1-Score: 0.5607