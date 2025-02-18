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
