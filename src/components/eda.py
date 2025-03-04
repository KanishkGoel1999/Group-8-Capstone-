import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, file_path):
        """
        Initializes the EDA class with the dataset.
        :param file_path: Path to the CSV file.
        """
        self.file_path = file_path
        self.df = None
    
    def load_data(self):
        """Loads the dataset from the file path."""
        self.df = pd.read_csv(self.file_path)
    
    def preprocess_data(self, columns):
        """Drops non-relevant columns from the dataset."""
        if self.df is not None:
            self.df = self.df.drop(columns=columns, errors='ignore')
    
    def compute_correlation(self):
        """Computes and returns the correlation matrix."""
        if self.df is not None:
            return self.df.corr()
        return None
    
    def plot_heatmap(self):
        """Plots a heatmap to visualize collinearity."""
        corr_matrix = self.compute_correlation()
        if corr_matrix is not None:
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap of Features and Target Variable")
            plt.show()
        else:
            print("Data not loaded. Please load the data first.")