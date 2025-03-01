import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

# function for checking missing values #TODO
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(df, columns_to_remove):
    """
    Removes specified columns from the dataset.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        columns_to_remove (list): List of column names to drop.
    
    Returns:
        pd.DataFrame: Processed dataframe.
    """
    return df.drop(columns=columns_to_remove, errors='ignore')

