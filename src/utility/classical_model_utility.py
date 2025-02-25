import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

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

def compute_metrics(y_true, y_pred):
    """
    Computes evaluation metrics including recall, precision, f1-score, and accuracy.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        dict: Dictionary containing all computed metrics.
    """
    return {
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        "accuracy": accuracy_score(y_true, y_pred)
    }

def compute_confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        np.ndarray: Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)