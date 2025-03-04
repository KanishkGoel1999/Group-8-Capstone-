import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.constants import EDGE_TYPES

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

def get_edge_index_dict(batch):
    """
    Constructs the edge index dictionary for a given batch dynamically.
    
    Parameters:
        batch (dict): A batch of data from the NeighborLoader containing edge indices.
    
    Returns:
        dict: Edge index dictionary mapping edge types to their respective edge indices.
    """
    return {
        EDGE_TYPES['ASKS']: batch[EDGE_TYPES['ASKS']].edge_index,
        EDGE_TYPES['REV_ASKS']: batch[EDGE_TYPES['REV_ASKS']].edge_index,
        EDGE_TYPES['HAS']: batch[EDGE_TYPES['HAS']].edge_index,
        EDGE_TYPES['REV_HAS']: batch[EDGE_TYPES['REV_HAS']].edge_index,
        EDGE_TYPES['ANSWERS']: batch[EDGE_TYPES['ANSWERS']].edge_index,
        EDGE_TYPES['REV_ANSWERS']: batch[EDGE_TYPES['REV_ANSWERS']].edge_index,
        EDGE_TYPES['ACCEPTED_ANSWER']: batch[EDGE_TYPES['ACCEPTED_ANSWER']].edge_index,
        EDGE_TYPES['REV_ACCEPTED']: batch[EDGE_TYPES['REV_ACCEPTED']].edge_index,
        EDGE_TYPES['SELF_LOOP']: batch[EDGE_TYPES['SELF_LOOP']].edge_index,
    }