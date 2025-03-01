# metrics.py
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

class Metrics:
    """
    A class to compute evaluation metrics and confusion matrix for model performance.
    """
    
    @staticmethod
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
            "recall": recall_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)
        }
    
    @staticmethod
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