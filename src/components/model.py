from xgboost import XGBClassifier

class Models:
    """
    A class to define and retrieve different machine learning models.
    """
    
    @staticmethod
    def get_xgboost_model(random_state=42):
        """
        Returns an instance of the XGBoost classifier with default parameters.
        
        Parameters:
            random_state (int): Random state for reproducibility.
        
        Returns:
            XGBClassifier: An instance of XGBoost classifier.
        """
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)