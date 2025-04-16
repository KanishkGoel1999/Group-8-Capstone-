# train_models.py
import sys

sys.path.append('../../component')

import os
import argparse
import prettytable
import argparse
import xgboost as xgb
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import torch

from component.preprocess import *
from component.classical_machine_learning_models.utils import *

#data_path = '/home/ubuntu/processed_train.csv'
data_path = '/home/ubuntu/synthetic_fraud_data.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42

np.random.seed(seed)
parent_dir = os.getcwd()



def save_predictions(model, X_test, y_test, output_path):
    '''
    Generate predictions using the trained model and save them to a file along with the original dataframe

    :param model: Trained model
    :param X_test: Test features
    :param y_test: Test target
    :param output_path: Path to save the predictions

    :returns None
    '''
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        y_pred = (y_prob > 0.5).astype(int)
        predictions_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        predictions_df['Prediction'] = y_pred
        predictions_df['Probability'] = y_prob

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        predictions_df.to_csv(output_path, index=False)

        print(f"Predictions saved to {output_path}")

    except Exception as e:
        print(f"Error generating predictions: {e}")


def save_model(model, model_path):
    '''
    Save the trained model to a file

    :param model: Trained model
    :param model_path: Path to save the model

    :returns None
    '''
    try:
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

        print(f"Model saved to {model_path}")

    except Exception as e:
        print(f"Error saving the model: {e}")


def main():
    target = 'is_fraud'
    data = pd.read_csv(data_path)

    #data = preprocess_data_ML(data, 'trans_date_trans_time', 'merchant', 'trans_num')
    data, test = preprocessDS2(data)

    data = data[
        ['transaction_id', 'amount', 'weekend_transaction', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
         'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_JPY', 'currency_RUB', 'currency_SGD', 'currency_USD', 'device_Android App', 'device_Chip Reader', 'device_Magnetic Stripe', 'device_NFC Payment',
         'device_Safari', 'device_iOS App', 'card_number', 'country_Canada', 'country_France',
         'country_Germany', 'country_Japan', 'country_Russia', 'country_Singapore', 'country_UK',
         'country_USA', 'merchant_id', 'merchant_category_Education', 'merchant_category_Entertainment', 'merchant_category_Gas', 'merchant_category_Grocery', 'is_fraud']]

    X = data.drop(columns=[target])
    y = data[target]

    train_models(X, y)


if __name__ == '__main__':
    main()
