import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import prettytable

import sys
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from component.preprocess import *
from component.classical_machine_learning_models.utils import *

import argparse
import warnings

warnings.filterwarnings("ignore")

#data_path = '/home/ubuntu/processed_test.csv'
data_path = '/home/ubuntu/synthetic_fraud_data.csv'

def generate_aucpr_plot(y_test, y_prob_dict):
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in y_prob_dict.items():
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AUC={average_precision_score(y_test, y_prob):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('plots/pr_curve.png')
    plt.show()

def generate_aucroc_plot(y_test, y_prob_dict):
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in y_prob_dict.items():
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_score(y_test, y_prob):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('plots/roc_curve.png')
    plt.show()


def evaluate():
    target = 'is_fraud'

    # Load test data
    print("Loading test data...")
    data = pd.read_csv(data_path)

    #data = preprocess_data_ML(data, 'trans_date_trans_time', 'merchant', 'trans_num')
    #data = preprocess_data2(data, 'timestamp', 'merchant', 'transaction_id')
    train, data = preprocessDS2((data))

    data = data[
        ['transaction_id', 'amount', 'weekend_transaction', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
         'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_JPY', 'currency_RUB', 'currency_SGD',
         'currency_USD', 'device_Android App', 'device_Chip Reader', 'device_Magnetic Stripe', 'device_NFC Payment',
         'device_Safari', 'device_iOS App', 'card_number', 'country_Canada', 'country_France',
         'country_Germany', 'country_Japan', 'country_Russia', 'country_Singapore', 'country_UK',
         'country_USA', 'merchant_id', 'merchant_category_Education', 'merchant_category_Entertainment',
         'merchant_category_Gas', 'merchant_category_Grocery', 'is_fraud']]

    print(data.info())
    X_test = data.drop(columns=[target])
    y_test = data[target]

    results = test_model(X_test, y_test)

    print(results)


if __name__ == "__main__":
    evaluate()