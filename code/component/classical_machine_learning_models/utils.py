import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, \
    roc_auc_score, confusion_matrix, average_precision_score
import xgboost as xgb
import prettytable
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
import pickle


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
    aucpr = auc(recall_curve, precision_curve)
    return y_pred, accuracy, precision, recall, f1, cm, roc_auc, aucpr


def display_results(model, X_test, y_test, title=None):
    _, accuracy, precision, recall, f1, cm, roc_auc, aucpr = evaluate_model(model, X_test, y_test)

    results = prettytable.PrettyTable(title=title)
    results.field_names = ["Metric", "Value"]
    results.add_row(["Accuracy", accuracy])
    results.add_row(["Precision", precision])
    results.add_row(["Recall", recall])
    results.add_row(["F1 Score", f1])
    results.add_row(["ROC AUC", roc_auc])
    results.add_row(["AUCPR", aucpr])
    print(results)
    print("Confusion Matrix")
    print(cm)


def generate_aucpr_plot(y_test, y_prob):
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    aucpr = auc(recall_curve, precision_curve)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, marker='.', label='AUCPR = {:.2f}'.format(aucpr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def dataset_stats(data, target, X_train, y_train, X_val=None, X_test=None, y_val=None, y_test=None):
    table = prettytable.PrettyTable()

    print("Dataset Stats")

    table.field_names = ["Data", "Rows", "Columns", "Target True", "Target False", "Target %"]
    table.add_row(
        ["Complete Dataset", data.shape[0], data.shape[1], data[target].sum(), data.shape[0] - data[target].sum(),
         f"{round(data[target].sum() / data.shape[0] * 100, 2)}%"])

    table.add_row(["Train", X_train.shape[0], X_train.shape[1], y_train.sum(), y_train.shape[0] - y_train.sum(),
                   f"{round(y_train.sum() / y_train.shape[0] * 100, 2)}%"])
    if X_test is not None and y_test is not None:
        table.add_row(["Test", X_test.shape[0], X_test.shape[1], y_test.sum(), y_test.shape[0] - y_test.sum(),
                       f"{round(y_test.sum() / y_test.shape[0] * 100, 2)}%"])

    if X_val is not None and y_val is not None:
        table.add_row(["Validation", X_val.shape[0], X_val.shape[1], y_val.sum(), y_val.shape[0] - y_val.sum(),
                       f"{round(y_val.sum() / y_val.shape[0] * 100, 2)}%"])
    print(table)


def train_and_evaluate_models(X_train, y_train, X_test, y_test, random_state=42):
    if isinstance(X_train, pd.DataFrame):
        X_train.columns = X_train.columns.astype(str)


    # Ensure test features match training features
    X_test = X_test[X_train.columns]

    # Define models
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'XGBoost': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Random Forest': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=random_state),
        'LightGBM': LGBMClassifier(random_state=random_state)
    }


    # Store results
    results = []

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_prob)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC_PR': auc_pr
        })

    return pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)


def train_models(X_train, y_train, random_state=42):
    if isinstance(X_train, pd.DataFrame):
        X_train.columns = X_train.columns.astype(str)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
    }

    os.makedirs("models", exist_ok=True)

    # Train and save models
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_path = os.path.join("models", f'{name.replace(" ", "_")}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)



def test_model(X_test, y_test):
    if isinstance(X_test, pd.DataFrame):
        X_test.columns = X_test.columns.astype(str)

    model_names = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest', 'XGBoost']
    results = []

    for name in model_names:
        model_path = os.path.join("models", f'{name.replace(" ", "_")}.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_prob)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC_PR': auc_pr
        })

    return pd.DataFrame(results).sort_values(by='AUC_PR', ascending=False)
