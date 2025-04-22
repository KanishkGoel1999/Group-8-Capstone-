from component.packages import *


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



def train_models(X_train, y_train, random_state=42, dir_name="models"):
    if isinstance(X_train, pd.DataFrame):
        X_train.columns = X_train.columns.astype(str)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
    }

    os.makedirs(dir_name, exist_ok=True)

    # Train and save models
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_path = os.path.join(dir_name, f'{name.replace(" ", "_")}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

def test_model(X_test, y_test, dir_name="models"):
    if isinstance(X_test, pd.DataFrame):
        X_test.columns = X_test.columns.astype(str)

    model_names = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest', 'XGBoost']
    results = []

    for name in model_names:
        model_path = os.path.join(dir_name, f'{name.replace(" ", "_")}.pkl')
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

        # Add to unified PR curve plot
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall_curve, precision_curve, label=f'{name} (AUCPR = {auc_pr:.2f})')

    # Finalize the plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return pd.DataFrame(results).sort_values(by='AUC_PR', ascending=False)