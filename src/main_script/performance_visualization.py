# import os
# import pickle
# import matplotlib.pyplot as plt
# import argparse

# artifact_dir = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

# def parse_args():
#     parser = argparse.ArgumentParser(description="Visualize GNN model performance")
#     parser.add_argument(
#         "dataset",
#         type=str,
#         choices=["STACK_OVERFLOW", "ASK_REDDIT"],
#         help="Dataset name (STACK_OVERFLOW or ASK_REDDIT)"
#     )
#     return parser.parse_args()

# def load_metrics(dataset_name):
#     with open(os.path.join(artifact_dir, f"train_metrics_{dataset_name}.pkl"), "rb") as f:
#         train_metrics = pickle.load(f)
#     with open(os.path.join(artifact_dir, f"test_metrics_{dataset_name}.pkl"), "rb") as f:
#         test_metrics = pickle.load(f)
#     return train_metrics, test_metrics

# def plot_metrics(train_metrics, test_metrics, dataset_name):
#     epochs = list(range(1, len(train_metrics["train_losses"]) + 1))
#     test_loss_line = [test_metrics["test_losses"][0]] * len(epochs)
#     test_acc_line = [test_metrics["test_accuracies"][0]] * len(epochs)
#     test_auc_line = [test_metrics["test_auc"][0]] * len(epochs)
#     test_precision_line = [test_metrics["test_precision"][0]] * len(epochs)

#     plt.figure(figsize=(15, 10))
    
#     # Plot Loss
#     # plt.subplot(2, 2, 1)
#     # plt.plot(epochs, train_metrics["train_losses"], label="Train Loss", color="red")
#     # plt.plot(epochs, test_loss_line, label="Test Loss", color="blue")
#     # plt.title(f"{dataset_name} - Loss Comparison")
#     # plt.xlabel("Epochs")
#     # plt.ylabel("Loss")
#     # plt.legend()

#     # # Plot Accuracy
#     # plt.subplot(2, 2, 2)
#     # plt.plot(epochs, train_metrics["train_accuracies"], label="Train Accuracy", color="green")
#     # plt.plot(epochs, test_acc_line, label="Test Accuracy", color="orange")
#     # plt.title(f"{dataset_name} - Accuracy Comparison")
#     # plt.xlabel("Epochs")
#     # plt.ylabel("Accuracy")
#     # plt.legend()

#     # AUC
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_metrics["train_aucs"], label="Train AUC", color="purple")
#     plt.plot(epochs, test_auc_line, label="Test AUC", color="black")
#     plt.title(f"{dataset_name} - AUC")
#     plt.xlabel("Epochs")
#     plt.ylabel("AUC")
#     plt.legend()

#     # Precision
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_metrics["train_precisions"], label="Train Precision", color="cyan")
#     plt.plot(epochs, test_precision_line, label="Test Precision", color="magenta")
#     plt.title(f"{dataset_name} - Precision")
#     plt.xlabel("Epochs")
#     plt.ylabel("Precision")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     args = parse_args()
#     train_metrics, test_metrics = load_metrics(args.dataset)
#     plot_metrics(train_metrics, test_metrics, args.dataset)
#     # for dataset in ["STACK_OVERFLOW", "ASK_REDDIT"]:
#     #     train_metrics, test_metrics = load_metrics(dataset)
#     #     plot_metrics(train_metrics, test_metrics, dataset)

import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

artifact_dir = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

def plot_roc(dataset_name):
    with open(os.path.join(artifact_dir, f"test_metrics_{dataset_name}.pkl"), "rb") as f:
        test_data = pickle.load(f)

    probs = test_data["probs"]
    labels = test_data["labels"]

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    for dataset in ["STACK_OVERFLOW", "ASK_REDDIT"]:
        plot_roc(dataset)
