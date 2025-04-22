import os
import pickle
import matplotlib.pyplot as plt

artifact_dir = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

def load_metrics(dataset_name):
    with open(os.path.join(artifact_dir, f"train_metrics_{dataset_name}.pkl"), "rb") as f:
        train_metrics = pickle.load(f)
    with open(os.path.join(artifact_dir, f"test_metrics_{dataset_name}.pkl"), "rb") as f:
        test_metrics = pickle.load(f)
    return train_metrics, test_metrics

def plot_metrics(train_metrics, test_metrics, dataset_name):
    epochs = list(range(1, len(train_metrics["train_losses"]) + 1))
    test_loss_line = [test_metrics["test_losses"][0]] * len(epochs)
    test_acc_line = [test_metrics["test_accuracies"][0]] * len(epochs)

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_metrics["train_losses"], label="Train Loss", color="red")
    plt.plot(epochs, test_loss_line, label="Test Loss", color="blue")
    plt.title(f"{dataset_name} - Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics["train_accuracies"], label="Train Accuracy", color="green")
    plt.plot(epochs, test_acc_line, label="Test Accuracy", color="orange")
    plt.title(f"{dataset_name} - Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for dataset in ["STACK_OVERFLOW", "ASK_REDDIT"]:
        train_metrics, test_metrics = load_metrics(dataset)
        plot_metrics(train_metrics, test_metrics, dataset)
