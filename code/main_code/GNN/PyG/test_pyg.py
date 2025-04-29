from component.packages import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from component.preprocess import *
from component.gnn.utils import *
from component.gnn.model_pytorch import *

config_path="config/config.yaml"

warnings.filterwarnings("ignore")


# Set device as cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(save_dir, exist_ok=True)

m_name = "GNN"

def load_config(config_path="/home/ubuntu/code/component/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# val_losses = [0.0292, 0.0073, 0.0062, 0.0056, 0.0052, 0.0046, 0.0045, 0.0045, 0.0046, 0.0040, 0.0039, 0.0040, 0.0065, 0.0036, 0.0039, 0.0030, 0.0028, 0.0027, 0.0030, 0.0026, 0.0031, 0.0060, 0.0025, 0.0025, 0.0027, 0.0026, 0.0029, 0.0025, 0.0028, 0.0025, 0.0025, 0.0024, 0.0033, 0.0022]
# f1-scores = [0.5056, 0.7431, 0.7974, 0.7949, 0.8123, 0.8695, 0.8409, 0.8384, 0.8608, 0.8938, 0.8631, 0.8439, 0.7311, 0.8643, 0.8817, 0.8771, 0.9167, 0.8958, 0.8847, 0.9166, 0.8866, 0.8498, 0.9071, 0.9048, 0.9065, 0.9169, 0.8950, 0.9179, 0.9037, 0.9048, 0.9149, 0.9075, 0.9160]


def evaluate():
    config = load_config()

    hidden_size = config["gnn"]["hidden_size"]
    n_layers  = config["gnn"]["n_layers"]
    out_size = config["gnn"]["out_size"]
    target_node = config["gnn"]["target_node"]


    # For dataset 1
    data_path = config["data"]["test_data_path1"]
    data_path = os.path.join(os.path.expanduser("~"), data_path)
    print("Loading test data...")
    df = pd.read_csv(data_path)
    model_path = os.path.join("models", 'best_pyg_model_DS1.pth')
    df = preprocess_data_1(df, 'trans_date_trans_time', 'merchant', 'trans_num')
    transaction_feats = df[
        ['transaction_id', 'amt', 'is_weekend', 'Month_Sin', 'Month_Cos', 'hour_sin', 'hour_cos', 'day_sin',
         'day_cos']].drop_duplicates(
        subset=['transaction_id'])
    user_feats = df[
        ['card_number', 'age', 'gender', 'lat', 'long']].drop_duplicates(subset=['card_number'])
    merchant_feats = df[['merchant_id', 'fraud_merchant_pct', 'merch_lat', 'merch_long']].drop_duplicates(
        subset=['merchant_id'])


    # For dataset 2
    # data_path = config["data"]["data_path2"]
    # data_path = os.path.join(os.path.expanduser("~"), data_path)
    # model_path = os.path.join("models", 'best_pyg_model_DS2.pth')
    # print("Loading test data...")
    # df = pd.read_csv(data_path)
    # train, df = preprocess_data_2(df)
    # transaction_feats = df[['transaction_id', 'amount', 'weekend_transaction', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'currency_AUD', 'currency_BRL', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_JPY', 'currency_MXN', 'currency_NGN', 'currency_RUB', 'currency_SGD', 'currency_USD', 'device_Android App', 'device_Chip Reader', 'device_Chrome', 'device_Edge', 'device_Firefox', 'device_Magnetic Stripe', 'device_NFC Payment', 'device_Safari', 'device_iOS App']]
    # user_feats = df[['card_number', 'country_Australia', 'country_Brazil', 'country_Canada', 'country_France', 'country_Germany', 'country_Japan', 'country_Mexico', 'country_Nigeria', 'country_Russia', 'country_Singapore', 'country_UK', 'country_USA']].drop_duplicates(subset=['card_number'])
    # merchant_feats = df[['merchant_id', 'merchant_category_Education', 'merchant_category_Entertainment', 'merchant_category_Gas', 'merchant_category_Grocery', 'merchant_category_Healthcare', 'merchant_category_Restaurant', 'merchant_category_Retail', 'merchant_category_Travel']].drop_duplicates(
    #     subset=['merchant_id'])


    # Create PyG graph
    data = create_graph_pyg(df, transaction_feats, merchant_feats, user_feats)
    print_graph_info_pyg(data)

    # Extract labels
    labels = data['transaction'].y.long().to(device)

    # Load trained model
    print("Loading trained model...")
    in_size_dict = {
        'transaction': data['transaction'].x.shape[1],
        'user': data['user'].x.shape[1],
        'merchant': data['merchant'].x.shape[1]
    }

    etypes = data.edge_types

    model = HeteroGNN(in_size_dict, hidden_size, out_size, n_layers, etypes, target_node)

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    loaded_state_dict = torch.load(model_path)
    print(loaded_state_dict.keys())

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Move data to device
    data = data.to(device)
    model = model.to(device)

    # Forward pass (entire dataset)
    with torch.no_grad():
        out = model(data)

    # Predictions
    probs = F.softmax(out, dim=1)
    preds = out.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    # Compute evaluation metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = average_precision_score(labels, preds)

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-PR Score: {auc:.4f}")

    # Save results
    result_df = pd.DataFrame({'true_labels': labels, 'predictions': preds})
    result_df.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to 'test_predictions.csv'.")

    y_prob_dict = {"HeteroGNN": probs.cpu().numpy()}  # dict of model_name -> probability array
    generate_aucpr_plot(labels, y_prob_dict)
    visualize_network_graph(data, sample_size=700)


if __name__ == "__main__":
    evaluate()
