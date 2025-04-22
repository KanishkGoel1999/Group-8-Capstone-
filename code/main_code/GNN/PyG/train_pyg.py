from component.packages import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from component.preprocess import *
from component.gnn.utils import *
from component.gnn.model_pytorch import *

import argparse
import warnings

warnings.filterwarnings("ignore")

# Set device as cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m_name = "GNN"

def load_config(config_path="/home/ubuntu/code/component/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    print("Loading data...")
    # For dataset 1
    m_name = "best_pyg_model_DS1.pth"
    data_path = config["data"]["train_data_path1"]
    data_path = os.path.join(os.path.expanduser("~"), data_path)
    df = pd.read_csv(data_path)
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
    # m_name="best_pyg_model_DS2.pth"
    # data_path = config["data"]["data_path2"]
    # data_path = os.path.join(os.path.expanduser("~"), data_path)
    # df = pd.read_csv(data_path)
    # df, test = preprocess_data_2(df)
    # transaction_feats = df[
    #     ['transaction_id', 'amount', 'weekend_transaction', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_JPY', 'currency_USD']]
    # user_feats = df[
    #     ['card_number', 'country_Canada', 'country_France',
    #      'country_Germany', 'country_Japan', 'country_Russia', 'country_Singapore', 'country_UK',
    #      'country_USA']].drop_duplicates(subset=['card_number'])
    # merchant_feats = df[
    #     ['merchant_id', 'merchant_category_Education', 'merchant_category_Entertainment', 'merchant_category_Gas', 'merchant_category_Grocery']].drop_duplicates(
    #     subset=['merchant_id'])

    classified_idx = torch.tensor(transaction_feats.index.values, dtype=torch.long)

    # Create PyG graph
    data = create_graph_pyg(df, transaction_feats, merchant_feats, user_feats)

    # Train-validation split
    train_idx, valid_idx = train_test_split(classified_idx.numpy(), random_state=42, test_size=0.2, stratify=df['is_fraud'])

    # Print graph info
    print_graph_info_pyg(data)

    # Extract labels
    labels = data['transaction'].y.long()
    print(f"Labels dim: {labels.dim()}")

    # Dictionary mapping node types to their input feature dimensions
    in_size_dict = {
        'transaction': data['transaction'].x.shape[1],
        'user': data['user'].x.shape[1],
        'merchant': data['merchant'].x.shape[1]
    }

    # Define model
    etypes = data.edge_types
    NUM_EPOCHS = config["gnn"]["num_epochs"]
    batch_size = config["gnn"]["batch_size"]
    hidden_size = config["gnn"]["hidden_size"]
    n_layers = config["gnn"]["n_layers"]
    lr = config["gnn"]["learning_rate"]
    weight_decay = float(config["gnn"]["weight_decay"])
    patience = config["gnn"]["patience"]
    out_size = config["gnn"]["out_size"]
    target_node = config["gnn"]["target_node"]
    min_epochs = config["gnn"]["min_epochs"]

    model = HeteroGNN(in_size_dict, hidden_size, out_size, n_layers, etypes, target_node)

    # Train model
    f1_scores, losses = train_pyg_model_ES(model, data, train_idx, valid_idx, NUM_EPOCHS, lr, weight_decay, batch_size,
                    m_name, patience=patience, min_epochs=min_epochs)





if __name__ == "__main__":
    main()
