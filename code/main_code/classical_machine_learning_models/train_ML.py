import sys

import yaml

sys.path.append('../../component')

from component.packages import *
from component.preprocess import *
from component.classical_machine_learning_models.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path="/home/ubuntu/code/component/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


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
    config = load_config()
    target = config['ml']['target']
    seed = config

    np.random.seed(seed)

    # For dataset 1
    data_path = config['data']['train_data_path1']
    data_path = os.path.join(os.path.expanduser("~"), data_path)
    data = pd.read_csv(data_path)
    dir_name = "models_DS1"
    data = preprocess_data_1(data, 'trans_date_trans_time', 'merchant', 'trans_num')
    data.drop(['card_number', 'fraud_merchant_pct', 'merchant_id', 'transaction_id'],
            axis=1, inplace=True)


    # For dataset 2
    # data_path = config['data']['data_path2']
    # data_path = os.path.join(os.path.expanduser("~"), data_path)
    # data = pd.read_csv(data_path)
    # dir_name = "models_DS2"
    # data, test = preprocess_data_2(data)
    # data = data[
    #     ['transaction_id', 'amount', 'weekend_transaction', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    #      'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_JPY', 'currency_SGD', 'currency_USD',
    #      'device_Android App', 'device_Chip Reader', 'device_Magnetic Stripe', 'device_NFC Payment', 'card_number',
    #      'country_Canada', 'country_France',
    #      'country_Germany', 'country_Japan', 'country_Russia', 'country_Singapore', 'country_UK',
    #      'country_USA', 'merchant_id', 'merchant_category_Education', 'merchant_category_Entertainment',
    #      'merchant_category_Gas', 'merchant_category_Grocery', 'is_fraud']]

    X = data.drop(columns=[target])
    y = data[target]

    print("Training starts")
    train_models(X, y,dir_name)


if __name__ == '__main__':


    main()
