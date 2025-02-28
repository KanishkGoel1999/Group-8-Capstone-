import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def preprocess_data(dataframe):
    df = dataframe
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day

    # Encode hour cyclically
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Encode day of the month cyclically
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    # Create a unique ID mapping for transactions
    trans_id_map = {trans: i for i, trans in enumerate(df["trans_num"].unique())}

    # Apply mapping
    df["trans_id"] = df["trans_num"].map(trans_id_map)

    # Create a unique ID mapping for transactions
    merch_id_map = {trans: i for i, trans in enumerate(df["merchant"].unique())}

    # Apply mapping
    df["merch_id"] = df["merchant"].map(merch_id_map)

    df.drop(
        columns=['trans_date_trans_time', 'day_period', 'category', 'first', 'last', 'street', 'merchant', 'trans_num'],
        inplace=True)

    return df
