import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import warnings

warnings.filterwarnings("ignore")

def preprocess_data2(dataframe, timestamp, merchant="", transaction_id=""):
    df = dataframe

    df[timestamp] = pd.to_datetime(df[timestamp], format='mixed')
    df["hour"] = df[timestamp].dt.hour
    df["day"] = df[timestamp].dt.day

    # Encode hour cyclically
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Encode day of the month cyclically
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df.drop(columns=[timestamp], inplace=True)


    if transaction_id!="":
        # Create a unique ID mapping for transactions
        trans_id_map = {trans: i for i, trans in enumerate(df[transaction_id].unique())}

        # Apply mapping
        df["transactionn_id"] = df[transaction_id].map(trans_id_map)
        df.drop(columns=[transaction_id], inplace=True)
        df.rename(columns={"transactionn_id": "transaction_id"}, inplace=True)



    if merchant!="":
        # Create a unique ID mapping for transactions
        merch_id_map = {trans: i for i, trans in enumerate(df[merchant].unique())}

        # Apply mapping
        df["merchant_id"] = df[merchant].map(merch_id_map)
        df.drop(merchant, axis=1, inplace=True)


    cols_to_drop = ['first', 'last', 'street', 'category', 'day_period', 'hour', 'day', 'distance', 'trans_num']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    if 'cc_num' in df.columns:
        df['card_number'] = df['cc_num']
        df.drop(columns=['cc_num'], inplace=True)

    print(df.columns)

    return df

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

def preprocess_data_cl(dataframe, timestamp, merchant="", transaction_id=""):
    df = dataframe

    df[timestamp] = pd.to_datetime(df[timestamp], format='mixed')
    df["hour"] = df[timestamp].dt.hour
    df["day"] = df[timestamp].dt.day

    # Encode hour cyclically
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Encode day of the month cyclically
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df.drop(columns=[timestamp], inplace=True)


    if transaction_id!="":
        # Create a unique ID mapping for transactions
        trans_id_map = {trans: i for i, trans in enumerate(df[transaction_id].unique())}

        # Apply mapping
        df["transactionn_id"] = df[transaction_id].map(trans_id_map)
        df.drop(transaction_id, axis=1, inplace=True)
        df.rename(columns={'transactionn_id': "transaction_id"})

    if merchant!="":
        # Create a unique ID mapping for transactions
        merch_id_map = {trans: i for i, trans in enumerate(df[merchant].unique())}

        # Apply mapping
        df["merchant_id"] = df[merchant].map(merch_id_map)
        df.drop(merchant, axis=1, inplace=True)

    # One hot encoding & Ordinal Encoding
    if 'day_period' in df.columns:
        oe_day_period = OrdinalEncoder(categories=[["Night", "Evening", "Afternoon", "Morning"]])
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        df['day_period'] = oe_day_period.fit_transform(df[["day_period"]])
        df['category'] = ohe.fit_transform(df[["category"]])




    cols_to_drop = ['category', 'first', 'last', 'street', 'day_period', 'hour', 'day', 'distance', 'trans_num']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    if 'cc_num' in df.columns:
        df['card_number'] = df['cc_num']
        df.drop(columns=['cc_num'], inplace=True)

    print(df.info())
    return df


def preprocess_data_ML(dataframe, timestamp, merchant="", transaction_id=""):
    df = dataframe

    df[timestamp] = pd.to_datetime(df[timestamp], format='mixed')
    df["hour"] = df[timestamp].dt.hour
    df["day"] = df[timestamp].dt.day

    # Encode hour cyclically
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Encode day of the month cyclically
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df.drop(columns=[timestamp], inplace=True)

    # oe_day_period = OrdinalEncoder(categories = [["Night", "Evening", "Afternoon", "Morning"]])
    # df['day_period'] = oe_day_period.fit_transform(df[["day_period"]])

    ohe = OneHotEncoder(sparse_output = False, drop = 'first')
    ohe_array = ohe.fit_transform(df[["category"]])
    ohe_column = ohe.get_feature_names_out()
    print(ohe_column)
    ohe_df = pd.DataFrame(ohe_array, columns = ohe_column, index=df.index)
    df = pd.concat([df, ohe_df], axis = 1)

    df.drop(['category_grocery_pos', 'category_home', 'category_misc_net', 'category_misc_pos', 'category_shopping_pos'], axis = 1, inplace = True)
    df.drop(transaction_id, axis=1, inplace=True)
    df.drop(merchant, axis=1, inplace=True)

    # city_pop, distance
    df.drop(['category', 'first', 'last', 'street', 'day_period', 'hour', 'day', 'cc_num', 'distance', 'city_pop'], axis=1, inplace=True)

    return df


def preprocessDS2(df):
    trans_id_map = {trans: i for i, trans in enumerate(df['transaction_id'].unique())}

    # Apply mapping
    df["transactionn_id"] = df['transaction_id'].map(trans_id_map)
    df.drop(columns=['transaction_id'], inplace=True)
    df.rename(columns={"transactionn_id": "transaction_id"}, inplace=True)

    # Create a unique ID mapping for transactions
    merch_id_map = {trans: i for i, trans in enumerate(df['merchant'].unique())}
    # Apply mapping
    df["merchant_id"] = df['merchant'].map(merch_id_map)


    df.drop(columns=['customer_id', 'merchant', 'merchant_type', 'card_type', 'city', 'city_size',
                     'velocity_last_hour', 'device_fingerprint', 'ip_address', 'high_risk_merchant', 'transaction_hour',
                     'distance_from_home', ], inplace=True)


    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df["hour"] = df['timestamp'].dt.hour
    df["day"] = df['timestamp'].dt.day

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Encode day of the month cyclically
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    df_encoded = pd.get_dummies(df, columns=['currency'], dtype=int)
    df_encoded = pd.get_dummies(df_encoded, columns=['device'], dtype=int)
    df_encoded = pd.get_dummies(df_encoded, columns=['merchant_category'], dtype=int)
    df_encoded.drop(['hour', 'day', 'channel'], axis=1, inplace=True)
    df_encoded = pd.get_dummies(df_encoded, columns=['country'], dtype=int)

    df_encoded.drop(['card_present'], axis=1, inplace=True)
    df_encoded['weekend_transaction'] = df_encoded['weekend_transaction'].astype(int)
    df_encoded['is_fraud'] = df_encoded['is_fraud'].astype(int)

    # split Front
    train = df_encoded[(df_encoded['timestamp'] >= '2024-09-01') & (df_encoded['timestamp'] <= '2024-10-23')]
    test = df_encoded[(df_encoded['timestamp'] > '2024-10-23') & (df_encoded['timestamp'] <= '2024-10-30')]

    train.drop(['timestamp'], axis=1, inplace=True)
    test.drop(['timestamp'], axis=1, inplace=True)


    return train, test

def preprocessDS2_rough(df):
    trans_id_map = {trans: i for i, trans in enumerate(df['transaction_id'].unique())}

    # Apply mapping
    df["transactionn_id"] = df['transaction_id'].map(trans_id_map)
    df.drop(columns=['transaction_id'], inplace=True)
    df.rename(columns={"transactionn_id": "transaction_id"}, inplace=True)

    # Create a unique ID mapping for transactions
    merch_id_map = {trans: i for i, trans in enumerate(df['merchant'].unique())}
    # Apply mapping
    df["merchant_id"] = df['merchant'].map(merch_id_map)


    df.drop(columns=['customer_id', 'merchant', 'merchant_type', 'card_type', 'city', 'city_size',
                     'velocity_last_hour', 'device_fingerprint', 'ip_address', 'high_risk_merchant', 'transaction_hour',
                     'distance_from_home', ], inplace=True)


    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df["hour"] = df['timestamp'].dt.hour
    df["day"] = df['timestamp'].dt.day

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Encode day of the month cyclically
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    df_encoded = pd.get_dummies(df, columns=['currency'], dtype=int)
    df_encoded = pd.get_dummies(df_encoded, columns=['device'], dtype=int)
    #df_encoded = pd.get_dummies(df, columns=['merchant_category'], dtype=int)
    df_encoded.drop(['hour', 'day', 'channel'], axis=1, inplace=True)
    df_encoded = pd.get_dummies(df_encoded, columns=['country'], dtype=int)

    df_encoded.drop(['card_present', 'currency', 'device'], axis=1, inplace=True)
    df_encoded['weekend_transaction'] = df_encoded['weekend_transaction'].astype(int)
    df_encoded['is_fraud'] = df_encoded['is_fraud'].astype(int)

    # split Front
    train = df_encoded[(df_encoded['timestamp'] >= '2024-09-01') & (df_encoded['timestamp'] <= '2024-10-23')]
    test = df_encoded[(df_encoded['timestamp'] > '2024-10-23') & (df_encoded['timestamp'] <= '2024-10-30')]

    train.drop(['timestamp'], axis=1, inplace=True)
    test.drop(['timestamp'], axis=1, inplace=True)


    return train, test


def preprocess_data_cat(df, timestamp, merchant="", transaction_id=""):
    df[timestamp] = pd.to_datetime(df[timestamp], format='mixed')
    df["hour"] = df[timestamp].dt.hour
    df["day"] = df[timestamp].dt.day

    # Encode hour cyclically
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Encode day of the month cyclically
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    if transaction_id != "":
        # Create a unique ID mapping for transactions
        trans_id_map = {trans: i for i, trans in enumerate(df[transaction_id].unique())}

        # Apply mapping
        df["transactionn_id"] = df[transaction_id].map(trans_id_map)
        df.drop(columns=[transaction_id], inplace=True)
        df.rename(columns={"transactionn_id": "transaction_id"}, inplace=True)

    if merchant != "":
        # Create a unique ID mapping for transactions
        merch_id_map = {trans: i for i, trans in enumerate(df[merchant].unique())}

        # Apply mapping
        df["merchant_id"] = df[merchant].map(merch_id_map)

    df.rename(columns={"cc_num": "card_number"}, inplace=True)

    # Target Encode gender
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])

    # Target Encode city, state, zip
    for col in ['city', 'state', 'zip']:
        target_mean = df.groupby(col)['is_fraud'].mean()
        df[f'{col}_encoded'] = df[col].map(target_mean)

    # Compute age
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    age_target_mean = df.groupby('age')['is_fraud'].mean()
    df['age_encoded'] = df['age'].map(age_target_mean)

    # Step 1: Calculate fraud percentages for each merchant
    merchant_groups = df.groupby(merchant)
    fraud_percentage_dict = {}

    for me, group in merchant_groups:
        total_trans = group.shape[0]
        total_fraud_trans = group[group["is_fraud"] == 1].shape[0]
        fraud_percentage_dict[me] = (total_fraud_trans / total_trans) * 100

    # Step 2: Map the calculated percentages back to the DataFrame
    df["fraud_merchant_pct"] = df["merchant"].map(fraud_percentage_dict)

    # def day_period(x):
    #     if x >=0 and x < 6: return "Night"
    #     elif x>= 6 and x <= 12: return "Morning"
    #     elif x> 12 and x <= 15: return "Afternoon"
    #     elif x> 15 and x <= 20: return "Evening"
    #     elif x> 20 and x <= 24: return "Night"

    # df["day_period"] = df["trans_date_trans_time"].dt.hour.apply(day_period)

    df.drop(merchant, axis=1, inplace=True)
    df.drop(columns=[timestamp], inplace=True)

    # city_pop, distance
    df.drop(['Unnamed: 0', 'unix_time', 'job', 'age', 'category', 'first', 'last', 'street', 'hour', 'day', 'city_pop',
             'gender', 'city', 'state', 'zip', 'dob'], axis=1, inplace=True)

    return df