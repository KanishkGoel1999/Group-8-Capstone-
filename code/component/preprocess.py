from component.packages import *

warnings.filterwarnings("ignore")

def preprocess_data_1(dataframe, timestamp, merchant="", transaction_id=""):
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



    ohe = OneHotEncoder(sparse_output = False, drop = 'first')
    ohe_array = ohe.fit_transform(df[["category"]])
    ohe_column = ohe.get_feature_names_out()
    ohe_df = pd.DataFrame(ohe_array, columns = ohe_column, index=df.index)
    df = pd.concat([df, ohe_df], axis = 1)

    # 'category_home', 'category_misc_net', 'category_shopping_net'
    df.drop(['category_grocery_pos', 'category_misc_pos', 'category_shopping_pos', 'category_kids_pets', 'category_personal_care', 'category_health_fitness'], axis = 1, inplace = True)

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


    cols_to_drop = ['first', 'last', 'street', 'category', 'day_period', 'hour', 'day']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    if 'cc_num' in df.columns:
        df['card_number'] = df['cc_num']
        df.drop(columns=['cc_num'], inplace=True)

    return df


def preprocess_data_2(df):
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


def preprocess_data_raw(dataframe, timestamp):

    df = dataframe
    df[timestamp] = pd.to_datetime(df[timestamp])
    df["month"] = df[timestamp].dt.month

    # Encode hour cyclically
    df["Month_Sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = df['dob'].apply(calculate_age)

    merchant_groups = df.groupby('merchant')
    fraud_percentage_dict = {}

    for merchant, group in merchant_groups:
        total_trans = group.shape[0]
        total_fraud_trans = group[group["is_fraud"] == 1].shape[0]
        fraud_percentage_dict[merchant] = (total_fraud_trans / total_trans) * 100

    # Step 2: Map the calculated percentages back to the DataFrame
    df["fraud_merchant_pct"] = df["merchant"].map(fraud_percentage_dict)
    df["is_weekend"] = df["trans_date_trans_time"].dt.day_name().apply(
        lambda x: int((x == "Friday") | (x == "Sunday") | (x == "saturday")))
    df["gender"] = df["gender"].apply(lambda x: int(x == "M"))
    df.drop(['Unnamed: 0', 'unix_time', 'job', 'dob', 'city', 'state', 'zip'], axis=1, inplace=True)

    return df

def calculate_age(dob):
    today = datetime.today()
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age