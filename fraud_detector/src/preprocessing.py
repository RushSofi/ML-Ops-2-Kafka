import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

def load_train_data():
    logger.info('Loading training data...')
    train = pd.read_csv('./train_data/train.csv')
    train = preprocess_data(train, is_train=True)
    logger.info('Train data processed. Shape: %s', train.shape)
    return train

def preprocess_data(df, is_train=False):
    df["log_amount"] = np.log1p(df["amount"])
    df["log_population_city"] = np.log1p(df["population_city"])
    
    threshold_amount = df["amount"].quantile(0.99) if is_train else df["amount"].quantile(0.99)
    df["high_amount"] = (df["amount"] > threshold_amount).astype(int)
    
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])
    df["hour"] = df["transaction_time"].dt.hour
    df["night_transaction"] = (df["hour"] < 12).astype(int)
    
    high_risk = ["shopping_net", "misc_net", "grocery_pos"]
    medium_risk = ["shopping_pos", "gas_transport"]
    df["cat_id_risk"] = df["cat_id"].apply(
        lambda x: 2 if x in high_risk else (1 if x in medium_risk else 0)
    )
    
    cat_cols = ["merch", "name_1", "name_2", "jobs", "us_state", "street", "one_city"]
    for col in cat_cols:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[col + "_freq"] = df[col].map(freq)
    
    cols_to_drop = ["merchant_lat", "merchant_lon", "post_code", "night_transaction"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def run_preproc(train, input_df):
    return preprocess_data(input_df, is_train=False)