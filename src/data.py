# src/data.py
import pandas as pd
import os

def load_and_merge_data(identity_path: str, transaction_path: str) -> pd.DataFrame:
    df_identity = pd.read_csv(identity_path)
    df_transaction = pd.read_csv(transaction_path)
    df_merged = pd.merge(df_identity, df_transaction, on="TransactionID", how="left")
    return df_merged

def clean_data(df: pd.DataFrame, null_threshold: float = 0.4) -> pd.DataFrame:
    df = df.copy()
    null_ratio = df.isnull().mean()
    cols_to_drop = null_ratio[null_ratio > null_threshold].index
    df.drop(columns=cols_to_drop, inplace=True)
    df.dropna(inplace=True)
    return df

def create_user_id(df: pd.DataFrame) -> pd.DataFrame:
    df["user_id"] = (
        df["card1"].astype(str) + "_" +
        df["card2"].astype(str) + "_" +
        df["card3"].astype(str) + "_" +
        df["card5"].astype(str) + "_" +
        df["card4"].astype(str) + "_" +
        df["card6"].astype(str) + "_" +
        #df["addr1"].astype(str) + "_" +
        df["dist1"].astype(str) + "_" +
        df["P_emaildomain"].astype(str) + "_" +
        df["R_emaildomain"].astype(str) + "_" +
        df["id_02"].astype(str) + "_" +
        df["id_05"].astype(str) + "_" +
        df["id_06"].astype(str) + "_" +
        df["id_15"].astype(str) + "_" +
        df["id_30"].astype(str) + "_" +
        df["id_31"].astype(str) + "_" +
        df["DeviceInfo"].astype(str)
    )
    df.set_index('user_id', inplace=True)
    return df

def load_preprocess_data(identity_path: str, transaction_path: str, null_threshold: float = 0.4) -> pd.DataFrame:
    df = load_and_merge_data(identity_path, transaction_path)
    df = clean_data(df, null_threshold)
    df = create_user_id(df)
    return df
