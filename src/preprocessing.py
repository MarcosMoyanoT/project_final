import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTETomek


def encode_and_scale(df: pd.DataFrame, categorical_columns: list, target_column: str = "isFraud"):
    """
    Encode categorical columns and scale numeric columns.
    """
    df_encoded = df.copy()

    # Label Encoding
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # Scale numeric columns
    numeric_columns = [col for col in df_encoded.columns if col not in categorical_columns + [target_column]]
    scaler = StandardScaler()
    df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])

    return df_encoded


def split_data(df: pd.DataFrame, target_column: str = "isFraud"):
    """
    Split dataset into train and validation sets.
    """
    X = df.drop(columns=[target_column, "TransactionID"] if "TransactionID" in df.columns else [target_column])
    y = df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_val, y_train, y_val


def balance_data(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Balance the training data using SMOTETomek.
    """
    smt = SMOTETomek(random_state=42)
    X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
    return X_train_sm, y_train_sm
