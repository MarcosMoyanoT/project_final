import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTETomek
import joblib
import fsspec

def encode_and_scale(
    df: pd.DataFrame,
    categorical_columns: list,
    target_column: str = "isFraud",
    preprocessors: dict = None
) -> tuple:
    """
    Encode categorical columns and scale numeric columns.
    If preprocessors are provided, it uses them for transformation.
    If not, it fits new ones and returns them.
    """
    df_encoded = df.copy()

    # Determine if we need to fit or just transform
    fit_preprocessors = preprocessors is None

    if fit_preprocessors:
        preprocessors = {
            'scalers': {},
            'label_encoders': {}
        }

    # Label Encoding
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = preprocessors['label_encoders'].get(col, LabelEncoder())
            if fit_preprocessors:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                preprocessors['label_encoders'][col] = le
            else:
                df_encoded[col] = le.transform(df_encoded[col].astype(str))

    # Scale numeric columns
    numeric_columns = [
        col for col in df_encoded.columns
        if col not in categorical_columns + [target_column] and pd.api.types.is_numeric_dtype(df_encoded[col])
    ]
    scaler = preprocessors['scalers'].get('scaler', StandardScaler())

    if fit_preprocessors:
        df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])
        preprocessors['scalers']['scaler'] = scaler
    else:
        df_encoded[numeric_columns] = scaler.transform(df_encoded[numeric_columns])

    if fit_preprocessors:
        return df_encoded, preprocessors
    else:
        return df_encoded, None


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


def save_preprocessors(preprocessors: dict, path: str):
    """
    Saves preprocessor objects (scalers and label encoders) to GCS or local file.
    """
    with fsspec.open(path, "wb") as f:
        joblib.dump(preprocessors, f)


def load_preprocessors(path: str):
    """
    Loads preprocessor objects from GCS or local file.
    """
    try:
        with fsspec.open(path, "rb") as f:
            preprocessors = joblib.load(f)
            return preprocessors
    except FileNotFoundError:
        print(f"Error: Preprocessor file not found at {path}")
        return None
