# src/data.py
import pandas as pd
import os

def load_and_merge_data(
    identity_path: str,
    transaction_path: str,
    nrows: int = None
) -> pd.DataFrame:
    """
    Loads and merges the identity and transaction datasets.
    `nrows` is used to load a limited number of rows for testing.
    """
    # Load a limited number of rows from both files
    df_identity = pd.read_csv(identity_path, nrows=nrows)
    df_transaction = pd.read_csv(transaction_path, nrows=nrows)

    # Use a 'right' merge to ensure all transactions (with `isFraud` label) are kept.
    # Rows from df_identity that don't have a match will have NaN values.
    df_merged = pd.merge(df_identity, df_transaction, on="TransactionID", how="right")
    return df_merged



def clean_data(df: pd.DataFrame, null_threshold: float = 0.4) -> pd.DataFrame:
    """
    Elimina las columnas con un alto porcentaje de valores nulos,
    excepto la columna objetivo 'isFraud'.
    """
    cols_to_drop = [
        col for col in df.columns
        if col != 'isFraud' and df[col].isnull().sum() / len(df) > null_threshold
    ]
    print(f"Columnas con más del {null_threshold*100}% de valores nulos que se eliminarán: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True)
    # Borrar filas que tienen NaNs en columnas relevantes, no sabemos bien como reemplazarlos
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

def load_preprocess_data(
    identity_path: str,
    transaction_path: str,
    null_threshold: float = 0.4,
    nrows: int = None
) -> pd.DataFrame:
    """
    Carga, limpia y crea un user_id para los datos.
    `nrows` permite cargar solo un número limitado de filas.
    """
    df = load_and_merge_data(identity_path, transaction_path, nrows=nrows)
    df = create_user_id(df)
    df = clean_data(df, null_threshold)

    return df

def load_services(path: str) -> pd.DataFrame:
    """
    Carga los servicios financieros desde un archivo CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo {path} no existe.")

    df_services = pd.read_csv(path)
    return df_services

def calculate_service_fraud_metrics(base_path: str = "raw_data/servicios") -> pd.DataFrame:
    """
    Calcula la probabilidad de fraude y el costo potencial de cada servicio financiero.

    Args:
        base_path (str): Ruta base de la carpeta de servicios.

    Returns:
        pd.DataFrame: DataFrame con las métricas de cada servicio.
    """
    services_metrics = []

    # 1. Cargar y procesar Credit Card
    try:
        df_cc = load_services(os.path.join(base_path, 'creditcard.csv'))
        total_cc = len(df_cc)
        fraud_cc = df_cc[df_cc['Class'] == 1]
        proba_cc = len(fraud_cc) / total_cc
        costo_cc = fraud_cc['Amount'].mean()
        services_metrics.append({
            'servicio': 'creditcard',
            'probabilidad_fraude': proba_cc,
            'costo_potencial_fraude': costo_cc
        })
    except FileNotFoundError:
        print("Advertencia: No se encontró el archivo creditcard.csv. Omitiendo.")

    # 2. Cargar y procesar Loan Applications
    try:
        df_loan = load_services(os.path.join(base_path, 'loan_applications.csv'))
        total_loan = len(df_loan)
        fraud_loan = df_loan[df_loan['fraud_type'] == 1]
        proba_loan = len(fraud_loan) / total_loan
        costo_loan = fraud_loan['loan_amount_requested'].mean()
        services_metrics.append({
            'servicio': 'loan_application',
            'probabilidad_fraude': proba_loan,
            'costo_potencial_fraude': costo_loan
        })
    except FileNotFoundError:
        print("Advertencia: No se encontró el archivo loan_applications.csv. Omitiendo.")

    # 3. Cargar y procesar Transactions
    try:
        df_transactions = load_services(os.path.join(base_path, 'transactions.csv'))
        total_trans = len(df_transactions)
        fraud_trans = df_transactions[df_transactions['fraud_flag'] == 1]
        proba_trans = len(fraud_trans) / total_trans
        costo_trans = fraud_trans['transaction_amount'].mean()
        services_metrics.append({
            'servicio': 'transactions',
            'probabilidad_fraude': proba_trans,
            'costo_potencial_fraude': costo_trans
        })
    except FileNotFoundError:
        print("Advertencia: No se encontró el archivo transactions.csv. Omitiendo.")

    df_metrics = pd.DataFrame(services_metrics)
    return df_metrics

import pandas as pd
import os


def load_services_from_gcs(path: str) -> pd.DataFrame:
    """
    Carga los servicios financieros desde un archivo CSV en Google Cloud Storage.
    """
    # Pandas puede leer directamente de GCS si tienes las credenciales configuradas
    try:
        df_services = pd.read_csv(path)
        return df_services
    except Exception as e:
        print(f"Error al cargar el archivo desde GCS: {e}")
        return None

def calculate_service_fraud_metrics_gcs(base_path: str) -> pd.DataFrame:
    """
    Calcula la probabilidad de fraude y el costo potencial de cada servicio financiero
    a partir de archivos en Google Cloud Storage.
    """
    services_metrics = []

    # Cargar y procesar Credit Card
    path_cc = os.path.join(base_path, 'creditcard.csv')
    df_cc = load_services_from_gcs(path_cc)
    if df_cc is not None:
        total_cc = len(df_cc)
        fraud_cc = df_cc[df_cc['Class'] == 1]
        proba_cc = len(fraud_cc) / total_cc
        costo_cc = fraud_cc['Amount'].mean()
        services_metrics.append({
            'servicio': 'creditcard',
            'probabilidad_fraude': proba_cc,
            'costo_potencial_fraude': costo_cc
        })
    else:
        print(f"Advertencia: No se pudo cargar el archivo {path_cc}. Omitiendo.")

    # Cargar y procesar Loan Applications
    path_loan = os.path.join(base_path, 'loan_applications.csv')
    df_loan = load_services_from_gcs(path_loan)
    if df_loan is not None:
        total_loan = len(df_loan)
        fraud_loan = df_loan[df_loan['fraud_type'] == 1]
        proba_loan = len(fraud_loan) / total_loan
        costo_loan = fraud_loan['loan_amount_requested'].mean()
        services_metrics.append({
            'servicio': 'loan_application',
            'probabilidad_fraude': proba_loan,
            'costo_potencial_fraude': costo_loan
        })
    else:
        print(f"Advertencia: No se pudo cargar el archivo {path_loan}. Omitiendo.")

    # Cargar y procesar Transactions
    path_trans = os.path.join(base_path, 'transactions.csv')
    df_transactions = load_services_from_gcs(path_trans)
    if df_transactions is not None:
        total_trans = len(df_transactions)
        fraud_trans = df_transactions[df_transactions['fraud_flag'] == 1]
        proba_trans = len(fraud_trans) / total_trans
        costo_trans = fraud_trans['transaction_amount'].mean()
        services_metrics.append({
            'servicio': 'transactions',
            'probabilidad_fraude': proba_trans,
            'costo_potencial_fraude': costo_trans
        })
    else:
        print(f"Advertencia: No se pudo cargar el archivo {path_trans}. Omitiendo.")

    df_metrics = pd.DataFrame(services_metrics)
    return df_metrics
