# train.py
import pandas as pd
from src.data import load_preprocess_data, calculate_service_fraud_metrics_gcs
from src.preprocessing import (
    encode_and_scale,
    split_data,
    balance_data,
    save_preprocessors
)
from src.model import train_xgb_model, predict, evaluate_model, save_model, assign_groups_and_services_from_proba, find_optimal_threshold
from imblearn.combine import SMOTETomek


def main():
    # -------------------------------------------------------------------------
    # COMPONENTE DE NEGOCIO (ANÁLISIS DE RIESGO DE SERVICIOS)
    # -------------------------------------------------------------------------
    gcs_services_path = "gs://fraud-detection-lewagon"
    df_service_metrics = calculate_service_fraud_metrics_gcs(base_path=gcs_services_path)
    print("---------------------------------------------------------------")
    print("📊 ANÁLISIS DE RIESGO DE SERVICIOS FINANCIEROS:")
    print("---------------------------------------------------------------")
    print(df_service_metrics)

    # -------------------------------------------------------------------------
    # COMPONENTE ML (PREDICCIÓN DE FRAUDE DE TRANSACCIÓN)
    # -------------------------------------------------------------------------
    print("\n---------------------------------------------------------------")
    print("🧠 ENTRENAMIENTO DEL MODELO ML PARA DETECCIÓN DE FRAUDE")
    print("---------------------------------------------------------------")

    # 1. Cargar datos de entrenamiento desde GCS
    # Añadimos `nrows=50000` para cargar solo 50,000 filas.
    df = load_preprocess_data(
        identity_path="gs://fraud-detection-lewagon/train_identity.csv",
        transaction_path="gs://fraud-detection-lewagon/train_transaction.csv",
        null_threshold=0.4,
        nrows=30000
    )
    print(f"Datos cargados y preprocesados. Shape: {df.shape}")

    categorical_columns = [
    'DeviceType', 'DeviceInfo', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
    'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20',
    'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
    ]

    # 2. Preprocesamiento (encoding y escalado)
    df_encoded, preprocessors = encode_and_scale(df, categorical_columns=categorical_columns, target_column='isFraud')
    print(f"Preprocesamiento completo. X shape: {df_encoded.shape}")

    # 3. División en train/val
    X_train, X_val, y_train, y_val = split_data(df=df_encoded, target_column='isFraud')

    # 4. Balanceo de clases
    X_train_resampled, y_train_resampled = balance_data(X_train, y_train)
    print(f"Balanceo completo. X_train shape: {X_train_resampled.shape}")

    # 5. Entrenar modelo y buscar mejor threshold
    model, threshold = train_xgb_model(X_train_resampled, y_train_resampled, X_val, y_val)
    print(f"Threshold óptimo: {threshold:.2f}")

    # 6. Predecir con threshold ajustado
    y_pred = predict(model, X_val, threshold)

    # 7. Evaluar modelo
    y_proba = model.predict_proba(X_val)[:, 1]
    evaluate_model(y_val, y_pred, y_proba)

    # 8. Asignar grupos de fraude y paquetes financieros (con umbrales fijos)
    user_ids = X_val.index
    grupos_df = assign_groups_and_services_from_proba(y_proba=y_proba, user_ids=user_ids)
    print("\n📊 Distribución de paquetes financieros asignados:")
    print(grupos_df["paquete_servicio"].value_counts())

    # 9. Guardar modelo y preprocesadores
    save_model(model, "gs://fraud-detection-lewagon/models/xgb_model.joblib")
    save_preprocessors(preprocessors, "gs://fraud-detection-lewagon/models/preprocessors.joblib")


if __name__ == "__main__":
    main()
