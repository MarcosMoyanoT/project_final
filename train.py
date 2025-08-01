# train.py

from src.data import load_preprocess_data
from src.preprocessing import encode_and_scale, split_data, balance_data
from src.model import train_xgb_model, predict, evaluate_model, save_model, assign_groups_and_services_from_proba
from imblearn.combine import SMOTETomek


def main():
    # 1. Cargar datos
    df = load_preprocess_data(
        identity_path="gs://fraud-detection-lewagon/train_identity.csv",
        transaction_path="gs://fraud-detection-lewagon/train_transaction.csv",
        null_threshold=0.4
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
    X, y = encode_and_scale(df, categorical_columns=categorical_columns, target_column='isFraud')
    print(f"Preprocesamiento completo. X shape: {X.shape}, y shape: {y.shape}")

    # 3. División en train/val
    X_train, X_val, y_train, y_val = split_data(df=df, target_column='isFraud')

    # 4. Balanceo de clases
    X_train_resampled, y_train_resampled = balance_data(X_train, y_train)
    print(f"Balanceo completo. X_train shape: {X_train_resampled.shape}")

    # 5. Entrenar modelo y buscar mejor threshold
    model, threshold = train_xgb_model(X_train_resampled, y_train_resampled, X_val, y_val)
    print(f"Modelo entrenado. Threshold óptimo: {threshold:.2f}")

    # 6. Predecir con threshold ajustado
    y_pred = predict(model, X_val, threshold)

    # 7. Evaluar modelo
    evaluate_model(y_val, y_pred, model.predict_proba(X_val)[:, 1])

    # 8. Asignar grupos de fraude y paquetes financieros
    y_proba = model.predict_proba(X_val)[:, 1]
    user_ids = X_val.index  # Si tu index es el user_id
    grupos_df = assign_groups_and_services_from_proba(y_proba, user_ids=user_ids)

    print("\n📊 Distribución de paquetes financieros asignados:")

    # 8. Guardar modelo
    # Guardar en GCS
    save_model(model, "gs://fraud-detection-lewagon/models/xgb_model.joblib")


if __name__ == "__main__":
    main()
