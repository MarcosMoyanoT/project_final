import pandas as pd
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify
import joblib
import os
import io
from google.cloud import storage

app = Flask(__name__)

# --- CONFIGURACIÓN ---
BUCKET_NAME = os.environ.get("BUCKET_NAME", "fraud-detection-lewagon")
MODEL_PATH = "models/xgb_model.joblib"
PREPROCESSORS_PATH = "models/preprocessors.joblib"

# --- FUNCIONES DE CARGA ---
def load_gcs_object(bucket_name, file_path):
    """Carga un objeto de GCS usando su ruta."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        file_bytes = blob.download_as_bytes()
        return joblib.load(io.BytesIO(file_bytes))
    except Exception as e:
        print(f"Error al cargar el archivo desde GCS: {file_path}. Error: {e}")
        return None

# Cargar el modelo y preprocesadores al iniciar la aplicación
# Se cargan una sola vez cuando la instancia de Cloud Run arranca
model = None
preprocessors = None
try:
    print(f"DEBUG: Intentando cargar modelo desde gs://{BUCKET_NAME}/{MODEL_PATH}...")
    model = load_gcs_object(BUCKET_NAME, MODEL_PATH)

    print(f"DEBUG: Intentando cargar preprocesadores desde gs://{BUCKET_NAME}/{PREPROCESSORS_PATH}...")
    preprocessors = load_gcs_object(BUCKET_NAME, PREPROCESSORS_PATH)

    if model and preprocessors:
        print("DEBUG: Modelo y preprocesadores cargados correctamente.")
        print(f"DEBUG: Claves del diccionario de preprocesadores: {preprocessors.keys()}")
    else:
        raise Exception("No se pudo cargar el modelo o los preprocesadores. Revisa los logs.")

except Exception as e:
    print(f"DEBUG: Error fatal al inicializar la aplicación: {e}")
    model = None
    preprocessors = None

# --- FUNCIONES DE PREPROCESAMIENTO ---
def preprocess_for_api(df_raw: pd.DataFrame, preprocessors: dict):
    """
    Aplica las transformaciones necesarias a los datos de una nueva transacción,
    asegurando que las columnas coincidan con las del modelo.
    """
    df = df_raw.copy()

    # 1. Obtener la lista de features esperadas del modelo
    if 'features_at_training' in preprocessors:
        all_features = preprocessors['features_at_training']
    elif 'features' in preprocessors: # Compatibilidad con nombres antiguos si aplica
        all_features = preprocessors['features']
    else:
        print("Advertencia: No se encontró la lista de features. Usando las columnas del DataFrame de entrada.")
        all_features = df_raw.columns.tolist()

    # --- LÍNEA CLAVE: FILTRAR Y REINDEXAR PRIMERO ---
    # Asegura que df tenga las mismas columnas y orden que all_features
    df = df.reindex(columns=all_features, fill_value=0)

    # 2. Rellenar nulos en columnas categóricas ANTES de la codificación
    # Identifica columnas categóricas que estaban en el entrenamiento
    if 'categorical_cols_for_imputation' in preprocessors: # Asumo que guardaste esta lista
        cols_to_fill = [col for col in preprocessors['categorical_cols_for_imputation'] if col in df.columns]
        if cols_to_fill:
            df[cols_to_fill] = df[cols_to_fill].fillna('missing').astype('category')
    else: # Fallback si no está la lista específica
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            df[categorical_cols] = df[categorical_cols].fillna('missing').astype('category')


    # 3. Imputar valores nulos en columnas numéricas
    if 'imputer' in preprocessors and preprocessors['imputer'] is not None:
        # Asegúrate de que las columnas para el imputer existan en df
        imputer_features = preprocessors['imputer'].feature_names_in_
        df_imputed_subset = df[imputer_features]

        df_imputed_transformed = pd.DataFrame(
            preprocessors['imputer'].transform(df_imputed_subset),
            columns=imputer_features,
            index=df.index
        )
        df[imputer_features] = df_imputed_transformed

    # 4. Aplicar Label Encoding (si se usa)
    if 'label_encoders' in preprocessors and preprocessors['label_encoders'] is not None:
        for col, encoder in preprocessors['label_encoders'].items():
            if col in df.columns:
                # Manejo de etiquetas desconocidas: asigna a la primera clase conocida o a una etiqueta de 'unknown'
                known_labels = list(encoder.classes_)
                df[col] = df[col].apply(lambda x: x if x in known_labels else known_labels[0]) # O 'unknown_label' si lo manejas así
                df[col] = encoder.transform(df[col])

    # 5. Aplicar One-Hot Encoding (si se usa)
    if 'one_hot_encoder' in preprocessors and preprocessors['one_hot_encoder'] is not None:
        ohe = preprocessors['one_hot_encoder']
        # Asegúrate de que las columnas para OHE existan y estén en el orden esperado
        ohe_features_in = ohe.feature_names_in_
        df_ohe_ready = df[ohe_features_in] # Filtrar solo las columnas que el OHE espera

        # Manejo de categorías desconocidas en OHE (handle_unknown='ignore' es común)
        encoded_cols = ohe.transform(df_ohe_ready).toarray()
        encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(ohe_features_in), index=df.index)

        # Quita las columnas originales y concatena las codificadas
        df = pd.concat([df.drop(columns=ohe_features_in, errors='ignore'), encoded_df], axis=1)

    # 6. Aplicar el scaler al final
    if 'scaler' in preprocessors and preprocessors['scaler'] is not None:
        scaler_features = preprocessors['scaler'].feature_names_in_
        # Asegúrate de que las columnas para el scaler existan en df
        df_scaler_subset = df[scaler_features]
        df[scaler_features] = preprocessors['scaler'].transform(df_scaler_subset)

    # Último reindexado para asegurar el orden final de las columnas
    df = df.reindex(columns=all_features, fill_value=0)

    return df


# --- API ENDPOINT ---
@app.route("/", methods=["POST"])
def predict_endpoint(): # Renombrado para evitar conflicto con la función 'predict' de arriba
    print("DEBUG: Solicitud recibida en /.")
    if model is None or preprocessors is None:
        print("DEBUG: Modelo o preprocesadores no disponibles en la solicitud.")
        return jsonify({"error": "Modelo no disponible. Por favor, revisa los logs del servidor."}), 500

    try:
        json_data = request.get_json(force=True)
        print(f"DEBUG: Datos JSON recibidos. Longitud: {len(json_data)}.")

        df_transactions = pd.DataFrame(json_data)
        print("DEBUG: DataFrame creado a partir de JSON.")

        # --- CRÍTICO: Eliminar 'TransactionID' antes del preprocesamiento ---
        if 'TransactionID' in df_transactions.columns:
            df_transactions = df_transactions.drop(columns=['TransactionID'])
            print("DEBUG: 'TransactionID' eliminado del DataFrame de entrada.")

        # --- Buscar el umbral con la clave correcta ---
        threshold = preprocessors.get('prediction_threshold') # <-- Usar la clave guardada
        if threshold is None:
            print("Advertencia: No se encontró 'prediction_threshold' en el preprocesador. Usando 0.5 por defecto.")
            threshold = 0.5
        else:
            print(f"DEBUG: Umbral encontrado: {threshold}.")

        X_processed = preprocess_for_api(df_transactions, preprocessors)
        print("DEBUG: Datos preprocesados correctamente.")

        y_probas = model.predict_proba(X_processed)[:, 1]
        print("DEBUG: Predicciones de probabilidad generadas.")

        results = [
            {
                "prediction": float(y_proba),
                "is_fraud": bool(y_proba >= threshold)
            }
            for y_proba in y_probas
        ]
        print("DEBUG: Resultados formateados.")

        return jsonify(results)

    except Exception as e:
        print(f"Error en la solicitud: {e}")
        return jsonify({"error": str(e), "message": f"Error en la API: {str(e)}. Revisa los datos de entrada y la estructura de tu preprocesador."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
