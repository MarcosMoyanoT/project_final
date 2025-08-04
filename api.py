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
try:
    print(f"Cargando modelo desde gs://{BUCKET_NAME}/{MODEL_PATH}...")
    model = load_gcs_object(BUCKET_NAME, MODEL_PATH)

    print(f"Cargando preprocesadores desde gs://{BUCKET_NAME}/{PREPROCESSORS_PATH}...")
    preprocessors = load_gcs_object(BUCKET_NAME, PREPROCESSORS_PATH)

    if model and preprocessors:
        print("Modelo y preprocesadores cargados correctamente.")
        print(f"Claves del diccionario de preprocesadores: {preprocessors.keys()}")
    else:
        raise Exception("No se pudo cargar el modelo o los preprocesadores. Revisa los logs.")

except Exception as e:
    print(f"Error fatal al inicializar la aplicación: {e}")
    model = None
    preprocessors = None

# --- FUNCIONES DE PREPROCESAMIENTO (ORDEN DE OPERACIONES CORREGIDO) ---
def preprocess_for_api(df_raw: pd.DataFrame, preprocessors: dict):
    """
    Aplica las transformaciones necesarias a los datos de una nueva transacción,
    asegurando que las columnas coincidan con las del modelo.
    """
    df = df_raw.copy()

    # 1. Obtener la lista de features esperadas del modelo
    if 'features_at_training' in preprocessors:
        all_features = preprocessors['features_at_training']
    elif 'features' in preprocessors:
        all_features = preprocessors['features']
    else:
        print("Advertencia: No se encontró la lista de features. Usando las columnas del DataFrame de entrada.")
        all_features = df_raw.columns.tolist()

    # --- LÍNEA CLAVE: FILTRAR Y REINDEXAR PRIMERO ---
    df = df.reindex(columns=all_features, fill_value=0)

    # 2. Rellenar nulos en columnas categóricas ANTES de la codificación
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].fillna('missing').astype('category')

    # 3. Imputar valores nulos en columnas numéricas
    if 'imputer' in preprocessors:
        df_imputed = pd.DataFrame(preprocessors['imputer'].transform(df[preprocessors['imputer'].feature_names_in_]),
                                  columns=preprocessors['imputer'].feature_names_in_,
                                  index=df.index)
        df[preprocessors['imputer'].feature_names_in_] = df_imputed

    # 4. Aplicar Label Encoding (si se usa)
    if 'label_encoders' in preprocessors:
        for col, encoder in preprocessors['label_encoders'].items():
            if col in df.columns:
                known_labels = list(encoder.classes_)
                df[col] = df[col].apply(lambda x: x if x in known_labels else known_labels[0])
                df[col] = encoder.transform(df[col])

    # 5. Aplicar One-Hot Encoding (si se usa)
    if 'one_hot_encoder' in preprocessors:
        ohe = preprocessors['one_hot_encoder']
        df_ohe_ready = df.reindex(columns=ohe.feature_names_in_, fill_value='missing')
        encoded_cols = ohe.transform(df_ohe_ready).toarray()
        encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out())
        df = pd.concat([df.drop(ohe.feature_names_in_, axis=1), encoded_df], axis=1)

    # 6. Reindexar el DataFrame final (este paso es opcional si el paso 1 funciona bien)
    df = df.reindex(columns=all_features, fill_value=0)

    # 7. Aplicar el scaler al final
    if 'scaler' in preprocessors:
        df[preprocessors['scaler'].feature_names_in_] = preprocessors['scaler'].transform(df[preprocessors['scaler'].feature_names_in_])

    return df

# --- API ENDPOINT ---
@app.route("/", methods=["POST"])
def predict():
    if model is None or preprocessors is None:
        return jsonify({"error": "Modelo no disponible. Por favor, revisa los logs del servidor."}), 500

    try:
        json_data = request.get_json(force=True)

        df_transactions = pd.DataFrame(json_data)

        threshold_keys = [k for k in preprocessors.keys() if 'threshold' in k.lower()]
        if threshold_keys:
            threshold = preprocessors.get(threshold_keys[0])
        else:
            print("Advertencia: No se encontró un umbral en el preprocesador. Usando 0.5 por defecto.")
            threshold = 0.5

        X_processed = preprocess_for_api(df_transactions, preprocessors)

        y_probas = model.predict_proba(X_processed)[:, 1]

        results = [
            {
                "prediction": float(y_proba),
                "is_fraud": bool(y_proba >= threshold)
            }
            for y_proba in y_probas
        ]

        return jsonify(results)

    except Exception as e:
        print(f"Error en la solicitud: {e}")
        return jsonify({"error": str(e), "message": "Revisa los datos de entrada y la estructura de tu preprocesador."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
