# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import fsspec
import os

from src.preprocessing import load_preprocessors
from src.model import load_model, assign_groups_and_services_from_proba, predict

# Inicializar la API
app = FastAPI()

# --- Carga de recursos ---
# Se cargan los recursos una sola vez al inicio del servicio
# Si los archivos están en GCS, FastAPI los cargará al iniciar.
MODEL_PATH = os.environ.get("MODEL_PATH", "gs://fraud-detection-lewagon/models/xgb_model.joblib")
PREPROCESSORS_PATH = os.environ.get("PREPROCESSORS_PATH", "gs://fraud-detection-lewagon/models/preprocessors.joblib")

# Cargar el modelo y los preprocesadores
try:
    model = load_model(MODEL_PATH)
    preprocessors = load_preprocessors(PREPROCESSORS_PATH)
    if model is None or preprocessors is None:
        raise FileNotFoundError("Error: Modelo o preprocesadores no cargados.")
except FileNotFoundError as e:
    print(e)
    model = None
    preprocessors = None

# Definir la estructura de la solicitud
# Pydantic nos ayuda a validar el tipo de datos que recibimos.
# Aquí debes listar todas las columnas que espera tu modelo.
# Por ejemplo, si una columna se llama 'TransactionDT', la incluyes aquí.
class TransactionFeatures(BaseModel):
    # Aquí irían todas las 300 columnas de tu modelo.
    # Por ejemplo:
    TransactionDT: int
    TransactionAmt: float
    ProductCD: str
    card1: int
    # ... y las demás 296 columnas

# --- Endpoints de la API ---
@app.get("/")
def read_root():
    return {"message": "API de Detección de Fraude y Asignación de Servicios. ¡Todo listo!"}

@app.post("/predict")
def predict_fraud_and_service(transaction: TransactionFeatures):
    if model is None or preprocessors is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. El servicio no está listo.")

    # Convertir los datos de la solicitud a un DataFrame de pandas
    df_transaccion = pd.DataFrame([transaction.dict()])

    # Preprocesar la nueva transacción usando los preprocesadores guardados
    # Nota: Aquí solo usamos `transform`, no `fit_transform`.
    # También hay que considerar la creación de la columna `user_id`
    # si es necesaria para la asignación de grupos.
    df_preprocesado, _ = encode_and_scale(
        df=df_transaccion,
        categorical_columns=preprocessors['label_encoders'].keys(), # Usar las mismas columnas categoricas
        preprocessors=preprocessors
    )

    # El `data.py` crea un `user_id` a partir de varias columnas.
    # Necesitas replicar esta lógica aquí para poder asignarle el grupo.
    # `user_id` es el índice de tu `X_val`, por lo que es necesario.
    # Esta es una parte crítica a implementar en el preprocesamiento de la API.
    # Por ahora, usaremos un índice genérico.
    user_ids = pd.Index([df_transaccion.index[0]]) # o cualquier lógica que tengas para user_id

    # Realizar la predicción
    y_pred, y_proba = predict(model, df_preprocesado, threshold=0.5)

    # Asignar grupo y servicio
    grupos_df = assign_groups_and_services_from_proba(y_proba=y_proba, user_ids=user_ids)

    # Devolver la predicción
    response = {
        "prediction": int(y_pred[0]),
        "probability": float(y_proba[0]),
        "risk_group": grupos_df.iloc[0]["grupo_fraude"],
        "assigned_service": grupos_df.iloc[0]["paquete_servicio"]
    }

    return response
