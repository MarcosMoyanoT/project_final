from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import joblib
import numpy as np
import os
import fsspec



def train_xgb_model(X_train, y_train, params=None):
    """
    Entrena un modelo XGBoost con los parámetros especificados o por defecto.
    """
    default_params = {
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.2,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "gamma": 0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }

    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def predict(model, X, threshold=0.5):
    """
    Realiza predicciones con un modelo entrenado y un threshold dado.
    """
    y_proba = model.predict_proba(X)[:, 1]
    return (y_proba >= threshold).astype(int), y_proba


import pandas as pd

def assign_groups_and_services_from_proba(y_proba, user_ids=None):
    """
    Asigna grupos de fraude y paquetes de servicios financieros a partir de la probabilidad de fraude.

    Args:
        y_proba (array-like): Probabilidad de fraude por usuario.
        user_ids (array-like or None): Índices o IDs de usuario. Si None, se usan índices por defecto.

    Returns:
        pd.DataFrame: DataFrame con columnas user_id, grupo_fraude, paquete_servicio
    """

    def asignar_grupo_fraude(prob):
        if prob < 0.3:
            return "Grupo 3"
        elif prob < 0.6:
            return "Grupo 2"
        elif prob < 0.9:
            return "Grupo 1"
        else:
            return "Fraudulento"

    def asignar_paquete(grupo):
        if grupo == "Grupo 1":
            return "Paquete Simple"
        elif grupo == "Grupo 2":
            return "Paquete Intermedio"
        elif grupo == "Grupo 3":
            return "Paquete Pro"
        else:
            return "Sin Paquete"

    df = pd.DataFrame({
        "user_id": user_ids if user_ids is not None else range(len(y_proba)),
        "prob_fraude": y_proba
    })

    df["grupo_fraude"] = df["prob_fraude"].apply(asignar_grupo_fraude)
    df["paquete_servicio"] = df["grupo_fraude"].apply(asignar_paquete)

    return df


def evaluate_model(y_true, y_pred, y_proba):
    """
    Calcula y muestra métricas de evaluación del modelo.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)

    print("\n--- Evaluación del modelo ---")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"ROC AUC      : {roc:.4f}")
    print("\n" + classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": acc,
        "f1": f1,
        "recall": rec,
        "precision": prec,
        "roc_auc": roc
    }


def save_model(model, model_path: str):
    """
    Guarda el modelo en local o GCS.
    Ejemplo path:
        - Local: "models/xgb_model.joblib"
        - GCS:   "gs://fraud-detection-lewagon/models/xgb_model.joblib"
    """
    with fsspec.open(model_path, "wb") as f:
        joblib.dump(model, f)


def load_model(path="model/xgb_model.pkl"):
    """
    Carga un modelo desde local o desde GCS usando fsspec.
    Ejemplo path:
        - Local: "model/xgb_model.pkl"
        - GCS:   "gs://fraud-detection-lewagon/models/xgb_model.pkl"
    """
    try:
        with fsspec.open(path, "rb") as f:
            model = joblib.load(f)
            print(f"📦 Modelo cargado desde: {path}")
            return model
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {path}")
