from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import joblib
import numpy as np
import os
import fsspec

def find_optimal_threshold(y_true, y_probs, metric='f1'):
    """
    Encuentra el mejor threshold según la métrica (default: F1-score)
    """
    thresholds = np.linspace(0.1, 0.9, 100)
    scores = []

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError("Metric must be 'f1', 'precision' or 'recall'")
        scores.append(score)

    best_idx = np.argmax(scores)
    return thresholds[best_idx]

def train_xgb_model(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo XGBoost con los parámetros especificados o por defecto.
    """
    default_params = {
    "n_estimators": 443,
    "max_depth": 7,
    "min_child_weight": 5,
    "gamma": 0.07695212100160928,
    "subsample": 0.865237466656049,
    "colsample_bytree": 0.7377072170135907,
    "reg_alpha": 0.8922143189961961,
    "reg_lambda": 0.9920686938647577,
    "learning_rate": 0.15753081640792435,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    }

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    # Calculamos el umbral con los datos de validación
    y_proba_val = model.predict_proba(X_val)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val, y_proba_val)

    return model, optimal_threshold


def predict(model, X_data, threshold):
    """
    Realiza predicciones binarias utilizando un umbral dado.

    Args:
        model: El modelo entrenado.
        X_data (pd.DataFrame): Los datos de entrada (features).
        threshold (float): El umbral para clasificar las predicciones.

    Returns:
        np.array: Un array de predicciones binarias (0 o 1).
    """
    y_proba = model.predict_proba(X_data)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred


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
