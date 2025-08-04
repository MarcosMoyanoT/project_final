import streamlit as st
import pandas as pd
import requests
import io
import json
import os
# import openai
# from pandasai import SmartDataframe
# from pandasai.llm.openai import OpenAI

# ---------- CONFIGURACIÓN DE PÁGINA Y API ----------
st.set_page_config(page_title="Simulador de Fraude + CFO IA", layout="wide")

API_URL = "https://fraud-detector-api-567985136734.us-central1.run.app"
# ---------- CARGA DE DATOS Y LÓGICA DE PREDICCIÓN ----------
st.title("Sistema de Detección de Fraude")
st.markdown("Carga tus archivos `train_transaction.csv` y `train_identity.csv` para obtener predicciones.")

# Widgets para subir archivos
uploaded_transaction_file = st.file_uploader("Elige el archivo de Transacciones (train_transaction.csv)", type="csv")
uploaded_identity_file = st.file_uploader("Elige el archivo de Identidad (train_identity.csv)", type="csv")

df_scores = None # Inicializamos df_scores como None

if uploaded_transaction_file and uploaded_identity_file:
    try:
        # Leer y fusionar los DataFrames
        df_transactions = pd.read_csv(uploaded_transaction_file)
        df_identity = pd.read_csv(uploaded_identity_file)

        # Usar un merge para combinar los datos de transacción e identidad
        # Se asume que la columna para hacer el merge es 'TransactionID'
        df_raw_input = pd.merge(df_transactions, df_identity, on='TransactionID', how='left')
        st.success("Archivos cargados y fusionados correctamente.")

        # Convertir el DataFrame a un formato JSON para la API
        json_data = df_raw_input.to_json(orient='records')

        # Realizar la llamada a la API
        st.info("Enviando datos a la API para predicción...")
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, data=json_data, headers=headers)

        if response.status_code == 200:
            predictions = response.json()
            st.success("Predicciones recibidas de la API.")

            # Agregar los resultados de la API a un DataFrame
            df_predictions = pd.DataFrame(predictions)

            # Combina el DataFrame original con las predicciones
            df_scores = df_raw_input.copy()
            df_scores['fraud_score'] = df_predictions['prediction']

        else:
            st.error(f"Error al conectar con la API. Código de estado: {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

# ---------- LÓGICA DEL SIMULADOR DE COSTOS (solo se ejecuta si hay datos) ----------
if df_scores is not None:
    st.sidebar.header("Ajustá los umbrales de riesgo:")
    low_risk_threshold = st.sidebar.slider("Máximo score para Bajo riesgo", 0.0, 1.0, 0.3, 0.01)
    medium_risk_threshold = st.sidebar.slider("Máximo score para Riesgo medio", low_risk_threshold, 1.0, 0.6, 0.01)
    high_risk_threshold = st.sidebar.slider("Máximo score para Riesgo alto", medium_risk_threshold, 1.0, 0.9, 0.01)

    st.sidebar.header("Ajustá el costo promedio asumido por servicio:")
    cost_completo = st.sidebar.number_input("Costo % - Tarjeta de Crédito", 0.0, 1.0, 0.015, 0.001)
    cost_medio = st.sidebar.number_input("Costo % - Préstamos", 0.0, 1.0, 0.01, 0.001)
    cost_simple = st.sidebar.number_input("Costo % - Transacciones", 0.0, 1.0, 0.003, 0.001)

    # --- Funciones de asignación de riesgo ---
    def assign_risk_group(score):
        if score < low_risk_threshold:
            return "Bajo riesgo"
        elif score < medium_risk_threshold:
            return "Riesgo medio"
        elif score < high_risk_threshold:
            return "Riesgo alto"
        else:
            return "Fraude"

    # Se asume que tienes las columnas 'TransactionAmt' para los cálculos
    df_scores["risk_group"] = df_scores["fraud_score"].apply(assign_risk_group)

    # --- ANÁLISIS y COSTOS ---
    # Asumo que la columna de monto es 'TransactionAmt'
    df_scores["estimated_cost"] = df_scores["TransactionAmt"] * df_scores["fraud_score"]

    # Costo sin modelo (baseline)
    df_scores['fraud_score_baseline'] = 0.08
    df_scores['estimated_cost_baseline'] = df_scores['TransactionAmt'] * df_scores['fraud_score_baseline']

    Costo_total_fraude_con_modelo = df_scores['estimated_cost'].sum()
    Costo_total_fraude_sin_modelo = df_scores['estimated_cost_baseline'].sum()

    ahorro_total = Costo_total_fraude_sin_modelo - Costo_total_fraude_con_modelo
    porcentaje_ahorro = ahorro_total / Costo_total_fraude_sin_modelo if Costo_total_fraude_sin_modelo > 0 else 0

    Monto_total_movimiento = df_scores['TransactionAmt'].sum()
    Porcentaje_costo_fraude = Costo_total_fraude_con_modelo / Monto_total_movimiento if Monto_total_movimiento > 0 else 0

    # ---------- INTERFAZ STREAMLIT CON VISUALIZACIONES ----------
    st.title("Resultados de la Predicción y Análisis de Costos")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Riesgo por Modelo")
        st.bar_chart(df_scores["risk_group"].value_counts())

    with col2:
        st.subheader("Métricas del modelo")
        st.metric("Costo con modelo (USD)", f"${Costo_total_fraude_con_modelo:,.2f}")
        st.metric("Costo sin modelo (USD)", f"${Costo_total_fraude_sin_modelo:,.2f}")
        st.metric("Ahorro estimado (USD)", f"${ahorro_total:,.2f}")
        st.metric("Porcentaje de ahorro", f"{porcentaje_ahorro:.2%}")

    st.markdown("### Vista previa de asignaciones y costos")
    st.dataframe(df_scores.head(20))

    st.markdown("### Costos estimados por grupo de riesgo")
    st.table(df_scores.groupby("risk_group")["estimated_cost"].sum().reset_index())

    # ---------- AGENTE CFO INTELIGENTE (COMENTADO) ----------
    # st.markdown("## 🤖 Consultá al CFO Asistente (IA)")

    # openai.api_key = os.environ.get("OPENAI_API_KEY")
    # if openai.api_key is None:
    #     st.error("La variable de entorno OPENAI_API_KEY no está configurada.")
    # else:
    #     llm_pandasai = OpenAI(api_token=openai.api_key)
    #     smart_df = SmartDataframe(df_scores, config={"llm": llm_pandasai, "verbose": True})

    #     user_query = st.text_input("Hacé tu pregunta financiera sobre los datos (ej: ¿cuánto cuesta el fraude en Bajo riesgo?)")

    #     if user_query:
    #         try:
    #             response = smart_df.chat(user_query)
    #             st.success("Respuesta del CFO (PandasAI):")
    #             st.write(response)
    #         except Exception as e:
    #             st.warning("Falla PandasAI. Asegúrate de que tu query sea sobre los datos.")
    #             st.write(f"Error: {e}")
