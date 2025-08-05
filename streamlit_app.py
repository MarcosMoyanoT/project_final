import streamlit as st
import pandas as pd
import requests
import io
import json
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Cargar variables de entorno al inicio
load_dotenv()

# ---------- CONFIGURACIÓN DE PÁGINA Y API ----------
st.set_page_config(page_title="Simulador de Fraude + CFO IA", layout="wide")

# Inicializar st.session_state
if 'df_scores' not in st.session_state:
    st.session_state.df_scores = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ---------- CARGA DE DATOS Y LÓGICA DE PREDICCIÓN ----------
st.title("Sistema de Detección de Fraude")
st.markdown("Carga tus archivos `transaction.csv` y `identity.csv` para obtener predicciones.")

# Widgets para subir archivos
uploaded_transaction_file = st.file_uploader("Elige el archivo de Transacciones (transaction.csv)", type="csv")
uploaded_identity_file = st.file_uploader("Elige el archivo de Identidad (identity.csv)", type="csv")

if uploaded_transaction_file and uploaded_identity_file:
    if st.session_state.df_scores is None or \
       (uploaded_transaction_file.name, uploaded_identity_file.name) != \
       (st.session_state.get('last_trans_file_name'), st.session_state.get('last_id_file_name')):

        try:
            df_transactions = pd.read_csv(uploaded_transaction_file)
            df_identity = pd.read_csv(uploaded_identity_file)
            df_raw_input = pd.merge(df_transactions, df_identity, on='TransactionID', how='left')
            st.success("Archivos cargados y fusionados correctamente.")

            json_data = df_raw_input.to_json(orient='records')
            st.info("Enviando datos a la API para predicción...")
            headers = {'Content-Type': 'application/json'}
            API_URL = "https://fraud-detector-api-567985136734.us-central1.run.app"
            response = requests.post(API_URL, data=json_data, headers=headers)

            if response.status_code == 200:
                predictions = response.json()
                st.success("Predicciones recibidas de la API.")
                df_predictions = pd.DataFrame(predictions)
                st.session_state.df_scores = df_raw_input.copy()
                st.session_state.df_scores['fraud_score'] = df_predictions['prediction']
                st.session_state.last_trans_file_name = uploaded_transaction_file.name
                st.session_state.last_id_file_name = uploaded_identity_file.name

            else:
                st.error(f"Error al conectar con la API. Código de estado: {response.status_code}")
                st.json(response.json())
                st.session_state.df_scores = None

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
            st.session_state.df_scores = None

# ---------- LÓGICA DEL SIMULADOR DE COSTOS (solo se ejecuta si hay datos) ----------
if st.session_state.df_scores is not None:
    st.sidebar.header("Ajustá los umbrales de riesgo:")
    low_risk_threshold = st.sidebar.slider("Máximo score para Bajo riesgo", 0.0, 1.0, 0.3, 0.01)
    medium_risk_threshold = st.sidebar.slider("Máximo score para Riesgo medio", low_risk_threshold, 1.0, 0.6, 0.01)
    high_risk_threshold = st.sidebar.slider("Máximo score para Riesgo alto", medium_risk_threshold, 1.0, 0.9, 0.01)

    st.sidebar.header("Ajustá el costo promedio asumido por servicio:")
    cost_completo = st.sidebar.number_input("Costo % - Tarjeta de Crédito", 0.0, 1.0, 0.015, 0.001)
    cost_medio = st.sidebar.number_input("Costo % - Préstamos", 0.0, 1.0, 0.01, 0.001)
    cost_simple = st.sidebar.number_input("Costo % - Transacciones", 0.0, 1.0, 0.003, 0.001)

    def assign_risk_group(score):
        if score < low_risk_threshold:
            return "Bajo riesgo"
        elif score < medium_risk_threshold:
            return "Riesgo medio"
        elif score < high_risk_threshold:
            return "Riesgo alto"
        else:
            return "Fraude"

    if 'TransactionAmt' in st.session_state.df_scores.columns:
        st.session_state.df_display = st.session_state.df_scores.copy()
        st.session_state.df_display["risk_group"] = st.session_state.df_display["fraud_score"].apply(assign_risk_group)
        st.session_state.df_display["estimated_cost"] = st.session_state.df_display["TransactionAmt"] * st.session_state.df_display["fraud_score"]
        st.session_state.df_display['fraud_score_baseline'] = 0.08
        st.session_state.df_display['estimated_cost_baseline'] = st.session_state.df_display['TransactionAmt'] * st.session_state.df_display['fraud_score_baseline']
        Costo_total_fraude_con_modelo = st.session_state.df_display['estimated_cost'].sum()
        Costo_total_fraude_sin_modelo = st.session_state.df_display['estimated_cost_baseline'].sum()
        ahorro_total = Costo_total_fraude_sin_modelo - Costo_total_fraude_con_modelo
        porcentaje_ahorro = ahorro_total / Costo_total_fraude_sin_modelo if Costo_total_fraude_sin_modelo > 0 else 0
        Monto_total_movimiento = st.session_state.df_display['TransactionAmt'].sum()

        st.title("Resultados de la Predicción y Análisis de Costos")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribución de Riesgo por Modelo")
            st.bar_chart(st.session_state.df_display["risk_group"].value_counts())
        with col2:
            st.subheader("Métricas del modelo")
            st.metric("Costo con modelo (USD)", f"${Costo_total_fraude_con_modelo:,.2f}")
            st.metric("Costo sin modelo (USD)", f"${Costo_total_fraude_sin_modelo:,.2f}")
            st.metric("Ahorro estimado (USD)", f"${ahorro_total:,.2f}")
            st.metric("Porcentaje de ahorro", f"{porcentaje_ahorro:.2%}")
        st.markdown("### Vista previa de asignaciones y costos")
        st.dataframe(st.session_state.df_display.head(20))
        st.markdown("### Costos estimados por grupo de riesgo")
        st.table(st.session_state.df_display.groupby("risk_group")["estimated_cost"].sum().reset_index())

        # ---------- AGENTE CFO INTELIGENTE CON CHAT ----------
        st.markdown("## 🤖 Agente AI")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        if OPENAI_API_KEY is None:
            st.error("La variable de entorno OPENAI_API_KEY no está configurada.")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Haz tu pregunta"):
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        llm_pandasai = OpenAI(api_token=OPENAI_API_KEY)
                        smart_df = SmartDataframe(st.session_state.df_display, config={"llm": llm_pandasai})

                        if user_query.lower() in ["hola", "hola como estas?", "saludos", "que tal?","buenas","quien eres?","como puedes ayudarme?"]:
                            response = "¡Hola! Soy tu agente financiero. Hazme consultas sobre los datos cargados y/o outputs generados"
                        try:
                            response = smart_df.chat(user_query)
                        except Exception as e:
                            response = f"Lo siento, ocurrió un error al procesar tu pregunta. Error: {e}"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(st.session_state.messages)
    else:
        st.warning("La columna 'TransactionAmt' no se encontró en los datos cargados. Necesaria para el cálculo de costos.")

else:
    st.warning("Primero sube los archivos para activar el agente inteligente y las visualizaciones.")
