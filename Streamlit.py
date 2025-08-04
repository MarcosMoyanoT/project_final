# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="Simulador de Fraude + CFO IA", layout="wide")

# ---------- CARGA DE DATOS ----------
df_scores = pd.read_csv("df_scores.csv")
#df_scores_baseline = pd.read_csv("df_scores_baseline.csv")
df_creditcard = pd.read_csv("creditcard.csv")
df_loan = pd.read_csv("loan_applications.csv")
df_transaction = pd.read_csv("transactions.csv")

df_creditcard.rename(columns={"Class": "fraud_flag"}, inplace=True)

# ---------- SIDEBAR ----------
st.sidebar.header("Ajustá los umbrales de riesgo:")
low_risk_threshold = st.sidebar.slider("Máximo score para Bajo riesgo", 0.0, 1.0, 0.3, 0.01)
medium_risk_threshold = st.sidebar.slider("Máximo score para Riesgo medio", low_risk_threshold, 1.0, 0.6, 0.01)
high_risk_threshold = st.sidebar.slider("Máximo score para Riesgo alto", medium_risk_threshold, 1.0, 0.9, 0.01)

st.sidebar.header("Ajustá el costo promedio asumido por servicio:")
cost_completo = st.sidebar.number_input("Costo % - Tarjeta de Crédito", 0.0, 1.0, 0.015, 0.001)
cost_medio = st.sidebar.number_input("Costo % - Préstamos", 0.0, 1.0, 0.01, 0.001)
cost_simple = st.sidebar.number_input("Costo % - Transacciones", 0.0, 1.0, 0.003, 0.001)

# ---------- FUNCIONES DE ASIGNACIÓN ----------
def assign_risk_group(score):
    if score < low_risk_threshold:
        return "Bajo riesgo"
    elif score < medium_risk_threshold:
        return "Riesgo medio"
    elif score < high_risk_threshold:
        return "Riesgo alto"
    else:
        return "Fraude"

def assign_service_package(risk_group):
    return {
        "Bajo riesgo": "Tarjeta de Crédito",
        "Riesgo medio": "Préstamos",
        "Riesgo alto": "Transacciones"
    }.get(risk_group, "Sin Servicio")

def estimate_cost(row):
    if row["service_assignment"] == "Tarjeta de Crédito":
        return row["TransactionAmt"] * cost_completo
    elif row["service_assignment"] == "Préstamos":
        return row["TransactionAmt"] * cost_medio
    elif row["service_assignment"] == "Transacciones":
        return row["TransactionAmt"] * cost_simple
    else:
        return 0.0

# ---------- APLICAR MODELO ----------
df_scores["risk_group"] = df_scores["fraud_score"].apply(assign_risk_group)
df_scores["service_assignment"] = df_scores["risk_group"].apply(assign_service_package)
df_scores["estimated_cost"] = df_scores.apply(estimate_cost, axis=1)

# ---------- ANÁLISIS HISTÓRICO ----------
fraud_amount_creditcard = df_creditcard[df_creditcard['fraud_flag'] == 1]['Amount'].sum()
fraud_amount_loan = df_loan[df_loan['fraud_flag'] == 1]['loan_amount_requested'].sum()
fraud_amount_transaction = df_transaction[df_transaction['fraud_flag'] == 1]['transaction_amount'].sum()

nofraud_amount_creditcard = df_creditcard[df_creditcard['fraud_flag'] == 0]['Amount'].sum()
nofraud_amount_loan = df_loan[df_loan['fraud_flag'] == 0]['loan_amount_requested'].sum()
nofraud_amount_transaction = df_transaction[df_transaction['fraud_flag'] == 0]['transaction_amount'].sum()

n_fraudes_creditcard = df_creditcard['fraud_flag'].value_counts().get(1)
n_fraudes_loan = df_loan['fraud_flag'].value_counts().get(1)
n_fraudes_transaction = df_transaction['fraud_flag'].value_counts().get(1)

Costo_promedio_fraude_TC = fraud_amount_creditcard / n_fraudes_creditcard
Costo_promedio_fraude_loan = fraud_amount_loan / n_fraudes_loan
Costo_promedio_fraude_transaction = fraud_amount_transaction / n_fraudes_transaction

costo_unitario = {
    'Tarjeta de Crédito': Costo_promedio_fraude_TC,
    'Préstamos': Costo_promedio_fraude_loan,
    'Transacciones': Costo_promedio_fraude_transaction
}

# ---------- COSTOS ESTIMADOS ----------
df_scores['costo_asignado'] = df_scores['service_assignment'].map(costo_unitario)
df_scores['costo_est_modelo'] = df_scores['fraud_score'] * df_scores['costo_asignado']
Costo_total_fraude_con_modelo = df_scores['costo_est_modelo'].sum()

df_scores['fraud_score'] = 0.08
df_scores['costo_asignado'] = df_scores['service_assignment'].map(costo_unitario)
df_scores['costo_est_modelo'] = df_scores['fraud_score'] * df_scores['costo_asignado']
Costo_total_fraude_sin_modelo = df_scores['costo_est_modelo'].sum()

ahorro_total = Costo_total_fraude_sin_modelo - Costo_total_fraude_con_modelo
porcentaje_ahorro = ahorro_total / Costo_total_fraude_sin_modelo if Costo_total_fraude_sin_modelo > 0 else 0

Monto_total_movimiento = (
    fraud_amount_creditcard + nofraud_amount_creditcard +
    fraud_amount_loan + nofraud_amount_loan +
    fraud_amount_transaction + nofraud_amount_transaction
)
Porcentaje_costo_fraude = Costo_total_fraude_con_modelo / Monto_total_movimiento if Monto_total_movimiento > 0 else 0

# ---------- INTERFAZ STREAMLIT ----------
st.title("Simulador de Servicios Financieros y Costos por Riesgo de Fraude")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución por tipo de servicio")
    st.bar_chart(df_scores["service_assignment"].value_counts())

with col2:
    st.subheader("Métricas del modelo")
    st.metric("Costo con modelo (USD)", f"${Costo_total_fraude_con_modelo:,.2f}")
    st.metric("Costo sin modelo (USD)", f"${Costo_total_fraude_sin_modelo:,.2f}")
    st.metric("Ahorro estimado (USD)", f"${ahorro_total:,.2f}")
    st.metric("Porcentaje de ahorro", f"{porcentaje_ahorro:.2%}")

st.markdown("### Vista previa de asignaciones y costos")
st.dataframe(df_scores.head(20))

st.markdown("### Costos estimados por tipo de servicio")
st.table(df_scores.groupby("service_assignment")["estimated_cost"].sum().reset_index())

# ---------- Asistente IA ----------
st.markdown("## 🤖  Agente IA:")

from pandasai import SmartDataframe
from openai import OpenAI

# Configuración API
import streamlit as st

import os

# Configuración API
from openai import OpenAI
import os
from dotenv import load_dotenv

# Cargar variables de entorno (si usas .env para guardar tu clave API)
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAI_OpenAI
from openai import OpenAI as OpenAIClient

# Cargar variables de entorno
load_dotenv()
df = pd.read_csv("df_scores.csv")

# Instanciar LLM y SmartDataframe
llm = PandasAI_OpenAI(api_token=os.getenv("OPENAI_API_KEY"))
sdf = SmartDataframe(df, config={"llm": llm})

# Título inicial
st.markdown("---")
st.subheader("💬 Bienvenido!")

# Inicializar historial de mensajes si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Esperar nueva entrada del usuario
user_input = st.chat_input("Haz una pregunta")

# Si hay input, procesarlo
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Pensando..."):
        try:
            # Intentar responder usando el DataFrame
            reply = sdf.chat(user_input)
        except Exception:
            try:
                # Si falla, usar LLM como backup (respuesta contextual)
                openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
                completion = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": user_input}]
                )
                reply = completion.choices[0].message.content
            except Exception as e:
                reply = f"❌ Error al generar respuesta: {e}"

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
