# app.py
import streamlit as st
import pandas as pd

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="Asignación de Servicios", layout="wide")

# ---------- CARGA DE DATOS ----------
df_scores = pd.read_csv("df_scores.csv")  # Asegurate de tener user_id, fraud_score, TransactionAmt

st.title("Simulador de Asignación de Servicios y Costos por Riesgo de Fraude")

# ---------- SLIDERS DE RIESGO ----------
st.sidebar.header("Ajustá los umbrales de riesgo:")
low_risk_threshold = st.sidebar.slider("Máximo score para Bajo riesgo", 0.0, 1.0, 0.3, 0.01)
medium_risk_threshold = st.sidebar.slider("Máximo score para Riesgo medio", low_risk_threshold, 1.0, 0.6, 0.01)
high_risk_threshold = st.sidebar.slider("Máximo score para Riesgo alto", medium_risk_threshold, 1.0, 0.9, 0.01)

# ---------- SLIDERS DE COSTO ----------
st.sidebar.header("Ajustá el costo promedio asumido por servicio:")
cost_completo = st.sidebar.number_input("Costo % - Paquete completo", min_value=0.0, max_value=1.0, value=0.015, step=0.001)
cost_medio = st.sidebar.number_input("Costo % - Paquete medio", min_value=0.0, max_value=1.0, value=0.01, step=0.001)
cost_simple = st.sidebar.number_input("Costo % - Paquete simple", min_value=0.0, max_value=1.0, value=0.003, step=0.001)
cost_sin_paquete = 0.0

# ---------- FUNCIONES ----------
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
    if risk_group == "Bajo riesgo":
        return "Paquete de Servicios completo"
    elif risk_group == "Riesgo medio":
        return "Paquete de Servicios medio"
    elif risk_group == "Riesgo alto":
        return "Paquete de Servicios simple"
    else:
        return "Sin Paquete de Servicios"

def estimate_cost(row):
    if row["service_assignment"] == "Paquete de Servicios completo":
        return row["TransactionAmt"] * cost_completo
    elif row["service_assignment"] == "Paquete de Servicios medio":
        return row["TransactionAmt"] * cost_medio
    elif row["service_assignment"] == "Paquete de Servicios simple":
        return row["TransactionAmt"] * cost_simple
    else:
        return 0.0

# ---------- APLICAR LÓGICA ----------
df_scores["risk_group"] = df_scores["fraud_score"].apply(assign_risk_group)
df_scores["service_assignment"] = df_scores["risk_group"].apply(assign_service_package)
df_scores["estimated_cost"] = df_scores.apply(estimate_cost, axis=1)

# ---------- MOSTRAR RESULTADOS ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de usuarios por tipo de servicio")
    st.bar_chart(df_scores["service_assignment"].value_counts())

with col2:
    st.subheader("Costo total estimado")
    total_cost = df_scores["estimated_cost"].sum()
    st.metric("Costo estimado total (USD)", f"${total_cost:,.2f}")

st.markdown("### Vista previa de asignaciones y costos por usuario")
st.dataframe(df_scores.head(20))

# ---------- MOSTRAR RESUMEN POR GRUPO ----------
st.markdown("### Costos promedio por grupo de servicio")
cost_summary = df_scores.groupby("service_assignment")["estimated_cost"].sum().reset_index()
st.table(cost_summary)
