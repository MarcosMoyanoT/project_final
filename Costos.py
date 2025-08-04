import streamlit as st
import pandas as pd

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="Asignación de Servicios", layout="wide")

# ---------- CARGA DE DATOS ----------
df_scores = pd.read_csv("df_scores.csv")
df_scores_baseline = pd.read_csv("df_scores_baseline.csv")
df_creditcard = pd.read_csv("df_creditcard.csv")
df_loan = pd.read_csv("df_loan.csv")
df_transaction = pd.read_csv("df_transaction.csv")

# Renombrar columna para consistencia
df_creditcard.rename(columns={"Class": "fraud_flag"}, inplace=True)

st.title("Simulador de Asignación de Servicios y Costos por Riesgo de Fraude")

# ---------- SIDEBAR ----------
st.sidebar.header("Ajustá los umbrales de riesgo:")
low_risk_threshold = st.sidebar.slider("Máximo score para Bajo riesgo", 0.0, 1.0, 0.3, 0.01)
medium_risk_threshold = st.sidebar.slider("Máximo score para Riesgo medio", low_risk_threshold, 1.0, 0.6, 0.01)
high_risk_threshold = st.sidebar.slider("Máximo score para Riesgo alto", medium_risk_threshold, 1.0, 0.9, 0.01)

st.sidebar.header("Ajustá el costo promedio asumido por servicio:")
cost_completo = st.sidebar.number_input("Costo % - Paquete completo", 0.0, 1.0, 0.015, 0.001)
cost_medio = st.sidebar.number_input("Costo % - Paquete medio", 0.0, 1.0, 0.01, 0.001)
cost_simple = st.sidebar.number_input("Costo % - Paquete simple", 0.0, 1.0, 0.003, 0.001)
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

C_promedio_total = (
    Costo_promedio_fraude_TC +
    Costo_promedio_fraude_loan +
    Costo_promedio_fraude_transaction
)

# ---------- COSTO CON MODELO ----------
costo_unitario = {
    'Tarjeta de Crédito': Costo_promedio_fraude_TC,
    'Préstamos': Costo_promedio_fraude_loan,
    'Transacciones': Costo_promedio_fraude_transaction
}

df_scores['costo_asignado'] = df_scores['service_assignment'].map(costo_unitario)
df_scores['costo_est_modelo'] = df_scores['fraud_score'] * df_scores['costo_asignado']
Costo_total_fraude_con_modelo = df_scores['costo_est_modelo'].sum()

# ---------- COSTO BASELINE (SIN MODELO) ----------
df_scores_baseline['fraud_score'] = 0.08  # Score promedio estimado
df_scores_baseline['costo_asignado'] = df_scores_baseline['service_assignment'].map(costo_unitario)
df_scores_baseline['costo_est_modelo'] = df_scores_baseline['fraud_score'] * df_scores_baseline['costo_asignado']
Costo_total_fraude_sin_modelo = df_scores_baseline['costo_est_modelo'].sum()

# ---------- MÉTRICAS FINALES ----------
ahorro_total = Costo_total_fraude_sin_modelo - Costo_total_fraude_con_modelo
porcentaje_ahorro = ahorro_total / Costo_total_fraude_sin_modelo if Costo_total_fraude_sin_modelo > 0 else 0

Monto_total_movimiento = (
    fraud_amount_creditcard + nofraud_amount_creditcard +
    fraud_amount_loan + nofraud_amount_loan +
    fraud_amount_transaction + nofraud_amount_transaction
)
Porcentaje_costo_fraude = Costo_total_fraude_con_modelo / Monto_total_movimiento if Monto_total_movimiento > 0 else 0

# ---------- VISUALIZACIÓN ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de usuarios por tipo de servicio")
    st.bar_chart(df_scores["service_assignment"].value_counts())

with col2:
    st.subheader("Costos estimados")
    st.metric("Costo con modelo (USD)", f"${Costo_total_fraude_con_modelo:,.2f}")
    st.metric("Costo sin modelo (USD)", f"${Costo_total_fraude_sin_modelo:,.2f}")
    st.metric("Ahorro estimado (USD)", f"${ahorro_total:,.2f}")
    st.metric("Porcentaje ahorro", f"{porcentaje_ahorro:.2%}")

st.markdown("### Vista previa de asignaciones y costos por usuario")
st.dataframe(df_scores.head(20))

# ---------- TABLA DE COSTOS POR SERVICIO ----------
st.markdown("### Costos promedio por tipo de servicio")
cost_summary = df_scores.groupby("service_assignment")["estimated_cost"].sum().reset_index()
st.table(cost_summary)
