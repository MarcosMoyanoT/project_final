import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ---------- CONFIGURACIÓN DE PÁGINA Y API ----------
st.set_page_config(page_title="🚨 Simulador de Fraude + Agente IA 🤖", layout="wide")

API_URL = "https://fraud-detector-api-567985136734.us-central1.run.app"

# ---------- CARGA DE DATOS Y LÓGICA DE PREDICCIÓN ----------
st.title("🔍 Sistema Inteligente de Detección de Fraude")
st.markdown("Subí tus archivos `train_transaction.csv` y `train_identity.csv` para detectar fraudes automáticamente ⚠️")

uploaded_transaction_file = st.file_uploader("📂 Elige el archivo de Transacciones (train_transaction.csv)", type="csv")
uploaded_identity_file = st.file_uploader("📂 Elige el archivo de Identidad (train_identity.csv)", type="csv")

df_scores = None

if uploaded_transaction_file and uploaded_identity_file:
    try:
        df_transactions = pd.read_csv(uploaded_transaction_file)
        df_identity = pd.read_csv(uploaded_identity_file)

        df_raw_input = pd.merge(df_transactions, df_identity, on='TransactionID', how='left')
        st.success("✅ Archivos cargados y fusionados correctamente.")

        json_data = df_raw_input.to_json(orient='records')

        st.info("📡 Enviando datos a la API para predicción...")
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, data=json_data, headers=headers)

        if response.status_code == 200:
            predictions = response.json()
            st.success("🎯 Predicciones recibidas de la API.")

            df_predictions = pd.DataFrame(predictions)
            df_scores = df_raw_input.copy()
            df_scores['fraud_score'] = df_predictions['prediction']

        else:
            st.error(f"❌ Error al conectar con la API. Código de estado: {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"⚠️ Ocurrió un error: {e}")

# ---------- LÓGICA DEL SIMULADOR DE COSTOS ----------
if df_scores is not None:
    # --- Sidebar para configuración ---
    st.sidebar.header("🎚️ Ajustá los umbrales de riesgo:")
    low_risk_threshold = st.sidebar.slider("🟢 Máximo score para Bajo riesgo", 0.0, 1.0, 0.3, 0.01)
    medium_risk_threshold = st.sidebar.slider("🟡 Máximo score para Riesgo medio", low_risk_threshold, 1.0, 0.6, 0.01)
    high_risk_threshold = st.sidebar.slider("🔴 Máximo score para Riesgo alto", medium_risk_threshold, 1.0, 0.9, 0.01)

    riesgo_historico = {
        "prestamo": 0.02052,
        "transaccion": 0.01004,
        "tarjeta": 0.001727
    }
    volumen_unidades = {
        "prestamo": 50000,
        "transaccion": 50000,
        "tarjeta": 285000
    }

    st.sidebar.header("💰 Tasa de costo por unidad de negocio:")

    cost_prestamo = st.sidebar.number_input("🏦 Tasa - Préstamos", 0.0, 1.0, value=0.02052, step=0.0001, format="%.5f")
    cost_transaccion = st.sidebar.number_input("💳 Tasa - Transacciones", 0.0, 1.0, value=0.01004, step=0.0001, format="%.5f")
    cost_tarjeta = st.sidebar.number_input("🧾 Tasa - Tarjeta de Crédito", 0.0, 1.0, value=0.00173, step=0.0001, format="%.5f")

    def costo_paquete(unidades):
        costo_total = 0.0
        volumen_total = sum([volumen_unidades[u] for u in unidades])
        for u in unidades:
            if u == "prestamo": costo_unitario = cost_prestamo
            elif u == "transaccion": costo_unitario = cost_transaccion
            elif u == "tarjeta": costo_unitario = cost_tarjeta
            else: costo_unitario = 0.0
            peso = volumen_unidades[u] / volumen_total
            costo_total += costo_unitario * peso
        return costo_total

    costo_simple = costo_paquete(["tarjeta"])
    costo_medio = costo_paquete(["tarjeta", "transaccion"])
    costo_completo = costo_paquete(["tarjeta", "transaccion", "prestamo"])

    st.sidebar.header("📦 Costos promedio por paquete")
    st.sidebar.markdown(f"**Paquete Simple** 💳: `{costo_simple:.5f}`")
    st.sidebar.markdown(f"**Paquete Medio** 💳➕📤: `{costo_medio:.5f}`")
    st.sidebar.markdown(f"**Paquete Completo** 💳➕📤➕🏦: `{costo_completo:.5f}`")

    # --- Asignar grupos de riesgo ---
    def assign_risk_group(score):
        if score < low_risk_threshold: return "Bajo riesgo"
        elif score < medium_risk_threshold: return "Riesgo medio"
        elif score < high_risk_threshold: return "Riesgo alto"
        else: return "Fraude"

    df_scores["risk_group"] = df_scores["fraud_score"].apply(assign_risk_group)

    df_scores_baseline = df_scores.copy()
    df_scores_baseline["paquete_servicio"] = df_scores_baseline["fraud_score"].apply(lambda s: "Paquete Completo" if s < 0.9 else "Sin Paquete")

    def asignar_paquete_modelo(score):
        if score < low_risk_threshold: return "Paquete Completo"
        elif score < medium_risk_threshold: return "Paquete Medio"
        elif score < high_risk_threshold: return "Paquete Básico"
        else: return "Sin Paquete"

    df_scores["paquete_servicio"] = df_scores["fraud_score"].apply(asignar_paquete_modelo)

    paquete_a_costo = {
        "Paquete Básico": costo_simple,
        "Paquete Medio": costo_medio,
        "Paquete Completo": costo_completo,
        "Sin Paquete": 0.0
    }

    paquete_a_costo_baseline = {
        "Paquete Completo": costo_completo,
        "Sin Paquete": 0.0
    }

    df_scores["estimated_cost_ponderado"] = df_scores.apply(
        lambda row: row["TransactionAmt"] * row["fraud_score"] * paquete_a_costo.get(row["paquete_servicio"], 0.0), axis=1)

    df_scores_baseline["estimated_cost_ponderado"] = df_scores_baseline.apply(
        lambda row: row["TransactionAmt"] * row["fraud_score"] * paquete_a_costo_baseline.get(row["paquete_servicio"], 0.0), axis=1)

    Costo_total_fraude_con_modelo = df_scores['estimated_cost_ponderado'].sum()
    Costo_total_fraude_sin_modelo = df_scores_baseline['estimated_cost_ponderado'].sum()
    ahorro_total = Costo_total_fraude_sin_modelo - Costo_total_fraude_con_modelo
    porcentaje_ahorro = ahorro_total / Costo_total_fraude_sin_modelo if Costo_total_fraude_sin_modelo > 0 else 0

    Monto_total_movimiento = df_scores['TransactionAmt'].sum()
    Porcentaje_costo_fraude = Costo_total_fraude_con_modelo / Monto_total_movimiento if Monto_total_movimiento > 0 else 0

    # --- Gráfica circular de riesgo ---
    st.title("📊 Resultados de la Predicción y Análisis de Costos")
    st.subheader("🧪 Distribución de Riesgo por Modelo")

    risk_counts = df_scores["risk_group"].value_counts().reset_index()
    risk_counts.columns = ["Riesgo de Grupo", "Cantidad"]

    color_map = {
        "Bajo riesgo": "#4CAF50",    # verde
        "Riesgo medio": "#FFEB3B",   # amarillo
        "Riesgo alto": "#FF9800",    # naranja
        "Fraude": "#F44336"          # rojo
    }

    risk_counts["color"] = risk_counts["Riesgo de Grupo"].map(color_map)

    fig = px.pie(
        risk_counts,
        values="Cantidad",
        names="Riesgo de Grupo",
        color="Riesgo de Grupo",
        color_discrete_map=color_map,
        hole=0.4,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        insidetextfont=dict(size=18, color="black"),
        marker=dict(line=dict(color="#000000", width=2))
    )

    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(font=dict(size=14)),
        font=dict(family="Arial", size=16)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Métricas ---
    st.subheader("📈 Métricas del modelo")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💸 Costo con modelo (USD)", f"${Costo_total_fraude_con_modelo:,.2f}")
    col2.metric("💰 Costo sin modelo (USD)", f"${Costo_total_fraude_sin_modelo:,.2f}")
    col3.metric("🤑 Ahorro estimado (USD)", f"${ahorro_total:,.2f}")
    col4.metric("📉 Porcentaje de ahorro", f"{porcentaje_ahorro:.2%}")

    # --- Tabla ---
    st.markdown("### 🧾 Vista previa de asignaciones y costos")

    cols_a_mostrar = [
        "TransactionID", "TransactionAmt", "fraud_score", "risk_group", "paquete_servicio", "estimated_cost_ponderado"
    ]

    df_vista = df_scores[cols_a_mostrar].head(20).copy()

    df_vista.rename(columns={
        "TransactionID": "ID de Transacción",
        "TransactionAmt": "Monto de Transacción",
        "fraud_score": "Puntaje de Fraude",
        "risk_group": "Riesgo de Grupo",
        "paquete_servicio": "Paquete de Servicio",
        "estimated_cost_ponderado": "Costo Estimado Ponderado"
    }, inplace=True)

    df_vista["Puntaje de Fraude"] = (df_vista["Puntaje de Fraude"] * 100).map("{:.2f}%".format)
    df_vista["Monto de Transacción"] = df_vista["Monto de Transacción"].map("${:,.2f}".format)
    df_vista["Costo Estimado Ponderado"] = df_vista["Costo Estimado Ponderado"].map("${:,.2f}".format)

    def color_riesgo(val):
        colores = {
            "Bajo riesgo": "background-color: #d4f4dd; color: #116611;",
            "Riesgo medio": "background-color: #fff4b2; color: #665511;",
            "Riesgo alto": "background-color: #ffccbb; color: #aa3300;",
            "Fraude": "background-color: #ff4c4c; color: white;"
        }
        return colores.get(val, "")

    def style_table(df):
        estilos = pd.DataFrame("", index=df.index, columns=df.columns)
        estilos["Riesgo de Grupo"] = df["Riesgo de Grupo"].map(color_riesgo)
        return estilos

    st.dataframe(df_vista.style.apply(style_table, axis=None), use_container_width=True)

    # --- Costos por grupo de riesgo ---
    st.markdown("### 📌 Costos estimados por grupo de riesgo")

    df_costos = df_scores.groupby("risk_group")["estimated_cost_ponderado"].sum().reset_index()
    df_costos.rename(columns={
        "risk_group": "Riesgo de Grupo",
        "estimated_cost_ponderado": "Costo Estimado Ponderado"
    }, inplace=True)
    df_costos["Costo Estimado Ponderado"] = df_costos["Costo Estimado Ponderado"].map("${:,.2f}".format)
    st.table(df_costos)

    # --- NUEVAS GRÁFICAS E INFO DE NEGOCIO ---

    st.markdown("### 📈 Análisis adicional y métricas clave")

    col1, col2 = st.columns(2)

    # 1. Porcentaje de transacciones por grupo de riesgo
    risk_pct = df_scores["risk_group"].value_counts(normalize=True).reset_index()
    risk_pct.columns = ["Riesgo de Grupo", "Porcentaje"]
    risk_pct["Porcentaje"] = (risk_pct["Porcentaje"] * 100).round(2)

    fig_pct = px.bar(
        risk_pct,
        x="Riesgo de Grupo",
        y="Porcentaje",
        color="Riesgo de Grupo",
        color_discrete_map=color_map,
        text="Porcentaje",
        title="Distribución % de Transacciones por Grupo de Riesgo"
    )
    fig_pct.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_pct.update_layout(yaxis_title="Porcentaje (%)", xaxis_title="", showlegend=False, margin=dict(t=40))
    col1.plotly_chart(fig_pct, use_container_width=True)

    # 2. Monto total de transacciones por grupo de riesgo
    monto_group = df_scores.groupby("risk_group")["TransactionAmt"].sum().reset_index()
    monto_group.rename(columns={"TransactionAmt": "Monto Total"}, inplace=True)
    monto_group = monto_group.sort_values(by="Monto Total", ascending=False)

    fig_monto = px.bar(
        monto_group,
        x="risk_group",
        y="Monto Total",
        color="risk_group",
        color_discrete_map=color_map,
        text=monto_group["Monto Total"].map("${:,.0f}".format),
        title="Monto Total de Transacciones por Grupo de Riesgo"
    )
    fig_monto.update_traces(textposition="outside")
    fig_monto.update_layout(yaxis_title="Monto Total (USD)", xaxis_title="", showlegend=False, margin=dict(t=40))
    col2.plotly_chart(fig_monto, use_container_width=True)

# 3. Costo estimado ponderado total por paquete de servicio
paquete_costos = df_scores.groupby("paquete_servicio")["estimated_cost_ponderado"].sum().reset_index()
paquete_costos.rename(
    columns={
        "estimated_cost_ponderado": "Costo Estimado Ponderado",
        "paquete_servicio": "Paquete de Servicio"
    },
    inplace=True
)
paquete_costos = paquete_costos.sort_values(by="Costo Estimado Ponderado", ascending=False)

fig_paquete = px.pie(
    paquete_costos,
    names="Paquete de Servicio",
    values="Costo Estimado Ponderado",
    title="📦 Costo Estimado Ponderado Total por Paquete de Servicio",
    hole=0.3,
    color="Paquete de Servicio",
    color_discrete_map={
        "Sin Paquete": "#003366",           # azul oscuro
        "Paquete Completo": "#6699CC",      # azul claro
        "Paquete Medio": "#336699",         # otro azul intermedio (opcional)
        "Paquete Básico": "#99CCFF"         # azul muy claro (opcional)
    }
)

fig_paquete.update_traces(
    textinfo="percent+label",
    insidetextfont=dict(size=18, color="black")
)

fig_paquete.update_layout(
    title_font=dict(size=24, family="Arial, sans-serif", color="white"),
    legend=dict(font=dict(size=14, color="black")),
    font=dict(family="Arial", size=14),
    margin=dict(t=50, b=10, l=10, r=10)
)

st.plotly_chart(fig_paquete, use_container_width=True)
