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
st.set_page_config(page_title="Simulador de Fraude + Agente IA", layout="wide")

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

        # Datos históricos y volúmenes
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
    total_volumen = sum(volumen_unidades.values())

    # Calcular costos históricos ponderados para mostrar como valor inicial
    costo_inicial_prestamo = riesgo_historico["prestamo"]
    costo_inicial_transaccion = riesgo_historico["transaccion"]
    costo_inicial_tarjeta = riesgo_historico["tarjeta"]

    st.sidebar.header("Ajustá la tasa de costo por unidad de negocio:")

    # Inputs para cada unidad con valores históricos como default
    cost_prestamo = st.sidebar.number_input(
        "Tasa - Préstamos",
        min_value=0.0, max_value=1.0,
        value=round(costo_inicial_prestamo, 5),
        step=0.0001,
        format="%.5f"
    )

    cost_transaccion = st.sidebar.number_input(
        "Tasa - Transacciones",
        min_value=0.0, max_value=1.0,
        value=round(costo_inicial_transaccion, 5),
        step=0.0001,
        format="%.5f"
    )

    cost_tarjeta = st.sidebar.number_input(
        "Tasa - Tarjeta de Crédito",
        min_value=0.0, max_value=1.0,
        value=round(costo_inicial_tarjeta, 5),
        step=0.0001,
        format="%.5f"
    )

    # Calcular costos ponderados por volumen para cada paquete
    def costo_paquete(unidades):
        # unidades: lista con nombres de las unidades incluidas en el paquete
        costo_total = 0.0
        volumen_total = sum([volumen_unidades[u] for u in unidades])
        for u in unidades:
            # Tomamos el costo ajustado y ponderamos por volumen relativo dentro del paquete
            if u == "prestamo":
                costo_unitario = cost_prestamo
            elif u == "transaccion":
                costo_unitario = cost_transaccion
            elif u == "tarjeta":
                costo_unitario = cost_tarjeta
            else:
                costo_unitario = 0.0
            peso = volumen_unidades[u] / volumen_total
            costo_total += costo_unitario * peso
        return costo_total

    # Paquetes
    costo_simple = costo_paquete(["tarjeta"])
    costo_medio = costo_paquete(["tarjeta", "transaccion"])
    costo_completo = costo_paquete(["tarjeta", "transaccion", "prestamo"])

    st.sidebar.header("Costos promedio asumidos por paquete:")

    st.sidebar.write(f"Tasa de costo - Simple (Tarjeta): {costo_simple:.5f}")
    st.sidebar.write(f"Tasa de costo - Medio (Tarjeta + Transacción): {costo_medio:.5f}")
    st.sidebar.write(f"Tasa de costo - Completo (Tarjeta + Transacción + Préstamo): {costo_completo:.5f}")

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

    # --- Generar df_scores_baseline con lógica alternativa ---
    df_scores_baseline = df_scores.copy()

    def asignar_paquete_baseline(score):
        return "Paquete Completo" if score < 0.9 else "Sin Paquete"

    def asignar_paquete_modelo(score):
        if score < 0.3:
            return "Paquete Completo"
        elif score < 0.6:
            return "Paquete Medio"
        elif score < 0.9:
            return "Paquete Básico"
        else:
            return "Sin Paquete"


    df_scores_baseline["paquete_servicio"] = df_scores_baseline["fraud_score"].apply(asignar_paquete_baseline)
    df_scores["paquete_servicio"] = df_scores["fraud_score"].apply(asignar_paquete_modelo)

    # --- ANÁLISIS y COSTOS ---
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

    # Para df_scores (modelo)
    df_scores["estimated_cost_ponderado"] = df_scores.apply(
    lambda row: row["TransactionAmt"] * row["fraud_score"] * paquete_a_costo.get(row["paquete_servicio"], 0.0),
    axis=1
)

# Para df_scores_baseline (sin modelo)
    df_scores_baseline["estimated_cost_ponderado"] = df_scores_baseline.apply(
    lambda row: row["TransactionAmt"] * row["fraud_score"] * paquete_a_costo_baseline.get(row["paquete_servicio"], 0.0),
    axis=1
)

    Costo_total_fraude_con_modelo = df_scores['estimated_cost_ponderado'].sum()
    Costo_total_fraude_sin_modelo = df_scores_baseline['estimated_cost_ponderado'].sum()

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
    st.dataframe(df_scores.head(20)) #Ver de tomar las columnas que nos interesan

    st.markdown("### Costos estimados por grupo de riesgo")
    st.table(df_scores.groupby("risk_group")["estimated_cost_ponderado"].sum().reset_index())


    # ---------- AGENTE IA (COMENTADO) ----------
    # st.markdown("## 🤖 Consultá al Asistente IA")

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
    #             st.success("Respuesta del asistente (PandasAI):")
    #             st.write(response)
    #         except Exception as e:
    #             st.warning("Falla PandasAI. Asegúrate de que tu query sea sobre los datos.")
    #             st.write(f"Error: {e}")
