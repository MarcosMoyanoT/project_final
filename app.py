import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAI_OpenAI

# Cargar variables de entorno
load_dotenv()

# Leer CSV
df = pd.read_csv("df_scores.csv")

# Título principal
st.title("Análisis de Riesgo de Fraude")

# Sidebar con opciones
menu = st.sidebar.selectbox("Selecciona una vista:", ["Dashboard", "Asistente CFO (chatbot)"])

# --- DASHBOARD PRINCIPAL ---
if menu == "Dashboard":
    # Filtros
    risk_group_filter = st.multiselect("Selecciona grupo de riesgo:", df["risk_group"].unique(), default=df["risk_group"].unique())
    service_filter = st.multiselect("Selecciona tipo de servicio:", df["service_assignment"].unique(), default=df["service_assignment"].unique())

    # Aplicar filtros
    filtered_df = df[
        df["risk_group"].isin(risk_group_filter) & df["service_assignment"].isin(service_filter)
    ]

    # Mostrar datos
    st.dataframe(filtered_df)

    # Métrica
    st.write("### Suma total de montos filtrados")
    st.metric("Total TransactionAmt", f"${filtered_df['TransactionAmt'].sum():,.2f}")

# --- CHATBOT CFO ---
elif menu == "Asistente CFO (chatbot)":
    st.subheader("💬 Chat con el CFO Asistente (PandasAI)")

    # Inicializar PandasAI
    llm = PandasAI_OpenAI(api_token=os.getenv("OPENAI_API_KEY"))
    sdf = SmartDataframe(df, config={"llm": llm})

    # Historial de mensajes
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial anterior
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Entrada del usuario
    user_input = st.chat_input("Haz una pregunta sobre el análisis de fraude...")

    # Procesar entrada
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Pensando..."):
            try:
                reply = sdf.chat(user_input)
                st.chat_message("assistant").write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"❌ Error al generar respuesta: {e}")
