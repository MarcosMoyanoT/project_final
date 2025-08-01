import streamlit as st
import pandas as pd

# Cargar datos
df = pd.read_csv("/home/juanpablo/code/MarcosMoyanoT/project_final/df_scores.csv")

st.title("Análisis de Riesgo de Fraude")

# Filtros interactivos
risk_group_filter = st.multiselect("Selecciona grupo de riesgo:", df["risk_group"].unique(), default=df["risk_group"].unique())
service_filter = st.multiselect("Selecciona tipo de servicio:", df["service_assignment"].unique(), default=df["service_assignment"].unique())

# Aplicar filtros
filtered_df = df[
    df["risk_group"].isin(risk_group_filter) &
    df["service_assignment"].isin(service_filter)
]

# Mostrar tabla filtrada
st.dataframe(filtered_df)

# Mostrar resumen numérico
st.write("### Suma total de montos filtrados")
st.metric("Total TransactionAmt", f"${filtered_df['TransactionAmt'].sum():,.2f}")
