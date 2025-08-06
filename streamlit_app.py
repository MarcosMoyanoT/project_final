import streamlit as st
import pandas as pd
import requests
import io
import json
import os
from dotenv import load_dotenv
# Las siguientes líneas de pandasai deben estar COMENTADAS o ELIMINADAS
# from pandasai import SmartDataframe
# from pandasai.llm.openai import OpenAI
from openai import OpenAI as openai_client # Importar el cliente general de OpenAI
import plotly.express as px # Importar Plotly Express

# Cargar variables de entorno al inicio
load_dotenv()

# Inicializar cliente de OpenAI para chat general
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    # st.info(f"Clave de API cargada (parcial): {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}") # Línea de depuración
    openai_client_chat = openai_client(api_key=OPENAI_API_KEY)
else:
    st.error("La variable de entorno OPENAI_API_KEY no está configurada. Por favor, revisa tu archivo .env.")
    openai_client_chat = None

# ---------- CONFIGURACIÓN DE PÁGINA Y API ----------
st.set_page_config(page_title="🚨 Detección de Fraude + Agente IA 🤖", layout="wide")

# Inicializar st.session_state para df_scores y messages
if 'df_scores' not in st.session_state:
    st.session_state.df_scores = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# URL de tu API de Cloud Run
API_URL = "https://fraud-detector-api-567985136734.us-central1.run.app"

# ---------- CARGA DE DATOS Y LÓGICA DE PREDICCIÓN ----------
st.title("FRAUD RISK APP")
st.subheader("🔍 Modelo inteligente para la detección de fraude instantanea para la reducción de costos de tu Fintech")
st.markdown("Subí tus archivos `transaction.csv` y `identity.csv` para detectar fraudes automáticamente ⚠️")

uploaded_transaction_file = st.file_uploader("📂 Elige el archivo de Transacciones (transaction.csv)", type="csv")
uploaded_identity_file = st.file_uploader("📂 Elige el archivo de Identidad (identity.csv)", type="csv")

if uploaded_transaction_file and uploaded_identity_file:
    if st.session_state.df_scores is None or \
       (uploaded_transaction_file.name, uploaded_identity_file.name) != \
       (st.session_state.get('last_trans_file_name'), st.session_state.get('last_id_file_name')):
        
        try:
            df_transactions = pd.read_csv(uploaded_transaction_file)
            df_identity = pd.read_csv(uploaded_identity_file)
            df_raw_input = pd.merge(df_transactions, df_identity, on='TransactionID', how='left')
            st.success("✅ Archivos cargados y fusionados correctamente.")

            json_data = df_raw_input.to_json(orient='records')
            st.info("📡 Enviando datos a la API para predicción...")
            headers = {'Content-Type': 'application/json'}
            
            # --- Añadir logs antes y después de la llamada a la API ---
            print("DEBUG: Realizando llamada a la API de predicción...")
            response = requests.post(API_URL, data=json_data, headers=headers)
            print(f"DEBUG: Llamada a la API finalizada. Código de estado: {response.status_code}")

            if response.status_code == 200:
                predictions = response.json()
                st.success("🎯 Predicciones recibidas de la API.")
                df_predictions = pd.DataFrame(predictions)
                
                st.session_state.df_scores = df_raw_input.copy()
                st.session_state.df_scores['fraud_score'] = df_predictions['prediction']

                st.session_state.last_trans_file_name = uploaded_transaction_file.name
                st.session_state.last_id_file_name = uploaded_identity_file.name
                
            else:
                st.error(f"❌ Error al conectar con la API. Código de estado: {response.status_code}")
                st.json(response.json())
                st.session_state.df_scores = None
                
        except Exception as e:
            st.error(f"⚠️ Ocurrió un error: {e}")
            st.session_state.df_scores = None

# ---------- LÓGICA DEL SIMULADOR DE COSTOS (solo se ejecuta si hay datos) ----------
if st.session_state.df_scores is not None:
    # --- Sidebar para configuración ---
    st.sidebar.header("🎚️ Ajustá los umbrales de riesgo:")
    st.sidebar.markdown(
    """      
    Ajustá los valores para personalizar cómo se clasifica el riesgo de cada usuario según el modelo.
    """
    )
   
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

    st.sidebar.header("💰 Tasa histórica de fraude por unidad de negocio:")
    cost_tarjeta = st.sidebar.number_input("💳 Tasa - Tarjeta de Crédito", 0.0, 1.0, value=0.00173, step=0.0001, format="%.3f")
    cost_transaccion = st.sidebar.number_input("🧾 Tasa - Transacciones", 0.0, 1.0, value=0.01004, step=0.0001, format="%.3f")
    cost_prestamo = st.sidebar.number_input("🏦 Tasa - Préstamos", 0.0, 1.0, value=0.02052, step=0.0001, format="%.3f")
      
    def costo_paquete(unidades):
        costo_total = 0.0
        volumen_total = sum([volumen_unidades[u] for u in unidades])
        for u in unidades:
            if u == "prestamo": costo_unitario = cost_prestamo
            elif u == "transaccion": costo_unitario = cost_transaccion
            elif u == "tarjeta": costo_unitario = cost_tarjeta
            else: costo_unitario = 0.0
            peso = volumen_unidades[u] / volumen_total if volumen_total > 0 else 0
            costo_total += costo_unitario * peso
        return costo_total

    costo_simple = costo_paquete(["tarjeta"])
    costo_medio = costo_paquete(["tarjeta", "transaccion"])
    costo_completo = costo_paquete(["tarjeta", "transaccion", "prestamo"])

    
    st.sidebar.markdown(
    """      
    Según el nivel de riesgo estimado, cada usuario recibe un paquete de servicios financieros adaptado a su perfil.
    
    """
    )
    st.sidebar.markdown(f"**Paquete Simple** 💳") 
    st.sidebar.markdown(f"**Paquete Medio** 💳➕🧾")
    st.sidebar.markdown(f"**Paquete Completo** 💳➕🧾➕🏦") 

    def assign_risk_group(score):
        if score < low_risk_threshold: return "Bajo riesgo"
        elif score < medium_risk_threshold: return "Riesgo medio"
        elif score < high_risk_threshold: return "Riesgo alto"
        else: return "Fraude"

    if 'TransactionAmt' in st.session_state.df_scores.columns:
        st.session_state.df_display = st.session_state.df_scores.copy()
        st.session_state.df_display["risk_group"] = st.session_state.df_display["fraud_score"].apply(assign_risk_group)

        def asignar_paquete_modelo(score):
            if score < low_risk_threshold: return "Paquete Completo"
            elif score < medium_risk_threshold: return "Paquete Medio"
            elif score < high_risk_threshold: return "Paquete Básico"
            else: return "Sin Paquete"

        st.session_state.df_display["paquete_servicio"] = st.session_state.df_display["fraud_score"].apply(asignar_paquete_modelo)

        paquete_a_costo = {
            "Paquete Básico": costo_simple,
            "Paquete Medio": costo_medio,
            "Paquete Completo": costo_completo,
            "Sin Paquete": 0.0
        }

        df_display_baseline = st.session_state.df_display.copy()
        df_display_baseline["paquete_servicio"] = df_display_baseline["fraud_score"].apply(lambda s: "Paquete Completo" if s < 0.9 else "Sin Paquete")

        paquete_a_costo_baseline = {
            "Paquete Completo": costo_completo,
            "Sin Paquete": 0.0
        }

        st.session_state.df_display["estimated_cost_ponderado"] = st.session_state.df_display.apply(
            lambda row: row["TransactionAmt"] * row["fraud_score"] * paquete_a_costo.get(row["paquete_servicio"], 0.0), axis=1)

        df_display_baseline["estimated_cost_ponderado"] = df_display_baseline.apply(
            lambda row: row["TransactionAmt"] * row["fraud_score"] * paquete_a_costo_baseline.get(row["paquete_servicio"], 0.0), axis=1)


        Costo_total_fraude_con_modelo = st.session_state.df_display['estimated_cost_ponderado'].sum()
        Costo_total_fraude_sin_modelo = df_display_baseline['estimated_cost_ponderado'].sum()
        ahorro_total = Costo_total_fraude_sin_modelo - Costo_total_fraude_con_modelo
        porcentaje_ahorro = ahorro_total / Costo_total_fraude_sin_modelo if Costo_total_fraude_sin_modelo > 0 else 0

        Monto_total_movimiento = st.session_state.df_display['TransactionAmt'].sum()

        # --- Gráficas y Métricas ---
        st.title("📊 Resultados de la Predicción y Análisis de Costos")
        st.subheader("🧪 Distribución de Riesgo por Modelo")

        risk_counts = st.session_state.df_display["risk_group"].value_counts().reset_index()
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

        st.subheader("📈 Métricas del modelo")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💸 Costo con Fraud Risk App (USD)", f"${Costo_total_fraude_con_modelo:,.2f}")
        col2.metric("💰 Costo sin Fraud Risk App (USD)", f"${Costo_total_fraude_sin_modelo:,.2f}")
        col3.metric("🤑 Ahorro estimado (USD)", f"${ahorro_total:,.2f}")
        col4.metric("📉 Porcentaje de ahorro", f"{porcentaje_ahorro:.2%}")

        st.markdown("### 🧾 Vista previa de asignaciones y costos")

        cols_a_mostrar = [
            "TransactionID", "TransactionAmt", "fraud_score", "risk_group", "paquete_servicio"
        ]

        df_vista = st.session_state.df_display[cols_a_mostrar].head(20).copy()

        df_vista.rename(columns={
            "TransactionID": "ID de Transacción",
            "TransactionAmt": "Monto de Transacción",
            "fraud_score": "Puntaje de Fraude",
            "risk_group": "Riesgo de Grupo",
            "paquete_servicio": "Paquete de Servicio"
                }, inplace=True)

        df_vista["Puntaje de Fraude"] = (df_vista["Puntaje de Fraude"] * 100).map("{:.2f}%".format)
        df_vista["Monto de Transacción"] = df_vista["Monto de Transacción"].map("${:,.2f}".format)
        

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

        st.markdown("### 📌 Costos estimados por grupo de riesgo")

        # Filtrar el DataFrame para excluir el grupo 'fraude'
        df_filtrado = st.session_state.df_display[
            st.session_state.df_display["risk_group"].str.strip().str.lower() != "fraude"
        ]
        
        # Agrupar y calcular los costos
        df_costos = df_filtrado.groupby("risk_group")["estimated_cost_ponderado"].sum().reset_index()
        
        # Ordenar de mayor a menor según el costo
        df_costos = df_costos.sort_values(by="estimated_cost_ponderado", ascending=False)
        
        # Renombrar columnas
        df_costos.rename(columns={
            "risk_group": "Riesgo de Grupo",
            "estimated_cost_ponderado": "Costo Estimado Ponderado"
        }, inplace=True)
        
        # Formatear la columna de costos
        df_costos["Costo Estimado Ponderado"] = df_costos["Costo Estimado Ponderado"].map("${:,.2f}".format)
        
        # Mostrar la tabla
        st.table(df_costos.reset_index(drop=True))
        

        st.markdown("### 📈 Análisis adicional y métricas clave")
        col1, col2 = st.columns(2)

        risk_pct = st.session_state.df_display["risk_group"].value_counts(normalize=True).reset_index()
        risk_pct.columns = ["Riesgo de Grupo", "Porcentaje"]
        risk_pct["Porcentaje"] = (risk_pct["Porcentaje"] * 100).round(2)

        fig_pct = px.bar(
            risk_pct,
            x="Riesgo de Grupo",
            y="Porcentaje",
            color="Riesgo de Grupo",
            color_discrete_map=color_map,
            text="Porcentaje",
            title=" % de distribución de Transacciones por Grupo de Riesgo"
        )
        fig_pct.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_pct.update_layout(yaxis_title="Porcentaje (%)", xaxis_title="", showlegend=False, margin=dict(t=40))
        col1.plotly_chart(fig_pct, use_container_width=True)

        monto_group = st.session_state.df_display.groupby("risk_group")["TransactionAmt"].sum().reset_index()
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

        paquete_costos = st.session_state.df_display.groupby("paquete_servicio")["estimated_cost_ponderado"].sum().reset_index()
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
                "Sin Paquete": "#003366",
                "Paquete Completo": "#6699CC",
                "Paquete Medio": "#336699",
                "Paquete Básico": "#99CCFF"
            }
        )

        fig_paquete.update_traces(
            textinfo="percent+label",
            insidetextfont=dict(size=18, color="black")
        )

        fig_paquete.update_layout(
            title_font=dict(size=24, family="Arial, sans-serif", color="black"),
            legend=dict(font=dict(size=14, color="black")),
            font=dict(family="Arial", size=14),
            margin=dict(t=50, b=10, l=10, r=10)
        )

        st.plotly_chart(fig_paquete, use_container_width=True)

        # ---------- AGENTE CFO INTELIGENTE CON CHAT ----------
        st.markdown("## 🤖 Agente AI")
        
        if not openai_client_chat:
            st.error("No se pudo inicializar el cliente de OpenAI. Revisa tu API key.")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Hacé tu pregunta financiera sobre los datos."):
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        # --- CÓDIGO CORREGIDO Y MEJORADO ---
                        # Heurística para decidir si la pregunta es sobre los datos
                        data_keywords = ["cuánto", "promedio", "suma", "total", "monto", "riesgo", "usuarios", "transacciones", "costo", "fraude", "número", "porcentaje", "distribución", "fraudulentos"]
                        is_data_query = any(keyword in user_query.lower() for keyword in data_keywords)

                        if is_data_query:
                            # --- Lógica para preguntas de datos (sin PandasAI) ---
                            # Le pedimos a GPT que genere el código Python
                            prompt_for_code = f"""
                            Eres un asistente experto en análisis de datos. Dada la siguiente pregunta del usuario y un DataFrame de pandas llamado `st.session_state.df_display` (que está en st.session_state.df_scores), genera el código Python para responder a la pregunta.
                            Asegúrate de que el código sea completo y ejecutable. Si la pregunta es sobre "usuarios" o "transacciones", asume que se refiere a filas en el DataFrame.

                            El DataFrame `st.session_state.df_display` tiene las siguientes columnas clave:
                            - 'risk_group': Contiene las categorías 'Bajo riesgo', 'Riesgo medio', 'Riesgo alto', 'Fraude'.
                            - 'TransactionAmt': Contiene el monto de la transacción.
                            - 'fraud_score': Es el score de fraude del modelo.
                            - 'estimated_cost_ponderado': Es el costo estimado por transacción.

                            Pregunta del usuario: "{user_query}"

                            Si la pregunta se refiere a "fraude", "fraudulentos" o "transacciones fraudulentas", debes filtrar la columna `risk_group` para que sea igual a 'Fraude'.
                            Por favor, genera solo el código Python. No incluyas explicaciones ni texto adicional.
                            Ejemplo:
                            # Código para "cuántas filas hay?"
                            print(len(df))
                            """
                            try:
                                code_completion = openai_client_chat.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": prompt_for_code}
                                    ]
                                )
                                python_code = code_completion.choices[0].message.content
                                
                                # Ejecutar el código generado
                                # Creamos un entorno de ejecución seguro
                                local_vars = {'df': st.session_state.df_scores, 'pd': pd}
                                output_buffer = io.StringIO()
                                try:
                                    # Redirigir la salida estándar para capturar el print
                                    import sys
                                    old_stdout = sys.stdout
                                    sys.stdout = output_buffer
                                    
                                    with st.spinner("Ejecutando código..."):
                                        exec(python_code, globals(), local_vars)
                                        response = local_vars.get('result', output_buffer.getvalue().strip())
                                        if not response: # Si el código no hizo print o no asignó 'result'
                                            response = "No pude obtener una respuesta específica de los datos con ese código."
                                            st.code(python_code) # Mostrar el código para depuración si no hay respuesta
                                except Exception as exec_e:
                                    response = f"Error al ejecutar el código Python generado: {exec_e}. Intenta reformular tu pregunta."
                                    st.code(python_code) # Mostrar el código para depuración
                                finally:
                                    sys.stdout = old_stdout # Restaurar la salida estándar
                                    
                            except Exception as e:
                                response = f"Lo siento, ocurrió un error al generar el código Python para tu pregunta. Error: {e}"
                        else:
                            # Usar el cliente general de OpenAI para preguntas conversacionales
                            messages_for_chat = [
                                {"role": "system", "content": "You are a helpful CFO assistant for a fraud detection app. You respond in Spanish and your tone is professional. You can't access any data directly. If the user asks for data analysis, tell them you can only answer general questions and suggest they ask a specific data-related question."},
                                {"role": "user", "content": user_query}
                            ]
                            
                            try:
                                chat_completion = openai_client_chat.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=messages_for_chat
                                )
                                response = chat_completion.choices[0].message.content
                            except Exception as e:
                                response = f"Lo siento, no pude procesar tu pregunta con el modelo de chat. Error: {e}"
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.warning("La columna 'TransactionAmt' no se encontró en los datos cargados. Necesaria para el cálculo de costos y visualizaciones.")

else:
    st.warning("Primero sube los archivos para activar el agente inteligente y las visualizaciones.")
