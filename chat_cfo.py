import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import openai
import os

# Establece tu clave de API de OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]

# Carga el dataset
df = pd.read_csv("df_scores.csv")

# Muestra columnas para verificar
print("Columnas del DataFrame:", df.columns)
print(df.head())  # opcional

# Instancia LLM y conecta SmartDataframe
llm_pandasai = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
smart_df = SmartDataframe(df, config={"llm": llm_pandasai, "verbose": True})

# Función para preguntar al CFO
def ask_cfo(message):
    try:
        answer = smart_df.chat(message)
        print("✅ Usando PandasAI")
        return answer
    except Exception as e:
        print("⚠️ Fallback a GPT:", e)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un CFO asistente experto en análisis financiero."},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content

# Pregunta de ejemplo
print(ask_cfo("¿Cuántos clientes tienen un score menor a 0.3?"))
