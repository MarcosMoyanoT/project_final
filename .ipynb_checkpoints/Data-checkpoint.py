import pandas as pd

# 1. Definimos los nombres de los archivos (agrega la ruta si es necesario)
files = [
    "/home/juanpablo/code/MarcosMoyanoT/project_final/raw_data/ieee-fraud-detection/sample_submission.csv",
    "/home/juanpablo/code/MarcosMoyanoT/project_final/raw_data/ieee-fraud-detection/test_identity.csv",
    "/home/juanpablo/code/MarcosMoyanoT/project_final/raw_data/ieee-fraud-detection/test_transaction.csv",
    "/home/juanpablo/code/MarcosMoyanoT/project_final/raw_data/ieee-fraud-detection/train_identity.csv",
    "/home/juanpablo/code/MarcosMoyanoT/project_final/raw_data/ieee-fraud-detection/train_transaction.csv"
]

# 2. Leemos los archivos en un diccionario de DataFrames
dfs = {file.split('.')[0]: pd.read_csv(file) for file in files}

# 3. Mostramos información general (opcional)
for name, df in dfs.items():
    print(f"\n--- {name} ---")
    print(df.info())  # Para ver columnas y tipos de datos
    print(df.head(2)) # Primeras filas de muestra

# 4. Función para obtener valores únicos por columna
def unique_values_report(df, df_name):
    print(f"\nValores únicos por columna en {df_name}:")
    for col in df.columns:
        uniques = df[col].unique()
        num_uniques = len(uniques)
        print(f"  - {col}: {num_uniques} valores únicos")
        # Mostramos los primeros 10 valores únicos (para no saturar)
        print(f"    Ejemplos: {uniques[:10]}")

# 5. Aplicamos la función a todos los archivos
for name, df in dfs.items():
    unique_values_report(df, name)
