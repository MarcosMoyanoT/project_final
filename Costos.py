import pandas as pd

# -------------------------
# 1. Cargar los datasets
# -------------------------
df_scores = pd.read_csv("df_scores.csv")
df_scores_baseline = pd.read_csv("df_scores_baseline.csv")
df_creditcard = pd.read_csv("df_creditcard.csv")
df_loan = pd.read_csv("df_loan.csv")
df_transaction = pd.read_csv("df_transaction.csv")

# Renombrar columna de flag de fraude
df_creditcard.rename(columns={"Class": "fraud_flag"}, inplace=True)

# -------------------------
# 2. Riesgo histórico
# -------------------------
riesgos_historicos = {
    'Tarjeta de Crédito': df_creditcard['fraud_flag'].mean(),
    'Préstamos': df_loan["fraud_flag"].mean(),
    'Transacciones': df_transaction["fraud_flag"].mean()
}
riesgos_ordenados = dict(sorted(riesgos_historicos.items(), key=lambda x: x[1], reverse=True))

print("\n--- Riesgo Histórico ---")
for unidad, riesgo in riesgos_ordenados.items():
    print(f'{unidad}: {riesgo:.4%} de riesgo histórico')

# -------------------------
# 3. Montos de fraude y no fraude
# -------------------------
fraud_amount_creditcard = df_creditcard[df_creditcard['fraud_flag'] == 1]['Amount'].sum()
fraud_amount_loan = df_loan[df_loan['fraud_flag'] == 1]['loan_amount_requested'].sum()
fraud_amount_transaction = df_transaction[df_transaction['fraud_flag'] == 1]['transaction_amount'].sum()

nofraud_amount_creditcard = df_creditcard[df_creditcard['fraud_flag'] == 0]['Amount'].sum()
nofraud_amount_loan = df_loan[df_loan['fraud_flag'] == 0]['loan_amount_requested'].sum()
nofraud_amount_transaction = df_transaction[df_transaction['fraud_flag'] == 0]['transaction_amount'].sum()

# -------------------------
# 4. Totales y porcentajes
# -------------------------
total_amount_creditcard = fraud_amount_creditcard + nofraud_amount_creditcard
total_amount_loan = fraud_amount_loan + nofraud_amount_loan
total_amount_transaction = fraud_amount_transaction + nofraud_amount_transaction

print("\n--- Porcentajes de fraude ---")
print(f"Tarjeta de Crédito: {fraud_amount_creditcard/total_amount_creditcard:.2%}")
print(f"Préstamos: {fraud_amount_loan/total_amount_loan:.2%}")
print(f"Transacciones: {fraud_amount_transaction/total_amount_transaction:.2%}")

# -------------------------
# 5. Costo promedio por usuario
# -------------------------
# Obtener cantidad de fraudes reales (flag = 1)
n_fraudes_creditcard = df_creditcard['fraud_flag'].value_counts().get(1)
n_fraudes_loan = df_loan['fraud_flag'].value_counts().get(1)
n_fraudes_transaction = df_transaction['fraud_flag'].value_counts().get(1)

# Cálculo del costo promedio
Costo_promedio_fraude_TC = fraud_amount_creditcard / n_fraudes_creditcard
Costo_promedio_fraude_loan = fraud_amount_loan / n_fraudes_loan
Costo_promedio_fraude_transaction = fraud_amount_transaction / n_fraudes_transaction

C_promedio_total = (
    Costo_promedio_fraude_TC +
    Costo_promedio_fraude_loan +
    Costo_promedio_fraude_transaction
)

print("\n--- Costo Promedio de Fraude por Unidad ---")
print(f"Tarjeta de Crédito: {Costo_promedio_fraude_TC:,.2f}")
print(f"Préstamos: {Costo_promedio_fraude_loan:,.2f}")
print(f"Transacciones: {Costo_promedio_fraude_transaction:,.2f}")


# -------------------------
# 6. Costo con modelo (real)
# -------------------------
# Mapear los costos promedio por servicio
costo_unitario = {
    'Tarjeta de Crédito': Costo_promedio_fraude_TC,
    'Préstamos': Costo_promedio_fraude_loan,
    'Transacciones': Costo_promedio_fraude_transaction
}

# Calcular costo total con modelo
df_scores['costo_asignado'] = df_scores['service_assignment'].map(costo_unitario)
df_scores['costo_est_modelo'] = df_scores['fraud_score'] * df_scores['costo_asignado']
Costo_total_fraude_con_modelo = df_scores['costo_est_modelo'].sum()

# -------------------------
# 7. Costo sin modelo (baseline)
# -------------------------

# Baseline: a todos los usuarios se les asignan todos los servicios, cada fila representa un servicio asignado
df_scores_baseline['costo_asignado'] = df_scores_baseline['service_assignment'].map(costo_unitario)
df_scores_baseline['costo_est_modelo'] = df_scores_baseline['fraud_score'] * df_scores_baseline['costo_asignado']

Costo_total_fraude_sin_modelo = df_scores_baseline['costo_est_modelo'].sum()

# -------------------------
# 8. Indicadores
# -------------------------
ahorro_total = Costo_total_fraude_sin_modelo - Costo_total_fraude_con_modelo
porcentaje_ahorro = ahorro_total / Costo_total_fraude_sin_modelo

Monto_total_movimiento = total_amount_creditcard + total_amount_loan + total_amount_transaction
Porcentaje_costo_fraude = Costo_total_fraude_con_modelo / Monto_total_movimiento

print("\n--- Indicadores Finales ---")
print(f"Costo total sin modelo: {Costo_total_fraude_sin_modelo:,.2f}")
print(f"Costo total con modelo: {Costo_total_fraude_con_modelo:,.2f}")
print(f"Ahorro estimado: {ahorro_total:,.2f}")
print(f"Porcentaje de ahorro: {porcentaje_ahorro:.2%}")
print(f"Monto total de movimientos: {Monto_total_movimiento:,.2f}")
print(f"Porcentaje del costo de fraude sobre los movimientos: {Porcentaje_costo_fraude:.2%}")

# -------------------------
# 9. Extras: comparación de servicios por usuario
# -------------------------
avg_servicios_con_modelo = df_scores.groupby('user_id')['service_assignment'].nunique().mean()
avg_servicios_baseline = df_scores_baseline.groupby('user_id')['service_assignment'].nunique().mean()

print("\n--- Servicios asignados promedio por usuario ---")
print(f"Con modelo: {avg_servicios_con_modelo:.2f}")
print(f"Sin modelo: {avg_servicios_baseline:.2f}")

# -------------------------
# 10. Usuarios con riesgo alto sin servicios
# -------------------------
umbral_alto = 0.9
usuarios_altoriesgo_sin_servicio = df_scores[
    (df_scores['fraud_score'] >= umbral_alto) & (df_scores['service_assignment'].isna())
]['user_id'].nunique()

print(f"\nUsuarios con score >= {umbral_alto} y sin servicios asignados: {usuarios_altoriesgo_sin_servicio}")
