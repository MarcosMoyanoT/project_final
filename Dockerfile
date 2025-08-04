# Usamos una imagen base oficial de Python para tu proyecto
FROM python:3.9-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos solo el archivo requirements.txt primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalamos las dependencias
# El comando --no-cache-dir y --upgrade son buenas prácticas
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos de tu proyecto
COPY . .

# Exponemos el puerto en el que correrá la aplicación (Cloud Run usará esto)
EXPOSE 8080

# Comando para iniciar el servidor de la API con Uvicorn
# El "app:app" significa que busque la instancia "app" dentro del archivo "app.py"
# --host 0.0.0.0 le dice al servidor que escuche en todas las interfaces de red disponibles
# --port $PORT le indica que use el puerto asignado por Cloud Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
