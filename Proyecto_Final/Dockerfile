FROM python:3.10-slim

# Instalación básica
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos solo requirements primero (para aprovechar cache)
COPY Proyecto_Final/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto (incluyendo models/, api/, data si la quieres dentro)
COPY Proyecto_Final/ .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando por defecto: levantar la API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
