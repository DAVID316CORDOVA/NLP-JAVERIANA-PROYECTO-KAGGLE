#Instrucciones para correr por ahora.

## 1. Levantar el entorno virtual e instalar las dependencias.
pip install -r requirements.txt

## 2. Correr los 3 archivos de modelos (que generan los pkl).

### desde Proyecto_Final (y con venv activo)
uvicorn api.main:app --reload

# POST ejemplo (PowerShell)
$body = @{
  headline = "Gobierno anuncia vacuna 100% efectiva contra COVID"
  text     = "Publicación en redes sociales asegura que..."
} | ConvertTo-Json

Invoke-RestMethod -Method Post "http://127.0.0.1:8000/predict?model=tfidf" -Body $body -ContentType "application/json"
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/predict?model=sbert" -Body $body -ContentType "application/json"
