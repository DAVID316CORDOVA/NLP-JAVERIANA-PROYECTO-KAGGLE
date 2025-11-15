# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Literal
from pathlib import Path
import joblib
import numpy as np
import warnings

from sentence_transformers import SentenceTransformer
import json

# ===== Rutas base =====
BASE_DIR = Path(r"C:\Users\santiagoBairesDev\Desktop\NLP\proyectos_grupo_PLN\NLP-JAVERIANA-PROYECTO-KAGGLE\kaggle_toxic\Proyecto_Final")
MODELS_DIR = BASE_DIR / "models"

# ===== Cargar Modelo 1 (TF-IDF + LogReg) =====
TFIDF_MODEL_PATH = MODELS_DIR / "model_tfidf_logreg.pkl"
pipe_tfidf = None
if TFIDF_MODEL_PATH.exists():
    pipe_tfidf = joblib.load(TFIDF_MODEL_PATH)
else:
    warnings.warn(f"No se encontró {TFIDF_MODEL_PATH}. El endpoint 'model=tfidf' fallará si se usa.")

# ===== Cargar Modelo 2 (SBERT + LogReg) =====
SBERT_CLF_PATH = MODELS_DIR / "model_sbert_logreg.pkl"
SBERT_CFG_PATH = MODELS_DIR / "model_sbert_config.json"
clf_sbert = None
sbert_model = None
if SBERT_CLF_PATH.exists() and SBERT_CFG_PATH.exists():
    clf_sbert = joblib.load(SBERT_CLF_PATH)
    cfg = json.loads(SBERT_CFG_PATH.read_text(encoding="utf-8"))
    sbert_model = SentenceTransformer(cfg["embedding_model"])
else:
    warnings.warn("No se encontraron archivos del Modelo 2 (SBERT). El endpoint 'model=sbert' fallará si se usa.")

# ===== FastAPI =====
app = FastAPI(title="Fake News ES API", version="1.0")

class NewsIn(BaseModel):
    headline: str = ""
    text: str = ""

class PredictionOut(BaseModel):
    model: str
    label: Literal["Fake", "True"]
    prob_fake: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_tfidf_loaded": TFIDF_MODEL_PATH.exists(),
        "model_sbert_loaded": SBERT_CLF_PATH.exists() and SBERT_CFG_PATH.exists()
    }

def preprocess(headline: str, text: str) -> str:
    full = (headline or "") + " " + (text or "")
    s = full.lower()
    # limpieza mínima consistente con entrenamiento
    import re
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\*number\*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@app.post("/predict", response_model=PredictionOut)
def predict(payload: NewsIn, model: Literal["tfidf", "sbert"] = Query("tfidf")):
    text = preprocess(payload.headline, payload.text)

    if model == "tfidf":
        if pipe_tfidf is None:
            raise HTTPException(status_code=500, detail="Modelo TF-IDF no disponible en el servidor.")
        proba = float(pipe_tfidf.predict_proba([text])[0][1])
        label = "Fake" if proba >= 0.5 else "True"
        return PredictionOut(model="tfidf", label=label, prob_fake=proba)

    elif model == "sbert":
        if clf_sbert is None or sbert_model is None:
            raise HTTPException(status_code=500, detail="Modelo SBERT no disponible en el servidor.")
        emb = sbert_model.encode([text], normalize_embeddings=True)
        proba = float(clf_sbert.predict_proba(emb)[0][1])
        label = "Fake" if proba >= 0.5 else "True"
        return PredictionOut(model="sbert", label=label, prob_fake=proba)

    else:
        raise HTTPException(status_code=400, detail="Parámetro 'model' inválido. Usa 'tfidf' o 'sbert'.")
