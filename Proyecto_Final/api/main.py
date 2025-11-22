from pathlib import Path
import joblib
import json
import warnings
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Literal
from sentence_transformers import SentenceTransformer

# ===== Rutas base relativas al archivo =====
BASE_DIR = Path(__file__).resolve().parents[1]   # Proyecto_Final
MODELS_DIR = BASE_DIR / "models"


# ===== Cargar Modelo 1 original (TF-IDF + LogReg) =====
TFIDF_MODEL_PATH = MODELS_DIR / "model_tfidf_logreg.pkl"
pipe_tfidf = joblib.load(TFIDF_MODEL_PATH) if TFIDF_MODEL_PATH.exists() else None

# ===== Cargar Modelo 1 tunado (TF-IDF best) =====
TFIDF_BEST_PATH = MODELS_DIR / "model_tfidf_best.pkl"
pipe_tfidf_best = joblib.load(TFIDF_BEST_PATH) if TFIDF_BEST_PATH.exists() else None

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
    warnings.warn("Modelo SBERT no disponible (falta .pkl o .json).")


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
        "model_tfidf_loaded": pipe_tfidf is not None,
        "model_tfidf_best_loaded": pipe_tfidf_best is not None,
        "model_sbert_loaded": clf_sbert is not None and sbert_model is not None,
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

class NewsIn(BaseModel):
    headline: str = ""
    text: str = ""

class PredictionOut(BaseModel):
    model: str
    label: Literal["Fake", "True"]
    prob_fake: float


def preprocess(headline: str, text: str) -> str:
    full = (headline or "") + " " + (text or "")
    import re
    s = full.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\*number\*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@app.post("/predict", response_model=PredictionOut)
def predict(payload: NewsIn,
           model: Literal["tfidf", "tfidf_best", "sbert"] = Query("tfidf_best")):
    text = preprocess(payload.headline, payload.text)

    if model == "tfidf":
        if pipe_tfidf is None:
            raise HTTPException(status_code=500, detail="Modelo TF-IDF (original) no disponible.")
        proba = float(pipe_tfidf.predict_proba([text])[0][1])
        label = "Fake" if proba >= 0.5 else "True"
        return PredictionOut(model="tfidf", label=label, prob_fake=proba)

    if model == "tfidf_best":
        if pipe_tfidf_best is None:
            raise HTTPException(status_code=500, detail="Modelo TF-IDF best no disponible.")
        proba = float(pipe_tfidf_best.predict_proba([text])[0][1])
        label = "Fake" if proba >= 0.5 else "True"
        return PredictionOut(model="tfidf_best", label=label, prob_fake=proba)

    if model == "sbert":
        if clf_sbert is None or sbert_model is None:
            raise HTTPException(status_code=500, detail="Modelo SBERT no disponible.")
        emb = sbert_model.encode([text], normalize_embeddings=True)
        proba = float(clf_sbert.predict_proba(emb)[0][1])
        label = "Fake" if proba >= 0.5 else "True"
        return PredictionOut(model="sbert", label=label, prob_fake=proba)

    raise HTTPException(status_code=400, detail="Modelo inválido.")
