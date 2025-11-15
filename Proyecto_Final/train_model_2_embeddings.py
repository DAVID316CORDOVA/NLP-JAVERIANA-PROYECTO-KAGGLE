# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from train_utils import get_paths, load_corpus, save_json, SEED

EMB_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"
# Puedes cambiar por otro SBERT multilingüe compatible

def embed_texts(model: SentenceTransformer, texts, batch_size: int = 32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

def main():
    base_dir, _, models_dir, _ = get_paths()
    df = load_corpus()
    if df.empty:
        raise RuntimeError("No se cargaron datos. Revisa los .xlsx en la carpeta data.")

    X = df["text"].tolist()
    y = df["label"].values.astype(int)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    sbert = SentenceTransformer(EMB_MODEL_NAME)
    E_tr = embed_texts(sbert, X_tr)
    E_va = embed_texts(sbert, X_va)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED
    )
    clf.fit(E_tr, y_tr)

    y_pred = clf.predict(E_va)
    y_prob = clf.predict_proba(E_va)[:, 1]

    report = classification_report(y_va, y_pred, target_names=["True(0)", "Fake(1)"], output_dict=True)
    cm = confusion_matrix(y_va, y_pred).tolist()
    f1m = f1_score(y_va, y_pred, average="macro")

    # Guardar SOLO el clasificador y además un config con el nombre del modelo de embeddings
    clf_path = models_dir / "model_sbert_logreg.pkl"
    joblib.dump(clf, clf_path)

    cfg_path = models_dir / "model_sbert_config.json"
    save_json(cfg_path, {"embedding_model": EMB_MODEL_NAME})

    metrics_path = models_dir / "model_sbert_logreg_metrics.json"
    save_json(metrics_path, {
        "f1_macro": f1m,
        "classification_report": report,
        "confusion_matrix": cm
    })

    print(f"[OK] Modelo 2 (clasificador) guardado en: {clf_path}")
    print(f"[OK] Config embeddings en: {cfg_path}")
    print(f"[OK] Métricas guardadas en: {metrics_path}")

if __name__ == "__main__":
    main()
