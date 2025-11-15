# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from train_utils import get_paths, load_corpus, save_json, SEED

def main():
    base_dir, _, models_dir, _ = get_paths()
    df = load_corpus()
    if df.empty:
        raise RuntimeError("No se cargaron datos. Revisa los .xlsx en la carpeta data.")

    X = df["text"].values
    y = df["label"].values.astype(int)

    # split interno para validación (dejamos test.xlsx para evaluación final si quieres)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # Pipeline TF-IDF + LogReg (probability)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        ))
    ])

    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    y_proba = pipe.predict_proba(X_va)[:, 1]

    report = classification_report(y_va, y_pred, target_names=["True(0)", "Fake(1)"], output_dict=True)
    cm = confusion_matrix(y_va, y_pred).tolist()
    f1m = f1_score(y_va, y_pred, average="macro")

    # guardar modelo completo como .pkl
    out_path = models_dir / "model_tfidf_logreg.pkl"
    joblib.dump(pipe, out_path)

    # guardar métricas
    metrics_path = models_dir / "model_tfidf_logreg_metrics.json"
    save_json(metrics_path, {
        "f1_macro": f1m,
        "classification_report": report,
        "confusion_matrix": cm
    })

    print(f"[OK] Modelo 1 guardado en: {out_path}")
    print(f"[OK] Métricas guardadas en: {metrics_path}")

if __name__ == "__main__":
    main()
