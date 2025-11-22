# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from train_utils import get_paths, load_corpus, save_json, SEED


def main():
    base_dir, _, models_dir, _ = get_paths()
    df = load_corpus()
    if df.empty:
        raise RuntimeError("No se cargaron datos. Revisa los .xlsx en la carpeta data.")

    X = df["text"].values
    y = df["label"].values.astype(int)

    # Pipeline base
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ))
    ])

    # Espacio de búsqueda (moderado para no morir en CV)
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_df": [0.85, 0.9, 0.95],
        "clf__C": [0.3, 1.0, 3.0, 10.0],
    }

    print("[INFO] Iniciando GridSearchCV...")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X, y)

    print("[INFO] Mejores parámetros encontrados:")
    print(grid.best_params_)
    print(f"[INFO] Mejor f1_macro CV: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    # Evaluación rápida en el propio entrenamiento (no hay hold-out aquí,
    # porque ya usamos CV; es solo para guardar un reporte consistente)
    y_pred = best_model.predict(X)
    report = classification_report(y, y_pred, target_names=["True(0)", "Fake(1)"], output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()
    f1m = f1_score(y, y_pred, average="macro")

    # Guardar modelo tunado
    out_path = models_dir / "model_tfidf_best.pkl"
    joblib.dump(best_model, out_path)

    metrics_path = models_dir / "model_tfidf_best_metrics.json"
    save_json(metrics_path, {
        "cv_best_params": grid.best_params_,
        "cv_best_f1_macro": grid.best_score_,
        "train_f1_macro": f1m,
        "train_classification_report": report,
        "train_confusion_matrix": cm,
    })

    print(f"[OK] Modelo TF-IDF tunado guardado en: {out_path}")
    print(f"[OK] Métricas CV + train guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
