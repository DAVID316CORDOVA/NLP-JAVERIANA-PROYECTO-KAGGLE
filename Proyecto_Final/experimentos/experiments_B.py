# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

# --- HACK para poder hacer "from train_utils import ..." estando en /experimentos ---
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer

from train_utils import load_corpus, SEED


N_EXPERIMENTS = 10
TEST_SIZE = 0.15  # 15% estratificado para test
EMB_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"
# Puedes cambiarlo por otro modelo multilingual de sentence-transformers si quieres.


def run_experiments_B1():
    # ====== Cargar corpus ======
    df = load_corpus()
    if df.empty:
        raise RuntimeError("No se cargaron datos en load_corpus().")

    X_text = df["text"].values
    y = df["label"].values.astype(int)

    print(f"[INFO] Total de instancias (train+dev): {len(X_text)}")
    print(f"[INFO] Distribución de etiquetas: True(0)={np.sum(y==0)}, Fake(1)={np.sum(y==1)}")
    print(f"[INFO] Experimentos: {N_EXPERIMENTS}, test_size = {TEST_SIZE}")

    # ====== Cargar modelo SBERT y precomputar embeddings ======
    print(f"[INFO] Cargando modelo SBERT: {EMB_MODEL_NAME}")
    sbert = SentenceTransformer(EMB_MODEL_NAME)

    print("[INFO] Calculando embeddings para todo el corpus (una sola vez)...")
    X_emb = sbert.encode(
        list(X_text),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    X_emb = np.array(X_emb, dtype=np.float32)

    # ====== Definir split estratificado repetido ======
    splitter = StratifiedShuffleSplit(
        n_splits=N_EXPERIMENTS,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    results_B1 = []

    exp_id = 0
    for train_idx, test_idx in splitter.split(X_emb, y):
        exp_id += 1
        print("\n" + "=" * 70)
        print(f"=== EXPERIMENTO B1 {exp_id}/{N_EXPERIMENTS} (SBERT + LogReg) ===")
        print("=" * 70)

        X_tr = X_emb[train_idx]
        X_te = X_emb[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=SEED + exp_id,  # pequeña variación por experimento
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        f1 = f1_score(y_te, y_pred, average="macro")
        acc = accuracy_score(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred)

        results_B1.append({
            "f1_macro": f1,
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
        })

        print("[B1 - SBERT+LogReg] f1_macro = {:.4f}, accuracy = {:.4f}".format(f1, acc))
        print("[B1 - SBERT+LogReg] Matriz de confusión (True(0) filas, Fake(1) filas):")
        print(cm)

    # ====== Resumen final ======
    f1s = np.array([r["f1_macro"] for r in results_B1])
    accs = np.array([r["accuracy"] for r in results_B1])

    print("\n" + "#" * 70)
    print(f"### RESUMEN B1 - SBERT + LogReg sobre {N_EXPERIMENTS} experimentos ###")
    print("#" * 70)
    print("f1_macro: media = {:.4f}, std = {:.4f}".format(f1s.mean(), f1s.std()))
    print("accuracy: media = {:.4f}, std = {:.4f}".format(accs.mean(), accs.std()))


if __name__ == "__main__":
    run_experiments_B1()
