# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

# Add parent folder to import train_utils
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix, hstack

from train_utils import load_corpus, SEED
from utils_stylometry import compute_stylometric_features

N_EXPERIMENTS = 10
TEST_SIZE = 0.15  # 15% estratificado para test


# ================== MODELOS A1, A2, A3 ==================

def build_model_A1() -> Pipeline:
    """
    A1: TF-IDF (1,2) + Logistic Regression
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=SEED,
        )),
    ])
    return pipe


def build_model_A2() -> Pipeline:
    """
    A2: CountVectorizer (1,2) + MultinomialNB
    """
    pipe = Pipeline([
        ("vec", CountVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            lowercase=True,
        )),
        ("clf", MultinomialNB(alpha=0.5)),
    ])
    return pipe


def train_eval_A3(X_text, y, train_idx, test_idx):
    """
    A3: TF-IDF + rasgos estilométricos combinados + Logistic Regression.

    1) Ajusta TF-IDF en X_train.
    2) Obtiene matriz TF-IDF para train y test.
    3) Calcula rasgos estilométricos para train y test.
    4) Concatena [TF-IDF | stylometry] y entrena LogisticRegression.
    """
    # textos
    X_tr_text = X_text[train_idx]
    X_te_text = X_text[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    # TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        lowercase=True,
    )
    X_tr_tfidf = tfidf.fit_transform(X_tr_text)
    X_te_tfidf = tfidf.transform(X_te_text)

    # Rasgos estilométricos
    X_tr_sty = compute_stylometric_features(X_tr_text)
    X_te_sty = compute_stylometric_features(X_te_text)

    # Convertir rasgos a sparse e hstack
    X_tr_sty_sp = csr_matrix(X_tr_sty)
    X_te_sty_sp = csr_matrix(X_te_sty)

    X_tr_final = hstack([X_tr_tfidf, X_tr_sty_sp])
    X_te_final = hstack([X_te_tfidf, X_te_sty_sp])

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=SEED,
    )
    clf.fit(X_tr_final, y_tr)
    y_pred = clf.predict(X_te_final)

    f1 = f1_score(y_te, y_pred, average="macro")
    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)

    return {
        "f1_macro": f1,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
    }


# ================== LOOP DE EXPERIMENTOS ==================

def run_experiments_A():
    df = load_corpus()
    if df.empty:
        raise RuntimeError("No se cargaron datos en load_corpus().")

    X_text = df["text"].values
    y = df["label"].values.astype(int)

    print(f"[INFO] Total de instancias (train+dev): {len(X_text)}")
    print(f"[INFO] Distribución de etiquetas: True(0)={np.sum(y==0)}, Fake(1)={np.sum(y==1)}")
    print(f"[INFO] Experimentos: {N_EXPERIMENTS}, test_size = {TEST_SIZE}")

    splitter = StratifiedShuffleSplit(
        n_splits=N_EXPERIMENTS,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    results_A1 = []
    results_A2 = []
    results_A3 = []

    exp_id = 0
    for train_idx, test_idx in splitter.split(X_text, y):
        exp_id += 1
        print("\n" + "=" * 70)
        print(f"=== EXPERIMENTO {exp_id}/{N_EXPERIMENTS} (15% test estratificado) ===")
        print("=" * 70)

        X_tr = X_text[train_idx]
        X_te = X_text[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        # ---------- A1: TF-IDF + LogReg ----------
        model_A1 = build_model_A1()
        model_A1.fit(X_tr, y_tr)
        y_pred_A1 = model_A1.predict(X_te)

        f1_A1 = f1_score(y_te, y_pred_A1, average="macro")
        acc_A1 = accuracy_score(y_te, y_pred_A1)
        cm_A1 = confusion_matrix(y_te, y_pred_A1)

        results_A1.append({
            "f1_macro": f1_A1,
            "accuracy": acc_A1,
            "confusion_matrix": cm_A1.tolist(),
        })

        print("[A1 - TFIDF+LogReg] f1_macro = {:.4f}, accuracy = {:.4f}".format(f1_A1, acc_A1))
        print("[A1 - TFIDF+LogReg] Matriz de confusión (True(0) filas, Fake(1) filas):")
        print(cm_A1)

        # ---------- A2: CountVectorizer + MultinomialNB ----------
        model_A2 = build_model_A2()
        model_A2.fit(X_tr, y_tr)
        y_pred_A2 = model_A2.predict(X_te)

        f1_A2 = f1_score(y_te, y_pred_A2, average="macro")
        acc_A2 = accuracy_score(y_te, y_pred_A2)
        cm_A2 = confusion_matrix(y_te, y_pred_A2)

        results_A2.append({
            "f1_macro": f1_A2,
            "accuracy": acc_A2,
            "confusion_matrix": cm_A2.tolist(),
        })

        print("[A2 - Count+NB]   f1_macro = {:.4f}, accuracy = {:.4f}".format(f1_A2, acc_A2))
        print("[A2 - Count+NB]   Matriz de confusión (True(0) filas, Fake(1) filas):")
        print(cm_A2)

        # ---------- A3: TF-IDF + Stylometry + LogReg ----------
        res_A3 = train_eval_A3(X_text, y, train_idx, test_idx)
        results_A3.append(res_A3)

        print("[A3 - TFIDF+Sty]  f1_macro = {:.4f}, accuracy = {:.4f}".format(
            res_A3["f1_macro"], res_A3["accuracy"]
        ))
        print("[A3 - TFIDF+Sty]  Matriz de confusión (True(0) filas, Fake(1) filas):")
        print(np.array(res_A3["confusion_matrix"]))

    # ================== RESÚMENES ==================
    def summarize(results, name: str):
        f1s = np.array([r["f1_macro"] for r in results])
        accs = np.array([r["accuracy"] for r in results])
        print("\n" + "#" * 70)
        print(f"### RESUMEN {name} sobre {N_EXPERIMENTS} experimentos ###")
        print("#" * 70)
        print("f1_macro: media = {:.4f}, std = {:.4f}".format(f1s.mean(), f1s.std()))
        print("accuracy: media = {:.4f}, std = {:.4f}".format(accs.mean(), accs.std()))

    summarize(results_A1, "A1 - TFIDF + LogReg")
    summarize(results_A2, "A2 - CountVectorizer + MultinomialNB")
    summarize(results_A3, "A3 - TFIDF + Stylometry + LogReg")


if __name__ == "__main__":
    run_experiments_A()
