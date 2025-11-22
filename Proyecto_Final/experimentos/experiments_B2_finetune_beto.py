# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

# --- HACK para poder importar train_utils desde /experimentos ---
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))

import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from train_utils import load_corpus, get_paths, SEED


MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
TEST_SIZE = 0.15  # 15% estratificado para validación


class NewsDataset(Dataset):
    """
    Dataset simple para usar con HuggingFace Trainer.
    Recibe 'encodings' (dict de tensores) y etiquetas.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_texts(tokenizer, texts, max_length=256):
    """
    Tokeniza una lista de textos para BERT.
    """
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return enc


def main():
    # ====== Cargar corpus ======
    base_dir, data_dir, models_dir, api_dir = get_paths()
    print(f"[INFO] base_dir = {base_dir}")
    print(f"[INFO] models_dir = {models_dir}")

    df = load_corpus()
    if df.empty:
        raise RuntimeError("No se cargaron datos en load_corpus().")

    X = df["text"].values
    y = df["label"].values.astype(int)

    print(f"[INFO] Total instancias: {len(X)}")
    print(f"[INFO] Distribución etiquetas: True(0)={np.sum(y==0)}, Fake(1)={np.sum(y==1)}")

    # ====== Partición estratificada 85% / 15% ======
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    train_idx, val_idx = next(splitter.split(X, y))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"[INFO] Tamaño train: {len(X_train)}, tamaño val: {len(X_val)}")

    # ====== Cargar tokenizer y modelo BETO ======
    print(f"[INFO] Cargando tokenizer y modelo: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # ====== Tokenizar ======
    print("[INFO] Tokenizando textos...")
    train_encodings = tokenize_texts(tokenizer, X_train, max_length=256)
    val_encodings = tokenize_texts(tokenizer, X_val, max_length=256)

    train_dataset = NewsDataset(train_encodings, y_train)
    val_dataset = NewsDataset(val_encodings, y_val)

    # ====== Definir función de métricas ======
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
        }

    # ====== Configuración de entrenamiento (CPU-friendly) ======
    output_dir = models_dir / "beto_finetuned_fake_news"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    no_cuda=True
    )

    # ====== Trainer ======
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ====== Entrenamiento ======
    print("[INFO] Comenzando entrenamiento en CPU (puede tardar un rato)...")
    trainer.train()

    # ====== Evaluación final en el conjunto de validación ======
    print("[INFO] Evaluando en el conjunto de validación (15%)...")
    preds_output = trainer.predict(val_dataset)
    logits = preds_output.predictions
    y_true = preds_output.label_ids
    y_pred = np.argmax(logits, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred,
        target_names=["True(0)", "Fake(1)"],
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    print("\n" + "=" * 70)
    print("=== RESULTADOS FINALES BETO FINETUNED (VALIDACIÓN 15%) ===")
    print("=" * 70)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-macro:  {f1_macro:.4f}")
    print("Matriz de confusión (True(0) filas, Fake(1) filas):")
    print(np.array(cm))

    # ====== Guardar modelo y tokenizer ======
    print(f"[INFO] Guardando modelo y tokenizer en: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ====== Guardar métricas en JSON ======
    import json
    metrics_path = output_dir / "metrics_beto_finetuned_val.json"
    payload = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
