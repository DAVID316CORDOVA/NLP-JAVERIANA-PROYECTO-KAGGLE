# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import json
import pandas as pd

SEED = 42


def get_paths() -> Tuple[Path, Path, Path, Path]:
    """
    base_dir  -> carpeta donde vive este archivo (Proyecto_Final)
    data_dir  -> Proyecto_Final/data
    models_dir-> Proyecto_Final/models
    api_dir   -> Proyecto_Final/api
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    api_dir = base_dir / "api"

    models_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, data_dir, models_dir, api_dir


def _read_xl(fp: Path) -> pd.DataFrame:
    """
    Lee un .xlsx si existe, imprime info de debug.
    """
    if not fp.exists():
        print(f"[WARN] Archivo NO encontrado: {fp}")
        return pd.DataFrame()
    try:
        print(f"[INFO] Leyendo archivo: {fp}")
        df = pd.read_excel(fp)
        print(f"[INFO] -> cargadas {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"[ERROR] Falló la lectura de {fp}: {e}")
        return pd.DataFrame()


def load_corpus() -> pd.DataFrame:
    """
    Lee train.xlsx, development.xlsx y test.xlsx del directorio data.
    Une train y dev para entrenar y devuelve un DataFrame con:
        text: headline + text
        label: 0 (True), 1 (Fake)
    """
    _, data_dir, _, _ = get_paths()
    print(f"[INFO] Carpeta data: {data_dir}")
    print(f"[INFO] Contenido de data/: {[p.name for p in data_dir.glob('*')]}")

    train_fp = data_dir / "train.xlsx"
    dev_fp   = data_dir / "development.xlsx"
    test_fp  = data_dir / "test.xlsx"  # si se llamara diferente, aquí habría que cambiarlo

    df_train = _read_xl(train_fp)
    df_dev   = _read_xl(dev_fp)
    df_test  = _read_xl(test_fp)

    def normalize(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if df.empty:
            print(f"[WARN] DataFrame vacío para {name}, se omite.")
            return df

        # mapeo de nombres de columnas a minúsculas
        cols_lower = {c.lower(): c for c in df.columns}

        # headline
        headline_col = cols_lower.get("headline")
        if headline_col is None:
            print(f"[WARN] No se encontró columna 'Headline' en {name}, se usa cadena vacía.")
            headline = pd.Series([""] * len(df))
        else:
            headline = df[headline_col].fillna("")

        # text
        text_col = cols_lower.get("text")
        if text_col is None:
            print(f"[WARN] No se encontró columna 'Text' en {name}, se usa cadena vacía.")
            text = pd.Series([""] * len(df))
        else:
            text = df[text_col].fillna("")

        text_final = (headline.astype(str) + "\n" + text.astype(str)).str.strip()

        # categoría
        cat_col = cols_lower.get("category")
        if cat_col is None:
            raise ValueError(f"No se encontró la columna 'Category' en {name}. Columnas: {df.columns.tolist()}")
        label = df[cat_col].astype(str).str.lower().map({"true": 0, "fake": 1})

        out = pd.DataFrame({"text": text_final, "label": label})
        out = out.dropna().reset_index(drop=True)
        print(f"[INFO] Normalizado {name}: {out.shape[0]} filas")
        return out

    tr = normalize(df_train, "train") if not df_train.empty else pd.DataFrame(columns=["text", "label"])
    dv = normalize(df_dev, "development") if not df_dev.empty else pd.DataFrame(columns=["text", "label"])
    te = normalize(df_test, "test") if not df_test.empty else pd.DataFrame(columns=["text", "label"])

    # Limpieza de texto
    def clean_text(s: pd.Series) -> pd.Series:
        s = s.str.lower()
        s = s.str.replace(r"http\S+|www\.\S+", " ", regex=True)
        s = s.str.replace(r"\*number\*", " ", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        return s

    for df in (tr, dv, te):
        if not df.empty:
            df["text"] = clean_text(df["text"])

    train_dev = pd.concat([tr, dv], axis=0).reset_index(drop=True)
    print(f"[INFO] Total filas train+dev: {train_dev.shape[0]}")

    if train_dev.empty:
        print("[ERROR] train+dev está vacío. Revisa nombres de archivos y columnas.")
    return train_dev


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
