# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Sequence, Tuple, Optional, List

import numpy as np

# Intentamos cargar spaCy en español de forma opcional
try:
    import spacy
    try:
        _NLP_ES = spacy.load("es_core_news_sm")
    except OSError:
        # Modelo no instalado; se puede instalar con:
        # python -m spacy download es_core_news_sm
        print("[WARN] spaCy 'es_core_news_sm' no está instalado. Se usarán solo rasgos básicos.")
        _NLP_ES = None
except ImportError:
    print("[WARN] spaCy no instalado. Se usarán solo rasgos estilométricos básicos.")
    _NLP_ES = None


def _basic_features_for_text(text: str) -> List[float]:
    """
    Rasgos estilométricos básicos, independientes de spaCy:
      - avg_token_len
      - std_token_len
      - n_tokens
      - n_sentences (aprox)
      - avg_sentence_len_tokens
      - punctuation_ratio
      - uppercase_ratio
      - digit_ratio
      - exclamation_ratio
      - question_ratio
      - type_token_ratio
    """
    if text is None:
        text = ""
    text = str(text)

    chars = len(text)
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    n_tokens = len(tokens)

    if n_tokens > 0:
        token_lens = [len(tok) for tok in tokens]
        avg_token_len = float(np.mean(token_lens))
        std_token_len = float(np.std(token_lens))
        ttr = float(len(set(tokens)) / n_tokens)
    else:
        avg_token_len = 0.0
        std_token_len = 0.0
        ttr = 0.0

    # oraciones aproximadas por . ! ?
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    n_sentences = len(sentences)
    if n_sentences > 0 and n_tokens > 0:
        avg_sent_len = float(n_tokens / n_sentences)
    else:
        avg_sent_len = float(n_tokens)

    punct_chars = ".,;:!?\"'()[]{}¿¡-"
    n_punct = sum(1 for c in text if c in punct_chars)
    punctuation_ratio = float(n_punct) / max(chars, 1)

    n_upper = sum(1 for c in text if c.isupper())
    n_letters = sum(1 for c in text if c.isalpha())
    uppercase_ratio = float(n_upper) / max(n_letters, 1)

    n_digits = sum(1 for c in text if c.isdigit())
    digit_ratio = float(n_digits) / max(chars, 1)

    n_excl = text.count("!") + text.count("¡")
    exclamation_ratio = float(n_excl) / max(chars, 1)

    n_ques = text.count("?") + text.count("¿")
    question_ratio = float(n_ques) / max(chars, 1)

    return [
        avg_token_len,
        std_token_len,
        float(n_tokens),
        float(n_sentences),
        avg_sent_len,
        punctuation_ratio,
        uppercase_ratio,
        digit_ratio,
        exclamation_ratio,
        question_ratio,
        ttr,
    ]


def _spacy_features_for_texts(texts: Sequence[str]) -> np.ndarray:
    """
    Rasgos adicionales usando spaCy (si está disponible):
      - conteo normalizado de POS: NOUN, VERB, ADJ, ADV, PROPN, PRON
      - conteo normalizado de entidades: PER, ORG, LOC, MISC
    Si spaCy o el modelo no están disponibles, devuelve matriz de ceros.
    """
    if _NLP_ES is None:
        # devolvemos matriz de ceros
        return np.zeros((len(texts), 10), dtype=float)

    pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON"]
    ent_labels = ["PER", "ORG", "LOC", "MISC"]
    n_pos = len(pos_tags)
    n_ent = len(ent_labels)

    feats = np.zeros((len(texts), n_pos + n_ent), dtype=float)

    for i, doc in enumerate(_NLP_ES.pipe(texts, batch_size=16)):
        # POS
        total_tokens = len(doc)
        for token in doc:
            if token.pos_ in pos_tags:
                j = pos_tags.index(token.pos_)
                feats[i, j] += 1.0
        if total_tokens > 0:
            feats[i, :n_pos] /= float(total_tokens)

        # entidades
        total_ents = len(doc.ents)
        for ent in doc.ents:
            if ent.label_ in ent_labels:
                k = ent_labels.index(ent.label_)
                feats[i, n_pos + k] += 1.0
        if total_ents > 0:
            feats[i, n_pos:] /= float(total_ents)

    return feats


def compute_stylometric_features(texts: Sequence[str]) -> np.ndarray:
    """
    Función principal: recibe una lista/array de textos y devuelve una matriz
    de rasgos estilométricos (básicos + opcionales spaCy).

    shape = (n_texts, n_features)
    """
    basic = np.array([_basic_features_for_text(t) for t in texts], dtype=float)
    spacy_f = _spacy_features_for_texts(texts)

    # concatenamos
    return np.hstack([basic, spacy_f])
