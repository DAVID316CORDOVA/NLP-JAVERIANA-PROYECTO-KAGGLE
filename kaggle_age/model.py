# ======================= AGE-RANGE SVM: ONE-CELL, RICH FEATURES =======================
# Stages:
#   A) Baseline (LinearSVC sobre features ricos precomputados)
#   B) HalvingGridSearch rápido (pilot de 15% del train)
#   C) HalvingGridSearch "profundo" centrado en el mejor modelo de Stage B
#   -> Genera 3 archivos de submission y un resumen en consola
# ======================================================================================

import os, re, time, json, math, warnings
import numpy as np
import pandas as pd
from collections import Counter

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- helper para logging -----------------------------
def log(msg):
    """Imprime mensajes con timestamp para seguir el flujo del pipeline."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ----------------------------- hiperparámetros globales -----------------------------
# Máximo de features para los distintos canales de TF-IDF
WORD_MAX_FEAT   = 50000   # palabras (uni+bi)
CHAR_MAX_FEAT   = 12000   # n-gramas de caracteres
HASH_MAX_FEAT   = 4000    # hashtags
MENT_MAX_FEAT   = 4000    # menciones

# Parámetros para seleccionar términos salientes por rango de edad
AGE_TOP_MIN_DF  = 10      # frecuencia mínima de documento para candidatos léxicos
AGE_TOP_K1      = 80      # top-K unigrams por edad
AGE_TOP_K2      = 40      # top-K bigrams por edad

# Número de menciones/hashtags “populares” a considerar
TOPK_MENTIONS   = 1000
TOPK_HASHTAGS   = 1000


# ===================== 1) CARGA DE DATOS =====================
log("Loading data ...")
train = pd.read_csv("age_train.csv", sep=";")
test  = pd.read_csv("age_test.csv",  sep=";")

# Aseguramos que la columna de texto no tenga NaNs
train["text"] = train["text"].fillna("")
test["text"]  = test["text"].fillna("")
log(f"train: {train.shape} | test: {test.shape}")

# Lista de etiquetas de edad (rangos)
ages = train["age_range"].unique().tolist()


# ===================== 2) MANEJO DE EMOJIS =====================
# Intentamos usar la librería 'emoji'; si no está, devolvemos conteo 0
try:
    import emoji
    EMOJI_SET = set(emoji.EMOJI_DATA.keys())

    def count_emojis(s):
        """Cuenta cuántos caracteres en el texto son emojis conocidos."""
        return sum(1 for ch in str(s) if ch in EMOJI_SET)

    log("Emoji lib available ")

except Exception:
    EMOJI_SET = set()

    def count_emojis(s):
        """Fallback: si no tenemos librería emoji, devolvemos siempre 0."""
        return 0

    log("Emoji lib NOT available; emoji_count=0 ⚠")


# ===================== 3) NORMALIZACIÓN (aware de Twitter) =====================
# Lista básica de stopwords en español, usada en varios vectorizadores
SPANISH_STOP = [
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
    "con","no","una","su","al","lo","como","mas","pero","sus","le","ya","o","este",
    "si","porque","esta","entre","cuando","muy","sin","sobre","tambien","me","hasta",
    "hay","donde","quien","desde","todo","nos","durante","todos","uno","les","ni",
    "contra","otros","fueron","ese","eso","habia","ante","e","esto","mi","antes","algunos",
    "que","tu","te","yo"
]

# Patrones de regex para URLs, menciones y hashtags
url_pat     = re.compile(r"(https?://\S+|www\.\S+)")
mention_pat = re.compile(r"@\w+")
hashtag_pat = re.compile(r"#\w+")

# Lista de TLDs de interés (para extraer info de países o dominios web)
TLD_LIST = ["co","mx","ar","es","us","cl","pe","ve","uy","ec","bo","pr","cr","pa","hn","ni","sv","gt","do"]


def english_ratio(t: str) -> float:
    """
    Calcula la proporción de letras 'a-z' (ASCII) sobre el total de letras en el texto.
    Idea: tweets con alto ratio pueden implicar contenido en inglés.
    """
    letters = [c for c in str(t) if c.isalpha()]
    return 0.0 if not letters else sum('a' <= c <= 'z' for c in letters) / len(letters)


def extract_domain_and_tld(url_text: str):
    """
    Dada una URL, extrae el host (dominio) y el TLD (co, es, com, etc.).
    Además, limpiamos 'www.' para quedarnos con la parte relevante.
    """
    u = re.sub(r"^https?://", "", url_text)
    host = u.split("/")[0].lower()
    host = host.replace("www.","")
    tld_bits = host.split(".")
    tld = tld_bits[-1] if tld_bits else ""
    return host, tld


def normalize_tweet(text: str) -> str:
    """
    Normalización 'twitter-aware':
      - Pasa a minúsculas.
      - Colapsa secuencias largas de menciones (@user1 @user2 @user3 ...) a las dos primeras.
      - Reemplaza URLs por un token 'URLTOKEN' e inyecta tokens url_<dominio> y tld_<tld>.
      - Separa '@' y '#' como tokens explícitos para TF-IDF.
    """
    t = str(text).lower().strip()

    # Reducimos ruido de muchas menciones seguidas: dejamos solo 2
    t = re.sub(
        r"(?:@\w+\s+){3,}",
        lambda m: " ".join(m.group(0).split()[:2]) + " ",
        t
    )

    domains = []
    tlds    = []

    # Función helper que se usará en la sustitución de URLs
    def _repl(m):
        u = m.group(0)
        dom, tld = extract_domain_and_tld(u)
        if dom:
            # Convertimos dominio en un token estable tipo 'url_twitter_com'
            domains.append("url_" + re.sub(r"[^a-z0-9]+","_", dom))
        if tld and tld in TLD_LIST:
            # Marcador de TLD específico, p.ej. tld_co, tld_es
            tlds.append("tld_" + tld)
        return " URLTOKEN "

    # Sustituimos URLs por el token y recogemos dominios/TLDs
    t = url_pat.sub(_repl, t)

    # Forzamos que '@' y '#' queden separados para que el vectorizador los identifique
    t = t.replace("@", " @").replace("#", " #")

    # Colapsamos espacios múltiples
    t = re.sub(r"\s+", " ", t).strip()

    # Inyectamos tokens de dominio y TLD al final del texto
    if domains:
        t += " " + " ".join(domains)
    if tlds:
        t += " " + " ".join(tlds)

    # Si por alguna razón queda vacío, devolvemos un token de seguridad
    return t if t else "empty_tweet"


log("Normalizing tweets ...")
train["clean_text"] = train["text"].apply(normalize_tweet)
test["clean_text"]  = test["text"].apply(normalize_tweet)
log(f"[AUDIT] empty clean: train={(train['clean_text'].str.strip()=='' ).sum()} | test={(test['clean_text'].str.strip()=='' ).sum()}")


# ===================== 4) TOP-K GLOBAL DE MENCIONES/HASHTAGS =====================
def collect_topk(series, pattern, topk):
    """
    Recorre una serie de textos, cuenta los tokens que matchean 'pattern'
    y devuelve el conjunto de los 'topk' más frecuentes.
    """
    cnt = Counter()
    for s in series:
        cnt.update(pattern.findall(str(s)))
    return {tok for tok, _ in cnt.most_common(topk)}


log("Collecting Top-K mentions/hashtags ...")
top_mentions = collect_topk(train["text"], mention_pat, TOPK_MENTIONS)
top_hashtags = collect_topk(train["text"], hashtag_pat, TOPK_HASHTAGS)
log(f"Top mentions: {len(top_mentions)} | Top hashtags: {len(top_hashtags)}")


def bucket_mention_hashtag(text):
    """
    Separa menciones/hashtags en:
      - famosos (aparecen en el top global)
      - raros (no están en el top)
    para capturar el "tipo" de interacción del usuario.
    """
    s = str(text)
    ms = mention_pat.findall(s)
    hs = hashtag_pat.findall(s)
    m_famous = sum(1 for m in ms if m in top_mentions)
    m_rare   = len(ms) - m_famous
    h_famous = sum(1 for h in hs if h in top_hashtags)
    h_rare   = len(hs) - h_famous
    return m_famous, m_rare, h_famous, h_rare


# ===================== 5) LÉXICOS SALIENTES POR EDAD (log-odds) =====================
log("Computing age-salient lexicons (log-odds with prior, min_df filtering) ...")

# Contamos unigrams y bigrams con min_df para no incluir términos ultra-escasos
cv_uni = CountVectorizer(
    ngram_range=(1,1),
    stop_words=SPANISH_STOP,
    min_df=AGE_TOP_MIN_DF,
    max_features=60000
)
cv_bi  = CountVectorizer(
    ngram_range=(2,2),
    stop_words=SPANISH_STOP,
    min_df=AGE_TOP_MIN_DF,
    max_features=60000
)

Xu = cv_uni.fit_transform(train["clean_text"])
Xb = cv_bi.fit_transform(train["clean_text"])
Vu = np.array(cv_uni.get_feature_names_out())
Vb = np.array(cv_bi.get_feature_names_out())


def log_odds_top(X, vocab, labels, which_age, alpha=0.5, top_k=50):
    """
    Calcula, para una edad concreta 'which_age', los términos más característicos
    usando log-odds con suavizado (alpha).
    Devuelve un conjunto de las top_k palabras/frases más salientes.
    """
    m = (labels == which_age)         # máscara: docs de esa edad
    A = X[m].sum(axis=0).A1 + alpha   # frecuencia + suavizado para esa edad
    B = X[~m].sum(axis=0).A1 + alpha  # frecuencia + suavizado para el resto
    s = np.log(A/A.sum()) - np.log(B/B.sum())  # score de log-odds
    return set(vocab[np.argsort(s)[::-1][:top_k]])


labels = train["age_range"].values
age_lex_uni = {
    a: log_odds_top(Xu, Vu, labels, a, alpha=0.5, top_k=AGE_TOP_K1)
    for a in ages
}
age_lex_bi  = {
    a: log_odds_top(Xb, Vb, labels, a, alpha=0.5, top_k=AGE_TOP_K2)
    for a in ages
}

# Vista rápida de algunas palabras salientes por edad
preview = {a: sorted(list(list(age_lex_uni[a])[:6])) for a in ages}
log(f"Age-salient (uni) preview: {json.dumps(preview, ensure_ascii=False)}")

# Compilamos regex por edad para contar ocurrencias de términos salientes
AGELEX_REGEX = {}
for a in ages:
    words = [re.escape(w) for w in sorted(list(age_lex_uni[a] | age_lex_bi[a])) if w.strip()]
    AGELEX_REGEX[a] = (
        re.compile(r"\b(?:" + "|".join(words) + r")\b", flags=re.IGNORECASE)
        if words else re.compile(r"$a")  # regex imposible si no hay palabras
    )


# ===================== 6) FEATURES NUMÉRICOS ESTRUCTURADOS =====================
# Listas de dominios típicos por categoría (social, video, news, e-commerce)
SOCIAL  = ["twitter","tiktok","instagram","facebook","fb","threads","x_com","x"]
VIDEO   = ["youtube","youtu_be","twitch"]
NEWS    = ["cnn","bbc","nytimes","washingtonpost","elpais","eltiempo","guardian","reuters"]
ECOM    = ["amazon","mercadolibre","ebay","aliexpress"]


def domain_buckets(text: str):
    """
    Cuenta cuántos dominios de tipo social / video / news / ecom / otros
    aparecen en los tokens 'url_<dominio>' del texto normalizado.
    """
    t = str(text).lower()
    doms = re.findall(r"url_([a-z0-9_]+)", t)
    out = {"url_social":0,"url_video":0,"url_news":0,"url_ecom":0,"url_other":0}
    for d in doms:
        if any(k in d for k in SOCIAL):  out["url_social"] += 1
        elif any(k in d for k in VIDEO): out["url_video"]  += 1
        elif any(k in d for k in NEWS):  out["url_news"]   += 1
        elif any(k in d for k in ECOM):  out["url_ecom"]   += 1
        else:                            out["url_other"]  += 1
    return out


def build_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un dataframe de features numéricos a partir del texto original y del clean_text:
      - longitud de texto (caracteres/palabras)
      - conteo de menciones / hashtags / URLs
      - RT, empieza con @
      - conteo de emojis
      - buckets de menciones/hashtags famosos vs raros
      - buckets de dominios por tipo (social/video/news/ecom/other)
      - english_ratio, proporción de puntuación, dígitos, signos, etc.
      - conteos de términos salientes por edad (agelex_*)
      - flags por TLD inyectados (tld_xx)
    """
    d = df.copy()

    # Longitud y conteos elementales
    d["char_count"]    = d["text"].str.len()
    d["word_count"]    = d["clean_text"].str.split().str.len()
    d["mention_count"] = d["text"].str.count(r"@\w+")
    d["hashtag_count"] = d["text"].str.count(r"#\w+")
    d["url_count"]     = d["text"].str.count(url_pat)

    # Marcadores tipo RT / si empieza con @
    d["is_rt"]         = d["text"].str.lower().str.startswith("rt ").fillna(False).astype(int)
    d["starts_with_at"]= d["text"].str.startswith("@").astype(int)

    # Emojis
    d["emoji_count"]   = d["text"].apply(count_emojis)

    # Buckets famosos/raros
    mf, mr, hf, hr = zip(*d["text"].apply(bucket_mention_hashtag))
    d["mention_famous"] = np.array(mf)
    d["mention_rare"]   = np.array(mr)
    d["hashtag_famous"] = np.array(hf)
    d["hashtag_rare"]   = np.array(hr)

    # Buckets de dominio (social/video/news/ecom/other)
    buckets = d["clean_text"].apply(domain_buckets).apply(pd.Series).fillna(0).astype(int)
    d = pd.concat([d, buckets], axis=1)

    # Razones y otros contadores
    d["english_ratio"] = d["text"].apply(english_ratio)
    d["punct_ratio"]   = d["text"].str.count(r"[.!?,;:]")/(d["char_count"]+1e-6)
    d["digit_count"]   = d["text"].str.count(r"\d")
    d["has_exclam"]    = d["text"].str.contains("!").astype(int)
    d["has_question"]  = d["text"].str.contains(r"\?|\u00bf").astype(int)
    d["ends_with_url"] = d["text"].str.contains(
        r"(?:https?://\S+|www\.\S+)\s*$", regex=True
    ).astype(int)

    # Conteos de léxico saliente por edad (uni+bi)
    for a, rg in AGELEX_REGEX.items():
        d["agelex_" + a.replace("-", "_")] = d["clean_text"].str.count(rg)

    # Conteos de tokens tld_* que inyectamos en clean_text
    for tld in TLD_LIST:
        token = f"tld_{tld}"
        d[token] = d["clean_text"].str.count(rf"\b{re.escape(token)}\b")

    return d


log("Building structured numeric features ...")
train_num = build_numeric(train)
test_num  = build_numeric(test)

# Columnas numéricas finales (excluimos metadatos y texto)
num_cols = [c for c in train_num.columns if c not in ["id","text","age_range","clean_text"]]
log(f"Numeric feature count: {len(num_cols)}")


# ===================== 7) CANALES DE TEXTO (TF-IDF, fit una vez) =====================
log("Vectorizing text channels ...")

# Canal de palabras (1-2 grams)
word_tfidf = TfidfVectorizer(
    max_features=WORD_MAX_FEAT,
    ngram_range=(1,2),
    stop_words=SPANISH_STOP,
    sublinear_tf=True,
    dtype=np.float32
)
# Canal de caracteres (3-5 chars)
char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    max_features=CHAR_MAX_FEAT,
    sublinear_tf=True,
    dtype=np.float32
)
# Hashtags como tokens separados
hash_tfidf = TfidfVectorizer(
    token_pattern=r"(?u)#\w+",
    lowercase=False,
    max_features=HASH_MAX_FEAT,
    dtype=np.float32
)
# Menciones como tokens separados
ment_tfidf = TfidfVectorizer(
    token_pattern=r"(?u)@\w+",
    lowercase=False,
    max_features=MENT_MAX_FEAT,
    dtype=np.float32
)

# Ajustamos en train y transformamos train/test
Xw_tr = word_tfidf.fit_transform(train["clean_text"])
Xc_tr = char_tfidf.fit_transform(train["clean_text"])
Xh_tr = hash_tfidf.fit_transform(train["clean_text"])
Xm_tr = ment_tfidf.fit_transform(train["clean_text"])

Xw_te = word_tfidf.transform(test["clean_text"])
Xc_te = char_tfidf.transform(test["clean_text"])
Xh_te = hash_tfidf.transform(test["clean_text"])
Xm_te = ment_tfidf.transform(test["clean_text"])

log(f"TF-IDF shapes train: word={Xw_tr.shape}, char={Xc_tr.shape}, hash={Xh_tr.shape}, ment={Xm_tr.shape}")


# ===================== 8) SIMILITUDES A CENTROIDES POR EDAD (word + char) =====================
log("Computing age-centroid cosine similarities (word+char) ...")

def centroid_rows(X, labels, ages_list):
    """
    Calcula el centroide (media de vectores TF-IDF) para cada edad.
    Devuelve un dict edad -> centroide (CSR 1 x vocab_size).
    """
    cents = {}
    for a in ages_list:
        m = (labels == a)
        cent_dense = np.asarray(X[m].mean(axis=0)).ravel()
        cents[a] = csr_matrix(cent_dense)
    return cents


def sims_from(X, cents, ages_list):
    """
    Para cada doc en X, calcula la similitud coseno con el centroide de cada edad.
    Devuelve una matriz (n_docs, n_edades) en formato sparse.
    """
    cols = []
    for a in ages_list:
        s = cosine_similarity(X, cents[a], dense_output=False)  # (n,1)
        cols.append(csr_matrix(s))
    return hstack(cols, format="csr")  # (n, len(ages))


age_cents_w = centroid_rows(Xw_tr, labels, ages)
age_cents_c = centroid_rows(Xc_tr, labels, ages)

Sw_tr = sims_from(Xw_tr, age_cents_w, ages)
Sw_te = sims_from(Xw_te, age_cents_w, ages)
Sc_tr = sims_from(Xc_tr, age_cents_c, ages)
Sc_te = sims_from(Xc_te, age_cents_c, ages)

log(f"Centroid sims (word,char) train: {Sw_tr.shape}, {Sc_tr.shape} | test: {Sw_te.shape}, {Sc_te.shape}")


# ===================== 9) ESCALAR NUMÉRICO + ENSAMBLAR MATRIZ FINAL =====================
log("Scaling numeric features & assembling final sparse matrices ...")

# Escalamos solo las features numéricas (StandardScaler sin centrar, para sparse)
scaler = StandardScaler(with_mean=False)
Xn_tr = scaler.fit_transform(csr_matrix(train_num[num_cols].values))
Xn_te = scaler.transform(csr_matrix(test_num[num_cols].values))

# Matriz final sparse combinando:
#   - TF-IDF word/char/hash/mention
#   - numéricas escaladas
#   - similitudes a centroides de edad (word+char)
X_tr = hstack(
    [Xw_tr, Xc_tr, Xh_tr, Xm_tr, Xn_tr, Sw_tr, Sc_tr],
    format="csr"
)
X_te = hstack(
    [Xw_te, Xc_te, Xh_te, Xm_te, Xn_te, Sw_te, Sc_te],
    format="csr"
)

y = train["age_range"].values
log(f"FINAL Shapes -> X_tr: {X_tr.shape} | X_te: {X_te.shape}")

# Split hold-out del 20% para evaluar localmente macro-F1
X_train, X_val, y_train, y_val = train_test_split(
    X_tr, y, test_size=0.2, random_state=42, stratify=y
)
log(f"Hold-out split -> X_train: {X_train.shape}, X_val: {X_val.shape}")


# ===================== 10) STAGE A: BASELINE =====================
log("Stage A: LinearSVC baseline (C=0.25, tol=1e-4) ...")

# OneVsRestClassifier envuelve LinearSVC para problema multi-clase
baseline = OneVsRestClassifier(
    LinearSVC(C=0.25, loss="squared_hinge", tol=1e-4),
    n_jobs=-1
)
baseline.fit(X_train, y_train)
yA = baseline.predict(X_val)

# Métricas hold-out
a_f1  = f1_score(y_val, yA, average="macro")
a_acc = accuracy_score(y_val, yA)

log("\n=== [Stage A] Baseline report ===")
print(classification_report(y_val, yA))
log(f"[Stage A] macro_f1={a_f1:.4f} acc={a_acc:.4f}")

# Ajustamos en todo el train y generamos submission
log("Fitting baseline on FULL train & writing CSV ...")
baseline_full = OneVsRestClassifier(
    LinearSVC(C=0.25, loss="squared_hinge", tol=1e-4),
    n_jobs=-1
)
baseline_full.fit(X_tr, y)
pd.DataFrame({
    "id": test["id"],
    "age_range": baseline_full.predict(X_te)
}).to_csv(
    "submission_stageA_baseline.csv",
    index=False,
    encoding="utf-8"
)
log("Wrote submission_stageA_baseline.csv")


# ===================== 11) STAGE B: HALVINGGRIDSEARCH (PILOT 15%) =====================
log("Stage B: HalvingGridSearch on 15% pilot ...")

# Tomamos ~15% de X_train para explorar hiperparámetros rápidamente
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.85, random_state=42)
for keep_idx, _ in sss.split(X_train, y_train):
    X_small = X_train[keep_idx]
    y_small = y_train[keep_idx]
log(f"Pilot split: {X_small.shape}")

# Grid pequeño de hiperparámetros para LinearSVC
param_grid_B = {
    "estimator__C":           [0.08, 0.10, 0.12, 0.15, 0.2],
    "estimator__tol":         [1e-4, 5e-4],
    "estimator__class_weight": [None, "balanced"],
}

# HalvingGridSearchCV reduce el número de candidatos en iteraciones sucesivas,
# entrenando con más recursos (más datos) a los modelos que van "ganando".
hsB = HalvingGridSearchCV(
    estimator=OneVsRestClassifier(LinearSVC(), n_jobs=-1),
    param_grid=param_grid_B,
    factor=3,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
    verbose=1
)

log("Fitting Stage B halving search ...")
hsB.fit(X_small, y_small)
log(f"[Stage B] best={hsB.best_params_} | cv_f1={hsB.best_score_:.4f}")


# Helper: extrae los parámetros del estimador interno "estimator__*"
def unwrap_estimator_params(best_params: dict):
    """
    Convierte claves del tipo 'estimator__C' en {'C': valor},
    para poder instanciar LinearSVC(**params).
    """
    return {
        k.split("estimator__",1)[1]: v
        for k,v in best_params.items()
        if k.startswith("estimator__")
    }


paramsB = unwrap_estimator_params(hsB.best_params_)
bestB = OneVsRestClassifier(LinearSVC(**paramsB), n_jobs=-1)

log("Refitting Stage B best on FULL train & evaluating on hold-out ...")
bestB.fit(X_tr, y)
yB = bestB.predict(X_val)

b_f1  = f1_score(y_val, yB, average="macro")
b_acc = accuracy_score(y_val, yB)

log("\n=== [Stage B] Report ===")
print(classification_report(y_val, yB))
log(f"[Stage B] macro_f1={b_f1:.4f} acc={b_acc:.4f}")

# Guardamos submission de Stage B
pd.DataFrame({
    "id": test["id"],
    "age_range": bestB.predict(X_te)
}).to_csv(
    "submission_stageB_halving.csv",
    index=False,
    encoding="utf-8"
)
log("Wrote submission_stageB_halving.csv")


# ===================== 12) STAGE C: HALVING "DEEP" ALREDEDOR DEL GANADOR B =====================
# Tomamos el mejor C de Stage B como punto central
c_star = hsB.best_params_["estimator__C"]

# Construimos un grid más fino alrededor de C*
gridC = sorted(set([
    max(0.05, c_star*0.6),
    max(0.05, c_star*0.75),
    max(0.05, c_star*0.9),
    c_star,
    min(3.0, c_star*1.1),
    min(3.0, c_star*1.25),
    min(3.0, c_star*1.5)
]))

param_grid_C = {
    "estimator__C":           gridC,
    "estimator__tol":         [1e-4, 5e-4, 1e-5],
    "estimator__class_weight": [None, "balanced"],
}

log(f"Stage C: deep grid around C*={c_star} -> {gridC}")

hsC = HalvingGridSearchCV(
    estimator=OneVsRestClassifier(LinearSVC(), n_jobs=-1),
    param_grid=param_grid_C,
    factor=2,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
    verbose=1
)

log("Fitting Stage C halving search ...")
hsC.fit(X_train, y_train)  # usamos el 80% (X_train) para esta búsqueda más profunda
log(f"[Stage C] best={hsC.best_params_} | cv_f1={hsC.best_score_:.4f}")

paramsC = unwrap_estimator_params(hsC.best_params_)
bestC = OneVsRestClassifier(LinearSVC(**paramsC), n_jobs=-1)

log("Refitting Stage C best on FULL train & evaluating ...")
bestC.fit(X_tr, y)
yC = bestC.predict(X_val)

c_f1  = f1_score(y_val, yC, average="macro")
c_acc = accuracy_score(y_val, yC)

log("\n=== [Stage C] Report ===")
print(classification_report(y_val, yC))
log(f"[Stage C] macro_f1={c_f1:.4f} acc={c_acc:.4f}")

# Submission de Stage C
pd.DataFrame({
    "id": test["id"],
    "age_range": bestC.predict(X_te)
}).to_csv(
    "submission_stageC_deep.csv",
    index=False,
    encoding="utf-8"
)
log("Wrote submission_stageC_deep.csv")


# ===================== 13) RESUMEN FINAL =====================
report = pd.DataFrame([
    {"stage":"A_baseline", "macro_f1":a_f1, "accuracy":a_acc, "csv":"submission_stageA_baseline.csv"},
    {"stage":"B_halving",  "macro_f1":b_f1, "accuracy":b_acc, "csv":"submission_stageB_halving.csv"},
    {"stage":"C_deep",     "macro_f1":c_f1, "accuracy":c_acc, "csv":"submission_stageC_deep.csv"},
]).sort_values("macro_f1", ascending=False)

log("\n================= SUMMARY (macro_f1 desc) =================")
print(report.to_string(index=False))
log("Done.")

