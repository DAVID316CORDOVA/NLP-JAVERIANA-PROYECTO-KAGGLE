# Predicción de rango de edad a partir de tweets (Kaggle)

Este proyecto aborda un problema de clasificación de rango de edad a partir de textos cortos (tweets).  
El objetivo es predecir la clase `age_range` para cada tweet, maximizando la métrica **macro-F1** en Kaggle.

La solución se basa en:

- **Ingeniería de características de NLP estructurada**.
- Un modelo lineal robusto (**LinearSVC**) entrenado sobre un gran conjunto de features.
- Búsquedas de hiperparámetros con **HalvingGridSearchCV** en dos etapas (Stage B y C).

---

## 1. Estructura general del pipeline

El flujo principal del script `AGE-RANGE SVM: ONE-CELL, RICH FEATURES` es:

1. Cargar datos (`age_train.csv` y `age_test.csv`).
2. Normalizar texto de forma específica para Twitter.
3. Extraer:
   - Léxicos discriminativos por rango de edad.
   - Features numéricas estructuradas (longitudes, conteos, dominios, etc.).
   - Representaciones TF-IDF en múltiples canales (palabras, caracteres, hashtags, menciones).
   - Similitudes a centroides de edad (word y char).
4. Combinar todo en una gran matriz dispersa de características.
5. Entrenar y evaluar:
   - **Stage A:** LinearSVC baseline.
   - **Stage B:** HalvingGridSearch sobre un 15% piloto.
   - **Stage C:** HalvingGridSearch “profundo” centrado en el mejor modelo de Stage B.
6. Generar 3 archivos de `submission` y un resumen final de resultados.

---

## 2. Estrategias de NLP y motivación

### 2.1 Normalización “Twitter-aware”

En lugar de usar un preprocesamiento genérico, se diseñó una función `normalize_tweet` que:

- Convierte todo a minúsculas.
- **Colapsa secuencias largas de menciones** (`@user1 @user2 @user3 ...`) en solo las 2 primeras para reducir ruido.
- **Detecta URLs** y las reemplaza con un token `URLTOKEN`, pero además:
  - Extrae el dominio y lo inyecta como `url_<dominio>` (p. ej. `url_twitter_com`).
  - Extrae el TLD (co, es, etc.) y lo inyecta como `tld_co`, `tld_es`, etc.
- Separa `@` y `#` del texto para que TF-IDF pueda explotar menciones y hashtags como tokens aparte.

**Motivación:**

- Los tweets contienen mucha señal en la estructura: URLs, dominios, menciones, hashtags, etc.
- Los dominios y TLDs pueden correlacionarse con edad (ej. jóvenes usando más TikTok/Instagram; adultos compartiendo noticias o bancos).
- Colapsar menciones reduce ruido de nombres de usuario aleatorios.

---

### 2.2 Léxicos salientes por edad (log-odds)

Se construyen BoW de unigrams y bigrams con `CountVectorizer` (min_df para estabilidad) y se usa una función `log_odds_top` que:

- Para cada `age_range`, calcula una puntuación de **log-odds con suavizado** (tipo análisis de estilo político / sociolingüístico).
- Selecciona los **Top-K unigrams** (`AGE_TOP_K1`) y **Top-K bigrams** (`AGE_TOP_K2`) más característicos de cada grupo de edad.
- Se compila una regex por edad para contar cuántas veces aparecen estos términos en cada tweet (`agelex_<edad>`).

**Motivación:**

- Diferentes edades usan vocabularios característicos: expresiones, argot, referencias culturales.
- Los conteos `agelex_*` añaden un nivel explícito de “léxico saliente” por grupo, que complementa el TF-IDF estándar.

---

### 2.3 Features numéricas estructuradas

La función `build_numeric` genera un conjunto amplio de features:

- **Longitud y conteos básicos:**
  - `char_count`, `word_count`
  - `mention_count`, `hashtag_count`, `url_count`
  - `digit_count`, `punct_ratio`
- **Indicadores estructurales:**
  - `is_rt` (si empieza con `RT`)
  - `starts_with_at` (si comienza con mención)
  - `ends_with_url` (si el tweet termina en URL)
- **Emojis y señales básicas:**
  - `emoji_count`
  - `english_ratio` (proporción de letras ASCII, proxy de uso de inglés)
- **Menciones/hashtags famosos vs raros:**
  - `mention_famous`, `mention_rare`
  - `hashtag_famous`, `hashtag_rare`
- **Buckets de dominio:**
  - `url_social`, `url_video`, `url_news`, `url_ecom`, `url_other`
- **Léxicos por edad:**
  - `agelex_<age_range>`: conteo de términos salientes de esa edad.
- **TLDs:**
  - `tld_co`, `tld_es`, etc.: número de veces que se menciona ese TLD en el texto normalizado.

**Motivación:**

- No toda la información está en las palabras: la **forma** del tweet (longitud, estructura, URLs, emojis) también lleva señal sobre la edad del usuario.
- Los patrones de uso de redes, noticias o e-commerce pueden variar según la edad.
- Las features densas numéricas ayudan a que un modelo lineal aproveche correlaciones simples sin depender únicamente de n-gramas.

---

### 2.4 Representaciones TF-IDF en múltiples canales

Se usan cuatro canales de texto:

1. **Palabras (unigrams+bigrams)**  
   `TfidfVectorizer` con n-gramas `(1,2)`, stopwords en español y `sublinear_tf=True`.  
   Captura el contenido semántico estándar del texto.

2. **Caracteres (3-5 chars)**  
   `TfidfVectorizer` con `analyzer='char'`, n-gramas `(3,5)`.  
   Captura información morfológica, ortográfica, abreviaturas, emoticonos, errores típicos según edad, etc.

3. **Hashtags**  
   `TfidfVectorizer` con `token_pattern='#\w+'`.  
   Los hashtags encapsulan temas, eventos, comunidades y suelen ser muy informativos.

4. **Menciones**  
   `TfidfVectorizer` con `token_pattern='@\w+'`.  
   A quién mencionas (influencers, amigos, marcas) puede correlacionarse con edad.

**Motivación:**

- Separar canales permite explotar diferentes vistas del texto:
  - Semántica de palabras.
  - Estilo y morfología (chars).
  - Temas explícitos (hashtags).
  - Red social (menciones).

---

### 2.5 Similitudes a centroides por edad

A partir de los TF-IDF de palabras y de caracteres, se construyen “centroides” para cada `age_range`:

- Para cada edad, se promedia el vector TF-IDF de todos los tweets de ese grupo.
- Para cada tweet, se calcula la **similitud coseno** con el centroide de cada edad, tanto en el espacio de palabras como en el de caracteres.
- Estas similitudes se añaden como columnas adicionales (una por edad y por canal).

**Motivación:**

- Proporciona al modelo una especie de “puntaje de cercanía” a cada grupo de edad:
  - “¿Este tweet se parece más al habla típica de 18-24, 25-34…?”.
- Es una forma barata de añadir estructura de clase (tipo prototipos) encima de TF-IDF.

---

### 2.6 Ensamblaje y escalado

La matriz final de características `X_tr` / `X_te` se construye como:

- `hstack([Xw, Xc, Xh, Xm, Xn, Sw, Sc])`, donde:
  - `Xw`: TF-IDF palabras,
  - `Xc`: TF-IDF caracteres,
  - `Xh`: TF-IDF hashtags,
  - `Xm`: TF-IDF menciones,
  - `Xn`: features numéricas escaladas (numeric),
  - `Sw`: similitudes a centroides en espacio de palabras,
  - `Sc`: similitudes a centroides en espacio de caracteres.

Las features numéricas se escalan con `StandardScaler(with_mean=False)` para mantener compatibilidad con matrices dispersas.

**Motivación:**

- Combinamos **representaciones dispersas** (TF-IDF) y **numéricas densas** en un solo espacio.
- El escalado numérico evita que algunas features tengan magnitudes desproporcionadas e impacten indebidamente al modelo lineal.

---

## 3. Modelos y etapas (Stage A, B, C)

### 3.1 Stage A – Baseline

- Modelo: `OneVsRestClassifier(LinearSVC)` con:
  - `C = 0.25`,
  - `loss = 'squared_hinge'`,
  - `tol = 1e-4`.
- Se usa un **hold-out** (80% train, 20% val) estratificado para evaluar macro-F1 y accuracy.
- Después se entrena en todo el train y se genera `submission_stageA_baseline.csv`.

**Motivación:**

- LinearSVC es robusto, rápido y tiende a funcionar muy bien con TF-IDF + features lineales.
- Esta etapa sirve como referencia para ver si la ingeniería de características está aportando valor.

---

### 3.2 Stage B – HalvingGridSearch (pilot 15%)

- Se toma un **pilot** de ~15% de `X_train` mediante `StratifiedShuffleSplit`.
- Se define un grid pequeño de hiperparámetros para `LinearSVC`:
  - `C` en `[0.08, 0.10, 0.12, 0.15, 0.2]`
  - `tol` en `[1e-4, 5e-4]`
  - `class_weight` en `[None, "balanced"]`
- Se usa `HalvingGridSearchCV` con:
  - `factor = 3`,
  - `cv = 3`,
  - `scoring = 'f1_macro'`.
- Se obtiene el mejor conjunto de parámetros (`best_params_`) y se reentrena un `OneVsRestClassifier(LinearSVC(**paramsB))` en todo `X_tr`.
- Se evalúa en `X_val` y se genera `submission_stageB_halving.csv`.

**Motivación:**

- HalvingGridSearch permite explorar hiperparámetros de forma más eficiente que un grid search estándar, reduciendo candidatos a medida que se usan más datos.
- El pilot del 15% reduce el coste computacional, pero es suficiente para “orientar” un buen valor de `C`.

---

### 3.3 Stage C – HalvingGridSearch “deep” alrededor de C\*

- Se toma el mejor `C*` de Stage B y se construye un grid **más fino** alrededor:
  - Escalando `C*` por `[0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]` con límites `[0.05, 3.0]`.
- Se define un grid:
  - `C` en esa lista,
  - `tol` en `[1e-4, 5e-4, 1e-5]`,
  - `class_weight` en `[None, "balanced"]`.
- Se llama a `HalvingGridSearchCV` de nuevo, esta vez sobre `X_train` (80% del total, split de validación se mantiene aparte).
- Se reentrena el mejor modelo sobre todo `X_tr` y se evalúa en `X_val`.
- Se genera `submission_stageC_deep.csv`.

**Motivación:**

- Stage B da una “aproximación gruesa” de un buen `C`.
- Stage C refina alrededor de ese punto para exprimir un poco más de macro-F1 sin explotar excesivamente el tiempo de cómputo.

---

## 4. Archivos de salida

El script genera tres archivos de predicción:

- `submission_stageA_baseline.csv`
- `submission_stageB_halving.csv`
- `submission_stageC_deep.csv`

Cada uno con el formato requerido por Kaggle:

```csv
id,age_range
12345,25-34
12346,18-24
...
