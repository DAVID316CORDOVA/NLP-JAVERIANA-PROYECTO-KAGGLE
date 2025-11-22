Proyecto Final – Detección de Fake News en Español
Modelos Clásicos, Embeddings y Modelos de Lenguaje – Evaluación Experimental Completa
1. Introducción

El presente proyecto desarrolla un sistema de clasificación automática de noticias falsas en español, utilizando el Spanish Fake News Corpus (IberLEF 2021 & 2020). El objetivo es construir y evaluar distintos enfoques de NLP, desde métodos clásicos hasta modelos contextualizados basados en embeddings, para posteriormente seleccionar el modelo más eficiente para una API de predicción y despliegue en Docker.

2. Dataset: Spanish Fake News Corpus

El corpus incluye noticias reales y falsas, recopiladas desde:

Sitios de fact-checking verificados (IFCN),

Periódicos reconocidos,

Redes sociales (casos de fake news).

Existen dos versiones:

| Archivo            | Tamaño                                         | Uso en este proyecto                            |
| ------------------ | ---------------------------------------------- | ----------------------------------------------- |
| `train.xlsx`       | 676 instancias                                 | Entrenamiento                                   |
| `development.xlsx` | 295 instancias                                 | Validación interna                              |
| `test.xlsx`        | 572 instancias (pero normalizado a 286 útiles) | Solo para referencia, no usado en entrenamiento |

Columnas principales:

Category (True/Fake)

Topic

Headline

Text (texto completo usado para el entrenamiento)

Source, Link

Total después de unir train + development:

 971 instancias

491 True

480 Fake

Balanceado → adecuado para clasificación binaria.

3. Metodología General de Experimentos

Se sigue un enfoque experimental inspirado en competencias tipo IberLEF:

Esquema experimental (A / B / C):
A – Modelos clásicos

A1: TF-IDF + Logistic Regression

A2: Bag-of-Words (CountVectorizer) + Naive Bayes

A3: TF-IDF + Stylometric Features + Logistic Regression

B – Modelos basados en embeddings

B1: SBERT Multilingüe + Logistic Regression

(C – Fine-tuning de transformers)

Intentado con BETO, pero descartado por incompatibilidades del entorno CPU+Windows+HF Trainer.
El enfoque fue reemplazado por una versión manual no incluida aquí.

3.1 Protocolo experimental

Para todos los experimentos de la serie A y B:

División estratificada 85%/15%.

10 experimentos independientes, cada uno con una partición distinta.

Métricas reportadas:

Accuracy

F1-Macro

Matriz de confusión

Esto permite medir estabilidad, varianza y robustez del modelo.

4. Resultados
4.1 Serie A – Modelos clásicos
 A1 – TF-IDF + Logistic Regression

El mejor modelo global del proyecto.

Resultados agregados (10 experimentos):
Accuracy promedio: 0.8116  (std = 0.0404)
F1-macro promedio: 0.8111  (std = 0.0404)

Ejemplo de una de las matrices de confusión:

[[69,  5],
 [20, 52]]


Interpretación:

Excelente balance entre clases.

Detecta bien tanto True como Fake.

Estabilidad muy buena (std ≈ 0.04).

 A2 – CountVectorizer + Naive Bayes

Resultados:
Accuracy promedio: 0.7562 (std = 0.0393)
F1-macro promedio: 0.7535 (std = 0.0397)


Más simple, más rápido, pero menos preciso.

 A3 – TF-IDF + Stylometric + Logistic Regression

Resultados:

Accuracy promedio: 0.7932 (std = 0.0414)
F1-macro promedio: 0.7921 (std = 0.0423)

Los rasgos estilométricos ayudan, pero no superan a A1.

4.2 Serie B – SBERT + Logistic Regression

Modelo probado:

sentence-transformers/distiluse-base-multilingual-cased-v1

Resultados (10 experimentos):
Accuracy promedio: 0.6452 (std = 0.0300)
F1-macro promedio: 0.6443 (std = 0.0298)


Matríz típica:
[[55, 19],
 [25, 47]]

Interpretación:

SBERT no captura bien el estilo del español del corpus (uso de ironía, sensacionalismo, modo narrativo).

Peor que TF-IDF debido al fuerte contenido léxico de la fake news en español.

5. Análisis Comparativo
5.1 Rendimiento general

| Modelo                             | Accuracy   | F1-macro   | Observaciones                      |
| ---------------------------------- | ---------- | ---------- | ---------------------------------- |
| **A1 – TF-IDF + LogReg**           | **0.8116** | **0.8111** |  Mejor modelo                    |
| A2 – Count + NB                    | 0.7562     | 0.7535     | Simple, baseline sólido            |
| A3 – TF-IDF + Stylometric + LogReg | 0.7932     | 0.7921     | Mejor que NB, peor que TF-IDF puro |
| **B1 – SBERT + LogReg**            | 0.6452     | 0.6443     | No útil para este corpus           |

5.2 Observaciones clave
 ¿Por qué gana TF-IDF + LogReg?

Porque:

El corpus está lleno de expresiones característicamente sensacionalistas o exageradas:

Titulares fabulosos

Repetición de palabras

Estereotipos

Números falsos

TF-IDF captura con mucha precisión estos patrones léxicos.

 ¿Por qué SBERT rinde peor?

SBERT generalista:

Está optimizado para similitud semántica, no para clasificación.

No captura ironías, sarcasmo ni estructuras típicas de fake news.

El corpus mezcla español latino, ibérico y redes sociales → SBERT sufre.

6. Conclusiones Técnicas

A1 – TF-IDF + Logistic Regression es el modelo ideal.

Buenas métricas

Rápido

Ligero

Fácil de dockerizar

Inferencia inmediata en CPU

Los métodos basados en embeddings sin fine-tuning no superan a los clásicos.

Intentos de fine-tuning con transformers fueron descartados por limitaciones en:

CPU-only Windows

Incompatibilidades del HF Trainer

Tiempo de entrenamiento

Recomendación final:

Modelo productivo: TF-IDF + LogReg

API FastAPI + Docker utilizando este modelo.

7. Conclusión Final del Proyecto

Este proyecto demuestra que:

Los métodos clásicos siguen siendo altamente competitivos para tareas de detección de fake news en español.

La disponibilidad de un corpus con fuerte carga léxica favorece modelos basados en n-gramas.

Los modelos contextualizados sin fine-tuning no superan a los clásicos.

El modelo TF-IDF + LogReg es ideal para despliegue en producción:

ligero

rápido

fácil de trasladar a Docker

métricas competitivas

La API desarrollada permite consumir este modelo desde aplicaciones externas (periodistas, entidades públicas, dashboards analíticos, etc.), lo cual cumple con los criterios del proyecto final del curso de PLN.