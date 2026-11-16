# NewsRetrieval

Sistema de recuperación de noticias que utiliza el Modelo de Espacio Vectorial, similitud coseno y representaciones de frecuencia, binaria y TF-IDF para recuperar y evaluar noticias mediante consultas de lenguaje natural.

## Características

- Corpus fijo de 2,000 noticias en inglés: 1,781 de Wikinews, 180 de Global Voices y 39 de EFF.
- Normalización común para documentos y consultas con spaCy: tokenización, etiquetado POS, filtrado selectivo y lematización.
- Doce configuraciones VSM: título o cuerpo, unigramas o bigramas, y frecuencia, binaria o TF-IDF.
- Recuperación por similitud coseno con diez resultados y puntuaciones.
- Benchmark congelado de quince consultas: cinco fáciles, cinco intermedias y cinco difíciles.
- Precision@10, Recall@10, F1@10, MRR y nDCG@10.
- Notebook demostrativo completo y ejecutado.

## Requisitos

El proyecto fue ejecutado con Python 3.11.5. Las versiones exactas de las bibliotecas están fijadas en `requirements.txt`, incluido el modelo `en_core_web_sm` 3.8.0 de spaCy.

## Instalación y ejecución en Windows

Desde CMD o PowerShell, dentro de la carpeta del proyecto:

```cmd
py -m pip install --no-cache-dir -r requirements.txt
py main.py
```

Comandos independientes:

```cmd
py main.py --collect
py main.py --prepare
py main.py --benchmark
py main.py --benchmark --frozen
py main.py --query "government surveillance and privacy" --field body --ngram 1 --representation tfidf
py -m unittest -v test_project
```

`--collect` conserva el corpus fijo: cuando ya existe, únicamente lo valida y no realiza una descarga. `--prepare` normaliza ese corpus y vuelve a generar las doce matrices.

## Instalación y ejecución en Linux

```bash
python3 -m pip install --no-cache-dir -r requirements.txt
python3 main.py
python3 main.py --collect
python3 main.py --prepare
python3 main.py --benchmark
python3 main.py --benchmark --frozen
python3 main.py --query "government surveillance and privacy" --field body --ngram 1 --representation tfidf
python3 -m unittest -v test_project
```

Si la distribución administra Python externamente y bloquea instalaciones globales, puede crear un entorno aislado con `python3 -m venv .venv`, activarlo y ejecutar allí la instalación. No es un requisito del proyecto.

## Instalación y ejecución en macOS

```bash
python3 -m pip install --no-cache-dir -r requirements.txt
python3 main.py
python3 main.py --collect
python3 main.py --prepare
python3 main.py --benchmark
python3 main.py --benchmark --frozen
python3 main.py --query "government surveillance and privacy" --field body --ngram 1 --representation tfidf
python3 -m unittest -v test_project
```

Estos comandos corresponden a la interfaz multiplataforma del proyecto; la validación final de esta entrega se realizó en Windows.

## Menú principal

`python main.py` abre un menú con estas operaciones:

1. Introducir una consulta libre, elegir título o cuerpo, unigramas o bigramas y frecuencia, binaria o TF-IDF; se muestran diez noticias ordenadas y sus puntuaciones.
2. Ejecutar el benchmark congelado y calcular las cinco métricas.
3. Mostrar el resumen comparativo de las doce configuraciones.
4. Salir.

La adquisición protegida y la preparación también pueden ejecutarse por separado con `--collect` y `--prepare`. `--benchmark` valida la referencia sin recalcular; al agregar `--frozen` se reproducen las 180 evaluaciones definitivas.

## Archivos del proyecto

| Ruta | Contenido |
|---|---|
| `data/news_raw_corpus.tsv` | Corpus original con texto, fuente, categoría, fecha, URL, recuperación, autor y licencia. |
| `data/news_normalized_corpus.tsv` | Corpus alineado con títulos y cuerpos normalizados. |
| `results/vectors.joblib` | Doce matrices dispersas y sus vectorizadores. |
| `data/queries.tsv` | Quince consultas y sus niveles de dificultad. |
| `data/qrels.tsv` | 303 juicios positivos congelados con grados 1 y 2. |
| `results/rankings.tsv` | Top-10 de la configuración principal `body_1_tfidf`. |
| `results/rankings_all_configurations.tsv` | Top-10 de las 180 combinaciones consulta-configuración. |
| `results/metrics.tsv` | Métricas por consulta y promedio de la configuración principal. |
| `results/metrics_by_query_configuration.tsv` | Métricas de las doce configuraciones. |
| `results/evaluation_by_configuration/` | Métricas separadas en un TSV por configuración. |
| `results/comparison.tsv` | Resumen comparativo por configuración y dificultad. |
| `NewsRetrieval_Pipeline.ipynb` | Demostración ejecutada, tablas, gráficas y análisis de errores. |
| [`docs/NewsRetrieval_Informe_Final.pdf`](docs/NewsRetrieval_Informe_Final.pdf) | Informe académico final. |

Los qrels conservan más de quince documentos positivos en tres necesidades informativas porque se encontraron documentos pertinentes adicionales. No se eliminaron para aparentar el intervalo recomendado. La referencia corresponde a una evaluación analítica delegada, no a una aprobación humana institucional exhaustiva.

## Resultados principales

Promedios reales de `body_1_tfidf`:

| Métrica | Valor |
|---|---:|
| Precision@10 | 0.3200 |
| Recall@10 | 0.1727 |
| F1@10 | 0.2196 |
| MRR | 0.5763 |
| nDCG@10 | 0.3349 |

## Informe académico

El informe con metodología, resultados, análisis de errores e integrantes está disponible en [`docs/NewsRetrieval_Informe_Final.pdf`](docs/NewsRetrieval_Informe_Final.pdf).



## Stage II — Traditional Machine Learning for NLP

Stage II extends the news retrieval system with supervised
news classification and unsupervised document clustering.

### Tasks

- Supervised classification using MultinomialNB,
  LogisticRegression, and LinearSVC.
- Unsupervised document clustering using K-Means.
- Comparison of text normalization and vectorization
  configurations.
- Model selection using development data.
- Final evaluation using a held-out test partition.

### Stage II execution

Run the following commands from the repository root:

```bash
python split_data.py
python train_classifier.py
python train_unsupervised.py
```

### Generated classification files

- labeled_documents.tsv
- classification_predictions.tsv
- classification_metrics.tsv
- classification_per_class_metrics.tsv
- classification_confusion_matrices.tsv
- classification_error_analysis.tsv

### Generated clustering files

- clustering_results.tsv
- clustering_metrics.tsv
- cluster_summary.tsv

### Experiment tracking

All supervised and K-Means experiments are recorded in:

- experiments.tsv

The best trained models and their metadata are saved
inside the results/ directory.

### Evaluation protocol

- Train data is used to fit models and vectorizers.
- Dev data is used for configuration selection.
- Test data is reserved for final evaluation.
- Classification uses macro-F1 as its primary selection metric.
- K-Means configurations are selected using dev silhouette.
- ARI and NMI are used only for external interpretation.

### Reproducibility

The data split uses random_state=0.
Software versions and environment information are
recorded in results/environment_info.json.
