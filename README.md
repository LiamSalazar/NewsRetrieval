# NewsRetrieval

Proyecto universitario de PLN tradicional sobre 2,000 noticias en inglés. Etapa I recupera noticias con VSM y similitud coseno; Etapa II clasifica sus categorías y compara agrupamientos con K-Means. El informe académico final en PDF está pendiente deliberadamente.

## Instalación

Entorno validado: Python 3.12.3 en Linux, scikit-learn 1.6.1, spaCy 3.8.5 y modelo inglés 3.8.0. Las dependencias están en `requirements.txt`; las versiones ejecutadas, en `results/environment_info.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

En Windows, activar con `.venv\Scripts\activate`. Ejecutar los comandos desde la raíz del repositorio.

## Estructura y entregables

- `collector.py`, `retrieval.py`, `evaluation.py`: adquisición protegida, normalización, doce espacios VSM y evaluación de Etapa I.
- `main.py`: consultas e inspección de ambas etapas.
- `split_data.py`, `pipelines.py`, `experiment_utils.py`, `train_classifier.py`, `train_unsupervised.py`: procesos de Etapa II y registro experimental.
- `data/`: corpus original/normalizado, consultas, qrels y evidencia de la referencia congelada.
- `results/`: vectores de recuperación, rankings, métricas, figuras, modelos finales, metadatos y entorno.
- `labeled_documents.tsv`, `split_summary.tsv`, `label_distribution.tsv`, `split_audit.tsv`: texto, etiquetas y evidencia de particiones.
- `classification_*.tsv`: predicciones, métricas agregadas/por clase, confusión y diez errores dev analizados.
- `clustering_results.tsv`, `clustering_metrics.tsv`, `cluster_summary.tsv`: asignaciones, evaluación del seleccionado, tamaños, diez términos y dos ejemplos por cluster.
- `clustering_dev_experiments.tsv`: comparación de los 29 ensayos con tamaños y espacio geométrico; complementa `experiments.tsv`, no lo duplica.
- `clustering_stability.tsv`, `clustering_contingency.tsv`: estabilidad entre semillas y evaluación externa.
- `experiments.tsv`: 101 experimentos (72 clasificación, 29 K-Means), parámetros y decisiones.
- `test_project.py`: pruebas con unittest.
- `PhasesInstructions/`: enunciado disponible de Etapa I. El PDF oficial de Etapa II no está disponible; el cumplimiento se contrasta con la especificación proporcionada, no directamente con ese PDF.

## Ejecución e inspección

Los artefactos necesarios ya están incluidos. Para consultarlos sin entrenar ni descargar:

```bash
python main.py --query "government surveillance and privacy" --field body --ngram 1 --representation tfidf
python main.py --supervised
python main.py --unsupervised
python -m unittest -v test_project
```

Para reproducir Etapa II (entrena los experimentos y actualiza sus entregables):

```bash
python split_data.py
python train_classifier.py
python train_unsupervised.py
```

Para Etapa I: `python main.py --collect` valida el corpus existente; `--prepare` regenera representaciones; `--benchmark` valida la referencia y `--benchmark --frozen` recalcula las doce configuraciones y sobrescribe sus salidas. No es necesario regenerar el benchmark para usar los artefactos entregados. El menú `python main.py` también permite consultar los resultados.

## Notebooks ilustrativos

Abrir desde la raíz con un entorno Jupyter que utilice las dependencias anteriores:

- `NewsRetrieval_Pipeline.ipynb`: notebook original de Etapa I, conservado. Su celda de evaluación recalcula y escribe resultados; no ejecutarla si se desea mantener los exports históricos byte a byte.
- `NewsRetrieval_Etapa_II.ipynb`: corpus, normalización, representaciones, clasificación, diez errores, K-Means, interpretación, estabilidad/contingencia y resultados finales. Completamente ejecutado; lee artefactos y realiza una inferencia puntual, sin descargas, reentrenamientos completos ni escritura de resultados. Matplotlib se usa solo para tamaños de clusters.

Ejecución programática del notebook nuevo, sin instalar otra interfaz:

```python
import nbformat
from nbclient import NotebookClient
n = nbformat.read('NewsRetrieval_Etapa_II.ipynb', as_version=4)
NotebookClient(n, timeout=120, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
nbformat.write(n, 'NewsRetrieval_Etapa_II.ipynb')
```

## Protocolo y resultados

Texto = título + cuerpo; categorías originales `deportes`, `finanzas`, `tecnologia`. Dos divisiones estratificadas con shuffle=True y random_state=0 conservan 1400/300/300 documentos; dev/test tienen 100 por categoría. No se ajustan vectorizadores ni SVD con dev/test durante la selección.

Tres normalizaciones: minúsculas/espacios; limpieza y stopwords; limpieza, stopwords y lematización. Representaciones: binaria (1,1), frecuencia (1,2), TF-IDF (1,1) y (1,2), con min_df=2. La comparación binaria/frecuencia cambia también n-gramas, por lo que no aísla solo la ponderación.

MultinomialNB, LogisticRegression y LinearSVC se seleccionan por macro-F1 dev. Solo el ganador se reajusta con train+dev para test. EXP_A_046 (LogisticRegression C=5, stopwords, TF-IDF 1–2) obtiene macro-F1 dev 0.89655 y test 0.89374. Las predicciones contienen los tres mejores modelos dev y solo el ganador test (`experiment_id=test`). Las diez notas son análisis asistidos sobre dev, preservados por ID y etiquetas real/predicha.

K-Means conserva EXP_B_001–017 y añade 018–029: TF-IDF lematizado, SVD 50/100, con/sin L2 posterior, k=3/4/5. Comparar con 012/013/014 aísla SVD para el mismo k; pares con igual k/dimensión aíslan L2. La selección maximiza `silhouette_dev × H(tamaños_dev)/log(k)`; desempata por orden. No usa categorías, ARI/NMI externos ni test. Es una regla operacional no supervisada, no una garantía temática: favorece equilibrio y compara geometrías diferentes, cuyas silhouettes euclidianas no son directamente equivalentes.

La regla conserva EXP_B_002: silhouette dev 0.28094, tamaños dev 79/1/215/5 y train+dev 406/1255/36/3. Estabilidad ARI entre asignaciones dev, semillas 0–4: media 0.97857, mínimo 0.95987, máximo 1.0. Esa repetibilidad no elimina el desequilibrio ni la mezcla temática. Los términos genéricos y documentos representativos sustentan esa limitación. SVD+L2 ofrece alternativas más equilibradas, sin cambiar el seleccionado a partir de etiquetas.

Los términos de clusters son promedios de pesos del vectorizador original entre miembros train+dev; los ejemplos son los más cercanos al centroide en el espacio de ajuste. Contingencia dev procede del ajuste train y test del ajuste train+dev; los IDs de cluster entre ajustes no se equiparan. ARI/NMI respecto a categorías son exclusivamente evaluación externa, distintos del ARI entre semillas.

## Integridad y limitaciones

El corpus raw y el benchmark permanecen congelados. Su SHA-256 histórico se reproduce serializando raw con CRLF; el archivo entregado usa LF. Normalizado y vectores fueron recuperados del commit `b02a7f9a`, con hashes exactos de `results/report_data.json`; no se reemplazaron hashes históricos. Las métricas históricas de las 180 evaluaciones fueron comprobadas; body_1_tfidf conserva P@10 0.3200, R@10 0.1727, F1@10 0.2196, MRR 0.5763, nDCG@10 0.3349.

Se mantienen los 303 juicios positivos: nueve consultas exceden el intervalo literal de 10–15 de Etapa I. La validación histórica es analítica delegada, no aprobación humana institucional. Los resultados con similitud cero se ordenan por posición original y se evalúan sin alterar el protocolo.

Test ya fue observado históricamente; no puede llamarse inédito. No se usa para seleccionar. Configuraciones secundarias pueden variar entre entornos (hasta 0.00672 en macro-F1 dev y 0.00199 en silhouette histórica); los ganadores reproducen sus resultados. SciPy puede emitir la advertencia `Unknown solver options: iprint` durante entrenamiento con sklearn 1.6.1. El notebook nuevo utiliza resultados guardados y no ejecuta esos ajustes.
