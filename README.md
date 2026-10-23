# NewsRetrieval

Sistema académico de recuperación léxica sobre 2,000 noticias auténticas en inglés. El proyecto utiliza un modelo de espacio vectorial, doce configuraciones, similitud coseno y un benchmark de quince consultas.

## Estado de la referencia

`data/qrels.tsv` contiene 303 juicios positivos congelados para **evaluación analítica delegada**. Esta denominación no equivale a aprobación humana individual exhaustiva ni a aprobación institucional. La referencia procede de la adjudicación semántica consolidada y conserva relevancias 1 y 2.

La consigna académica solicita identificar entre 10 y 15 documentos relevantes por consulta. Comercio, fútbol y vigilancia superan el máximo literal porque se conservaron todos los positivos auténticos encontrados: 28, 20 y 25 documentos por consulta, respectivamente. Reducirlos habría ocultado relevancia y alterado el denominador de Recall@10. `data/academic_reference_core.tsv` es únicamente un ejemplo expositivo y nunca sustituye los qrels oficiales.

## Instalación y ejecución

```bash
python -m venv .venv
pip install -r requirements.txt
python -m unittest -v test_project
python main.py --benchmark
python main.py --benchmark --frozen
```

`python main.py --benchmark` valida corpus, consultas y qrels sin recalcular resultados. `--frozen` ejecuta las 180 búsquedas y sobrescribe de forma coherente el único conjunto de resultados finales en `results/`.

Para una consulta libre:

```bash
python main.py --query "government surveillance and privacy" --field body --ngram 1 --representation tfidf
```

## Corpus y extracción

El corpus fijo conserva 2,000 noticias: 1,781 de Wikinews, 180 de Global Voices y 39 de EFF. Cada registro incluye ID, título, cuerpo o extracto, fuente, categoría, fecha, URL, fecha de recuperación, autor, licencia y URL de licencia.

- Wikinews: API de MediaWiki y texto introductorio de páginas de la edición inglesa.
- Global Voices: feeds RSS temáticos, contenido atribuido bajo CC BY 3.0.
- EFF Deeplinks: feed RSS original bajo CC BY 4.0; se excluye contenido republicado de terceros.

La extracción inicial aplicó validación, deduplicación por ID, URL, título y cuerpo, comprobación conservadora de idioma y escritura atómica. El notebook usa el corpus persistido y no descarga nuevamente las noticias.

## Normalización y representaciones

`retrieval.py` aplica la misma normalización a documentos y consultas: limpieza de HTML y caracteres de control, tokenización y POS tagging con spaCy, eliminación selectiva de artículos, preposiciones, conjunciones y pronombres, y lematización en minúsculas. El corpus normalizado conserva los campos originales y añade `language`, `normalized_title` y `normalized_body`.

`results/vectors.joblib` almacena doce matrices CSR y sus vectorizadores:

- Campo: título o cuerpo.
- Características: unigramas o bigramas.
- Ponderación: frecuencia, binaria o TF-IDF.

No se utilizan embeddings ni modelos semánticos. La recuperación transforma la consulta con el vectorizador correspondiente, calcula similitud coseno y devuelve exactamente diez documentos. Cuando no hay características compartidas pueden aparecer empates con similitud cero; se conservan y se auditan semánticamente.

## Benchmark y auditoría Top-10

`data/queries.tsv` contiene quince consultas: cinco por dificultad y tres formulaciones para cada necesidad informativa. Las cinco necesidades son comercio internacional, precios de energía, fútbol como fenómeno político/social, controversias del cricket y vigilancia pública.

`data/qrels.tsv` contiene solo positivos. Los negativos confirmados y sus justificaciones se mantienen en `data/pool_human_review.tsv`. Los 1,800 resultados Top-10 de la evaluación final están vinculados a uno de esos juicios en `results/retrieval_judgments.tsv`; no quedan devoluciones sin juicio.

Antes de la evaluación final se revisaron individualmente los pares Top-10 que solo aparecían por desempate a similitud cero. La evidencia se conserva en `data/top10_zero_similarity_adjudication.tsv`. La puntuación nunca se utilizó como etiqueta.

## Métricas

Para cada consulta y configuración se calculan:

- Precision@10: positivos recuperados entre diez.
- Recall@10: positivos recuperados entre todos los qrels de la consulta.
- F1@10: media armónica de precisión y recall.
- MRR: rango recíproco del primer positivo.
- nDCG@10: ganancia graduada `2^relevance - 1`, descuento logarítmico y normalización por el ranking ideal.

Las pruebas comprueban ejemplos conocidos, ausencia de positivos, grados mixtos, resultados con score cero, duplicados, consistencia de las 180 evaluaciones y correspondencia matemática entre rankings y métricas.

## Resultados finales

- `results/rankings.tsv`: Top-10 de la configuración principal `body_1_tfidf`, con el formato solicitado.
- `results/rankings_all_configurations.tsv`: 1,800 devoluciones de las doce configuraciones.
- `results/retrieval_judgments.tsv`: auditoría individual con grado, estado y evidencia.
- `results/metrics_by_query_configuration.tsv`: 180 filas con conteos y cinco métricas.
- `results/metrics.tsv`: resultados de `body_1_tfidf` y su promedio.
- `results/comparison.tsv`: promedios por configuración y dificultad.
- `results/report_data.json`: datos consolidados para el informe académico.
- `results/figures/`: cinco gráficas SVG derivadas de la misma ejecución.

`NewsRetrieval_Pipeline.ipynb` demuestra el flujo completo y contiene una función `show_top10(query_id, configuration)` para inspeccionar cualquier combinación.

## Limitaciones

- La validación de relevancia es analítica delegada, no una evaluación humana institucional exhaustiva.
- El pool no demuestra exhaustividad sobre las 2,000 noticias.
- La recuperación léxica es sensible a sinónimos, paráfrasis, ambigüedad y vocabulario fuera del índice.
- Los extractos almacenados pueden contener menos contexto que el artículo completo.
- Los empates de similitud cero reflejan ausencia de características compartidas, no irrelevancia automática.
