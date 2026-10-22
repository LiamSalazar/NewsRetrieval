# NewsRetrieval

Proyecto escolar en Python para recuperación léxica de 2,000 noticias en inglés de Wikinews, Global Voices y EFF. Incluye un benchmark provisional reproducible; no incluye embeddings.

## Ejecutar

```bash
python -m venv .venv
pip install -r requirements.txt
python -m unittest -v test_project
python main.py
python main.py --benchmark
```

`main.py` valida el corpus fijo. Si `data/news_raw_corpus.tsv` no existe, ejecuta la adquisición inicial mediante RSS/Atom y la API de MediaWiki. Si existe, debe contener exactamente 2,000 registros válidos; un corpus incompleto o inválido se informa y no se sobrescribe.

Después valida o genera automáticamente `data/news_normalized_corpus.tsv` y `results/vectors.joblib`. Los resultados procesados se reutilizan mientras correspondan al corpus actual.

La interfaz permite elegir contenido (`title` o `body`), características (unigramas o bigramas), representación (frecuencia, binaria o TF-IDF) y texto libre. Los valores predeterminados son cuerpo, unigramas y TF-IDF. Cada consulta usa la misma normalización spaCy, similitud del coseno y muestra los diez resultados principales con su puntuación. Se pueden hacer consultas sucesivas sin reconstruir las matrices.

## Corpus y procedencia

El TSV original conserva IDs, autor, fuente, URL, fecha, licencia y enlace de licencia. La adquisición aplica deduplicación por ID, URL, título y cuerpo, validación, controles de idioma/procedencia y escritura atómica. No se descargan imágenes ni se sortean restricciones de acceso.

- Wikinews: texto introductorio de páginas inglesas publicadas desde la API pública de MediaWiki.
- Global Voices: resúmenes de sus feeds RSS temáticos bajo CC BY 3.0.
- EFF Deeplinks: entradas originales de su RSS bajo CC BY 4.0, excluyendo contenido de terceros.

## Normalización y VSM

`retrieval.py` usa spaCy para tokenización, POS tagging, eliminación selectiva de stopwords y lematización. scikit-learn genera representaciones de frecuencia, binarias y TF-IDF en matrices dispersas persistidas con joblib. Se conservan las doce combinaciones: 2 campos × 2 tipos de n-gramas × 3 representaciones.

El corpus normalizado mantiene los campos originales y añade `language`, `normalized_title` y `normalized_body`. `results/vectors.joblib` contiene los vectorizadores, matrices CSR, documentos, IDs, versiones y la huella del corpus.

`NewsRetrieval_Pipeline.ipynb` se conserva como material del proyecto. `evaluation.py`, los archivos de consultas y juicios, y los resultados históricos permanecen fuera del flujo de ejecución.

## Benchmark y evaluación

`data/queries.tsv` contiene 15 consultas en inglés: tres formulaciones (easy=1, medium=2, hard=3) para cinco necesidades informativas. `data/candidates_review.tsv` documenta candidatos, extractos del corpus original y relevancias propuestas. `data/qrels.tsv` contiene 10 juicios por consulta, pero está marcado como provisional y requiere revisión humana.

El comando `python main.py --benchmark` valida los archivos sin calcular resultados definitivos. Después de revisar y congelar humanamente los qrels, `python main.py --benchmark --frozen` ejecuta las 15 consultas en las 12 configuraciones y escribe `results/rankings.tsv`, `results/metrics.tsv` y `results/comparison.tsv`. Las métricas son Precision@10, Recall@10, F1@10, MRR y nDCG@10 con ganancia graduada `2^relevance - 1`. Las etiquetas humanas nunca se modifican usando resultados del VSM.

Las cinco necesidades son: comercio internacional y relaciones económicas; precios de energía y sus efectos; fútbol como fenómeno político/social; eventos y controversias del cricket; y vigilancia tecnológica, privacidad y supervisión pública. Las métricas no se presentan como definitivas mientras `qrels.tsv` siga provisional.
