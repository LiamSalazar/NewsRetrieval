# NewsRetrieval

Proyecto escolar en Python para recuperación léxica de 2,000 noticias en inglés de Wikinews, Global Voices y EFF. No incluye benchmark, métricas ni embeddings.

## Ejecutar

```bash
python -m venv .venv
pip install -r requirements.txt
python -m unittest -v test_project
python main.py
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
