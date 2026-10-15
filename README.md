# NewsRetrieval

Proyecto escolar basado en los apartados 4–8 de `01_News_Retrieval_Project_Stage_1.pdf` (el PDF está en el directorio superior). Adquisición HTTP pública, normalización inglesa con spaCy y recuperación léxica por coseno. No incluye todavía benchmark, métricas ni embeddings.

## Ejecutar

Python 3.12. En este equipo el ejecutable del sistema es `python3`; dentro del entorno se utiliza `python`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v test_project
python main.py init --target 2000
python main.py update --per-category 10 --dry-run
python main.py update --per-category 10
python main.py process
python main.py search
python main.py search --query "European Union semiconductor production Chips Act 43 billion"
python main.py search --query "satellite launch" --field title --ngram 2 --representation binary
```

`--field`: `title` o `body`; `--ngram`: `1` o `2`; `--representation`: `frequency`, `binary` o `tfidf`. Predeterminado: cuerpo, unigramas, TF-IDF, diez resultados. Las consultas deben estar en inglés. Sin coincidencias se muestran puntuaciones cero y un aviso; los empates conservan el orden del corpus.

`init` completa cuotas totales, no consume cuota semanal y no vuelve a descargar un corpus que ya alcanza el objetivo. Para adquisición histórica consulta hasta cinco páginas RSS por tema y después la API de Wikinews; `--max-batches` limita los lotes históricos. Si se interrumpe una carga inicial incompleta, puede reanudarse con `init`. No se declara completa sin el volumen, las categorías y tres medios representados.

`update` admite cuotas de 1 a 10 por categoría, acumuladas desde el viernes 00:00 de Ciudad de México hasta el jueves siguiente, mediante `results/collection_log.jsonl`. La ventana predeterminada es de ocho días de publicación (`--days`); nunca usa el archivo histórico como sustituto de novedades. Una semana con 1–10 incorporaciones por tema es válida (`complete`); `target_met` indica si alcanzó la meta solicitada. El informe siempre muestra faltantes respecto de esa meta.

Sin incorporaciones no cambia el corpus, la copia de seguridad ni la bitácora. `--dry-run` de adquisición tampoco escribe archivos. Hay deduplicación por ID, URL, título y cuerpo, validación y reemplazo atómico del TSV; antes del primer reemplazo se conserva `data/news_raw_corpus.tsv.bak`. No se debe borrar la bitácora semanal. Después de incorporar noticias, ejecutar `process`; la interfaz de búsqueda rechaza índices cuyo corpus cambió.

El workflow conserva viernes a las 16:00 UTC (10:00 CDMX), ejecución manual y exclusión de ejecuciones concurrentes. Instala dependencias/modelo, prueba y actualiza; su paso existente de guardado publica los cambios mediante GitHub Actions. Durante esta intervención no se ejecutó ese paso ni se hicieron commits o pushes.

## Corpus verificado el 15 de octubre de 2026

| Medio | Finanzas | Deportes | Tecnología | Total |
|---|---:|---:|---:|---:|
| Global Voices | 64 | 69 | 48 | 181 |
| Wikinews | 603 | 598 | 579 | 1,780 |
| EFF Deeplinks | 0 | 0 | 39 | 39 |
| Total | 667 | 667 | 666 | 2,000 |

| Categoría | Primera publicación | Última publicación |
|---|---|---|
| Finanzas | 2005-07-13 | 2026-10-14 |
| Deportes | 2006-06-10 | 2026-09-05 |
| Tecnología | 2004-11-16 | 2026-10-15 |

Es un corpus histórico y actual; finanzas incluye economía y negocios. Las categorías proceden de feeds temáticos y categorías editoriales de Wikinews. En su categoría amplia «Science and technology» se exige además vocabulario tecnológico; es un filtro léxico, no una clasificación manual exhaustiva.

Se conservaron exactamente los IDs de 1,672 artículos; se sustituyeron 328 artículos por otros con nuevos IDs. Las listas y hashes están en [results/corpus_changes.json](results/corpus_changes.json). La copia `.bak` conserva las 2,000 noticias anteriores, está disponible localmente y se excluye de Git. Los futuros juicios de relevancia deben usar los IDs actuales. `queries.tsv` y `qrels.tsv` mantienen solamente sus cabeceras; `evaluation.py` y `evaluate` quedan pendientes.

### Procedencia y permisos

Se comprobaron por HTTP las tres políticas oficiales (200). Se almacenan autor, fuente, URL, licencia y enlace de licencia por noticia; no se descargan imágenes ni se sortean restricciones de acceso.

- **Global Voices:** resúmenes de RSS de economía/negocios, deportes y tecnología, [CC BY 3.0 y política de atribución](https://globalvoices.org/about/global-voices-attribution-policy/). Se conservan autor y enlace; se rechazan republicaciones y avisos particulares detectados en el contenido/derechos del feed. Se respeta la pausa de diez segundos solicitada por su `robots.txt`.
- **Wikinews inglés:** texto introductorio del wikitext público, sin imágenes, tablas ni apartado de fuentes. Su [política oficial](https://en.wikinews.org/wiki/Wikinews:Copyright) permite atribuir a Wikinews. En este corpus hay 64 textos de dominio público, 1,701 CC BY 2.5 y 15 CC BY 4.0. Las marcas explícitas del artículo prevalecen sobre la regla por fecha. Solo se aceptan páginas publicadas, sin marcas de infracción. Es un [archivo de solo lectura desde mayo de 2026](https://en.wikinews.org/wiki/Wikinews:Closure_of_Wikinews), no una fuente semanal.
- **EFF Deeplinks:** medio editorial independiente sobre tecnología y derechos digitales, obtenido mediante [RSS oficial](https://www.eff.org/rss/updates.xml). Su [política](https://www.eff.org/copyright) autoriza reutilización y adaptación de material original bajo CC BY 4.0. Se excluyen republicaciones detectadas, entradas promocionales/podcasts y bloques HTML de citas de terceros; no se extiende esa licencia a materiales ajenos.

Los textos originales quedan en `news_raw_corpus.tsv`. La versión normalizada y los vectores son transformaciones para NLP; esta documentación identifica esos cambios y conserva la atribución. Los filtros de derechos son conservadores y dependen de los avisos publicados por las fuentes; no constituyen una auditoría individual de todos los derechos de terceros.

### Actualización real

Se ejecutaron `init`, `update --dry-run`, `update` y su repetición sobre el corpus definitivo. Resultado semanal: **0 finanzas, 0 deportes, 0 tecnología**, con faltantes 10/10/10 frente a la meta. Las noticias recientes admisibles de finanzas/tecnología ya estaban almacenadas; el feed deportivo más reciente es del 5 de septiembre, fuera de la ventana. Cinco avisos de exclusión por derechos aparecen en el informe; no hubo errores HTTP en esta ejecución. No se sustituyeron faltantes por noticias antiguas. Hashes y tiempos de modificación de los archivos de adquisición permanecieron iguales.

Los casos de incorporaciones positivas parciales, acumulación semanal y apertura del siguiente viernes se verifican con datos simulados en las pruebas unitarias. No se atribuyen esas incorporaciones a la actualización HTTP real. La disponibilidad deportiva semanal sigue dependiendo de Global Voices y no está garantizada.

## Normalización y VSM

Antes de aplicar el modelo se verifican los dominios de las ediciones inglesas y señales léxicas inglesas en título+cuerpo de cada registro. Los 2,000 registros superaron esta comprobación. Es una comprobación conservadora de procedencia/vocabulario, no un detector multilingüe universal; pueden existir nombres y citas breves en otros idiomas dentro de noticias inglesas.

Se usa `en_core_web_sm` 3.8.0 con spaCy 3.8.16, tokenización, POS y lemas; NER/parser están desactivados. Se limpian HTML, controles, espacios y puntuación; se pasa a minúsculas. Solo se eliminan stopwords etiquetadas como preposición, conjunción o pronombre y los artículos `a/an/the` con etiqueta DT. Se descartan clíticos posesivos POS; se conservan números, nombres, vocabulario técnico, negación y auxiliares. `normalize_text` es compartida por documentos y consultas.

Ejemplo real del corpus:

> Digital authoritarianism is not developing in isolation, but through the international exchange of technologies, standards, and practices.

```text
digital authoritarianism be not develop isolation international exchange technology standard practice
```

El TSV normalizado conserva todos los campos originales y añade `language`, `normalized_title` y `normalized_body`, sin cambiar IDs ni orden. El corpus original no se sobrescribe durante `process`.

Se generan **tres matrices por cada fila de esta tabla**, es decir, las doce combinaciones requeridas. El número de columnas coincide con el vocabulario de cada espacio:

| Contenido | N-gramas | Frecuencia | Binaria | TF-IDF |
|---|---|---|---|---|
| Título | 1 | 2000 × 4772 | 2000 × 4772 | 2000 × 4772 |
| Título | 2 | 2000 × 11392 | 2000 × 11392 | 2000 × 11392 |
| Cuerpo | 1 | 2000 × 28266 | 2000 × 28266 | 2000 × 28266 |
| Cuerpo | 2 | 2000 × 281917 | 2000 × 281917 | 2000 × 281917 |

`results/vectors.joblib` contiene un diccionario con `ids`, `documents`, `spaces`, versiones del modelo y hash lógico del corpus. Cada espacio, por ejemplo `body_1_tfidf`, tiene `vectorizer` y `matrix` CSR. Frecuencia/binaria usan CountVectorizer; TF-IDF usa TfidfVectorizer. Los n-gramas son exclusivamente `(1,1)` o `(2,2)`, sobre los tokens ya normalizados; se conservan también tokens de un carácter. El archivo (~53 MiB) está generado localmente y excluido de Git; se reconstruye con `process`.

### Recuperación comprobada

Configuración predeterminada, resultados ordenados por coseno:

| Consulta | Primer resultado | Puntuación |
|---|---|---:|
| European Union semiconductor production Chips Act 43 billion | EU agrees €43 billion Chips Act to boost semiconductor production | 0.407940 |
| chips investment markets | EU agrees €43 billion Chips Act to boost semiconductor production | 0.253131 |
| zzzxxyyqqvv | Sin coincidencias; diez documentos con cero | 0.000000 |

Los treinta resultados con ID, título, URL y puntuación están en [results/validation_report.json](results/validation_report.json), junto con fechas por año, dimensiones, vocabularios, políticas HTTP, ejemplos y resultados de adquisición. Estos ejemplos son pruebas funcionales, no el benchmark formal de quince consultas.

## Verificación y límites

Las 23 pruebas unitarias pasan y usan fixtures y no realizan tráfico HTTP. Cubren extracción, licencias, fechas, cuotas, deduplicación, escrituras, idioma, POS, lemas, stopwords selectivas, números, doce espacios, persistencia y las configuraciones de búsqueda. La integración ejecutada por separado verificó las 2,000 filas, metadatos/IDs, las doce matrices dispersas, valores enteros/binarios/TF-IDF, transformación tras recarga y las tres consultas reales. Dos ejecuciones completas produjeron el mismo SHA-256 del TSV normalizado y del archivo de vectores. Se validó el YAML del workflow y sus comandos localmente; no se ejecutó un job alojado en GitHub.

Quedan para otra iteración el benchmark y sus métricas. La principal limitación externa actual es la ausencia de novedades deportivas dentro de la ventana semanal; los faltantes se informan, no se ocultan.
