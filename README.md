# NewsRetrieval — Fase 1

Proyecto escolar de extracción de noticias en Python. Consulta RSS/Atom de Global Voices y el archivo histórico de Wikinews, limpia texto, valida metadatos, elimina duplicados por URL canónica y guarda TSV UTF-8. La única dependencia externa es `requests`.

`retrieval.py` y `evaluation.py` siguen como estructuras vacías. Los TSV de corpus normalizado, consultas, juicios, rankings y métricas conservan sus encabezados para las siguientes fases.

## Preparación y comandos

Requiere Python 3.10 o posterior y una base de zonas horarias del sistema; se verificó con Python 3.12 en Linux. En este entorno el ejecutable es `python3`; dentro del entorno virtual también está disponible `python`.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest -v test_project
python main.py init --target 30 --max-batches 2 --dry-run
python main.py init --target 2000
python main.py update --per-category 10
python main.py update --dry-run
```

`init --target` indica la meta total, repartida entre finanzas, deportes y tecnología (667/667/666 para 2,000). Conserva lo ya guardado y busca los faltantes por categoría. Primero consulta los feeds y luego pagina el archivo en lotes de 50 páginas, con un máximo de 80 lotes por categoría. Si se agota una categoría, el límite o falla una consulta, informa la cantidad obtenida y el motivo. Una ejecución posterior vuelve a calcular los faltantes; los resultados de una ejecución se guardan al terminar, mediante reemplazo atómico del TSV.

`update` consulta solamente los feeds, con una ventana de publicación reciente de 8 días (`--days`). La cuota se comparte entre todas las ejecuciones del mismo periodo: viernes 00:00 a jueves 23:59, hora de Ciudad de México. Con `--per-category 10`, volver a ejecutar el comando busca únicamente lo que falta de esas diez noticias por categoría. El siguiente viernes abre otra cuota. El archivo histórico nunca se usa para completar novedades semanales.

La cuota consumida se conserva en `results/collection_log.jsonl`: mantenerlo junto al corpus y no borrarlo entre actualizaciones. Solo se agrega una entrada cuando se guardan noticias. Las ejecuciones sin novedades muestran su informe en pantalla y no modifican archivos. La función `collect_news(path=..., log_path=...)` permite destinos temporales; sin `log_path`, un corpus alternativo utiliza un JSONL con su mismo nombre.

`--dry-run` consulta y valida, pero no crea ni modifica corpus, bitácora o cuotas persistentes. En su informe, `added` representa candidatos que se guardarían y `total_after` conserva el tamaño persistido. `per_category_missing` indica los faltantes de esta ejecución; `weekly_already_added` muestra la cuota consumida. Los rechazos por derechos particulares también se explican en `errors`. Los faltantes se avisan sin descartar los resultados válidos; los argumentos inválidos y errores locales de integridad terminan con error.

## Fuentes, categorías y atribución

| Categoría | RSS de Global Voices | Categoría histórica de Wikinews |
|---|---|---|
| finanzas | [Economics & Business](https://globalvoices.org/-/topics/economics-business/feed/) | [Economy and business](https://en.wikinews.org/wiki/Category:Economy_and_business) |
| deportes | [Sport](https://globalvoices.org/-/topics/sport/feed/) | [Sports](https://en.wikinews.org/wiki/Category:Sports) |
| tecnologia | [Technology](https://globalvoices.org/-/topics/technology/feed/) | [Science and technology](https://en.wikinews.org/wiki/Category:Science_and_technology) |

Son categorías editoriales amplias: finanzas incluye economía y negocios; tecnología incluye ciencia. La asignación procede del feed o de la categoría de Wikinews, sin un clasificador automático. Puede haber temas compartidos; se conserva una sola copia y la primera categoría que lo incorpora. No se garantiza pertinencia temática estricta de cada artículo. El muestreo histórico confirmó, por ejemplo, noticias sobre empleo de Hewlett-Packard, los Juegos Mediterráneos y propulsión espacial.

- **Global Voices:** se almacena únicamente la descripción/resumen editorial del feed, con autor, título y enlace, bajo [CC BY 3.0 y su política de atribución](https://globalvoices.org/about/global-voices-attribution-policy/). El contenido completo del feed se examina para excluir avisos de republicación, publicación original externa o derechos reservados; una declaración individual de derechos no reconocida también se omite. No se asigna CC BY 3.0 a esas republicaciones. No se descargan fotografías ni se guardan sus pies. Se rechazan entradas sin resumen o autor. Estos controles dependen de los avisos suministrados por la fuente; no constituyen una revisión jurídica individual de todo el sitio. [Directorio oficial de feeds](https://globalvoices.org/feeds/).
- **Wikinews:** la [política oficial de copyright](https://en.wikinews.org/wiki/Wikinews:Copyright) permite atribuir el texto a “Wikinews”. Antes del 25/09/2005 se registra dominio público; del 25/09/2005 al 15/12/2024, CC BY 2.5; desde el 16/12/2024, CC BY 4.0. La plantilla individual `Archived-cc-2.5`, `Archived-cc-4.0` o `Archived-pd`, si está presente, prevalece sobre la regla por fecha. Se exige una marca de publicación, fecha interpretable y al menos 18 palabras, y se rechazan marcas reconocidas de infracción de copyright. Se excluyen las recopilaciones `Digest/` y se eliminan medios, galerías, tablas, referencias, fuentes y plantillas auxiliares; se conservan los textos visibles de enlaces y plantillas `w`. La conversión es sencilla: se omiten entradas con delimitadores wiki sin resolver. No es una reproducción íntegra de la página original.
- El [cierre de Wikinews en mayo de 2026](https://meta.wikimedia.org/wiki/Wikinews) lo convierte en una fuente histórica. Su [API MediaWiki](https://en.wikinews.org/w/api.php) sigue siendo el mecanismo usado para leer categorías y revisiones; no se sortean bloqueos. Las consultas tienen timeout de 25 segundos y hasta tres intentos ante fallos de conexión o estados transitorios; los HTTP permanentes no se reintentan.

Al distribuir el corpus, acompañarlo con esta declaración: **“Textos obtenidos de las fuentes enlazadas; resúmenes RSS o texto de Wikinews convertido a texto plano, con eliminación de formato y material auxiliar. Se conservan título, autor, fuente, fecha, URL y licencia por registro.”** Mantener estos metadatos y las condiciones de cada licencia. El uso académico no concede permisos adicionales sobre textos o medios de terceros.

## Archivos persistentes

`data/news_raw_corpus.tsv` conserva exactamente estas once columnas:

```text
id  title  description/body  source  category  date  url  retrieved_at  author  license  license_url
```

El separador real es tabulación y la codificación es UTF-8. `date` contiene la fecha de publicación UTC; `retrieved_at`, la fecha y hora de recuperación UTC. El ID deriva de la URL, eliminando fragmentos y parámetros habituales de seguimiento. Se escribe mediante `csv`, de forma atómica, ordenando por fecha e ID. Un corpus con columnas faltantes o registros incompletos causa un error antes de escribir.

`results/collection_log.jsonl` registra las incorporaciones y permite mantener las cuotas semanales. Los demás TSV permanecen reservados, sin contenido generado para fases posteriores.

## Automatización

`.github/workflows/weekly.yml` programa `0 16 * * 5`: viernes a las 10:00 CDMX (UTC−6). Permite ejecución manual mediante `workflow_dispatch`, instala dependencias, ejecuta las pruebas y después `python main.py update --per-category 10 --days 8`. Publica el informe en el resumen de Actions y muestra advertencias de faltantes. Serializa las ejecuciones mediante `concurrency`.

El workflow guarda el corpus y la bitácora si hay cambios; una ejecución sin incorporaciones no provoca commit. Necesita permisos de escritura del token y estar instalado en un repositorio GitHub. Esta copia local no contiene `.git`; no se ejecutaron Actions ni pushes remotos. El horario y el comando local se revisaron; la ejecución remota sigue pendiente. GitHub puede retrasar los cron programados.

## Validación real de esta auditoría — 15/10/2026

Las pruebas unitarias usan ejemplos simulados; las siguientes comprobaciones de extracción utilizaron solicitudes HTTP reales, sin sustituir sus respuestas:

- `python3 -m unittest -v test_project`: 14 pruebas correctas, incluyendo RSS/Atom, atribución, limpieza wiki, paginación, TSV, cuotas entre ejecuciones y cambio de viernes, deduplicación, reanudación y modo de prueba.
- Los tres endpoints RSS y la API respondieron HTTP 200 con XML/JSON válidos. Se recuperaron documentos históricos completos mediante consultas de categorías y revisiones, no solamente respuestas de estado satisfactorio.
- Muestra de feeds: 15 entradas por tema; se aceptaron 12 de finanzas, 15 de deportes y 13 de tecnología antes de deduplicar entre temas. Se excluyeron cinco apariciones por avisos de republicación/derechos. Se extrajeron además dos artículos históricos válidos por categoría.
- `python3 main.py init --target 30 --max-batches 2 --dry-run`: 30 candidatos reales, 10 por tema, todos de Global Voices. Sin escrituras.
- Persistencia temporal: se guardaron 90 documentos (38 Global Voices y 52 Wikinews), se reejecutó con cero incorporaciones y después se amplió a 93 (31 por tema). Se verificaron las once columnas, UTF-8 con acentos, URLs e IDs únicos y conservación byte por byte de corpus y bitácora durante el modo de prueba.
- `python3 main.py update --dry-run`, sobre corpus vacío: 6 candidatos recientes (3 finanzas, 0 deportes, 3 tecnología), todos de Global Voices; faltaron 7/10/7 para la cuota de diez. La misma actualización en archivos temporales guardó seis y, al repetirla, añadió cero.

- `python3 main.py init --target 2000`: alcanzó la meta. La revisión del primer resultado descubrió galerías sin limpiar y cuatro recopilaciones `Digest/`; se corrigió el extractor y se repitió la carga completa en un destino temporal antes de guardar el corpus definitivo del proyecto, que originalmente solo tenía encabezados. La carga corregida realizó **99 solicitudes HTTP en 23.5 segundos**. Los resultados definitivos son:

| Categoría | Global Voices | Wikinews | Total |
|---|---:|---:|---:|
| finanzas | 12 | 655 | 667 |
| deportes | 14 | 653 | 667 |
| tecnologia | 12 | 654 | 666 |
| **Total** | **38** | **1,962** | **2,000** |

El corpus definitivo está en `data/news_raw_corpus.tsv`, con su informe en `results/collection_log.jsonl`. Se comprobaron 2,000 IDs y 2,000 URLs distintos, campos obligatorios completos y ausencia de etiquetas HTML, delimitadores wiki, residuos de imágenes/galerías y páginas `Digest/`. Se revisaron textos de muestra y seis enlaces de artículos (una muestra por categoría y fuente) respondieron HTTP 200. Esto acredita el volumen inicial obtenido, sin extrapolar cifras de páginas catalogadas.

Repetir `init --target 2000` añadió cero noticias. `update --dry-run` y dos ejecuciones de `update --per-category 10`, sobre el corpus ya cargado, añadieron cero: las seis noticias recientes elegibles ya estaban guardadas. Se compararon hashes SHA-256 de corpus y bitácora para verificar que permanecían idénticos. Los resultados de la prueba semanal sobre corpus vacío se distinguen de estos resultados posteriores a la carga inicial.

La limitación semanal es real: el feed deportivo más reciente de la muestra era del 05/09/2026, fuera de la ventana de ocho días. Global Voices no garantiza diez noticias por tema cada semana. Si se exige completar esa cuota obligatoriamente, se necesitaría una fuente adicional con suficiente frecuencia y permiso explícito para almacenar su texto; no se añaden proveedores sin verificar esas condiciones ni se rellenan faltantes con históricos.
