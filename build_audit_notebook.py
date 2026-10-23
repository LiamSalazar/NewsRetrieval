"""Genera el notebook ejecutable para la auditoría académica de NewsRetrieval."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "NewsRetrieval_Pipeline.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(r"""
# NewsRetrieval — Pipeline reproducible y auditoría académica

Este notebook demuestra de principio a fin el sistema léxico de recuperación de 2,000 noticias en inglés. Reutiliza el corpus y las doce matrices persistidas: **no descarga noticias, no reconstruye vectores, no modifica consultas y no altera relevancias**.

El benchmark permanece **PROVISIONAL**. La ejecución incluida usa el modo provisional exclusivamente como verificación técnica; los 303 qrels candidatos requieren aprobación humana y no sustituyen automáticamente `data/qrels.tsv`.
"""),
    md(r"""
## 1. Introducción y configuración

El objetivo es recuperar noticias con el modelo de espacio vectorial (VSM) en doce configuraciones: dos campos (título/cuerpo), dos granularidades (unigramas/bigramas) y tres representaciones (frecuencia/binaria/TF-IDF). Todas las rutas se resuelven desde la raíz del proyecto y las funciones provienen de los módulos productivos.
"""),
    code(r"""
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
from html import escape
from importlib import metadata
from pathlib import Path
import csv
import platform

import joblib
import spacy
import sklearn
from IPython.display import HTML, Markdown, display
from sklearn.metrics.pairwise import cosine_similarity

from collector import read_corpus, validate_corpus
from evaluation import evaluate, load_benchmark, validate_benchmark
from retrieval import STOP_POS, clean_text, load_nlp, normalize_text, search_news


def find_project_root(start=Path.cwd()):
    for candidate in (start, *start.parents):
        if (candidate / "data" / "news_raw_corpus.tsv").exists() and (candidate / "retrieval.py").exists():
            return candidate
    raise FileNotFoundError("No se encontró la raíz de NewsRetrieval")


ROOT = find_project_root()
DATA = ROOT / "data"
RESULTS = ROOT / "results"
VECTORS = RESULTS / "vectors.joblib"


def read_tsv(path, comments=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        lines = (line for line in stream if not (comments and line.startswith("#")))
        return list(csv.DictReader(lines, delimiter="\t"))


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def html_table(rows, columns=None, max_rows=20, title=None):
    rows = list(rows)
    columns = columns or (list(rows[0]) if rows else [])
    shown = rows[:max_rows]
    def formatted(value):
        return f"{value:0.6f}" if isinstance(value, float) else str(value)
    head = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "".join("<tr>" + "".join(
        f"<td>{escape(formatted(row.get(column, '')))}</td>"
        for column in columns) + "</tr>" for row in shown)
    caption = f"<caption><strong>{escape(title)}</strong></caption>" if title else ""
    suffix = f"<p><em>Se muestran {len(shown)} de {len(rows)} filas.</em></p>" if len(rows) > len(shown) else ""
    display(HTML(f"<table>{caption}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{suffix}"))


def bar_chart(rows, label_key, value_key, title, width=760, height=330):
    rows = list(rows)
    maximum = max((float(row[value_key]) for row in rows), default=1.0) or 1.0
    left, top, plot_width, row_height = 155, 35, width - 190, max(18, (height - 55) / max(1, len(rows)))
    items = [f'<text x="{width/2}" y="20" text-anchor="middle" font-weight="bold">{escape(title)}</text>']
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = float(row[value_key])
        bar_width = plot_width * value / maximum
        items.append(f'<text x="{left-8}" y="{y+12}" text-anchor="end" font-size="11">{escape(str(row[label_key]))}</text>')
        items.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="13" fill="#377eb8"/>')
        items.append(f'<text x="{left+bar_width+5:.1f}" y="{y+11}" font-size="10">{value:.3f}</text>')
    display(HTML(f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">{"".join(items)}</svg>'))


versions = [{"componente": name, "versión": version} for name, version in [
    ("Python", platform.python_version()), ("spaCy", spacy.__version__),
    ("scikit-learn", sklearn.__version__), ("joblib", joblib.__version__),
    ("modelo en_core_web_sm", load_nlp().meta["version"]),
]]
display(Markdown(f"**Raíz detectada:** `{ROOT.name}/` — **ejecución:** {datetime.now().isoformat(timespec='seconds')}"))
html_table(versions, title="Dependencias efectivamente cargadas")
"""),
    md(r"""
## 2. Corpus: procedencia, cobertura y ejemplos originales

Se carga el TSV persistido; no se invocan funciones de adquisición. Cada noticia conserva ID, fuente, categoría, fecha, URL, autor y licencia. La unidad documental es una noticia y el cuerpo procede del extracto almacenado por el recolector.
"""),
    code(r"""
raw_rows = read_corpus(DATA / "news_raw_corpus.tsv")
normalized_rows = read_tsv(DATA / "news_normalized_corpus.tsv")
validate_corpus(raw_rows)
assert len(raw_rows) == len(normalized_rows) == 2000
assert [row["id"] for row in raw_rows] == [row["id"] for row in normalized_rows]

source_counts = Counter(row["source"] for row in raw_rows)
category_counts = Counter(row["category"] for row in raw_rows)
dates = [row["date"] for row in raw_rows]
corpus_summary = [
    {"indicador": "documentos", "valor": len(raw_rows)},
    {"indicador": "fuentes", "valor": ", ".join(f"{k}: {v}" for k, v in sorted(source_counts.items()))},
    {"indicador": "categorías", "valor": ", ".join(f"{k}: {v}" for k, v in sorted(category_counts.items()))},
    {"indicador": "intervalo de fechas", "valor": f"{min(dates)} — {max(dates)}"},
    {"indicador": "IDs únicos", "valor": len({row['id'] for row in raw_rows})},
    {"indicador": "URLs únicas", "valor": len({row['url'] for row in raw_rows})},
]
html_table(corpus_summary, title="Resumen del corpus persistido")

examples = []
for source in sorted(source_counts):
    row = next(item for item in raw_rows if item["source"] == source)
    examples.append({"id": row["id"], "fuente": source, "categoría": row["category"],
                     "fecha": row["date"], "título": row["title"],
                     "cuerpo_original": row["description/body"][:280] + ("…" if len(row["description/body"]) > 280 else ""),
                     "url": row["url"], "licencia": row["license"]})
html_table(examples, max_rows=10, title="Ejemplos originales por fuente")
"""),
    md(r"""
## 3. Normalización lingüística

Documentos y consultas pasan por el mismo `normalize_text`: limpieza de caracteres de control/HTML, tokenización de spaCy, POS tagging, eliminación selectiva de puntuación y de artículos, preposiciones, conjunciones y pronombres, y lematización en minúsculas. Cada campo se procesa por separado y el resultado se une con un único espacio.
"""),
    code(r"""
sample = next(row for row in raw_rows if len(row["description/body"].split()) >= 100)
analysis_text = f"{sample['title']} {sample['description/body']}"
doc = load_nlp()(clean_text(analysis_text))

def removal_reason(token):
    lemma = token.lemma_.lower().strip()
    if token.is_space: return "espacio"
    if token.is_punct: return "puntuación"
    if token.tag_ == "POS" or lemma in {"'s", "’s"}: return "posesivo"
    if not any(character.isalnum() for character in lemma): return "sin caracteres alfanuméricos"
    if token.is_stop and token.pos_ in STOP_POS: return "stopword por POS"
    if token.is_stop and token.tag_ == "DT" and token.lower_ in {"a", "an", "the"}: return "artículo"
    return "conservar"

token_rows = [{"token": token.text, "POS": token.pos_, "tag": token.tag_,
               "lema": token.lemma_.lower().strip(), "decisión": removal_reason(token)}
              for token in doc[:45]]
display(Markdown(f"**Documento auténtico:** `{sample['id']}` — {sample['title']}"))
html_table(token_rows, max_rows=45, title="Traza token/POS/lema")

normalized_sample = next(row for row in normalized_rows if row["id"] == sample["id"])
field_examples = [
    {"campo": "título", "original": sample["title"], "normalizado_calculado": normalize_text(sample["title"]),
     "normalizado_persistido": normalized_sample["normalized_title"]},
    {"campo": "cuerpo", "original": sample["description/body"][:420] + "…",
     "normalizado_calculado": normalize_text(sample["description/body"])[:420] + "…",
     "normalizado_persistido": normalized_sample["normalized_body"][:420] + "…"},
]
assert normalize_text(sample["title"]) == normalized_sample["normalized_title"]
assert normalize_text(sample["description/body"]) == normalized_sample["normalized_body"]
html_table(field_examples, title="Original frente a normalizado")
"""),
    md(r"""
## 4. Doce representaciones VSM persistidas

Las matrices se cargan de `results/vectors.joblib`. No se vuelven a ajustar los vectorizadores. Para cada combinación se informa dimensión, vocabulario y valores no nulos de un documento real.
"""),
    code(r"""
bundle = joblib.load(VECTORS)
assert len(bundle["ids"]) == 2000 and len(bundle["spaces"]) == 12
sample_index = bundle["ids"].index(sample["id"])
space_rows = []
for key, space in bundle["spaces"].items():
    matrix = space["matrix"]
    vectorizer = space["vectorizer"]
    vector = matrix.getrow(sample_index)
    names = vectorizer.get_feature_names_out()
    pairs = sorted(((names[column], float(value)) for column, value in zip(vector.indices, vector.data)),
                   key=lambda item: (-item[1], item[0]))[:5]
    field, ngram, representation = key.split("_", 2)
    space_rows.append({"configuración": key, "campo": field, "n": ngram,
                       "representación": representation, "forma": str(matrix.shape),
                       "vocabulario": len(names), "no_nulos_ejemplo": vector.nnz,
                       "términos_y_valores": "; ".join(f"{term}={value:.4f}" for term, value in pairs)})
html_table(space_rows, max_rows=12, title="Dimensiones y ejemplos reales de las 12 matrices")
"""),
    md(r"""
## 5. Recuperación y similitud coseno

La consulta se normaliza con la misma función, se transforma con el vectorizador ya ajustado y se compara mediante coseno con la matriz seleccionada. El Top-10 siguiente es real e incluye ID, título, puntuación y enlace.
"""),
    code(r"""
queries = read_tsv(DATA / "queries.tsv")
demo_query = queries[0]
configuration = "body_1_tfidf"
space = bundle["spaces"][configuration]
normalized_query = normalize_text(demo_query["query"])
query_vector = space["vectorizer"].transform([normalized_query])
manual_scores = cosine_similarity(query_vector, space["matrix"]).ravel()
manual_order = sorted(range(len(manual_scores)), key=lambda index: (-manual_scores[index], index))[:10]
top10 = search_news(demo_query["query"], field="body", ngram=1, representation="tfidf",
                    top_k=10, bundle=bundle)
assert [row["id"] for row in top10] == [bundle["ids"][index] for index in manual_order]
assert all(abs(row["score"] - manual_scores[index]) < 1e-12 for row, index in zip(top10, manual_order))
display(Markdown(f"**{demo_query['query_id']} original:** {demo_query['query']}  \n"
                 f"**Normalizada:** `{normalized_query}`  \n"
                 f"**Vector de consulta:** {query_vector.shape[1]} dimensiones, {query_vector.nnz} valores no nulos"))
retrieval_rows = [{"rango": rank, "id": row["id"], "título": row["title"],
                   "score_coseno": row["score"], "enlace": row["url"]}
                  for rank, row in enumerate(top10, 1)]
html_table(retrieval_rows, max_rows=10, title="Top-10 real — cuerpo, unigramas, TF-IDF")
"""),
    md(r"""
## 6. Benchmark provisional

Hay quince consultas: tres formulaciones (easy/medium/hard, codificadas 1/2/3) para cinco necesidades. `qrels.tsv` continúa vigente pero provisional. La referencia ampliada candidata contiene todos los positivos propuestos tras la adjudicación del pool y **no se aplica sin aprobación humana**. El intervalo académico de 10–15 se usa para exposición; no justifica descartar positivos auténticos ni crea una segunda referencia de evaluación.
"""),
    code(r"""
qrels_rows = read_tsv(DATA / "qrels.tsv", comments=True)
candidate_rows = read_tsv(DATA / "qrels_candidates_for_approval.tsv")
pool_review = read_tsv(DATA / "pool_human_review.tsv")
status_rows = read_tsv(DATA / "benchmark_status.tsv")
assert len(queries) == 15 and len(candidate_rows) == 303 and len(pool_review) == 941
assert all(row["approval_status"] == "propuesta_ia_pendiente_aprobacion_humana" for row in candidate_rows)

qrel_counts = Counter(row["query_id"] for row in qrels_rows)
candidate_counts = Counter(row["query_id"] for row in candidate_rows)
query_rows = [{**row, "qrels_vigentes_provisionales": qrel_counts[row["query_id"]],
               "qrels_candidatos_pendientes": candidate_counts[row["query_id"]]}
              for row in queries]
html_table(status_rows, title="Estado explícito del benchmark")
html_table(query_rows, max_rows=15, title="Las 15 consultas y sus juicios")

label_counts = Counter(row["proposed_relevance"] for row in pool_review)
html_table([{"artefacto": "pool adjudicado", "filas": len(pool_review), "detalle": str(dict(sorted(label_counts.items())))},
            {"artefacto": "qrels.tsv vigente", "filas": len(qrels_rows), "detalle": "provisional; no sustituido"},
            {"artefacto": "qrels candidatos", "filas": len(candidate_rows), "detalle": "303 pendientes de aprobación humana"}],
           title="Trazabilidad de juicios")
"""),
    md(r"""
## 7. Evaluación provisional de las 180 combinaciones

Se ejecutan 15 consultas × 12 configuraciones. Las métricas son P@10, Recall@10, F1@10, MRR y nDCG@10 (ganancia graduada $2^{rel}-1$). Esta corrida verifica el código y las salidas, pero **no es el experimento congelado ni un resultado definitivo**. Usa únicamente `qrels.tsv` vigente y comprueba que el archivo no cambie.
"""),
    code(r"""
qrels_hash_before = file_sha256(DATA / "qrels.tsv")
benchmark_queries, benchmark_qrels = load_benchmark(DATA / "queries.tsv", DATA / "qrels.tsv")
validation = validate_benchmark(benchmark_queries, benchmark_qrels, raw_rows)
evaluation_report = evaluate(benchmark_queries, benchmark_qrels, bundle=bundle,
                             output_dir=RESULTS / "provisional", provisional=True)
qrels_hash_after = file_sha256(DATA / "qrels.tsv")
assert qrels_hash_before == qrels_hash_after
assert evaluation_report == {"rankings": 150, "queries": 15, "configurations": 12,
                             "evaluations": 180, "provisional": True}

all_metric_rows = read_tsv(RESULTS / "provisional" / "metrics_by_query_configuration.tsv")
reference_metric_rows = read_tsv(RESULTS / "provisional" / "metrics.tsv")
comparison_rows = read_tsv(RESULTS / "provisional" / "comparison.tsv")
assert len(all_metric_rows) == 180 and len(comparison_rows) == 48
html_table([{"control": "benchmark", "resultado": str(validation)},
            {"control": "ejecución", "resultado": str(evaluation_report)},
            {"control": "SHA-256 qrels antes/después", "resultado": qrels_hash_before},
            {"control": "estado", "resultado": "PROVISIONAL — aprobación humana pendiente"}],
           title="Controles de la ejecución")
html_table(reference_metric_rows, max_rows=16, title="Resultados provisionales por consulta — referencia body_1_tfidf")
"""),
    md(r"""
## 8. Comparación por consulta, configuración y promedio

La tabla completa de 180 evaluaciones queda en `results/provisional/metrics_by_query_configuration.tsv`; los promedios por configuración y dificultad, en `comparison.tsv`. Las visualizaciones siguientes proceden de esa ejecución real y conservan la advertencia provisional.
"""),
    code(r"""
numeric_fields = ["precision@10", "recall@10", "f1@10", "mrr", "ndcg@10"]
for row in all_metric_rows + comparison_rows:
    for field in numeric_fields:
        row[field] = float(row[field])

mean_comparison = sorted((row for row in comparison_rows if row["difficulty_level"] == "all"),
                         key=lambda row: (-row["ndcg@10"], row["configuration"]))
html_table(mean_comparison, max_rows=12, title="Promedios provisionales de las 12 configuraciones")
bar_chart(mean_comparison, "configuration", "ndcg@10",
          "nDCG@10 medio por configuración — PROVISIONAL", height=350)

query_comparison = sorted((row for row in all_metric_rows if row["configuration"] == "body_1_tfidf"),
                          key=lambda row: row["query_id"])
bar_chart(query_comparison, "query_id", "ndcg@10",
          "nDCG@10 por consulta: cuerpo + unigramas + TF-IDF — PROVISIONAL", height=390)
"""),
    md(r"""
## 9. Análisis de errores preparado

La referencia primaria es TF-IDF sobre unigramas del cuerpo (`body_1_tfidf`). El código identifica cinco consultas de mayor y cinco de menor desempeño y aporta evidencia observable: cobertura del vocabulario normalizado, términos fuera del vocabulario y títulos recuperados. Esas señales permiten discutir vocabulario, sinónimos, ambigüedad, normalización, nombres propios y límites de coincidencia léxica. La interpretación es provisional hasta aprobar la referencia ampliada.
"""),
    code(r"""
reference_rows = sorted((row for row in all_metric_rows if row["configuration"] == "body_1_tfidf"),
                        key=lambda row: (row["ndcg@10"], row["mrr"], row["query_id"]))
worst_ids = {row["query_id"] for row in reference_rows[:5]}
best_ids = {row["query_id"] for row in reference_rows[-5:]}
query_by_id = {row["query_id"]: row for row in queries}
vocabulary = set(bundle["spaces"]["body_1_tfidf"]["vectorizer"].get_feature_names_out())

diagnostics = []
for metric in reference_rows:
    query_id = metric["query_id"]
    if query_id not in worst_ids | best_ids:
        continue
    normalized = normalize_text(query_by_id[query_id]["query"])
    tokens = normalized.split()
    oov = [token for token in tokens if token not in vocabulary]
    retrieved = search_news(query_by_id[query_id]["query"], "body", 1, "tfidf", bundle=bundle)
    signals = []
    if oov:
        signals.append("vocabulario fuera del índice: " + ", ".join(oov))
    else:
        signals.append("todos los unigramas están en el vocabulario")
    if metric["precision@10"] == 0:
        signals.append("sin positivos provisionales en Top-10: revisar sinónimos/ambigüedad")
    if any(token[:1].isupper() for token in query_by_id[query_id]["query"].split()):
        signals.append("posibles nombres propios")
    diagnostics.append({"grupo": "mejor" if query_id in best_ids else "peor",
                        "consulta": query_id, "dificultad": metric["difficulty_level"],
                        "nDCG@10": metric["ndcg@10"], "MRR": metric["mrr"],
                        "cobertura_vocabulario": f"{len(tokens)-len(oov)}/{len(tokens)}",
                        "señales_observables": "; ".join(signals),
                        "tres_primeros_títulos": " | ".join(row["title"] for row in retrieved[:3])})
diagnostics.sort(key=lambda row: (row["grupo"], -row["nDCG@10"], row["consulta"]))
html_table(diagnostics, max_rows=10, title="Cinco mejores y cinco peores: diagnóstico trazable provisional")

cross_configuration = []
for query_id in sorted(best_ids | worst_ids):
    rows = [row for row in all_metric_rows if row["query_id"] == query_id]
    reference = next(row for row in rows if row["configuration"] == "body_1_tfidf")
    strongest = max(rows, key=lambda row: (row["ndcg@10"], row["mrr"], row["configuration"]))
    weakest = min(rows, key=lambda row: (row["ndcg@10"], row["mrr"], row["configuration"]))
    cross_configuration.append({"consulta": query_id, "referencia_nDCG": reference["ndcg@10"],
                                "mejor_configuración": strongest["configuration"],
                                "mejor_nDCG": strongest["ndcg@10"],
                                "peor_configuración": weakest["configuration"],
                                "peor_nDCG": weakest["ndcg@10"]})
html_table(cross_configuration, max_rows=10, title="Comparación de las consultas extremas con las demás configuraciones")
"""),
    md(r"""
## 10. Reproducibilidad, controles y conclusiones

Fuentes: Wikinews en inglés, Global Voices y EFF, con URL y licencia por registro. Restricciones: recuperación léxica sin embeddings; sensibilidad a sinónimos, paráfrasis, ambigüedad, nombres propios y cobertura del extracto local. El corpus y los vectores son persistidos y su integridad se comprueba abajo.

Ejecución recomendada desde la raíz:

```text
python -m unittest -v test_project
python main.py --benchmark
python main.py --benchmark --provisional
```

No se usa `--frozen` mientras falte la aprobación humana. Tras esa aprobación se deberá ejecutar el experimento definitivo por separado.
"""),
    code(r"""
required_files = [
    DATA / "news_raw_corpus.tsv", DATA / "news_normalized_corpus.tsv", VECTORS,
    DATA / "queries.tsv", DATA / "qrels.tsv", DATA / "qrels_candidates_for_approval.tsv",
    DATA / "pool_human_review.tsv", RESULTS / "provisional" / "metrics_by_query_configuration.tsv",
]
integrity_rows = [{"archivo": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                   "sha256": file_sha256(path)} for path in required_files]

assert len(raw_rows) == 2000
assert len({row["id"] for row in raw_rows}) == 2000
assert len(bundle["spaces"]) == 12
assert len(queries) == 15 and len({row["query_id"] for row in queries}) == 15
assert len(pool_review) == len({(row["need_id"], row["document_id"]) for row in pool_review}) == 941
assert len(candidate_rows) == len({(row["query_id"], row["document_id"]) for row in candidate_rows}) == 303
assert file_sha256(DATA / "qrels.tsv") == qrels_hash_before
html_table(integrity_rows, max_rows=len(integrity_rows), title="Huella de artefactos críticos")
display(Markdown(
    "**Conclusión técnica:** el pipeline, las doce matrices, las quince consultas y la evaluación provisional "
    "son reproducibles desde archivos persistidos. El benchmark **no es definitivo**: los 303 qrels candidatos "
    "continúan pendientes de auditoría y aprobación humana; el núcleo expositivo de 10–15 no sustituye la "
    "referencia ampliada ni debe emplearse para ocultar positivos auténticos."
))
"""),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
nbf.write(notebook, NOTEBOOK)
print(f"Notebook generado: {NOTEBOOK} ({len(cells)} celdas)")
