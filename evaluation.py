
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from retrieval import VECTORS, search_news


CONFIGURATIONS = [(field, ngram, representation)
                  for field in ("title", "body")
                  for ngram in (1, 2)
                  for representation in ("frequency", "binary", "tfidf")]
MAIN_CONFIGURATION = "body_1_tfidf"
NEEDS = ("trade", "energy", "football", "cricket", "surveillance")
NEED_BY_QUERY = {f"Q{number:02d}": NEEDS[(number - 1) // 3] for number in range(1, 16)}
METRICS = ("precision@10", "recall@10", "f1@10", "mrr", "ndcg@10")


def _read_tsv(path, comments=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        lines = (line for line in stream if not (comments and line.startswith("#")))
        return list(csv.DictReader(lines, delimiter="\t"))


def load_benchmark(queries_path="data/queries.tsv", qrels_path="data/qrels.tsv"):
    
    queries = _read_tsv(queries_path, comments=True)
    qrels_rows = _read_tsv(qrels_path, comments=True)
    if not queries or set(queries[0]) != {"query_id", "difficulty_level", "query"}:
        raise ValueError("queries.tsv debe tener query_id, difficulty_level y query")
    if qrels_rows and set(qrels_rows[0]) != {"query_id", "document_id", "relevance"}:
        raise ValueError("qrels.tsv debe tener query_id, document_id y relevance")
    query_ids = {row["query_id"] for row in queries}
    if len(query_ids) != len(queries):
        raise ValueError("query_id repetido")
    for row in queries:
        if row["difficulty_level"] not in {"1", "2", "3"}:
            raise ValueError("difficulty_level debe ser 1, 2 o 3")
    qrels = defaultdict(dict)
    for row in qrels_rows:
        if row["query_id"] not in query_ids:
            raise ValueError("qrels contiene una consulta inexistente")
        relevance = int(row["relevance"])
        if relevance not in {1, 2}:
            raise ValueError("qrels.tsv solo incluye relevancias positivas 1 y 2")
        if row["document_id"] in qrels[row["query_id"]]:
            raise ValueError("documento repetido en qrels")
        qrels[row["query_id"]][row["document_id"]] = relevance
    return queries, dict(qrels)


def validate_benchmark(queries, qrels, documents=None, allow_expanded=True):
    
    if len(queries) != 15:
        raise ValueError("El benchmark debe tener 15 consultas")
    levels = {str(level): sum(row["difficulty_level"] == str(level) for row in queries)
              for level in (1, 2, 3)}
    if levels != {"1": 5, "2": 5, "3": 5}:
        raise ValueError("Debe haber cinco consultas por dificultad")
    if len({row["query_id"] for row in queries}) != 15:
        raise ValueError("query_id repetido")
    document_ids = {row["id"] for row in documents} if documents is not None else None
    exceeded = {}
    counts = {}
    for query in queries:
        query_id = query["query_id"]
        rows = qrels.get(query_id, {})
        counts[query_id] = len(rows)
        if len(rows) < 10:
            raise ValueError(f"{query_id} debe tener al menos 10 relevantes")
        if len(rows) > 15:
            exceeded[query_id] = len(rows)
        if set(rows.values()) != {1, 2}:
            raise ValueError(f"{query_id} debe incluir relevancia 1 y 2")
        if document_ids is not None and not set(rows).issubset(document_ids):
            raise ValueError(f"{query_id} contiene documentos inexistentes")
    for start in (1, 4, 7, 10, 13):
        query_ids = [f"Q{number:02d}" for number in range(start, start + 3)]
        references = [{(document_id, grade) for document_id, grade in qrels[qid].items()}
                      for qid in query_ids]
        if not references[0] == references[1] == references[2]:
            raise ValueError(f"Las tres consultas {query_ids} no comparten la misma referencia")
    if exceeded and not allow_expanded:
        raise ValueError(f"La referencia supera 15 positivos: {exceeded}")
    if exceeded:
        warnings.warn("Referencia ampliada: se conservan positivos auténticos por encima del intervalo "
                      f"académico de 10-15: {exceeded}", UserWarning, stacklevel=2)
    return {"queries": len(queries), "difficulty_levels": levels,
            "relevant_judgments": sum(counts.values()), "relevant_by_query": counts,
            "expanded_reference": bool(exceeded), "academic_range_exceeded": exceeded}


def _document_id(result):
    return result.get("id") if isinstance(result, dict) else str(result)


def calculate_metrics(results, relevant, k=10):
    
    ranked = list(results)[:k]
    ids = [_document_id(row) for row in ranked]
    if len(ids) != len(set(ids)):
        raise ValueError("El ranking contiene IDs duplicados")
    grades = [int(relevant.get(document_id, 0)) for document_id in ids]
    hits = sum(grade > 0 for grade in grades)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reciprocal_rank = next((1.0 / rank for rank, grade in enumerate(grades, 1) if grade > 0), 0.0)
    dcg = sum((2 ** grade - 1) / math.log2(rank + 1)
              for rank, grade in enumerate(grades, 1))
    ideal_grades = sorted((int(value) for value in relevant.values()), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 1)
               for rank, grade in enumerate(ideal_grades, 1))
    return {"precision@10": precision, "recall@10": recall, "f1@10": f1,
            "mrr": reciprocal_rank, "ndcg@10": dcg / idcg if idcg else 0.0}


def calculate_metric_details(results, relevant, k=10):
    ranked = list(results)[:k]
    metrics = calculate_metrics(ranked, relevant, k=k)
    grades = [int(relevant.get(_document_id(row), 0)) for row in ranked]
    first_rank = next((rank for rank, grade in enumerate(grades, 1) if grade > 0), 0)
    return {"relevant_retrieved": sum(grade > 0 for grade in grades),
            "irrelevant_retrieved": sum(grade == 0 for grade in grades),
            "total_relevant_reference": len(relevant), "first_relevant_rank": first_rank,
            "top10_grades": grades, **metrics}


def _mean(rows, keys):
    return {key: sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0 for key in keys}


def _load_review_map(path="data/pool_human_review.tsv"):
    rows = _read_tsv(path)
    review = {}
    for row in rows:
        key = (row["need_id"], row["document_id"])
        if key in review:
            raise ValueError(f"Juicio consolidado duplicado: {key}")
        value = row["proposed_relevance"].strip()
        if value not in {"0", "1", "2"}:
            raise ValueError(f"Juicio consolidado no resuelto: {key}")
        review[key] = {"grade": int(value), "status": row["review_status"],
                       "evidence": row["justification"]}
    return review


def _load_positive_evidence(path="data/qrels_validation_evidence.tsv"):
    source = Path(path)
    if not source.exists():
        return {}
    return {(row["query_id"], row["document_id"]): row for row in _read_tsv(source)}


def evaluate(queries, qrels, bundle=None, output_dir="results", frozen=False,
             provisional=False, review_path="data/pool_human_review.tsv"):
    
    if not frozen and not provisional:
        raise RuntimeError("La evaluación requiere frozen=True o provisional=True")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = bundle if bundle is not None else __import__("joblib").load(VECTORS)
    review = _load_review_map(review_path)
    positive_evidence = _load_positive_evidence()
    all_metrics = []
    rankings_all = []
    judgments = []

    for field, ngram, representation in CONFIGURATIONS:
        configuration = f"{field}_{ngram}_{representation}"
        for query in queries:
            query_id = query["query_id"]
            need_id = NEED_BY_QUERY[query_id]
            results = search_news(query["query"], field, ngram, representation,
                                  top_k=10, bundle=bundle)
            if len(results) != 10 or len({row["id"] for row in results}) != 10:
                raise ValueError(f"Top-10 inválido para {query_id}/{configuration}")
            details = calculate_metric_details(results, qrels[query_id], k=10)
            unjudged = 0
            for rank, row in enumerate(results, 1):
                document_id = row["id"]
                score = float(row["score"])
                base = {"query_id": query_id, "difficulty_level": query["difficulty_level"],
                        "query": query["query"], "configuration": configuration,
                        "field": field, "ngram": ngram, "representation": representation,
                        "rank": rank, "document_id": document_id, "title": row["title"],
                        "url": row["url"], "score": score,
                        "zero_similarity": int(score == 0.0)}
                rankings_all.append(base)
                audit = review.get((need_id, document_id))
                if audit is None:
                    grade, status, evidence = "", "unjudged", ""
                    unjudged += 1
                else:
                    grade, status, evidence = audit["grade"], audit["status"], audit["evidence"]
                official_grade = qrels[query_id].get(document_id)
                if official_grade is not None:
                    if grade == "" or int(grade) != official_grade:
                        raise ValueError(f"Qrel positivo no coincide con revisión: {query_id}/{document_id}")
                    source = positive_evidence.get((query_id, document_id))
                    if source:
                        status, evidence = source["validation_status"], source["evidence"]
                elif grade in {1, 2}:
                    raise ValueError(f"Positivo consolidado ausente de qrels: {query_id}/{document_id}")
                elif grade == 0:
                    status = "negativo_confirmado_revision_semantica_delegada"
                is_relevant = "" if grade == "" else int(int(grade) > 0)
                outcome = "unjudged" if grade == "" else ("relevant" if is_relevant else "irrelevant_confirmed")
                judgments.append({**base, "relevance_grade": grade, "judgment_status": status,
                                  "is_relevant": is_relevant, "retrieval_outcome": outcome,
                                  "evidence": evidence})
            if details["relevant_retrieved"] != sum(
                    row["is_relevant"] == 1 for row in judgments[-10:]):
                raise AssertionError("El conteo relevante no coincide con la auditoría Top-10")
            all_metrics.append({"query_id": query_id,
                                "difficulty_level": query["difficulty_level"],
                                "configuration": configuration, "field": field,
                                "ngram": ngram, "representation": representation,
                                "relevant_retrieved": details["relevant_retrieved"],
                                "irrelevant_retrieved": 10 - details["relevant_retrieved"] - unjudged,
                                "unjudged_retrieved": unjudged,
                                "total_relevant_reference": details["total_relevant_reference"],
                                "zero_similarity_results": sum(row["score"] == 0.0 for row in results),
                                "first_relevant_rank": details["first_relevant_rank"],
                                **{metric: details[metric] for metric in METRICS}})

    if len(rankings_all) != 1800 or len(judgments) != 1800 or len(all_metrics) != 180:
        raise AssertionError("La evaluación no produjo 180 búsquedas y 1,800 resultados")
    if any(row["unjudged_retrieved"] for row in all_metrics):
        raise ValueError("La evaluación contiene resultados Top-10 sin juicio")

    main_rankings = [{"query_id": row["query_id"], "document_id": row["document_id"],
                      "rank": row["rank"], "score": row["score"]}
                     for row in rankings_all if row["configuration"] == MAIN_CONFIGURATION]
    main_metrics = [row for row in all_metrics if row["configuration"] == MAIN_CONFIGURATION]
    comparison = []
    for field, ngram, representation in CONFIGURATIONS:
        configuration = f"{field}_{ngram}_{representation}"
        configuration_rows = [row for row in all_metrics if row["configuration"] == configuration]
        for level in ("1", "2", "3", "all"):
            subset = configuration_rows if level == "all" else [
                row for row in configuration_rows if row["difficulty_level"] == level]
            comparison.append({"configuration": configuration, "difficulty_level": level,
                               "query_count": len(subset), "retrieved_results": len(subset) * 10,
                               "relevant_retrieved": sum(row["relevant_retrieved"] for row in subset),
                               "irrelevant_retrieved": sum(row["irrelevant_retrieved"] for row in subset),
                               "unjudged_retrieved": sum(row["unjudged_retrieved"] for row in subset),
                               "zero_similarity_results": sum(row["zero_similarity_results"] for row in subset),
                               **_mean(subset, METRICS)})

    ranking_fields = ["query_id", "difficulty_level", "query", "configuration", "field", "ngram",
                      "representation", "rank", "document_id", "title", "url", "score", "zero_similarity"]
    judgment_fields = ranking_fields + ["relevance_grade", "judgment_status", "is_relevant",
                                        "retrieval_outcome", "evidence"]
    metric_fields = ["query_id", "difficulty_level", "configuration", "field", "ngram",
                     "representation", "relevant_retrieved", "irrelevant_retrieved", "unjudged_retrieved",
                     "total_relevant_reference", "zero_similarity_results", "first_relevant_rank", *METRICS]
    comparison_fields = ["configuration", "difficulty_level", "query_count", "retrieved_results",
                         "relevant_retrieved", "irrelevant_retrieved", "unjudged_retrieved",
                         "zero_similarity_results", *METRICS]
    _write_tsv(output / "rankings.tsv", ["query_id", "document_id", "rank", "score"], main_rankings)
    _write_tsv(output / "rankings_all_configurations.tsv", ranking_fields, rankings_all)
    _write_tsv(output / "retrieval_judgments.tsv", judgment_fields, judgments)
    _write_tsv(output / "metrics_by_query_configuration.tsv", metric_fields, all_metrics)
    write_metrics_by_configuration(output / "metrics_by_query_configuration.tsv")
    _write_tsv(output / "comparison.tsv", comparison_fields, comparison)
    mean_main = {"query_id": "MEAN", "difficulty_level": "all",
                 "configuration": MAIN_CONFIGURATION, "field": "body", "ngram": 1,
                 "representation": "tfidf",
                 **_mean(main_metrics, ["relevant_retrieved", "irrelevant_retrieved",
                                       "unjudged_retrieved", "total_relevant_reference",
                                       "zero_similarity_results", "first_relevant_rank", *METRICS])}
    _write_tsv(output / "metrics.tsv", metric_fields, main_metrics + [mean_main])
    _write_report_data(output, bundle, queries, qrels, all_metrics, comparison, judgments)
    _write_figures(output, comparison)
    return {"rankings": len(main_rankings), "rankings_all": len(rankings_all),
            "retrieval_judgments": len(judgments), "queries": len(queries),
            "configurations": len(CONFIGURATIONS), "evaluations": len(all_metrics),
            "unjudged": sum(row["unjudged_retrieved"] for row in all_metrics),
            "zero_similarity_results": sum(row["zero_similarity_results"] for row in all_metrics),
            "provisional": not frozen, "status": "final" if frozen else "provisional"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report_data(output, bundle, queries, qrels, all_metrics, comparison, judgments):
    documents = bundle["documents"]
    source_counts = Counter(row["source"] for row in documents)
    category_counts = Counter(row["category"] for row in documents)
    dates = [row["date"] for row in documents]
    reference_rows = [row for row in all_metrics if row["configuration"] == MAIN_CONFIGURATION]
    ordered = sorted(reference_rows, key=lambda row: (row["ndcg@10"], row["mrr"], row["query_id"]))
    environment = {}
    for package in ("spacy", "scikit-learn", "joblib", "nbformat", "nbclient"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "not-installed"
    report = {
        "evaluation_status": "frozen_delegated_analytical_validation_not_institutional_human_approval",
        "corpus": {"documents": len(documents), "sources": dict(source_counts),
                   "categories": dict(category_counts), "date_min": min(dates), "date_max": max(dates),
                   "licenses": sorted({row["license"] for row in documents})},
        "normalization": {"average_original_body_tokens": sum(len(row["description/body"].split()) for row in documents) / len(documents),
                          "average_normalized_body_tokens": sum(len(row["normalized_body"].split()) for row in documents) / len(documents)},
        "vector_spaces": [{"configuration": key, "rows": value["matrix"].shape[0],
                           "columns": value["matrix"].shape[1],
                           "vocabulary": len(value["vectorizer"].vocabulary_)}
                          for key, value in bundle["spaces"].items()],
        "queries": [{**row, "relevant_documents": len(qrels[row["query_id"]])} for row in queries],
        "reference": {"qrels": sum(len(rows) for rows in qrels.values()),
                      "by_query": {query_id: len(rows) for query_id, rows in qrels.items()},
                      "academic_max_exceeded": {query_id: len(rows) for query_id, rows in qrels.items() if len(rows) > 15}},
        "execution": {"query_configuration_runs": len(all_metrics), "top10_results": len(judgments),
                      "relevant_returns": sum(row["is_relevant"] == 1 for row in judgments),
                      "irrelevant_returns": sum(row["is_relevant"] == 0 for row in judgments),
                      "unjudged_returns": sum(row["is_relevant"] == "" for row in judgments),
                      "zero_similarity_returns": sum(row["zero_similarity"] == 1 for row in judgments)},
        "metrics_by_query_configuration": all_metrics,
        "comparison": comparison,
        "best_five_primary": list(reversed(ordered[-5:])),
        "worst_five_primary": ordered[:5],
        "limitations": ["Referencia aceptada mediante evaluación analítica delegada, no aprobación humana institucional.",
                        "El pool no demuestra exhaustividad sobre las 2,000 noticias.",
                        "La recuperación es léxica y sensible a sinónimos, paráfrasis y vocabulario fuera del índice.",
                        "Los Top-10 incluyen desempates con similitud cero cuando la consulta no comparte características."],
        "environment": {"python": platform.python_version(), **environment,
                        "spacy_model": bundle["model"], "spacy_runtime": bundle["spacy_version"]},
        "hashes": {name: _sha256(path) for name, path in {
            "raw_corpus": "data/news_raw_corpus.tsv", "normalized_corpus": "data/news_normalized_corpus.tsv",
            "vectors": VECTORS, "queries": "data/queries.tsv", "qrels": "data/qrels.tsv"}.items()},
    }
    (output / "report_data.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_figures(output, comparison):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    means = [row for row in comparison if row["difficulty_level"] == "all"]
    _bar_svg(figures / "01_ndcg_by_configuration.svg", means, "configuration", "ndcg@10",
             "nDCG@10 medio por configuración")
    _bar_svg(figures / "02_relevant_returns_by_configuration.svg", means, "configuration", "relevant_retrieved",
             "Resultados relevantes recuperados por configuración (150 devoluciones)")
    _bar_svg(figures / "03_zero_similarity_by_configuration.svg", means, "configuration", "zero_similarity_results",
             "Devoluciones con similitud cero por configuración")
    difficulty = []
    for level in ("1", "2", "3"):
        rows = [row for row in comparison if row["difficulty_level"] == level]
        difficulty.append({"difficulty": level, "ndcg@10": sum(row["ndcg@10"] for row in rows) / len(rows)})
    _bar_svg(figures / "04_ndcg_by_difficulty.svg", difficulty, "difficulty", "ndcg@10",
             "nDCG@10 medio por dificultad y configuración")
    _grouped_svg(figures / "05_precision_recall_by_configuration.svg", means, "configuration",
                 "precision@10", "recall@10", "Precision@10 y Recall@10 medios")


def _bar_svg(path, rows, label_key, value_key, title):
    width, left, right, top, row_height = 1000, 240, 90, 55, 32
    height = top + row_height * len(rows) + 35
    maximum = max(float(row[value_key]) for row in rows) or 1.0
    plot_width = width - left - right
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>']
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = float(row[value_key])
        bar = value / maximum * plot_width
        parts.extend([f'<text x="{left-8}" y="{y+16}" text-anchor="end" font-family="Arial" font-size="12">{row[label_key]}</text>',
                      f'<rect x="{left}" y="{y}" width="{bar:.2f}" height="20" fill="#3568a8"/>',
                      f'<text x="{left+bar+6:.2f}" y="{y+15}" font-family="Arial" font-size="11">{value:.4f}</text>'])
    parts.append('</svg>')
    path.write_text("".join(parts), encoding="utf-8")


def _grouped_svg(path, rows, label_key, first_key, second_key, title):
    width, left, right, top, row_height = 1000, 240, 90, 65, 42
    height = top + row_height * len(rows) + 40
    plot_width = width - left - right
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>',
             '<rect x="740" y="39" width="12" height="12" fill="#3568a8"/><text x="758" y="50" font-family="Arial" font-size="11">Precision</text>',
             '<rect x="830" y="39" width="12" height="12" fill="#d9792b"/><text x="848" y="50" font-family="Arial" font-size="11">Recall</text>']
    for index, row in enumerate(rows):
        y = top + index * row_height
        first, second = float(row[first_key]), float(row[second_key])
        parts.extend([f'<text x="{left-8}" y="{y+19}" text-anchor="end" font-family="Arial" font-size="12">{row[label_key]}</text>',
                      f'<rect x="{left}" y="{y}" width="{first*plot_width:.2f}" height="14" fill="#3568a8"/>',
                      f'<rect x="{left}" y="{y+18}" width="{second*plot_width:.2f}" height="14" fill="#d9792b"/>'])
    parts.append('</svg>')
    path.write_text("".join(parts), encoding="utf-8")


def _write_tsv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_by_configuration(consolidated_path="results/metrics_by_query_configuration.tsv"):
    
    consolidated_path = Path(consolidated_path)
    with consolidated_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "configuration" not in fields or len(rows) != 180:
        raise ValueError("El consolidado debe contener 180 evaluaciones y la columna configuration")

    expected = [f"{field}_{ngram}_{representation}"
                for field, ngram, representation in CONFIGURATIONS]
    grouped = {configuration: [] for configuration in expected}
    for row in rows:
        configuration = row["configuration"]
        if configuration not in grouped:
            raise ValueError(f"Configuración inesperada en el consolidado: {configuration}")
        grouped[configuration].append(row)
    for configuration, configuration_rows in grouped.items():
        if (len(configuration_rows) != 15
                or len({row["query_id"] for row in configuration_rows}) != 15):
            raise ValueError(f"{configuration} no contiene exactamente las 15 consultas")

    destination = consolidated_path.parent / "evaluation_by_configuration"
    destination.mkdir(parents=True, exist_ok=True)
    expected_files = {f"{configuration}.tsv" for configuration in expected}
    for stale in destination.glob("*.tsv"):
        if stale.name not in expected_files:
            stale.unlink()
    for configuration in expected:
        _write_tsv(destination / f"{configuration}.tsv", fields, grouped[configuration])

    exported = []
    for configuration in expected:
        exported.extend(_read_tsv(destination / f"{configuration}.tsv"))
    fingerprint = lambda row: tuple(row[field] for field in fields)
    if (len(exported) != len(rows)
            or Counter(map(fingerprint, exported)) != Counter(map(fingerprint, rows))):
        raise RuntimeError("El desglose por configuración no reproduce exactamente el consolidado")
    return {configuration: len(grouped[configuration]) for configuration in expected}
