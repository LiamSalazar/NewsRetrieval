"""Benchmark reproducible para las doce configuraciones VSM."""
import csv
from collections import defaultdict
from pathlib import Path

from retrieval import VECTORS, search_news


def load_benchmark(queries_path="data/queries.tsv", qrels_path="data/qrels.tsv"):
    """Carga consultas y qrels sin modificar sus juicios provisionales."""
    with Path(queries_path).open(encoding="utf-8", newline="") as stream:
        queries = list(csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t"))
    with Path(qrels_path).open(encoding="utf-8", newline="") as stream:
        qrels_rows = list(csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t"))
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
            raise ValueError("qrels.tsv solo incluye relevancias 1 y 2")
        if row["document_id"] in qrels[row["query_id"]]:
            raise ValueError("documento repetido en qrels")
        qrels[row["query_id"]][row["document_id"]] = relevance
    return queries, dict(qrels)


def _document_id(result):
    return result.get("id") if isinstance(result, dict) else str(result)


def calculate_metrics(results, relevant, k=10):
    """Calcula P, R, F1, MRR y nDCG con ganancia 2^relevance - 1."""
    ranked = list(results)[:k]
    grades = [int(relevant.get(_document_id(row), 0)) for row in ranked]
    hits = sum(grade > 0 for grade in grades)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reciprocal_rank = next((1.0 / (i + 1) for i, grade in enumerate(grades) if grade > 0), 0.0)
    dcg = sum((2 ** grade - 1) / __import__("math").log2(i + 2)
              for i, grade in enumerate(grades))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / __import__("math").log2(i + 2)
               for i, grade in enumerate(ideal))
    return {"precision@10": precision, "recall@10": recall, "f1@10": f1,
            "mrr": reciprocal_rank, "ndcg@10": dcg / idcg if idcg else 0.0}


def _mean(rows, keys):
    return {key: sum(row[key] for row in rows) / len(rows) if rows else 0.0 for key in keys}


def validate_benchmark(queries, qrels, documents=None):
    """Valida estructura, dificultades, cuotas y niveles sin congelar juicios."""
    if len(queries) != 15:
        raise ValueError("El benchmark debe tener 15 consultas")
    levels = {str(level): sum(q["difficulty_level"] == str(level) for q in queries)
              for level in (1, 2, 3)}
    if levels != {"1": 5, "2": 5, "3": 5}:
        raise ValueError("Debe haber cinco consultas por dificultad")
    document_ids = {row["id"] for row in documents} if documents is not None else None
    for query in queries:
        rows = qrels.get(query["query_id"], {})
        if not 10 <= len(rows) <= 15:
            raise ValueError(f"{query['query_id']} debe tener entre 10 y 15 relevantes")
        if set(rows.values()) != {1, 2}:
            raise ValueError(f"{query['query_id']} debe incluir relevancia 1 y 2")
        if document_ids is not None and not set(rows).issubset(document_ids):
            raise ValueError(f"{query['query_id']} contiene documentos inexistentes")
    return {"queries": len(queries), "difficulty_levels": levels,
            "relevant_judgments": sum(len(rows) for rows in qrels.values())}


def evaluate(queries, qrels, bundle=None, output_dir="results", frozen=False,
             provisional=False):
    """Ejecuta las 12 configuraciones; solo escribe definitivo con aprobación."""
    if not frozen and not provisional:
        raise RuntimeError("qrels.tsv es provisional; use frozen=True tras revisión humana")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configurations = [(field, ngram, rep) for field in ("title", "body")
                      for ngram in (1, 2) for rep in ("frequency", "binary", "tfidf")]
    all_metrics = []
    main_rankings = []
    comparison = []
    for field, ngram, representation in configurations:
        config_id = f"{field}_{ngram}_{representation}"
        config_metrics = []
        for query in queries:
            rows = search_news(query["query"], field, ngram, representation, bundle=bundle)
            metrics = calculate_metrics(rows, qrels.get(query["query_id"], {}))
            record = {"query_id": query["query_id"], "difficulty_level": query["difficulty_level"],
                      "configuration": config_id, **metrics}
            config_metrics.append(record)
            if config_id == "body_1_tfidf":
                main_rankings.extend({"query_id": query["query_id"], "document_id": row["id"],
                                      "rank": rank, "score": row["score"]}
                                      for rank, row in enumerate(rows, 1))
                all_metrics.append(record)
        for level in ("1", "2", "3", "all"):
            subset = config_metrics if level == "all" else [r for r in config_metrics
                       if r["difficulty_level"] == level]
            comparison.append({"configuration": config_id, "difficulty_level": level,
                               "query_count": len(subset), **_mean(subset, list(metrics))})
    _write_tsv(output / "rankings.tsv", ["query_id", "document_id", "rank", "score"], main_rankings)
    _write_tsv(output / "metrics.tsv", ["query_id", "difficulty_level", "configuration",
                                        "precision@10", "recall@10", "f1@10", "mrr", "ndcg@10"],
               all_metrics + [{"query_id": "MEAN", "difficulty_level": "all", "configuration": "body_1_tfidf",
                               **_mean(all_metrics, ["precision@10", "recall@10", "f1@10", "mrr", "ndcg@10"])}])
    _write_tsv(output / "comparison.tsv", ["configuration", "difficulty_level", "query_count",
                                            "precision@10", "recall@10", "f1@10", "mrr", "ndcg@10"], comparison)
    return {"rankings": len(main_rankings), "queries": len(queries),
            "configurations": len(configurations), "provisional": not frozen}


def _write_tsv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
