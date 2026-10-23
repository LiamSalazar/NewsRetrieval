"""Añade dos hallazgos externos al pool existente sin reconstruirlo."""
import csv
import hashlib
from pathlib import Path

DATA = Path("data")
QRELS_SHA = "71394f670bc92c278c4ea853bc554e1111204a769193c30e400becefea0457ad"
ADDITIONS = {
    ("surveillance", "Nbc0999a090a07dc280d0"),
    ("football", "N5caa0ad89d6f0b141f37"),
}


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write(path, fields, rows):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main():
    if hashlib.sha256((DATA / "qrels.tsv").read_bytes()).hexdigest() != QRELS_SHA:
        raise ValueError("qrels.tsv cambió antes del microcierre")
    corpus = {row["id"]: row for row in read(DATA / "news_raw_corpus.tsv")}
    pool_rows = read(DATA / "relevance_pool.tsv")
    review_rows = read(DATA / "pool_human_review.tsv")
    priority_rows = read(DATA / "pool_review_priority.tsv")
    pool = {(row["need_id"], row["document_id"]): row for row in pool_rows}
    review = {(row["need_id"], row["document_id"]): row for row in review_rows}
    priority = {(row["need_id"], row["document_id"]): row for row in priority_rows}
    if set(pool) != set(review) or set(pool) != set(priority):
        raise ValueError("Pool, revisión y prioridades no están alineados")
    for key in ADDITIONS:
        document = corpus[key[1]]
        if key not in pool:
            pool[key] = {"need_id": key[0], "document_id": key[1],
                         "discovery_provenance": "auditoria_independiente_adicional_cierre"}
            review[key] = {"need_id": key[0], "document_id": key[1],
                           "title": document["title"], "excerpt": document["description/body"],
                           "url": document["url"], "previous_relevance": "sin_juzgar",
                           "proposed_relevance": "", "justification": "Pendiente de integrar microcierre.",
                           "confidence": "", "review_status": "pendiente_microcorreccion_cierre"}
            priority[key] = {"need_id": key[0], "document_id": key[1],
                             "review_priority": "P0_auditoria_independiente_adicional"}
    pool_out = sorted(pool.values(), key=lambda row: (row["need_id"], row["document_id"]))
    review_out = sorted(review.values(), key=lambda row: (row["need_id"], row["document_id"]))
    priority_out = sorted(priority.values(), key=lambda row: (row["need_id"], row["document_id"]))
    write(DATA / "relevance_pool.tsv", list(pool_out[0]), pool_out)
    write(DATA / "pool_human_review.tsv", list(review_out[0]), review_out)
    write(DATA / "pool_review_priority.tsv", list(priority_out[0]), priority_out)
    print(f"Pool integrado: {len(pool_out)} pares; altas nuevas: {len(ADDITIONS)}")


if __name__ == "__main__":
    main()
