"""Congela la referencia delegada y documenta los Top-10 a similitud cero."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BACKUP = ROOT.parent / "NewsRetrieval_private_audit_backup_2026-10-23"

NEEDS = ("trade", "energy", "football", "cricket", "surveillance")
NEED_BY_QUERY = {f"Q{number:02d}": NEEDS[(number - 1) // 3] for number in range(1, 16)}

ARTICLE_FACTS = {
    "Nb52ef31d6f481a85dba5": "es una semblanza de Barry Steinhardt y su defensa histórica de libertades digitales",
    "N8002e77bdcfaa28419f1": "describe el mercado internacional de tecnologías y prácticas de vigilancia autoritaria",
    "N067c55f3ab4d13725c08": "documenta un litigio contra vigilancia gubernamental de expresión en redes sociales",
    "N7bb52e0b51a67d884772": "documenta fallos de privacidad en el reconocimiento facial de pasajeros de la TSA",
    "N6f87a799a62e2aa67140": "trata una demanda por entrega de datos de usuarios de Telenor a la junta de Myanmar",
    "N0975a070eab121babc9d": "trata la impugnación de búsquedas policiales sin orden en bases ALPR",
    "N8a9aa01ba55c134f1e57": "trata litigios sobre la red ALPR y el rastreo policial de vehículos",
    "Ncb2efe1cee545ca1fcad": "trata acuerdos de migración laboral y costos de contratación entre Bangladesh y Malasia",
    "Nf58730be4453a6de5b92": "analiza cables submarinos, IA e infraestructura petrolera en política militar y económica",
    "N6a7de4da9283111ad2f7": "describe represión estatal contra organizaciones LGBTQ+ y bloqueo de sus medios digitales",
}

EXCLUSION = {
    "trade": "no presenta una negociación, aplicación o disputa sobre un tratado o regla comercial entre gobiernos o bloques",
    "energy": "no describe un aumento de tarifa o combustible ni su efecto económico concreto en usuarios, hogares o empresas",
    "football": "no aborda un acontecimiento de fútbol ni una consecuencia política, social, diplomática o cultural derivada de él",
    "cricket": "no aborda cricket ni una controversia institucional, disciplinaria, política, cultural o de seguridad de ese deporte",
    "surveillance": "no documenta una práctica concreta de recopilación o rastreo de personas por una autoridad pública ni su supervisión",
}


def read_tsv(path: Path, comments: bool = False):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lines = (line for line in stream if not (comments and line.startswith("#")))
        return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    raw = read_tsv(DATA / "news_raw_corpus.tsv")
    queries = read_tsv(DATA / "queries.tsv")
    candidates = read_tsv(DATA / "qrels_candidates_for_approval.tsv")
    pool = read_tsv(DATA / "pool_human_review.tsv")
    document_by_id = {row["id"]: row for row in raw}

    if len(raw) != 2000 or len(document_by_id) != 2000:
        raise ValueError("El corpus original no contiene 2,000 IDs únicos")
    if len(queries) != 15 or Counter(row["difficulty_level"] for row in queries) != Counter({"1": 5, "2": 5, "3": 5}):
        raise ValueError("Las consultas no cumplen la distribución 15 y 5/5/5")
    if len(candidates) != 303:
        raise ValueError(f"Se esperaban 303 qrels candidatos y hay {len(candidates)}")
    candidate_keys = {(row["query_id"], row["document_id"]) for row in candidates}
    if len(candidate_keys) != 303:
        raise ValueError("Hay duplicados necesidad-documento en los qrels candidatos")
    if {row["relevance"] for row in candidates} != {"1", "2"}:
        raise ValueError("Los qrels candidatos deben incluir exactamente los grados positivos 1 y 2")
    if any(row["document_id"] not in document_by_id for row in candidates):
        raise ValueError("Hay qrels candidatos fuera del corpus original")
    if any(row["query_id"] not in NEED_BY_QUERY for row in candidates):
        raise ValueError("Hay qrels candidatos para consultas inexistentes")

    by_query = {query_id: {(row["document_id"], row["relevance"]) for row in candidates
                           if row["query_id"] == query_id}
                for query_id in NEED_BY_QUERY}
    for start in (1, 4, 7, 10, 13):
        group = [by_query[f"Q{number:02d}"] for number in range(start, start + 3)]
        if not (group[0] == group[1] == group[2]):
            raise ValueError(f"Las tres consultas de la necesidad Q{start:02d}-Q{start+2:02d} no comparten referencia")

    pool_map = {(row["need_id"], row["document_id"]): row for row in pool}
    for row in candidates:
        pair = (NEED_BY_QUERY[row["query_id"]], row["document_id"])
        if pair not in pool_map or pool_map[pair]["proposed_relevance"] != row["relevance"]:
            raise ValueError(f"Qrel sin correspondencia exacta con la adjudicación consolidada: {pair}")

    BACKUP.mkdir(parents=True, exist_ok=True)
    old_qrels = DATA / "qrels.tsv"
    backup_qrels = BACKUP / "qrels_pre_freeze_195.tsv"
    if not backup_qrels.exists():
        backup_qrels.write_bytes(old_qrels.read_bytes())
    old_hash = sha256(backup_qrels)

    qrels = sorted(({"query_id": row["query_id"], "document_id": row["document_id"],
                     "relevance": row["relevance"]} for row in candidates),
                   key=lambda row: (int(row["query_id"][1:]), row["document_id"]))
    write_tsv(DATA / "qrels.tsv", ["query_id", "document_id", "relevance"], qrels)

    evidence = sorted(({"query_id": row["query_id"], "document_id": row["document_id"],
                        "relevance": row["relevance"],
                        "validation_status": "aceptada_revision_analitica_delegada",
                        "evidence": row["evidence"]} for row in candidates),
                      key=lambda row: (int(row["query_id"][1:]), row["document_id"]))
    write_tsv(DATA / "qrels_validation_evidence.tsv",
              ["query_id", "document_id", "relevance", "validation_status", "evidence"], evidence)

    zero_pairs = {
        ("cricket", document_id) for document_id in ARTICLE_FACTS
    } | {
        ("energy", document_id) for document_id in ARTICLE_FACTS
    } | {
        ("football", document_id) for document_id in ARTICLE_FACTS
        if document_id not in {"Nb52ef31d6f481a85dba5", "Ncb2efe1cee545ca1fcad"}
    } | {
        ("trade", document_id) for document_id in ARTICLE_FACTS
        if document_id not in {"Nb52ef31d6f481a85dba5", "N8002e77bdcfaa28419f1",
                               "N067c55f3ab4d13725c08", "Ncb2efe1cee545ca1fcad"}
    } | {("surveillance", "Nf58730be4453a6de5b92")}
    if len(zero_pairs) != 35:
        raise AssertionError(f"Se esperaban 35 pares revisados y se definieron {len(zero_pairs)}")

    zero_rows = []
    newly_added = 0
    for need_id, document_id in sorted(zero_pairs):
        document = document_by_id[document_id]
        justification = (f"Lectura semántica independiente del score: el artículo {ARTICLE_FACTS[document_id]}; "
                         f"para la necesidad {need_id}, {EXCLUSION[need_id]}. La similitud cero solo explica "
                         "su aparición por desempate y no determina la etiqueta.")
        zero_rows.append({"need_id": need_id, "document_id": document_id,
                          "title": document["title"], "url": document["url"],
                          "similarity_context": "top10_score_exactly_zero",
                          "proposed_relevance": "0",
                          "review_status": "adjudicada_revision_analitica_delegada",
                          "evidence": justification})
        if (need_id, document_id) not in pool_map:
            pool.append({"need_id": need_id, "document_id": document_id,
                         "title": document["title"],
                         "excerpt": document["description/body"][:800], "url": document["url"],
                         "previous_relevance": "", "proposed_relevance": "0",
                         "justification": justification, "confidence": "high",
                         "review_status": "adjudicada_top10_cero_revision_analitica_delegada"})
            newly_added += 1
    write_tsv(DATA / "top10_zero_similarity_adjudication.tsv",
              ["need_id", "document_id", "title", "url", "similarity_context",
               "proposed_relevance", "review_status", "evidence"], zero_rows)
    pool.sort(key=lambda row: (row["need_id"], row["document_id"]))
    write_tsv(DATA / "pool_human_review.tsv", list(pool[0]), pool)

    status = [{"artifact": "qrels.tsv", "status": "frozen_delegated_analytical_validation",
               "approval_scope": "accepted_for_delegated_analytical_evaluation_not_institutional_human_approval",
               "note": ("Referencia ampliada de 303 qrels congelada. Supera 15 positivos en comercio, fútbol y vigilancia "
                        "para no ocultar documentos relevantes; academic_reference_core.tsv es solo expositivo.")}]
    write_tsv(DATA / "benchmark_status.tsv", list(status[0]), status)

    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status[0]["status"],
        "approval_scope": status[0]["approval_scope"],
        "input_archive": "NewsRetrieval_entradas_cierre_Sol_High_2026-10-23.zip",
        "input_archive_sha256": None,
        "input_archive_verification": "unavailable_in_workspace; local 303-row candidate reference validated under explicit user authorization",
        "local_candidate_source_sha256": sha256(DATA / "qrels_candidates_for_approval.tsv"),
        "previous_qrels_rows": 195,
        "previous_qrels_sha256": old_hash,
        "previous_qrels_backup": "external_private_backup/qrels_pre_freeze_195.tsv (not included in public delivery)",
        "frozen_qrels_rows": len(qrels),
        "frozen_qrels_sha256": sha256(DATA / "qrels.tsv"),
        "corpus_sha256": sha256(DATA / "news_raw_corpus.tsv"),
        "pool_rows_after_top10_zero_adjudication": len(pool),
        "zero_similarity_pairs_individually_adjudicated": len(zero_rows),
        "new_top10_pairs_added_to_pool": sum(
            row["review_status"] == "adjudicada_top10_cero_revision_analitica_delegada" for row in pool),
        "academic_max_exceeded": {query_id: len(rows) for query_id, rows in by_query.items() if len(rows) > 15},
    }
    (DATA / "reference_freeze_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
