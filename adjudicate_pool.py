"""Integra adjudicaciones semánticas sin reconstruir el pool ni tocar qrels."""
import csv
import hashlib
from collections import Counter
from pathlib import Path


DATA = Path("data")
REVIEW = DATA / "pool_human_review.tsv"
PRIORITY = DATA / "pool_review_priority.tsv"
P1 = DATA / "newsretrieval_p1_revision_independiente.tsv"
BATCH_DIR = DATA / "p2_adjudication_batches"
REMAINING_BATCH_DIR = DATA / "remaining_adjudication_batches"
PROGRESS = DATA / "p2_adjudication_progress.tsv"
REMAINING = DATA / "p2_adjudication_remaining.tsv"
CORRECTIONS = DATA / "semantic_corrections.tsv"
AMBIGUITY_RESOLUTIONS = DATA / "ambiguity_resolutions.tsv"
POSITIVE_SECOND_REVIEW = DATA / "positive_second_review.tsv"
CLOSING_AUDIT = DATA / "closing_audit_decisions.tsv"
MICRO_CLOSING = DATA / "microclosing_decisions.tsv"
QRELS_EXPECTED_SHA256 = "71394f670bc92c278c4ea853bc554e1111204a769193c30e400becefea0457ad"


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, fields, rows):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    if hashlib.sha256((DATA / "qrels.tsv").read_bytes()).hexdigest() != QRELS_EXPECTED_SHA256:
        raise ValueError("qrels.tsv cambió desde el inicio de la adjudicación")
    review = read_tsv(REVIEW)
    fields = list(review[0])
    by_key = {(row["need_id"], row["document_id"]): row for row in review}
    pool_keys = {(row["need_id"], row["document_id"])
                 for row in read_tsv(DATA / "relevance_pool.tsv")}
    if len(review) != len(by_key) or set(by_key) != pool_keys:
        raise ValueError("La tabla de revisión no coincide con los pares únicos del pool")

    p1 = read_tsv(P1)
    if len(p1) != 14:
        raise ValueError("La segunda revisión P1 no contiene 14 casos")
    for decision in p1:
        key = (decision["need_id"], decision["document_id"])
        row = by_key[key]
        label = decision["independent_proposed_relevance"]
        if decision["review_status"] == "revision_humana_necesaria":
            row["proposed_relevance"] = ""
            row["confidence"] = "baja"
            row["review_status"] = "pendiente_tras_dos_revisiones_independientes"
        else:
            if label not in {"0", "1", "2"}:
                raise ValueError(f"Etiqueta P1 inválida para {key}")
            row["proposed_relevance"] = label
            row["confidence"] = decision["confidence"]
            row["review_status"] = "propuesta_concordante_dos_revisiones_pendiente_humana"
        row["justification"] = decision["reason_from_original_text"]

    p2_keys = {(row["need_id"], row["document_id"]) for row in read_tsv(PRIORITY)
               if row["review_priority"] == "P2_lectura_prioritaria_sin_juicio"}
    decisions = {}
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(BATCH_DIR.glob("batch_*.tsv")):
        for decision in read_tsv(path):
            key = (decision["need_id"], decision["document_id"])
            if key not in p2_keys:
                raise ValueError(f"{path.name}: el par no pertenece a P2: {key}")
            if key in decisions:
                raise ValueError(f"Decisión P2 duplicada: {key}")
            if decision["proposed_relevance"] not in {"", "0", "1", "2"}:
                raise ValueError(f"Etiqueta P2 inválida: {key}")
            if not decision["justification"].strip():
                raise ValueError(f"Falta justificación P2: {key}")
            decisions[key] = decision

    for key, decision in decisions.items():
        row = by_key[key]
        row["proposed_relevance"] = decision["proposed_relevance"]
        row["justification"] = decision["justification"]
        row["confidence"] = decision["confidence"]
        row["review_status"] = ("propuesta_semantica_p2_pendiente_aprobacion_humana"
                                if decision["proposed_relevance"] else
                                "p2_examinado_evidencia_insuficiente_pendiente")

    # Adjudicaciones posteriores (P3-P5), guardadas en lotes reanudables.
    remaining_decisions = {}
    REMAINING_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(REMAINING_BATCH_DIR.glob("batch_*.tsv")):
        for decision in read_tsv(path):
            key = (decision["need_id"], decision["document_id"])
            if key not in by_key:
                raise ValueError(f"{path.name}: el par no pertenece al pool: {key}")
            if key in remaining_decisions:
                raise ValueError(f"Decisión posterior duplicada: {key}")
            if decision["proposed_relevance"] not in {"", "0", "1", "2"}:
                raise ValueError(f"Etiqueta posterior inválida: {key}")
            if not decision["justification"].strip():
                raise ValueError(f"Falta justificación posterior: {key}")
            remaining_decisions[key] = decision

    for key, decision in remaining_decisions.items():
        row = by_key[key]
        row["proposed_relevance"] = decision["proposed_relevance"]
        row["justification"] = decision["justification"]
        row["confidence"] = decision["confidence"]
        row["review_status"] = ("propuesta_semantica_{}_pendiente_aprobacion_humana".format(
            decision["priority"].lower()) if decision["proposed_relevance"] else
            "examinado_evidencia_insuficiente_pendiente")

    if CORRECTIONS.exists():
        for correction in read_tsv(CORRECTIONS):
            key = (correction["need_id"], correction["document_id"])
            if key not in by_key:
                raise ValueError(f"Corrección ajena al pool: {key}")
            row = by_key[key]
            if row["proposed_relevance"] == correction["corrected_proposed_relevance"]:
                continue
            if row["proposed_relevance"] != correction["previous_proposed_relevance"]:
                raise ValueError(f"La etiqueta anterior no coincide para {key}")
            row["proposed_relevance"] = correction["corrected_proposed_relevance"]
            row["justification"] = correction["justification"]
            row["confidence"] = correction["confidence"]
            row["review_status"] = "propuesta_corregida_control_semantico_pendiente_humana"

    if AMBIGUITY_RESOLUTIONS.exists():
        for resolution in read_tsv(AMBIGUITY_RESOLUTIONS):
            key = (resolution["need_id"], resolution["document_id"])
            if key not in by_key:
                raise ValueError(f"Resolución ajena al pool: {key}")
            row = by_key[key]
            row["proposed_relevance"] = resolution["proposed_relevance"]
            row["justification"] = resolution["justification"]
            row["confidence"] = resolution["confidence"]
            row["review_status"] = ("ambiguedad_resuelta_pendiente_aprobacion_humana"
                                    if resolution["proposed_relevance"] else
                                    "ambiguedad_reexaminada_evidencia_insuficiente")

    if POSITIVE_SECOND_REVIEW.exists():
        for confirmation in read_tsv(POSITIVE_SECOND_REVIEW):
            key = (confirmation["need_id"], confirmation["document_id"])
            row = by_key[key]
            if row["proposed_relevance"] != confirmation["confirmed_relevance"]:
                raise ValueError(f"Segunda revisión no coincide para {key}")
            row["justification"] = confirmation["second_review_justification"]
            row["confidence"] = confirmation["confidence"]
            row["review_status"] = "segunda_revision_positiva_completada_pendiente_humana"

    if CLOSING_AUDIT.exists():
        closing_rows = read_tsv(CLOSING_AUDIT)
        if len(closing_rows) != 10:
            raise ValueError("La auditoría de cierre debe contener exactamente 10 decisiones")
        for decision in closing_rows:
            key = (decision["need_id"], decision["document_id"])
            row = by_key[key]
            previous = decision["previous_proposed_relevance"]
            if row["proposed_relevance"] != previous:
                raise ValueError(f"La etiqueta previa de cierre no coincide para {key}")
            row["proposed_relevance"] = decision["final_proposed_relevance"]
            row["justification"] = decision["justification"]
            row["confidence"] = decision["confidence"]
            row["review_status"] = "auditoria_independiente_cierre_integrada_pendiente_humana"

    if MICRO_CLOSING.exists():
        micro_rows = read_tsv(MICRO_CLOSING)
        if len(micro_rows) != 3:
            raise ValueError("El microcierre debe contener exactamente tres decisiones")
        for decision in micro_rows:
            key = (decision["need_id"], decision["document_id"])
            row = by_key[key]
            if row["proposed_relevance"] != decision["previous_proposed_relevance"]:
                raise ValueError(f"La etiqueta previa del microcierre no coincide para {key}")
            row["proposed_relevance"] = decision["final_proposed_relevance"]
            row["justification"] = decision["justification"]
            row["confidence"] = decision["confidence"]
            row["review_status"] = "microcorreccion_cierre_integrada_pendiente_humana"

    write_tsv(REVIEW, fields, review)
    progress = []
    for need_id in sorted({need for need, _ in p2_keys}):
        selected_keys = [key for key in decisions if key[0] == need_id]
        labels = Counter(by_key[key]["proposed_relevance"] for key in selected_keys)
        total = sum(1 for need, _ in p2_keys if need == need_id)
        progress.append({"need_id": need_id, "p2_total": total, "examined": len(selected_keys),
                         "proposed_0": labels["0"], "proposed_1": labels["1"],
                         "proposed_2": labels["2"], "pending": labels[""],
                         "remaining": total - len(selected_keys)})
    write_tsv(PROGRESS, list(progress[0]), progress)
    remaining_rows = []
    for need_id, document_id in sorted(p2_keys - set(decisions)):
        row = by_key[(need_id, document_id)]
        remaining_rows.append({"need_id": need_id, "document_id": document_id,
                               "title": row["title"],
                               "review_priority": "P2_lectura_prioritaria_sin_juicio"})
    write_tsv(REMAINING, ["need_id", "document_id", "title", "review_priority"],
              remaining_rows)
    print(f"P1 integrado: {len(p1)}; P2 examinados: {len(decisions)}/{len(p2_keys)}; "
          f"P3-P5 examinados: {len(remaining_decisions)}")


if __name__ == "__main__":
    main()
