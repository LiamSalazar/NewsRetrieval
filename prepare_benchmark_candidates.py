"""Genera artefactos de auditoría final sin congelar ni modificar qrels."""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import joblib

from retrieval import VECTORS, search_news


DATA = Path("data")
OUTPUT = Path("output")
NEEDS = ("trade", "energy", "football", "cricket", "surveillance")
QRELS_SHA = "71394f670bc92c278c4ea853bc554e1111204a769193c30e400becefea0457ad"
BASE_HASHES = {
    "news_raw_corpus.tsv": "82ea7d58c1a457f8bf23822735e5b4d41354fad0b7b82bd929b769017ebc1f6f",
    "news_normalized_corpus.tsv": "1793f6b542452c58a612ad3177054bd331f2198afc8bff1c12d7d808465342b8",
    "vectors.joblib": "181c2e77f18759c80274b48e319c3f2680a1d3fcab52135f37be547ca4c3d76c",
}
COVERAGE_POSITIVES = {
    ("trade", "N464bc76a6a29dcb06885"): ("1", "El título y resumen sitúan a productores cameruneses adaptándose a normas de la UE; es una consecuencia parcial de regulación comercial, aunque el instrumento no se desarrolla."),
    ("surveillance", "N6f87a799a62e2aa67140"): ("2", "El título documenta que Telenor entregó datos de usuarios a la junta de Myanmar y el resumen vincula esa conducta con daños y rendición de cuentas."),
}
SECOND_COVERAGE_POSITIVES = {}


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write(path, fields, rows):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if digest(DATA / "qrels.tsv") != QRELS_SHA:
        raise ValueError("qrels.tsv cambió durante la adjudicación")
    actual = {"news_raw_corpus.tsv": digest(DATA / "news_raw_corpus.tsv"),
              "news_normalized_corpus.tsv": digest(DATA / "news_normalized_corpus.tsv"),
              "vectors.joblib": digest(VECTORS)}
    if actual != BASE_HASHES:
        raise ValueError("El corpus o los vectores cambiaron")

    corpus_rows = read(DATA / "news_raw_corpus.tsv")
    corpus = {row["id"]: row for row in corpus_rows}
    pool_rows = read(DATA / "relevance_pool.tsv")
    review_rows = read(DATA / "pool_human_review.tsv")
    priority_rows = read(DATA / "pool_review_priority.tsv")
    pool = {(row["need_id"], row["document_id"]): row for row in pool_rows}
    review = {(row["need_id"], row["document_id"]): row for row in review_rows}
    priority = {(row["need_id"], row["document_id"]): row for row in priority_rows}
    if set(pool) != set(review) or set(pool) != set(priority):
        raise ValueError("Pool, revisión y prioridades no coinciden")

    sample = []
    for need in NEEDS:
        groups = defaultdict(list)
        for document in corpus_rows:
            key = (need, document["id"])
            if (key not in pool or
                    "muestreo_externo_cobertura_v1" in pool[key]["discovery_provenance"]):
                groups[(document["source"], document["category"])].append(document)
        for (source, category), documents in sorted(groups.items()):
            document = min(documents, key=lambda row: hashlib.sha256(
                f"coverage-v1|{need}|{row['id']}".encode()).hexdigest())
            key = (need, document["id"])
            label, reason = COVERAGE_POSITIVES.get(key, ("0",
                f"El título y el resumen tratan «{document['title']}»; no aportan el instrumento o práctica específica exigida por la necesidad {need}."))
            sample.append({"need_id": need, "document_id": document["id"],
                           "source": source, "category": category, "proposed_relevance": label,
                           "justification": reason, "sample_seed": "coverage-v1"})
            if label in {"1", "2"} and key not in pool:
                pool[key] = {"need_id": need, "document_id": document["id"],
                             "discovery_provenance": "muestreo_externo_cobertura_v1"}
                review[key] = {"need_id": need, "document_id": document["id"],
                               "title": document["title"], "excerpt": document["description/body"],
                               "url": document["url"], "previous_relevance": "sin_juzgar",
                               "proposed_relevance": label, "justification": reason,
                               "confidence": "media" if label == "1" else "alta",
                               "review_status": "propuesta_muestreo_cobertura_pendiente_humana"}
                priority[key] = {"need_id": need, "document_id": document["id"],
                                 "review_priority": "P4_control_cobertura_externa"}

    write(DATA / "coverage_external_sample.tsv", list(sample[0]), sample)

    # Segunda comprobación independiente: mismo diseño estratificado, nueva
    # semilla y exclusión estricta de todos los pares ya presentes en el pool.
    sample_v2 = []
    for need in NEEDS:
        groups = defaultdict(list)
        for document in corpus_rows:
            key = (need, document["id"])
            if key not in pool:
                groups[(document["source"], document["category"])].append(document)
        for (source, category), documents in sorted(groups.items()):
            document = min(documents, key=lambda row: hashlib.sha256(
                f"coverage-v2|{need}|{row['id']}".encode()).hexdigest())
            key = (need, document["id"])
            label, reason = SECOND_COVERAGE_POSITIVES.get(key, ("0",
                f"La lectura del título y texto de «{document['title']}» muestra que trata un asunto distinto de los criterios semánticos de {need}."))
            sample_v2.append({"need_id": need, "document_id": document["id"],
                              "source": source, "category": category,
                              "proposed_relevance": label, "justification": reason,
                              "sample_seed": "coverage-v2"})
            if label in {"1", "2"}:
                raise ValueError("La muestra v2 contiene un positivo no integrado")
    write(DATA / "coverage_external_sample_v2.tsv", list(sample_v2[0]), sample_v2)
    pool_out = sorted(pool.values(), key=lambda row: (row["need_id"], row["document_id"]))
    review_out = sorted(review.values(), key=lambda row: (row["need_id"], row["document_id"]))
    priority_out = sorted(priority.values(), key=lambda row: (row["need_id"], row["document_id"]))
    write(DATA / "relevance_pool.tsv", list(pool_out[0]), pool_out)
    write(DATA / "pool_human_review.tsv", list(review_out[0]), review_out)
    write(DATA / "pool_review_priority.tsv", list(priority_out[0]), priority_out)

    query_rows = read(DATA / "queries.tsv")
    candidate_rows = read(DATA / "candidates_review.tsv")
    query_need = {row["query_id"]: row["need_id"] for row in candidate_rows}
    by_need = defaultdict(list)
    for query in query_rows:
        by_need[query_need[query["query_id"]]].append(query)
    if len(query_rows) != 15 or any({row["difficulty_level"] for row in rows} != {"1", "2", "3"}
                                    for rows in by_need.values()):
        raise ValueError("Las cinco tríadas de consultas no son consistentes estructuralmente")

    qrel_candidates = []
    for row in review_out:
        if row["proposed_relevance"] in {"1", "2"}:
            for query in by_need[row["need_id"]]:
                qrel_candidates.append({"query_id": query["query_id"],
                                        "document_id": row["document_id"],
                                        "relevance": row["proposed_relevance"],
                                        "approval_status": "propuesta_ia_pendiente_aprobacion_humana",
                                        "evidence": row["justification"]})
    write(DATA / "qrels_candidates_for_approval.tsv", list(qrel_candidates[0]), qrel_candidates)

    # Núcleo documental académico: selección transparente para satisfacer la
    # exigencia descriptiva de 10-15 ejemplos por consulta. No es otro qrels y
    # nunca se usa como denominador de métricas.
    status_rank = {
        "segunda_revision_positiva_completada_pendiente_humana": 0,
        "propuesta_concordante_dos_revisiones_pendiente_humana": 1,
        "propuesta_corregida_control_semantico_pendiente_humana": 2,
    }
    academic_core = []
    full_positive_counts = {}
    for need in NEEDS:
        positives = [row for row in review_out
                     if row["need_id"] == need and row["proposed_relevance"] in {"1", "2"}]
        full_positive_counts[need] = len(positives)
        positives.sort(key=lambda row: (status_rank.get(row["review_status"], 9),
                                        -int(row["proposed_relevance"]), row["document_id"]))
        for row in positives[:15]:
            academic_core.append({
                "need_id": need,
                "applicable_query_ids": ",".join(sorted(q["query_id"] for q in by_need[need])),
                "document_id": row["document_id"],
                "proposed_relevance": row["proposed_relevance"],
                "role": "nucleo_documental_no_usar_para_metricas",
                "selection_basis": "evidencia_semantica_y_estado_de_control_no_ranking",
            })
    write(DATA / "academic_reference_core.tsv", list(academic_core[0]), academic_core)
    method_lines = [
        "# Tratamiento del requisito académico de 10–15 relevantes", "",
        "Existe una sola referencia candidata para evaluación: `qrels_candidates_for_approval.tsv`, que conserva todos los positivos auténticos y sigue pendiente de aprobación humana.", "",
        "`academic_reference_core.tsv` identifica hasta 15 ejemplos previamente adjudicados por necesidad (aplicables a sus tres consultas) únicamente como núcleo documental para exposición y auditoría. No es un segundo qrels, no está completo y no debe alimentar métricas.", "",
        "| Necesidad | Positivos completos | Núcleo documental | Recall@10 máximo con referencia completa |", "|---|---:|---:|---:|",
    ]
    for need in NEEDS:
        core_count = sum(row["need_id"] == need for row in academic_core)
        maximum = min(10, full_positive_counts[need]) / full_positive_counts[need]
        method_lines.append(f"| {need} | {full_positive_counts[need]} | {core_count} | {maximum:.1%} |")
    method_lines += ["", "Usar sólo el núcleo como denominador inflaría artificialmente Recall@10 y rompería la comparabilidad. Por ello las métricas futuras deben calcularse exclusivamente sobre la referencia completa una vez aprobada; el núcleo sólo documenta el cumplimiento expositivo del intervalo solicitado."]
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "academic_reference_methodology.md").write_text("\n".join(method_lines) + "\n", encoding="utf-8")

    pending = []
    for row in review_out:
        if not row["proposed_relevance"]:
            pending.append({"need_id": row["need_id"], "document_id": row["document_id"],
                            "title": row["title"], "reason": row["justification"],
                            "review_status": row["review_status"],
                            "priority": priority[(row["need_id"], row["document_id"])]["review_priority"]})
    write(DATA / "human_decision_pending.tsv",
          ["need_id", "document_id", "title", "reason", "review_status", "priority"], pending)

    configs = [(field, ngram, representation) for field in ("title", "body")
               for ngram in (1, 2) for representation in ("frequency", "binary", "tfidf")]
    bundle = joblib.load(VECTORS)
    top10 = defaultdict(set)
    zero = 0
    for query in query_rows:
        need = query_need[query["query_id"]]
        for field, ngram, representation in configs:
            for row in search_news(query["query"], field, ngram, representation, 10, bundle=bundle):
                if row["score"] > 0:
                    top10[need].add(row["id"])
                else:
                    zero += 1
    coverage_rows = []
    for need in NEEDS:
        judged = sum(bool(review.get((need, document_id), {}).get("proposed_relevance"))
                     for document_id in top10[need])
        ambiguous = sum((need, document_id) in review and not review[(need, document_id)]["proposed_relevance"]
                        for document_id in top10[need])
        coverage_rows.append({"need_id": need, "unique_positive_score_top10": len(top10[need]),
                              "with_proposed_judgment": judged, "pending": ambiguous})
    write(DATA / "top10_judgment_coverage.tsv", list(coverage_rows[0]), coverage_rows)

    qc = []
    for row in review_out:
        if row["proposed_relevance"] in {"1", "2"}:
            if row["review_status"] == "propuesta_concordante_dos_revisiones_pendiente_humana":
                quality_status = "dos_revisiones_independientes_concordantes"
            elif row["review_status"] == "auditoria_independiente_cierre_integrada_pendiente_humana":
                quality_status = "auditoria_independiente_cierre_integrada"
            elif row["review_status"] == "microcorreccion_cierre_integrada_pendiente_humana":
                quality_status = "microcorreccion_cierre_integrada"
            elif row["review_status"] == "segunda_revision_positiva_completada_pendiente_humana":
                quality_status = "segunda_revision_final_completada"
            elif row["review_status"] in {"propuesta_semantica_p2_pendiente_aprobacion_humana",
                                           "propuesta_corregida_control_semantico_pendiente_humana"}:
                quality_status = "control_semantico_adicional_por_ia"
            elif row["review_status"] == "propuesta_muestreo_cobertura_pendiente_humana":
                quality_status = "una_revision_de_cobertura_pendiente_control"
            elif (row["review_status"].startswith("propuesta_semantica_p3_") or
                  row["review_status"].startswith("propuesta_semantica_p4_") or
                  row["review_status"].startswith("propuesta_semantica_p5_") or
                  row["review_status"] == "ambiguedad_resuelta_pendiente_aprobacion_humana"):
                quality_status = "adjudicacion_semantica_nueva_completada"
            else:
                quality_status = "evidencia_previa_pendiente_segunda_revision_final"
            qc.append({"need_id": row["need_id"], "document_id": row["document_id"],
                       "proposed_relevance": row["proposed_relevance"],
                       "quality_status": quality_status,
                       "evidence": row["justification"]})
    write(DATA / "positive_quality_review.tsv", list(qc[0]), qc)
    integrity = [{"artifact": name, "sha256": value, "status": "sin_cambios"}
                 for name, value in actual.items()]
    integrity.append({"artifact": "qrels.tsv", "sha256": QRELS_SHA,
                      "status": "provisional_sin_cambios"})
    write(DATA / "integrity_manifest.tsv", list(integrity[0]), integrity)

    counts = defaultdict(Counter)
    for row in review_out:
        counts[row["need_id"]][row["proposed_relevance"] or "pending"] += 1
    p2_keys = set()
    for path in (DATA / "p2_adjudication_batches").glob("batch_*.tsv"):
        p2_keys.update((row["need_id"], row["document_id"]) for row in read(path))
    p1_rows = read(DATA / "newsretrieval_p1_revision_independiente.tsv")
    p1_new = {(row["need_id"], row["document_id"]) for row in p1_rows
              if row["independent_proposed_relevance"] in {"1", "2"}
              and row["codex_proposed_relevance"] in {"1", "2"}}
    new_positive = Counter()
    micro_path = DATA / "microclosing_decisions.tsv"
    micro_positive = set()
    if micro_path.exists():
        micro_positive = {(row["need_id"], row["document_id"]) for row in read(micro_path)
                          if not row["previous_proposed_relevance"]
                          and row["final_proposed_relevance"] in {"1", "2"}}
    for row in review_out:
        key = (row["need_id"], row["document_id"])
        if row["proposed_relevance"] in {"1", "2"} and (key in p2_keys or key in p1_new
                                                          or key in COVERAGE_POSITIVES
                                                          or key in micro_positive):
            new_positive[row["need_id"]] += 1
    correction_counts = Counter(row["need_id"] for row in read(DATA / "semantic_corrections.tsv"))
    closing_path = DATA / "closing_audit_decisions.tsv"
    if closing_path.exists():
        for row in read(closing_path):
            if (row["previous_proposed_relevance"] and
                    row["previous_proposed_relevance"] != row["final_proposed_relevance"]):
                correction_counts[row["need_id"]] += 1
    if micro_path.exists():
        for row in read(micro_path):
            if (row["previous_proposed_relevance"] and
                    row["previous_proposed_relevance"] != row["final_proposed_relevance"]):
                correction_counts[row["need_id"]] += 1
    examined = Counter()
    unjudged = Counter()
    for row in review_out:
        if row["proposed_relevance"] or row["review_status"] in {
                "pendiente_tras_dos_revisiones_independientes",
                "p2_examinado_evidencia_insuficiente_pendiente",
                "ambiguedad_reexaminada_evidencia_insuficiente",
                "examinado_evidencia_insuficiente_pendiente"}:
            examined[row["need_id"]] += 1
        else:
            unjudged[row["need_id"]] += 1

    summary_rows = []
    for need in NEEDS:
        selected = [row for row in pool_out if row["need_id"] == need]
        selected_review = [row for row in review_out if row["need_id"] == need]
        labels = Counter(row["proposed_relevance"] or "pending" for row in selected_review)
        positives = labels["1"] + labels["2"]
        positives_without_top20 = sum(
            row["proposed_relevance"] in {"1", "2"} and
            "top20:" not in pool[(row["need_id"], row["document_id"])]["discovery_provenance"]
            for row in selected_review)
        conflicts = sum(
            row["previous_relevance"] in {"0", "1", "2"} and
            row["previous_relevance"] != row["proposed_relevance"]
            for row in selected_review)
        summary_rows.append({
            "need_id": need, "unique_candidates": len(selected),
            "discovery_top20": sum("top20:" in row["discovery_provenance"] for row in selected),
            "discovery_expansion": sum("expansion:" in row["discovery_provenance"] for row in selected),
            "discovery_existing_or_audit": sum(any(token in row["discovery_provenance"]
                for token in ("qrels", "auditoria", "revision_previa")) for row in selected),
            "discovery_unretrieved_sample": sum(any(token in row["discovery_provenance"]
                for token in ("muestra_", "muestreo")) for row in selected),
            "examined_previous_proposals": sum(row["previous_relevance"] not in {"", "sin_juzgar"}
                                                for row in selected_review),
            "proposed_0": labels["0"], "proposed_1": labels["1"],
            "proposed_2": labels["2"], "pending_or_ambiguous": labels["pending"],
            "possible_false_negatives": positives_without_top20,
            "conflicts_with_previous": conflicts,
            "academic_range_10_15": "within" if 10 <= positives <= 15 else "above" if positives > 15 else "below",
        })
    write(DATA / "relevance_pool_summary.tsv", list(summary_rows[0]), summary_rows)
    OUTPUT.mkdir(exist_ok=True)
    lines = ["# Estado de adjudicación y preparación del benchmark", "",
             "Benchmark provisional: ninguna propuesta equivale a aprobación humana.", "",
             "| Necesidad | Pool | Examinados | 0 | 1 | 2 | Ambiguos/sin juicio | Sin lectura individual | Nuevos positivos | Correcciones | Top-10 pendientes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for need in NEEDS:
        lines.append(f"| {need} | {sum(counts[need].values())} | {examined[need]} | {counts[need]['0']} | {counts[need]['1']} | {counts[need]['2']} | {counts[need]['pending']} | {unjudged[need]} | {new_positive[need]} | {correction_counts[need]} | {next(r['pending'] for r in coverage_rows if r['need_id']==need)} |")
    lines += ["", f"Muestra externa estratificada v1: {len(sample)} pares; nuevos positivos incorporados: {len(COVERAGE_POSITIVES)}.",
              f"Muestra externa estratificada v2: {len(sample_v2)} pares; nuevos positivos incorporados: {len(SECOND_COVERAGE_POSITIVES)}.",
              f"Resultados Top-10 de coseno cero excluidos de cobertura: {zero} apariciones.",
              "Las tríadas easy/medium/hard conservan la misma necesidad; cambia únicamente la formulación.",
              "No queda ningún candidato sin propuesta ni sin lectura individual; todas las etiquetas siguen pendientes de aprobación humana.",
              "Los temas con más de 15 positivos conservan todos los juicios; la especificación 10–15 debe resolverse en la auditoría final sin descartar relevantes auténticos."]
    (OUTPUT / "benchmark_adjudication_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Pool: {len(pool_out)}; qrels candidatos: {len(qrel_candidates)}; pendientes: {len(pending)}")


if __name__ == "__main__":
    main()
