"""Construye un pool provisional y ciego para la revisión humana de qrels.

Este módulo no cambia el corpus, las matrices ni los qrels.  Los rankings se
usan únicamente como mecanismo de descubrimiento; no se transforma la ausencia
en un ranking en un juicio de relevancia.
"""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import joblib

from collector import CORPUS, read_corpus
from retrieval import VECTORS, search_news


DATA = Path("data")
POOL = DATA / "relevance_pool.tsv"
HUMAN_REVIEW = DATA / "pool_human_review.tsv"
SUMMARY = DATA / "relevance_pool_summary.tsv"
AUDIT = DATA / "newsretrieval_revision_independiente.tsv"
AUDIT_REVIEW = DATA / "audit_integration_review.tsv"
PRIORITY_VIEW = DATA / "pool_review_priority.tsv"

CONFIGURATIONS = [(field, ngram, representation)
                  for field in ("title", "body")
                  for ngram in (1, 2)
                  for representation in ("frequency", "binary", "tfidf")]

# Consultas de descubrimiento, no nuevas definiciones de la necesidad.  Se
# conservan como procedencia y sus resultados quedan sin juicio si no tenían
# una revisión semántica previa.
EXPANSIONS = {
    "trade": ["free trade agreement tariff dispute treaty negotiation",
              "WTO trade rules customs duties retaliation",
              "bilateral commerce pact ratification trade bloc"],
    "energy": ["fuel price increase consumers household bills transport costs",
               "electricity tariff rise energy affordability businesses",
               "petrol gasoline prices economic impact drivers"],
    "football": ["football soccer discrimination politics diplomacy rights controversy",
                 "soccer match cultural identity government dispute",
                 "football social conflict protest racism"],
    "cricket": ["cricket ban sanction corruption safety governance controversy",
                "cricket political dispute tour security institution",
                "cricket disciplinary decision public controversy"],
    "surveillance": ["police facial recognition privacy government surveillance",
                     "public agency tracking data collection civil rights",
                     "license plate reader monitoring oversight safeguards"],
}

# Decisiones semánticas obtenidas al contrastar el texto original con
# relevance_criteria.tsv. Son propuestas para revisión, nunca qrels aprobados.
AUDIT_DECISIONS = {
    "N0b7dca731a735582ed42": ("aceptar", "2", "CAFTA y el tratado andino son instrumentos comerciales concretos y el texto describe su aprobación y negociación."),
    "Nea40d9de95d7c074e738": ("discrepar", "2", "La negociación bilateral, la queja ante la OMC, los aranceles y las obligaciones de China son el asunto central; corresponde grado 2, no 1."),
    "N4e7a4a9a840c2d5f33e9": ("aceptar", "1", "La apertura del mercado del arroz y las políticas de APEC explican directamente la protesta, aunque el foco incluye disturbios y seguridad."),
    "N3da8a56ee9ee6503a824": ("aceptar", "2", "El alza del gas natural elevó los costos de Calpine por encima de sus precios contratados y contribuyó directamente a su quiebra."),
    "N32cf6e0d73cbcfea086e": ("aceptar", "0", "Describe pérdidas por desperdicio de renovables, pero no una subida de tarifas o combustibles ni su efecto en usuarios o actividad económica."),
    "N362d35089bbee01302e6": ("aceptar", "0", "Describe restricciones de generación y cumplimiento ambiental, sin encarecimiento de energía ni consecuencias económicas por precios altos."),
    "Ne0bb64115b2cb69b6fbb": ("aceptar", "2", "La sanción de FIFA y UEFA nace de un conflicto de gobernanza vinculado explícitamente con la división política y étnica bosnia."),
    "N7f9a31c7d486608276b0": ("necesita_revision_humana", "", "El resumen sólo afirma que la crisis haitiana afecta al fútbol; no aporta evidencia suficiente para confirmar el grado 2 ni el vínculo exigido por el criterio."),
    "Nf68a8fabaed54bbd6b85": ("necesita_revision_humana", "", "La descripción menciona vagamente problemas de una serie, sin identificar la controversia institucional necesaria para sostener el grado 2."),
    "N6fbbf60f137b452cfa28": ("necesita_revision_humana", "", "La dimensión poscolonial sustenta posiblemente un grado 1 cultural, pero debe confirmarse que el alcance compartido de las tres consultas la admite."),
    "N4e1554d8ec70687dab2d": ("aceptar", "2", "Documenta la colaboración de AT&T con la NSA para interceptar comunicaciones y entregar registros de usuarios, con litigio sobre derechos."),
    "N3a0ee614dc1b84538217": ("aceptar", "2", "La directiva exige retener metadatos y entregarlos a policía y servicios secretos; la vigilancia pública y sus salvaguardas son el foco."),
    "Nee6523dc2c9c5f31ad13": ("aceptar", "2", "Documenta compras de sistemas de vigilancia con IA por once gobiernos africanos, una práctica pública concreta."),
    "N969cee0ff4444292306e": ("aceptar", "0", "El rastreo lo realizó la empresa privada Phorm/BT; las autoridades aparecen como reguladores y no como ejecutores de vigilancia pública."),
}


def tsv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, fields, rows):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sample_unretrieved(rows, excluded, need_id, size=6):
    """Muestra estable, estratificada por fuente y categoría, fuera del pool."""
    groups = defaultdict(list)
    for row in rows:
        if row["id"] not in excluded:
            groups[(row["source"], row["category"])].append(row)
    selected = []
    # Una vuelta sobre celdas fuente/categoría evita que una fuente o categoría
    # dominante convierta la muestra de control en una muestra accidentalmente homogénea.
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda r: hashlib.sha256(
            f"pool-v1|{need_id}|{r['id']}".encode("utf-8")).hexdigest())
        selected.append(ordered[0])
        if len(selected) == size:
            break
    return selected


def main():
    queries = tsv_rows(DATA / "queries.tsv")
    qrels = tsv_rows(DATA / "qrels.tsv")
    existing = tsv_rows(DATA / "candidates_review.tsv")
    audit = tsv_rows(AUDIT)
    corpus = read_corpus(CORPUS)
    documents = {row["id"]: row for row in corpus}
    if len(documents) != len(corpus):
        raise ValueError("El corpus contiene identificadores duplicados")

    # Las necesidades se infieren de la revisión existente, que ya une cada
    # query con su formulación y necesidad. No se modifica queries.tsv.
    query_meta = {}
    for row in existing:
        query_meta[row["query_id"]] = {key: row[key] for key in
                                        ("need_id", "information_need", "query")}
    if set(query_meta) != {row["query_id"] for row in queries}:
        raise ValueError("candidates_review.tsv no cubre las 15 consultas")
    if len(audit) != 14 or len({(row["need_id"], row["document_id"]) for row in audit}) != 14:
        raise ValueError("La auditoría independiente debe contener 14 casos únicos")
    if set(AUDIT_DECISIONS) != {row["document_id"] for row in audit}:
        raise ValueError("Falta una decisión semántica para algún caso de auditoría")
    invalid_audit = [row["document_id"] for row in audit if row["document_id"] not in documents]
    if invalid_audit:
        raise ValueError(f"Documentos de auditoría ausentes del corpus: {invalid_audit}")

    prior = {(row["need_id"], row["document_id"]): row for row in existing}
    qrel_ids = {(query_meta[row["query_id"]]["need_id"], row["document_id"])
                for row in qrels}
    candidates = defaultdict(lambda: defaultdict(set))
    top10_positive = defaultdict(set)
    bundle = joblib.load(VECTORS)
    zero_results = 0

    # Los 12 top-20 exigidos para cada una de las 15 formulaciones.
    for query in queries:
        meta = query_meta[query["query_id"]]
        for field, ngram, representation in CONFIGURATIONS:
            found = search_news(query["query"], field, ngram, representation,
                                top_k=20, bundle=bundle)
            ids = [row["id"] for row in found]
            if len(ids) != 20 or len(set(ids)) != 20 or not set(ids) <= documents.keys():
                raise ValueError(f"Ranking inválido: {query['query_id']} {field} {ngram} {representation}")
            label = f"top20:{query['query_id']}:{field}:ngram{ngram}:{representation}"
            for rank, row in enumerate(found, 1):
                if row["score"] <= 0.0:
                    zero_results += 1
                    continue
                candidates[meta["need_id"]][row["id"]].add(label)
                if rank <= 10:
                    top10_positive[meta["need_id"]].add(row["id"])

    # Expansión léxica trazable mediante sinónimos, paráfrasis y conceptos
    # relacionados. Sólo descubre; no puntúa semánticamente los documentos.
    for need_id, expansion_queries in EXPANSIONS.items():
        for number, text in enumerate(expansion_queries, 1):
            found = search_news(text, "body", 1, "tfidf", top_k=20, bundle=bundle)
            for row in found:
                if row["score"] > 0.0:
                    candidates[need_id][row["id"]].add(f"expansion:{number}")
                else:
                    zero_results += 1

    for need_id, document_id in qrel_ids:
        candidates[need_id][document_id].add("qrels_provisional_positive")
    for (need_id, document_id), row in prior.items():
        if row["relevance_previous"] in {"0", "0/no incluido"}:
            candidates[need_id][document_id].add("auditoria_o_negativo_sospechoso")
        else:
            candidates[need_id][document_id].add("revision_previa")
    pool_before_audit = {(need_id, document_id) for need_id, values in candidates.items()
                         for document_id in values}
    for row in audit:
        candidates[row["need_id"]][row["document_id"]].add("auditoria_independiente")

    # Sólo después de fijar todos los mecanismos de recuperación se añade la
    # muestra de control, garantizando que sea genuinamente no recuperada.
    for need_id in EXPANSIONS:
        excluded = set(candidates[need_id])
        for row in sample_unretrieved(corpus, excluded, need_id):
            candidates[need_id][row["id"]].add("muestra_no_recuperada_estratificada_v1")

    audit_by_key = {(row["need_id"], row["document_id"]): row for row in audit}
    pool_rows, review_rows = [], []
    for need_id in sorted(candidates):
        info = next(meta["information_need"] for meta in query_meta.values()
                    if meta["need_id"] == need_id)
        for document_id in sorted(candidates[need_id]):
            document = documents[document_id]
            sources = sorted(candidates[need_id][document_id])
            pool_rows.append({"need_id": need_id, "document_id": document_id,
                              "discovery_provenance": " | ".join(sources)})
            old = prior.get((need_id, document_id))
            audited = audit_by_key.get((need_id, document_id))
            if audited:
                decision, audit_label, rationale = AUDIT_DECISIONS[document_id]
                previous = audited["current_label"]
                if decision == "necesita_revision_humana":
                    proposed = old["relevance_proposed"] if old else ""
                    confidence = "baja"
                    status = "conflicto_auditoria_pendiente_revision_humana"
                else:
                    proposed = audit_label
                    confidence = audited["confidence"]
                    status = f"propuesta_auditoria_{decision}_pendiente_humana"
                justification = rationale
            elif old:
                # Se conserva exactamente la propuesta y la justificación
                # semántica anterior, pero nunca se declara aprobación humana.
                proposed = old["relevance_proposed"]
                justification = old["justification"]
                confidence = "media"
                status = "propuesta_previa_reexaminada_pendiente_humana"
                previous = old["relevance_previous"]
            else:
                proposed = ""
                previous = "0/no incluido"
                justification = ("No examinada semánticamente en este pool; su presencia "
                                 "es sólo de descubrimiento y no implica relevancia ni irrelevancia.")
                confidence = "no evaluada"
                status = "pendiente_sin_revision_semantica"
            review_rows.append({"need_id": need_id, "document_id": document_id,
                                "title": document["title"], "excerpt": document["description/body"],
                                "url": document["url"], "previous_relevance": previous,
                                "proposed_relevance": proposed, "justification": justification,
                                "confidence": confidence, "review_status": status})

    def priority(row):
        key = (row["need_id"], row["document_id"])
        sources = candidates[row["need_id"]][row["document_id"]]
        if key in audit_by_key:
            return 1
        if not row["proposed_relevance"] and row["document_id"] in top10_positive[row["need_id"]]:
            return 2
        if not row["proposed_relevance"] and any(source.startswith("expansion:") for source in sources):
            return 3
        if ("auditoria_o_negativo_sospechoso" in sources
                or "muestra_no_recuperada_estratificada_v1" in sources):
            return 4
        if not row["proposed_relevance"]:
            return 5
        return 6

    review_rows.sort(key=lambda row: (priority(row), row["need_id"], row["document_id"]))
    fields = ["need_id", "document_id", "discovery_provenance"]
    write_tsv(POOL, fields, pool_rows)
    review_fields = ["need_id", "document_id", "title", "excerpt", "url",
                     "previous_relevance", "proposed_relevance", "justification",
                     "confidence", "review_status"]
    write_tsv(HUMAN_REVIEW, review_fields, review_rows)

    priority_names = {
        1: "P1_conflicto_auditoria", 2: "P2_lectura_prioritaria_sin_juicio",
        3: "P3_busqueda_ampliada_sin_juicio", 4: "P4_control_o_negativo_sospechoso",
        5: "P5_demas_pendientes", 6: "P6_propuesta_previa_sin_conflicto",
    }
    priority_rows = [{"need_id": row["need_id"], "document_id": row["document_id"],
                      "review_priority": priority_names[priority(row)]} for row in review_rows]
    write_tsv(PRIORITY_VIEW, ["need_id", "document_id", "review_priority"], priority_rows)

    audit_rows = []
    for row in audit:
        decision, proposed, rationale = AUDIT_DECISIONS[row["document_id"]]
        audit_rows.append({"need_id": row["need_id"], "document_id": row["document_id"],
                           "title": documents[row["document_id"]]["title"],
                           "current_label": row["current_label"],
                           "auditor_suggested_label": row["suggested_label"],
                           "integration_decision": decision, "proposed_relevance": proposed,
                           "justification": rationale,
                           "present_before_audit": "si" if (row["need_id"], row["document_id"]) in pool_before_audit else "no",
                           "present_after_audit": "si"})
    write_tsv(AUDIT_REVIEW, list(audit_rows[0]), audit_rows)

    summary_rows = []
    for need_id in sorted(candidates):
        relevant = Counter(row["proposed_relevance"] for row in review_rows if row["need_id"] == need_id)
        # Un candidato que apareció en varias configuraciones cuenta una sola
        # vez por mecanismo en el resumen; la granularidad completa permanece
        # disponible en relevance_pool.tsv.
        mechanisms = Counter()
        for sources in candidates[need_id].values():
            mechanisms.update({source.split(":", 1)[0] for source in sources})
        pending = sum(row["need_id"] == need_id and not row["proposed_relevance"] for row in review_rows)
        conflicts = sum(row["need_id"] == need_id and row["previous_relevance"] not in {"", "0/no incluido"}
                        and row["proposed_relevance"] != row["previous_relevance"] for row in review_rows)
        positives = relevant["1"] + relevant["2"]
        summary_rows.append({"need_id": need_id, "unique_candidates": len(candidates[need_id]),
                             "discovery_top20": mechanisms["top20"],
                             "discovery_expansion": mechanisms["expansion"],
                             "discovery_existing_or_audit": mechanisms["qrels_provisional_positive"] + mechanisms["revision_previa"] + mechanisms["auditoria_o_negativo_sospechoso"],
                             "discovery_unretrieved_sample": mechanisms["muestra_no_recuperada_estratificada_v1"],
                             "examined_previous_proposals": len(candidates[need_id]) - pending,
                             "proposed_0": relevant["0"], "proposed_1": relevant["1"],
                             "proposed_2": relevant["2"], "pending_or_ambiguous": pending,
                             "possible_false_negatives": mechanisms["auditoria_o_negativo_sospechoso"],
                             "conflicts_with_previous": conflicts,
                             "academic_range_10_15": ("within" if 10 <= positives <= 15 else
                                                      "below" if positives < 10 else "above")})
    write_tsv(SUMMARY, list(summary_rows[0]), summary_rows)
    print(f"Pool provisional: {len(pool_rows)} candidatos únicos para {len(summary_rows)} necesidades.")
    print(f"Resultados de similitud cero omitidos como evidencia: {zero_results}.")


if __name__ == "__main__":
    main()
