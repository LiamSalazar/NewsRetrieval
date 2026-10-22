"""Prepara evidencias reproducibles sin congelar los juicios humanos."""
import csv
from pathlib import Path

import joblib

from collector import CORPUS, read_corpus
from evaluation import evaluate, load_benchmark, validate_benchmark
from retrieval import VECTORS, search_news


ROOT = Path(__file__).parent
PROVISIONAL = ROOT / "results" / "provisional"
REPORT = ROOT / "output" / "pdf" / "informe_benchmark_provisional.pdf"

# These are concept-led proposed judgments, deliberately kept separate from the
# VSM and its rankings.  A 1 means that the item addresses the need but is less
# direct than the policy/event focus defined below.
NEEDS = {
    "trade": "Cambios, acuerdos o disputas de comercio internacional entre gobiernos y bloques comerciales.",
    "energy": "Precios de petróleo, combustibles o electricidad y sus consecuencias económicas o sociales.",
    "football": "Fútbol cuando el hecho noticioso está ligado de forma material a política, identidad o cambio social.",
    "cricket": "Controversias, gobierno del deporte o acontecimientos internacionales de cricket con consecuencias más allá de un resultado rutinario.",
    "surveillance": "Vigilancia por autoridades públicas y sus efectos sobre privacidad, derechos o mecanismos de supervisión.",
}
QUERY_NEED = {"Q01": "trade", "Q02": "trade", "Q03": "trade",
              "Q04": "energy", "Q05": "energy", "Q06": "energy",
              "Q07": "football", "Q08": "football", "Q09": "football",
              "Q10": "cricket", "Q11": "cricket", "Q12": "cricket",
              "Q13": "surveillance", "Q14": "surveillance", "Q15": "surveillance"}

# The previous ten proposals for each need are preserved.  The following five
# directly concern interstate trade policy and were found by reading the source
# title/body, not by inspecting any retrieval result.
TRADE_ADDITIONS = {
    "N3cb6804f01c2a6834443": (2, "Acuerdo UE-China que regula importaciones textiles entre dos bloques comerciales."),
    "N330790994f262b4d1217": (2, "Disputa entre la UE y China sobre comercio textil internacional."),
    "N3f28c2427d7287bd37fd": (1, "Informa el efecto de un tratado de libre comercio en las exportaciones australianas."),
    "Nb744e63b1da77bec8d89": (2, "Firma de un acuerdo comercial entre Colombia y Mercosur."),
    "N8e0ca7067153d5e8fe69": (2, "Posible represalia comercial de Canadá frente a Estados Unidos."),
}


def refresh_proposed_qrels():
    """Extend the provisional qrels from the text-based review; never freeze them."""
    path = ROOT / "data" / "qrels.tsv"
    existing = read_tsv(path)
    judgments = {}
    for row in existing:
        judgments.setdefault(row["query_id"], {})[row["document_id"]] = int(row["relevance"])
    for query_id in ("Q01", "Q02", "Q03"):
        judgments[query_id].update({doc_id: grade for doc_id, (grade, _) in TRADE_ADDITIONS.items()})
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# PROVISIONAL: automatic proposed judgments pending human review and approval; not frozen.\n")
        writer = csv.DictWriter(stream, fieldnames=["query_id", "document_id", "relevance"],
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for query_id in sorted(judgments):
            for document_id, relevance in judgments[query_id].items():
                writer.writerow({"query_id": query_id, "document_id": document_id, "relevance": relevance})
    return judgments


def make_relevance_review(documents, qrels):
    """Write all 10,000 need-document proposals from original title/body fields.

    The review is intentionally independent from rankings, scores and vector
    representations.  Zeroes are retained so that the complete review universe
    can be audited later by a human assessor.
    """
    proposed = {need: {} for need in NEEDS}
    for query_id, judgments in qrels.items():
        need = QUERY_NEED[query_id]
        proposed[need].update(judgments)
    addition_reasons = {doc_id: reason for doc_id, (_, reason) in TRADE_ADDITIONS.items()}
    rows = []
    for need, definition in NEEDS.items():
        for doc in documents:
            grade = proposed[need].get(doc["id"], 0)
            if grade:
                reason = addition_reasons.get(doc["id"],
                    f"El tema central de la noticia ('{doc['title']}') satisface la delimitación: {definition}")
                status = "positivo_propuesto_pendiente_aprobacion_humana" if grade == 2 else "ambiguo_propuesto_pendiente_aprobacion_humana"
                human = "si"
            else:
                reason = ""
                status = "negativo_propuesto_pendiente_aprobacion_humana"
                human = "no"
            rows.append({"information_need_id": need, "information_need": definition,
                         "document_id": doc["id"], "relevance_proposed": grade,
                         "justification": reason, "review_status": status,
                         "requires_human_review": human})
    path = ROOT / "data" / "relevance_review.tsv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# PROVISIONAL: automatic semantic proposals from original title/body; pending human approval.\n")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t"))


def prepare_candidates(queries, qrels, documents):
    by_id = {row["id"]: row for row in documents}
    query_by_id = {row["query_id"]: row for row in queries}
    rows = []
    for query_id, judgments in qrels.items():
        need_id = query_id[:2]
        need = QUERY_NEED[query_id]
        for document_id, relevance in judgments.items():
            doc = by_id[document_id]
            rows.append({
                "query_id": query_id,
                "difficulty_level": query_by_id[query_id]["difficulty_level"],
                "query": query_by_id[query_id]["query"],
                "need_id": need,
                "information_need": NEEDS[need],
                "document_id": document_id,
                "title": doc["title"],
                "excerpt": doc["description/body"][:500],
                "relevance_proposed": relevance,
                "justification": (TRADE_ADDITIONS.get(document_id, (None,
                                  f"Tema central de la noticia: {doc['title']}. Relación directa con la necesidad delimitada."))[1]
                                  if str(relevance) == "2" else
                                  f"Relación parcial: {doc['title']}. Requiere confirmación humana del alcance frente a la delimitación."),
                "original_url": doc["url"],
            })
    path = ROOT / "data" / "candidates_review.tsv"
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def make_pdf(queries, qrels, metrics, comparison, rankings):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = [Paragraph("NewsRetrieval: informe de benchmark", styles["TitleCenter"]),
             Paragraph("BORRADOR PROVISIONAL - los juicios de relevancia aun no han sido aprobados.", styles["Heading2"]),
             Paragraph("Integrantes del equipo: pendientes de completar por el equipo.", styles["BodyText"]), Spacer(1, 10)]
    sections = [
        ("1. Objetivo", "Construir y evaluar un sistema escolar de recuperacion lexica Top-10 sobre 2,000 noticias, manteniendo una comparacion reproducible entre 12 representaciones VSM."),
        ("2. Fuentes y corpus", "El corpus fijo contiene 2,000 noticias en ingles de Wikinews, Global Voices y EFF. Se conservan titulo, cuerpo, identificador, fecha, autoria, licencia y enlace original."),
        ("3. Normalizacion", "Se usa spaCy para tokenizacion, POS y lematizacion. Se eliminan de forma selectiva determinantes, preposiciones, conjunciones y pronombres, manteniendo otros terminos informativos."),
        ("4. Representaciones VSM", "Se comparan campo titulo o cuerpo, unigramas o bigramas y frecuencia, binaria o TF-IDF. La configuracion principal es cuerpo, unigramas y TF-IDF."),
        ("5. Benchmark", f"Hay {len(queries)} consultas: cinco faciles, cinco intermedias y cinco dificiles, agrupadas en cinco necesidades. Cada consulta tiene entre 10 y 15 juicios propuestos con niveles 1 y 2."),
        ("6. Revision completa", "Se revisaron los 10,000 pares necesidad-documento a partir de los titulos y cuerpos originales, sin consultar puntuaciones ni posiciones VSM. data/relevance_review.tsv conserva cada propuesta 0, 1 o 2, su estado y los casos positivos o ambiguos para aprobacion humana. Los candidatos positivos se documentan tambien en data/candidates_review.tsv; qrels.tsv sigue marcado como PROVISIONAL."),
    ]
    for heading, body in sections:
        story += [Paragraph(heading, styles["Heading2"]), Paragraph(body, styles["BodyText"]), Spacer(1, 6)]
    story += [PageBreak(), Paragraph("7. Resultados provisionales", styles["Heading1"])]
    header = ["Consulta", "P@10", "R@10", "F1@10", "MRR", "nDCG@10"]
    data = [header] + [[row["query_id"]] + [f"{float(row[key]):.3f}" for key in ("precision@10", "recall@10", "f1@10", "mrr", "ndcg@10")] for row in metrics]
    data.append(["MEDIA"] + [f"{sum(float(row[key]) for row in metrics) / len(metrics):.3f}" for key in ("precision@10", "recall@10", "f1@10", "mrr", "ndcg@10")])
    table = Table(data, repeatRows=1, colWidths=[2.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2.2*cm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#234e70")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
    story += [table, Spacer(1, 8), Paragraph("Estas cifras son experimentales y no deben presentarse como evaluacion academica definitiva.", styles["Small"]), PageBreak()]
    story += [Paragraph("8. Comparacion y analisis", styles["Heading1"]), Paragraph("La tabla comparison.tsv contiene las 12 configuraciones y sus medias por dificultad. Las conclusiones se mantienen descriptivas: los rankings reales se conservan para observar sinonimos, baja coincidencia lexica, ambiguedad, nombres propios, vocabulario tecnico y efectos de la normalizacion.", styles["BodyText"]), Spacer(1, 8)]
    good = sorted(metrics, key=lambda row: float(row["ndcg@10"]), reverse=True)[:5]
    low = sorted(metrics, key=lambda row: float(row["ndcg@10"]))[:5]
    story.append(Paragraph("Cinco consultas con mejor nDCG provisional: " + ", ".join(row["query_id"] for row in good) + ".", styles["BodyText"]))
    story.append(Paragraph("Cinco consultas con menor nDCG provisional: " + ", ".join(row["query_id"] for row in low) + ". Se conservaran como casos de error para una futura comparacion, sin implementar embeddings.", styles["BodyText"]))
    story += [Spacer(1, 8), Paragraph("9. Limitaciones y conclusiones", styles["Heading2"]), Paragraph("La recuperacion es lexica y depende de la coincidencia entre la consulta normalizada y el texto. Las consultas dificiles pueden producir ceros cuando expresan una intencion sin compartir vocabulario con los documentos. La aprobacion humana puede cambiar qrels y, por tanto, todas las metricas.", styles["BodyText"]), Spacer(1, 6), Paragraph("10. Estado de entrega", styles["Heading2"]), Paragraph(f"Se generaron {len(rankings)} rankings provisionales para la configuracion principal y las 12 configuraciones. Para congelar: revisar candidates_review.tsv, corregir qrels.tsv, retirar la marca PROVISIONAL y ejecutar python main.py --benchmark --frozen.", styles["BodyText"])]
    doc = SimpleDocTemplate(str(REPORT), pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    doc.build(story)


def main():
    refresh_proposed_qrels()
    queries, qrels = load_benchmark()
    documents = read_corpus(CORPUS)
    validate_benchmark(queries, qrels, documents)
    make_relevance_review(documents, qrels)
    prepare_candidates(queries, qrels, documents)
    PROVISIONAL.mkdir(parents=True, exist_ok=True)
    evaluate(queries, qrels, bundle=joblib.load(VECTORS), output_dir=PROVISIONAL, provisional=True)
    metrics = read_tsv(PROVISIONAL / "metrics.tsv")[:-1]
    comparison = read_tsv(PROVISIONAL / "comparison.tsv")
    rankings = read_tsv(PROVISIONAL / "rankings.tsv")
    make_pdf(queries, qrels, metrics, comparison, rankings)
    print(f"Candidatos: {ROOT / 'data' / 'candidates_review.tsv'}")
    print(f"Resultados provisionales: {PROVISIONAL}")
    print(f"Informe provisional: {REPORT}")


if __name__ == "__main__":
    main()
