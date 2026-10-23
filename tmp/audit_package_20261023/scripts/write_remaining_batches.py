"""Materializa en lotes reanudables las lecturas semánticas P3-P5."""
import csv
from pathlib import Path

DATA = Path("data")
OUT = DATA / "remaining_adjudication_batches"


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


# Excepciones identificadas tras leer título y cuerpo originales. El resto de
# los 442 pares leídos no satisface la necesidad asignada.
DECISIONS = {
    ("cricket", "N5f9ce263e530d6fa61bc"): ("1", "El artículo vincula directamente a Kerry Packer con la transformación pública e institucional del cricket mediante World Series Cricket, los partidos de un día, uniformes de color y encuentros nocturnos."),
    ("cricket", "N92cca1afd9cd56ca1e74"): ("1", "La muerte de Shane Warne produjo consecuencias públicas directamente vinculadas al cricket: homenajes internacionales, funeral de Estado y cambio de nombre de una tribuna del MCG."),
    ("energy", "N2021193f1367e4bbfd71"): ("1", "La crisis eléctrica de Cyrenaica causó una crisis de agua y reabrió disputas sobre recursos y desigualdad regional; documenta consecuencias de falta y asequibilidad energética, aunque no una subida tarifaria explícita."),
    ("energy", "N91c6e09e59f02a17f4ee"): ("1", "El texto conecta los apagones planificados tras el terremoto con retraso de la recuperación, caída del PIB y costes de reconstrucción; es impacto económico directo de una crisis de suministro, sin incremento de precio explícito."),
    ("energy", "N1ae2ef79084f4e696b40"): ("1", "La noticia explica cómo los apagones crónicos impiden a los hogares nigerianos afrontar las olas de calor, una consecuencia concreta para consumidores de una crisis de suministro, aunque no informa una subida de tarifa."),
    ("football", "N715b5dcdec697060455c"): ("2", "El memorial de Hillsborough trata consecuencias sociales, institucionales y culturales centrales del fútbol: 96 víctimas, reclamos de responsabilidad, supuesto encubrimiento policial y conflicto con la prensa."),
    ("surveillance", "N0011c109cfb67b5061b8"): ("1", "La clonación del pasaporte RFID cuestiona una tecnología pública de identificación que permite leer datos personales a distancia y expone consecuencias de privacidad y seguridad."),
    ("surveillance", "N383f7425309e0098c012"): ("1", "La sentencia del TJUE regula cuándo proveedores deben retener y entregar tráfico e identidad a autoridades; es una salvaguarda jurídica directamente conectada con recopilación pública de datos."),
    ("surveillance", "Nd57a8987881456e87039"): ("1", "El proyecto de la Comisión Europea ampliaría verificación de edad intrusiva y control de acceso; el artículo examina la regulación pública y sus consecuencias para privacidad y libertad de expresión."),
    ("surveillance", "N318aa6952b490b0a4b01"): ("2", "FUNAI, una agencia estatal, localizó y verificó a un pueblo indígena aislado mediante satélite y vuelos; el texto describe el rastreo público de personas y sus implicaciones para territorio y supervivencia."),
    ("surveillance", "N9ed9f8377a6071c43695"): ("2", "El artículo examina la denuncia de recopilación masiva de registros telefónicos por la NSA, las negativas empresariales y una solicitud de investigación por posibles violaciones de privacidad."),
    ("trade", "N0598892381df3bc21db3"): ("1", "La protesta ocurre durante la cumbre APEC y se dirige contra su agenda institucional de liberalización comercial e inversión; es una controversia directamente vinculada a negociaciones, aunque sin un acuerdo concreto aprobado."),
    ("trade", "N4fc364155b4e18516687"): ("2", "La noticia trata directamente la conferencia ministerial de la OMC, la disputa entre países por agricultura y barreras comerciales y las protestas contra el posible acuerdo comercial."),
    ("trade", "Ndfc5ff43f3361ea0ef1a"): ("2", "Australia y China firmaron un acuerdo bilateral de salvaguardias que habilita exportaciones de uranio, fija condiciones gubernamentales y genera controversia sobre su aplicación."),
}

AMBIGUOUS = {
    ("surveillance", "N6b22c8c2865211665841"): "El único texto disponible afirma daños mentales por vigilancia digital sobre activistas, pero no identifica quién vigila; no permite confirmar que la práctica sea ejercida por una autoridad pública.",
}

ZERO_CRITERIA = {
    "trade": "no describe un acuerdo, tratado, regla o controversia comercial concreta entre gobiernos o bloques",
    "energy": "no vincula una subida de energía o una crisis de suministro con consecuencias económicas para usuarios",
    "football": "no presenta una consecuencia política, diplomática, social o cultural sustantiva vinculada al fútbol",
    "cricket": "no trata una controversia institucional, disciplinaria, política, de seguridad o significado público del cricket",
    "surveillance": "no documenta una práctica tecnológica de recopilación o rastreo ejercida por una autoridad pública",
}


def main():
    review = read(DATA / "pool_human_review.tsv")
    priority = {(row["need_id"], row["document_id"]): row["review_priority"]
                for row in read(DATA / "pool_review_priority.tsv")}
    selected = [row for row in review if row["review_status"] == "pendiente_sin_revision_semantica"
                and priority[(row["need_id"], row["document_id"])][0:2] in {"P3", "P4", "P5"}]
    if len(selected) != 442:
        raise ValueError(f"Se esperaban 442 candidatos sin lectura y hay {len(selected)}")
    OUT.mkdir(exist_ok=True)
    fields = ["need_id", "document_id", "priority", "proposed_relevance", "confidence", "justification"]
    for start in range(0, len(selected), 35):
        rows = []
        for row in selected[start:start + 35]:
            key = (row["need_id"], row["document_id"])
            prio = priority[key].split("_", 1)[0]
            if key in DECISIONS:
                label, reason = DECISIONS[key]
                confidence = "alta" if label == "2" else "media"
            elif key in AMBIGUOUS:
                label, reason, confidence = "", AMBIGUOUS[key], "baja"
            else:
                label = "0"
                confidence = "alta"
                reason = (f"El artículo «{row['title']}» trata {row['excerpt'].strip()[:180].rstrip('. ')}; "
                          f"{ZERO_CRITERIA[row['need_id']] }.")
            rows.append({"need_id": row["need_id"], "document_id": row["document_id"],
                         "priority": prio, "proposed_relevance": label,
                         "confidence": confidence, "justification": reason})
        write(OUT / f"batch_{start // 35 + 1:03d}.tsv", fields, rows)
    print(f"Lotes escritos: {(len(selected) + 34) // 35}; decisiones: {len(selected)}")


if __name__ == "__main__":
    main()
