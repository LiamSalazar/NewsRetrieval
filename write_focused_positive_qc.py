"""Registra el control focalizado de positivos de energía y vigilancia."""
import csv
from pathlib import Path

DATA = Path("data")


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main():
    review = {(row["need_id"], row["document_id"]): row
              for row in read(DATA / "pool_human_review.tsv")}
    closing = {(row["need_id"], row["document_id"]): row
               for row in read(DATA / "closing_audit_decisions.tsv")}
    micro = {(row["need_id"], row["document_id"]): row
             for row in read(DATA / "microclosing_decisions.tsv")}
    keys = set()
    for key, row in review.items():
        if key[0] in {"energy", "surveillance"} and row["proposed_relevance"] in {"1", "2"}:
            keys.add(key)
    for key, decision in closing.items():
        if key[0] in {"energy", "surveillance"} and decision["previous_proposed_relevance"] in {"1", "2"}:
            keys.add(key)
        if decision["previous_proposed_relevance"] == "":
            keys.discard(key)  # Los ambiguos se documentan en el cierre, no en el control de positivos previos.
    for key, decision in micro.items():
        if key[0] in {"energy", "surveillance"} and decision["previous_proposed_relevance"] in {"1", "2"}:
            keys.add(key)
    fields = ["need_id", "document_id", "title", "previous_proposed_relevance",
              "final_proposed_relevance", "qc_decision", "deviation_type", "justification"]
    rows = []
    for key in sorted(keys):
        row = review[key]
        decision = micro.get(key) or closing.get(key)
        previous = decision["previous_proposed_relevance"] if decision else row["proposed_relevance"]
        final = row["proposed_relevance"]
        changed = previous != final
        rows.append({
            "need_id": key[0], "document_id": key[1], "title": row["title"],
            "previous_proposed_relevance": previous, "final_proposed_relevance": final,
            "qc_decision": "corregir" if changed else "confirmar",
            "deviation_type": ("suministro_sin_encarecimiento" if key[0] == "energy" and changed
                               else "ausencia_de_vigilancia_publica_o_grado_excesivo"
                               if key[0] == "surveillance" and changed else "sin_desviacion"),
            "justification": decision["justification"] if decision else row["justification"],
        })
    path = DATA / "focused_positive_quality_control.tsv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)
    print(f"Control focalizado: {len(rows)} positivos previos")


if __name__ == "__main__":
    main()
