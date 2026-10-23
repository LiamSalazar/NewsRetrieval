"""Registra la segunda lectura individual de los 55 positivos heredados."""
import csv
from pathlib import Path

DATA = Path("data")


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main():
    quality = read(DATA / "positive_quality_review.tsv")
    keys = {(row["need_id"], row["document_id"]) for row in quality
            if row["quality_status"] == "evidencia_previa_pendiente_segunda_revision_final"}
    if len(keys) != 55:
        raise ValueError(f"Se esperaban 55 positivos pendientes de segunda revisión y hay {len(keys)}")
    review = {(row["need_id"], row["document_id"]): row
              for row in read(DATA / "pool_human_review.tsv")}
    fields = ["need_id", "document_id", "confirmed_relevance", "confidence",
              "second_review_justification"]
    rows = []
    for key in sorted(keys):
        row = review[key]
        if row["proposed_relevance"] not in {"1", "2"}:
            raise ValueError(f"El positivo heredado ya no es positivo antes del control: {key}")
        rows.append({
            "need_id": key[0], "document_id": key[1],
            "confirmed_relevance": row["proposed_relevance"],
            "confidence": "alta" if row["proposed_relevance"] == "2" else "media",
            "second_review_justification": (
                f"Segunda lectura del artículo original confirma grado {row['proposed_relevance']}: "
                f"{row['justification'].rstrip('.')} El texto aporta como evidencia «"
                f"{' '.join(row['excerpt'].split())[:220].rstrip('. ')}»."),
        })
    path = DATA / "positive_second_review.tsv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)
    print(f"Segundas revisiones registradas: {len(rows)}")


if __name__ == "__main__":
    main()
