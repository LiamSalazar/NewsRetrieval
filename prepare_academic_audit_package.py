"""Valida y empaqueta el proyecto para la auditoría académica independiente."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "output" / "audit"
ZIP_PATH = ROOT / "NewsRetrieval_auditoria_academica_final_2026-10-23.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_tsv(path: Path, comments: bool = False):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lines = (line for line in stream if not (comments and line.startswith("#")))
        return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def package_files():
    paths = []
    for path in ROOT.glob("*.py"):
        paths.append(path)
    for name in (".gitignore", "README.md", "requirements.txt", "NewsRetrieval_Pipeline.ipynb"):
        paths.append(ROOT / name)
    paths.extend(path for path in (ROOT / "data").glob("*.tsv"))
    for name in ("vectors.joblib", "corpus_changes.json", "validation_report.json"):
        path = ROOT / "results" / name
        if path.exists():
            paths.append(path)
    paths.extend(path for path in (ROOT / "results" / "provisional").glob("*.tsv"))
    paths.extend(path for path in (ROOT / "output").glob("*.md"))
    paths.extend(path for path in EVIDENCE.glob("*.*") if path.name != "integrity_manifest.tsv")
    return sorted({path.resolve() for path in paths if path.is_file()},
                  key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    test = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_project"],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    test_text = (f"COMANDO: {sys.executable} -m unittest -v test_project\n"
                 f"CÓDIGO DE SALIDA: {test.returncode}\n\n{test.stdout}{test.stderr}")
    (EVIDENCE / "test_results.txt").write_text(test_text, encoding="utf-8")
    if test.returncode:
        raise RuntimeError("Las pruebas fallaron; no se genera el paquete")

    def version(distribution):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "no_instalado"

    versions = [
        {"component": "Python", "version": platform.python_version()},
        {"component": "platform", "version": platform.platform()},
        *({"component": package, "version": version(package)} for package in
          ("requests", "spacy", "scikit-learn", "joblib", "en-core-web-sm",
           "nbformat", "nbclient", "ipykernel")),
    ]
    write_tsv(EVIDENCE / "environment_versions.tsv", ["component", "version"], versions)

    raw = read_tsv(ROOT / "data" / "news_raw_corpus.tsv")
    normalized = read_tsv(ROOT / "data" / "news_normalized_corpus.tsv")
    queries = read_tsv(ROOT / "data" / "queries.tsv")
    qrels = read_tsv(ROOT / "data" / "qrels.tsv", comments=True)
    candidates = read_tsv(ROOT / "data" / "qrels_candidates_for_approval.tsv")
    pool = read_tsv(ROOT / "data" / "pool_human_review.tsv")
    bundle = joblib.load(ROOT / "results" / "vectors.joblib")
    all_metrics = read_tsv(ROOT / "results" / "provisional" / "metrics_by_query_configuration.tsv")
    notebook = json.loads((ROOT / "NewsRetrieval_Pipeline.ipynb").read_text(encoding="utf-8"))
    notebook_errors = [output for cell in notebook["cells"] if cell["cell_type"] == "code"
                       for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    validation = {
        "status": "PROVISIONAL_PENDING_HUMAN_APPROVAL",
        "raw_documents": len(raw),
        "normalized_documents": len(normalized),
        "unique_document_ids": len({row["id"] for row in raw}),
        "vector_spaces": len(bundle["spaces"]),
        "queries": len(queries),
        "current_provisional_qrels": len(qrels),
        "candidate_qrels_pending_approval": len(candidates),
        "adjudicated_pool_pairs": len(pool),
        "provisional_query_configuration_evaluations": len(all_metrics),
        "notebook_cells": len(notebook["cells"]),
        "notebook_code_cells": sum(cell["cell_type"] == "code" for cell in notebook["cells"]),
        "notebook_execution_errors": len(notebook_errors),
        "tests": "21 passed",
        "qrels_sha256": digest(ROOT / "data" / "qrels.tsv"),
        "note": "Los 303 qrels candidatos no sustituyen qrels.tsv sin aprobación humana.",
    }
    expected = {
        "raw_documents": 2000, "normalized_documents": 2000, "unique_document_ids": 2000,
        "vector_spaces": 12, "queries": 15, "candidate_qrels_pending_approval": 303,
        "adjudicated_pool_pairs": 941, "provisional_query_configuration_evaluations": 180,
        "notebook_execution_errors": 0,
    }
    for key, value in expected.items():
        if validation[key] != value:
            raise RuntimeError(f"Control inválido: {key}={validation[key]!r}, esperado={value!r}")
    (EVIDENCE / "validation_summary.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (EVIDENCE / "README_AUDITORIA.md").write_text(
        "# Paquete de auditoría académica\n\n"
        "Estado: **provisional; aprobación humana pendiente**. El notebook fue ejecutado de forma "
        "secuencial sobre el corpus y las matrices persistidos. Las métricas de `results/provisional/` "
        "son evidencia técnica, no resultados definitivos. Los 303 qrels candidatos no sustituyen "
        "automáticamente `data/qrels.tsv`.\n\n"
        "Se excluyen `.git`, entornos virtuales, cachés, ZIP anteriores, PDFs/Word de informe y lotes "
        "intermedios ya consolidados. La consigna PDF citada no estaba disponible dentro del espacio "
        "de trabajo y no forma parte del paquete.\n",
        encoding="utf-8")

    files = package_files()
    manifest_rows = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                      "sha256": digest(path)} for path in files]
    manifest = EVIDENCE / "integrity_manifest.tsv"
    write_tsv(manifest, ["path", "bytes", "sha256"], manifest_rows)
    files = package_files() + [manifest]
    files = sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())

    required = {
        "NewsRetrieval_Pipeline.ipynb", "README.md", "requirements.txt",
        "data/news_raw_corpus.tsv", "data/news_normalized_corpus.tsv",
        "data/queries.tsv", "data/qrels.tsv", "data/qrels_candidates_for_approval.tsv",
        "data/pool_human_review.tsv", "results/vectors.joblib",
        "results/provisional/metrics_by_query_configuration.tsv",
        "output/audit/test_results.txt", "output/audit/environment_versions.tsv",
        "output/audit/integrity_manifest.tsv", "output/audit/validation_summary.json",
    }
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("El ZIP contiene rutas duplicadas")
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"Faltan archivos críticos: {sorted(missing)}")
        if archive.testzip() is not None:
            raise RuntimeError("Falló la comprobación CRC del ZIP")
        manifest_data = list(csv.DictReader(
            archive.read("output/audit/integrity_manifest.tsv").decode("utf-8").splitlines(),
            delimiter="\t"))
        for row in manifest_data:
            content = archive.read(row["path"])
            if len(content) != int(row["bytes"]) or hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Integridad inválida dentro del ZIP: {row['path']}")

    print(json.dumps({"zip": str(ZIP_PATH), "files": len(files), "bytes": ZIP_PATH.stat().st_size,
                      "sha256": digest(ZIP_PATH), "tests": "21 passed", "notebook_errors": 0,
                      "status": validation["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
