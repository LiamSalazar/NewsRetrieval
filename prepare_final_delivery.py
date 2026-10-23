"""Crea y valida la entrega académica limpia de NewsRetrieval."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
STAGE = (ROOT / "tmp" / "final_delivery_stage_v2").resolve()
SMOKE = (ROOT.parent / "NewsRetrieval_final_smoke_20261023").resolve()
ZIP_PATH = ROOT / "NewsRetrieval_ENTREGA_FINAL.zip"

ROOT_FILES = [
    ".gitignore", "README.md", "requirements.txt", "collector.py", "retrieval.py",
    "evaluation.py", "main.py", "test_project.py", "build_final_notebook.py",
    "NewsRetrieval_Pipeline.ipynb",
]
DATA_FILES = [
    "news_raw_corpus.tsv", "news_normalized_corpus.tsv", "queries.tsv", "qrels.tsv",
    "benchmark_status.tsv", "relevance_criteria.tsv", "pool_human_review.tsv",
    "qrels_validation_evidence.tsv", "reference_freeze_record.json",
    "top10_zero_similarity_adjudication.tsv", "academic_reference_core.tsv",
]
RESULT_FILES = [
    "vectors.joblib", "rankings.tsv", "rankings_all_configurations.tsv",
    "retrieval_judgments.tsv", "metrics_by_query_configuration.tsv", "metrics.tsv",
    "comparison.tsv", "report_data.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_recreate(path: Path, expected_name: str, parent: Path) -> None:
    if path.name != expected_name or path.parent.resolve() != parent.resolve():
        raise RuntimeError(f"Ruta temporal inesperada: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode:
        raise RuntimeError(f"Falló {' '.join(map(str, command))}:\n{output}")
    return output


def copy_source() -> None:
    for name in ROOT_FILES:
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = STAGE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in DATA_FILES:
        source = ROOT / "data" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = STAGE / "data" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in RESULT_FILES:
        source = ROOT / "results" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = STAGE / "results" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for source in sorted((ROOT / "results" / "figures").glob("*.svg")):
        destination = STAGE / "results" / "figures" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def execute_notebook() -> None:
    path = STAGE / "NewsRetrieval_Pipeline.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(STAGE)}}).execute()
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError(f"El notebook terminó con {len(errors)} errores")
    nbformat.write(notebook, path)


def validate_stage() -> dict:
    raw = read_tsv(STAGE / "data" / "news_raw_corpus.tsv")
    normalized = read_tsv(STAGE / "data" / "news_normalized_corpus.tsv")
    queries = read_tsv(STAGE / "data" / "queries.tsv")
    qrels = read_tsv(STAGE / "data" / "qrels.tsv")
    pool = read_tsv(STAGE / "data" / "pool_human_review.tsv")
    rankings = read_tsv(STAGE / "results" / "rankings_all_configurations.tsv")
    judgments = read_tsv(STAGE / "results" / "retrieval_judgments.tsv")
    metrics = read_tsv(STAGE / "results" / "metrics_by_query_configuration.tsv")
    comparison = read_tsv(STAGE / "results" / "comparison.tsv")
    bundle = joblib.load(STAGE / "results" / "vectors.joblib")
    notebook = json.loads((STAGE / "NewsRetrieval_Pipeline.ipynb").read_text(encoding="utf-8"))
    notebook_errors = [output for cell in notebook["cells"] if cell["cell_type"] == "code"
                       for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    checks = {
        "status": "frozen_delegated_analytical_validation",
        "documents": len(raw), "unique_document_ids": len({row["id"] for row in raw}),
        "sources": len({row["source"] for row in raw}),
        "normalized_documents": len(normalized),
        "aligned_corpus_ids": [row["id"] for row in raw] == [row["id"] for row in normalized],
        "vector_spaces": len(bundle["spaces"]),
        "all_vector_rows_2000": all(space["matrix"].shape[0] == 2000 for space in bundle["spaces"].values()),
        "queries": len(queries), "difficulty_distribution": dict(sorted(
            __import__("collections").Counter(row["difficulty_level"] for row in queries).items())),
        "qrels": len(qrels), "unique_qrels": len({(row["query_id"], row["document_id"]) for row in qrels}),
        "pool_pairs": len(pool), "query_configuration_runs": len(metrics),
        "top10_results": len(rankings), "retrieval_judgments": len(judgments),
        "unjudged_returns": sum(row["judgment_status"] == "unjudged" for row in judgments),
        "zero_similarity_returns": sum(row["zero_similarity"] == "1" for row in judgments),
        "comparison_rows": len(comparison), "notebook_cells": len(notebook["cells"]),
        "notebook_execution_errors": len(notebook_errors),
        "qrels_sha256": sha256(STAGE / "data" / "qrels.tsv"),
    }
    expected = {"documents": 2000, "unique_document_ids": 2000, "normalized_documents": 2000,
                "vector_spaces": 12, "queries": 15, "qrels": 303, "unique_qrels": 303,
                "query_configuration_runs": 180, "top10_results": 1800,
                "retrieval_judgments": 1800, "unjudged_returns": 0,
                "comparison_rows": 48, "notebook_execution_errors": 0}
    for key, value in expected.items():
        if checks[key] != value:
            raise RuntimeError(f"Control final fallido: {key}={checks[key]!r}; esperado={value!r}")
    if checks["sources"] < 3 or not checks["aligned_corpus_ids"] or not checks["all_vector_rows_2000"]:
        raise RuntimeError("Falló un control estructural del corpus o los vectores")
    return checks


def add_evidence(checks, command_outputs) -> None:
    evidence = STAGE / "output" / "final"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "test_results.txt").write_text(
        "\n\n".join(f"COMANDO: {label}\n{output}" for label, output in command_outputs), encoding="utf-8")
    packages = ["requests", "spacy", "scikit-learn", "joblib", "en-core-web-sm",
                "nbformat", "nbclient", "ipykernel"]
    versions = [{"component": "Python", "version": platform.python_version()},
                {"component": "platform", "version": platform.platform()}]
    for package in packages:
        versions.append({"component": package, "version": importlib.metadata.version(package)})
    write_tsv(evidence / "environment_versions.tsv", ["component", "version"], versions)
    (evidence / "validation_summary.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (evidence / "DELIVERY_CONTENTS.md").write_text(
        "# Entrega académica final\n\n"
        "Referencia congelada para evaluación analítica delegada; no equivale a aprobación humana institucional. "
        "Incluye 303 qrels, 180 evaluaciones, 1,800 devoluciones individualmente vinculadas con juicio, "
        "cinco métricas, tablas, SVG y notebook ejecutado.\n\n"
        "El archivo de entrada `NewsRetrieval_entradas_cierre_Sol_High_2026-10-23.zip` no estuvo disponible "
        "en el espacio de trabajo; por ello su SHA-256 externo no pudo comprobarse. La referencia local de 303 "
        "filas fue validada contra el corpus y promovida bajo autorización explícita.\n\n"
        "Se excluyeron Git, entornos, cachés, ZIP anteriores, resultados provisionales, PDF/Word provisionales, "
        "lotes intermedios y scripts históricos de adjudicación.\n",
        encoding="utf-8")


def build_zip() -> tuple[int, str]:
    evidence = STAGE / "output" / "final"
    files = sorted((path for path in STAGE.rglob("*") if path.is_file()
                    and path.name != "integrity_manifest.tsv"
                    and "__pycache__" not in path.parts and path.suffix != ".pyc"
                    and ".ipynb_checkpoints" not in path.parts),
                   key=lambda path: path.relative_to(STAGE).as_posix())
    manifest_rows = [{"path": path.relative_to(STAGE).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256(path)} for path in files]
    write_tsv(evidence / "integrity_manifest.tsv", ["path", "bytes", "sha256"], manifest_rows)
    files.append(evidence / "integrity_manifest.tsv")
    files.sort(key=lambda path: path.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        forbidden = [name for name in names if name.startswith((".git/", ".venv/", "tmp/", "__pycache__/"))
                     or name.endswith((".pyc", ".zip", ".docx", ".pdf")) or "/provisional/" in name]
        if forbidden:
            raise RuntimeError(f"Contenido prohibido en ZIP: {forbidden}")
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise RuntimeError("El ZIP contiene duplicados o errores CRC")
        manifest = list(csv.DictReader(
            archive.read("output/final/integrity_manifest.tsv").decode("utf-8").splitlines(), delimiter="\t"))
        for row in manifest:
            content = archive.read(row["path"])
            if len(content) != int(row["bytes"]) or hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Hash interno inválido: {row['path']}")
    return len(files), sha256(ZIP_PATH)


def smoke_test_zip() -> str:
    safe_recreate(SMOKE, "NewsRetrieval_final_smoke_20261023", ROOT.parent)
    try:
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(SMOKE)
        tests = run([sys.executable, "-m", "unittest", "-v", "test_project"], SMOKE)
        benchmark = run([sys.executable, "main.py", "--benchmark"], SMOKE)
        query = run([sys.executable, "main.py", "--query", "government surveillance and privacy",
                     "--field", "body", "--ngram", "1", "--representation", "tfidf"], SMOKE)
        if "Ran 25 tests" not in tests or "OK" not in tests:
            raise RuntimeError("La prueba limpia no confirmó 25 tests")
        if "Benchmark válido y congelado" not in benchmark or not query.strip():
            raise RuntimeError("La ejecución limpia de main.py no produjo la salida esperada")
        return "25 tests OK; benchmark validado; consulta libre funcional"
    finally:
        if SMOKE.exists() and SMOKE.name == "NewsRetrieval_final_smoke_20261023" and SMOKE.parent == ROOT.parent.resolve():
            shutil.rmtree(SMOKE)


def main() -> None:
    safe_recreate(STAGE, "final_delivery_stage_v2", (ROOT / "tmp").resolve())
    copy_source()
    run([sys.executable, "build_final_notebook.py"], STAGE)
    benchmark_output = run([sys.executable, "main.py", "--benchmark", "--frozen"], STAGE)
    execute_notebook()
    test_output = run([sys.executable, "-m", "unittest", "-v", "test_project"], STAGE)
    if "Ran 25 tests" not in test_output or "OK" not in test_output:
        raise RuntimeError("No se confirmaron las 25 pruebas en la carpeta limpia")
    checks = validate_stage()
    add_evidence(checks, [("python main.py --benchmark --frozen", benchmark_output),
                          ("python -m unittest -v test_project", test_output)])
    count, zip_hash = build_zip()
    smoke = smoke_test_zip()
    print(json.dumps({"zip": str(ZIP_PATH), "files": count, "bytes": ZIP_PATH.stat().st_size,
                      "sha256": zip_hash, "clean_smoke_test": smoke, **checks},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
