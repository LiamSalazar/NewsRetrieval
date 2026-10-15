"""Interfaz de línea de comandos para el proyecto escolar."""
import argparse
import json
import os

from collector import collect_news


def main():
    parser = argparse.ArgumentParser(description="News Retrieval — proyecto académico")
    parser.add_argument("command", choices=["init", "update", "process", "search", "evaluate"])
    parser.add_argument("--target", type=int, default=2000, help="Meta de noticias para init")
    parser.add_argument("--per-category", type=int, default=10, help="Noticias nuevas por tema")
    parser.add_argument("--days", type=int, default=8, help="Ventana reciente en días para update")
    parser.add_argument("--max-batches", type=int, default=80, help="Máximo de lotes históricos por tema")
    parser.add_argument("--dry-run", action="store_true", help="No modifica los TSV ni los registros")
    args = parser.parse_args()
    if min(args.target, args.per_category, args.days, args.max_batches) < 1:
        parser.error("target, per-category, days y max-batches deben ser positivos")
    if args.command in {"init", "update"}:
        report = collect_news(mode="init" if args.command == "init" else "weekly",
                              target=args.target, per_category=args.per_category,
                              days=args.days, max_batches=args.max_batches,
                              dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["complete"]:
            prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") else "AVISO: "
            print(prefix + "No se alcanzaron todas las cuotas. Consultar errores y faltantes.")
    else:
        print(f"{args.command}: reservado para la siguiente fase del proyecto.")


if __name__ == "__main__":
    main()
