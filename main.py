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
    parser.add_argument('--query', help='Consulta inglesa; si falta, se solicita por terminal')
    parser.add_argument('--field', choices=['title', 'body'], default='body')
    parser.add_argument('--ngram', type=int, choices=[1, 2], default=1)
    parser.add_argument('--representation', choices=['frequency', 'binary', 'tfidf'], default='tfidf')
    args = parser.parse_args()
    if min(args.target, args.per_category, args.days, args.max_batches) < 1:
        parser.error("target, per-category, days y max-batches deben ser positivos")
    if args.per_category > 10:
        parser.error('per-category debe estar entre 1 y 10')
    if args.dry_run and args.command not in {'init', 'update'}:
        parser.error('--dry-run está disponible para init/update')
    if args.command in {"init", "update"}:
        report = collect_news(mode="init" if args.command == "init" else "weekly",
                              target=args.target, per_category=args.per_category,
                              days=args.days, max_batches=args.max_batches,
                              dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["complete"]:
            prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") else "AVISO: "
            print(prefix + "No se alcanzaron todas las cuotas. Consultar errores y faltantes.")
    elif args.command == 'process':
        from retrieval import process_corpus
        print(json.dumps(process_corpus(), ensure_ascii=False, indent=2))
    elif args.command == 'search':
        from retrieval import CORPUS, VECTORS, corpus_digest, read_corpus, search_news
        import joblib
        bundle = joblib.load(VECTORS)
        if corpus_digest(read_corpus(CORPUS)) != bundle['raw_digest']:
            parser.error('El corpus cambió: ejecuta python main.py process')
        rows = search_news(args.query if args.query is not None else input('Consulta en inglés: '),
                           args.field, args.ngram, args.representation, bundle=bundle)
        if not any(row['score'] > 0 for row in rows):
            print('Sin coincidencias léxicas: todas las puntuaciones son cero.')
        for row in rows:
            print(f"{row['score']:.6f}\t{row['id']}\t{row['title']}\t{row['url']}")
    else:
        print(f"{args.command}: reservado para la siguiente fase del proyecto.")


if __name__ == "__main__":
    main()
