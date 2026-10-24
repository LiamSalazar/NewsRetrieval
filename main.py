"""Punto único de ejecución para adquirir, procesar y consultar el corpus."""
import argparse
import csv
from pathlib import Path
import joblib
import spacy

from collector import CORPUS, collect_news, read_corpus, validate_corpus
from evaluation import evaluate, load_benchmark, validate_benchmark
from retrieval import NORMALIZED, VECTORS, corpus_digest, load_nlp, process_corpus, search_news


def benchmark_status():
    path = Path('data/benchmark_status.tsv')
    if not path.exists():
        return ''
    with path.open(encoding='utf-8-sig', newline='') as stream:
        row = next(csv.DictReader(stream, delimiter='\t'), {})
    return row.get('status', '')


def corpus_is_current(bundle, rows):
    return (bundle.get('ids') == [row['id'] for row in rows]
            and bundle.get('raw_digest') == corpus_digest(rows)
            and bundle.get('model') == load_nlp().meta['version']
            and bundle.get('spacy_version') == spacy.__version__)


def ensure_data():
    if not CORPUS.exists():
        report = collect_news()
        if not report['complete']:
            raise RuntimeError('La adquisición inicial no alcanzó 2,000 noticias; el archivo no fue creado.')
    rows = read_corpus(CORPUS)
    validate_corpus(rows)
    if len(rows) != 2000:
        raise RuntimeError(f'El corpus contiene {len(rows)} noticias; se requieren 2,000. No se modificó.')

    rebuild = True
    if NORMALIZED.exists() and VECTORS.exists():
        try:
            bundle = joblib.load(VECTORS)
            normalized = read_corpus(NORMALIZED)
            rebuild = (not corpus_is_current(bundle, rows)
                       or [row['id'] for row in normalized] != [row['id'] for row in rows])
        except (OSError, ValueError, KeyError, EOFError):
            rebuild = True
    if rebuild:
        print('Procesando el corpus y generando las 12 matrices VSM...')
        process_corpus()
    else:
        print('Corpus y matrices actuales reutilizados.')


def collect_corpus():
    """Adquiere el corpus solo si no existe; un corpus fijo nunca se sobrescribe."""
    report = collect_news()
    action = 'validado sin descarga' if report['added'] == 0 else 'adquirido'
    print(f"Corpus {action}: {report['total_after']} noticias.")


def prepare_corpus():
    """Normaliza el corpus fijo y genera sus doce espacios vectoriales."""
    rows = read_corpus(CORPUS)
    validate_corpus(rows)
    if len(rows) != 2000:
        raise RuntimeError(f'El corpus contiene {len(rows)} noticias; se requieren 2,000. No se modificó.')
    spaces = process_corpus()
    print(f'Preparación completada: {len(rows)} noticias y {len(spaces)} espacios vectoriales.')


def ask_choice(prompt, options, default):
    try:
        text = input(f'{prompt} [{"/".join(options)}] (predeterminado: {default}): ').strip().lower()
    except EOFError:
        return default
    return default if not text else text if text in options else None


def search_loop(first_query=None, first_field='body', first_ngram=1, first_rep='tfidf'):
    bundle = joblib.load(VECTORS)
    direct_query = first_query is not None
    while True:
        was_direct = direct_query
        try:
            query = first_query if direct_query else input('\nConsulta (o escribe salir): ').strip()
        except EOFError:
            return
        first_query = None
        if query.lower() in {'salir', 'exit', 'quit', ''}:
            return
        if direct_query:
            field = first_field or ask_choice('Contenido', ['title', 'body'], 'body')
            ngram = first_ngram or ask_choice('Características', ['1', '2'], '1')
            representation = first_rep or ask_choice(
                'Representación', ['frequency', 'binary', 'tfidf'], 'tfidf')
        else:
            field = ask_choice('Contenido', ['title', 'body'], 'body')
            ngram = ask_choice('Características', ['1', '2'], '1')
            representation = ask_choice('Representación', ['frequency', 'binary', 'tfidf'], 'tfidf')
        direct_query = False
        if field is None or ngram is None or representation is None:
            print('Selección inválida.')
            continue
        rows = search_news(query, field, int(ngram), representation, bundle=bundle)
        for row in rows:
            print(f"{row['score']:.6f}\t{row['id']}\t{row['title']}\t{row['url']}")
        if was_direct:
            return


def comparison_menu():
    path = 'results/comparison.tsv'
    try:
        with open(path, encoding='utf-8') as stream:
            print(stream.read())
    except OSError:
        print('Aún no existe results/comparison.tsv; ejecute la evaluación aprobada.')


def print_metrics_summary(path='results/metrics.tsv'):
    with open(path, encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    mean = next((row for row in rows if row['query_id'] == 'MEAN'), None)
    if mean:
        print('Media: ' + ', '.join(f"{key}={float(mean[key]):.4f}"
                                    for key in ('precision@10', 'recall@10', 'f1@10', 'mrr', 'ndcg@10')))


def interactive_menu():
    while True:
        print('\n1. Consulta libre\n2. Ejecutar benchmark aprobado\n3. Ver resumen comparativo\n4. Salir')
        try:
            option = input('Opción: ').strip()
        except EOFError:
            return
        if option == '1':
            search_loop()
        elif option == '2':
            try:
                confirmation = input('Escriba EVALUAR para usar la referencia delegada congelada: ').strip()
            except EOFError:
                return
            if confirmation == 'EVALUAR':
                queries, qrels = load_benchmark()
                validate_benchmark(queries, qrels, read_corpus(CORPUS))
                report = evaluate(queries, qrels, bundle=joblib.load(VECTORS), frozen=True)
                print(f"Evaluación completada: {report['queries']} consultas, {report['configurations']} configuraciones.")
                print_metrics_summary()
            else:
                print('Evaluación no ejecutada.')
        elif option == '3':
            comparison_menu()
        elif option == '4':
            return
        else:
            print('Opción inválida.')


def main():
    parser = argparse.ArgumentParser(description='News Retrieval')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--collect', action='store_true',
                         help='Adquiere el corpus inicial o valida el corpus fijo existente')
    actions.add_argument('--prepare', action='store_true',
                         help='Normaliza el corpus fijo y genera las 12 representaciones VSM')
    actions.add_argument('--query', help='Ejecuta una consulta directamente')
    parser.add_argument('--field', choices=['title', 'body'], default=None)
    parser.add_argument('--ngram', type=int, choices=[1, 2], default=None)
    parser.add_argument('--representation', choices=['frequency', 'binary', 'tfidf'], default=None)
    actions.add_argument('--benchmark', action='store_true',
                         help='Valida el benchmark; con --frozen ejecuta las 12 configuraciones')
    parser.add_argument('--frozen', action='store_true',
                        help='Usa la referencia congelada por evaluación analítica delegada')
    args = parser.parse_args()
    try:
        if args.collect:
            collect_corpus()
            return 0
        if args.prepare:
            prepare_corpus()
            return 0
        ensure_data()
        if args.benchmark:
            queries, qrels = load_benchmark()
            validate_benchmark(queries, qrels, read_corpus(CORPUS))
            if benchmark_status() != 'frozen_delegated_analytical_validation':
                raise RuntimeError('benchmark_status.tsv no declara la referencia delegada congelada')
            if not args.frozen:
                print('Benchmark válido y congelado para evaluación analítica delegada; use --frozen para recalcular resultados.')
                return 0
            report = evaluate(queries, qrels, bundle=joblib.load(VECTORS),
                              output_dir='results', frozen=True)
            print(f"Evaluación completada: {report['queries']} consultas, {report['configurations']} configuraciones.")
            print_metrics_summary('results/metrics.tsv')
            return 0
        if args.query or any(value is not None for value in (args.field, args.ngram, args.representation)):
            search_loop(args.query, args.field, args.ngram, args.representation)
        else:
            interactive_menu()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Problema: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
