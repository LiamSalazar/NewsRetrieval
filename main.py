"""Punto único de ejecución para adquirir, procesar y consultar el corpus."""
import argparse
import joblib
import spacy

from collector import CORPUS, collect_news, read_corpus, validate_corpus
from evaluation import evaluate, load_benchmark
from retrieval import NORMALIZED, VECTORS, corpus_digest, load_nlp, process_corpus, search_news


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


def main():
    parser = argparse.ArgumentParser(description='News Retrieval')
    parser.add_argument('--query', help='Ejecuta una consulta directamente')
    parser.add_argument('--field', choices=['title', 'body'], default=None)
    parser.add_argument('--ngram', type=int, choices=[1, 2], default=None)
    parser.add_argument('--representation', choices=['frequency', 'binary', 'tfidf'], default=None)
    parser.add_argument('--benchmark', action='store_true',
                        help='Valida el benchmark; con --frozen ejecuta las 12 configuraciones')
    parser.add_argument('--frozen', action='store_true',
                        help='Confirma que los qrels fueron revisados y aprobados')
    args = parser.parse_args()
    try:
        ensure_data()
        if args.benchmark:
            queries, qrels = load_benchmark()
            counts = {query['query_id']: len(qrels.get(query['query_id'], {})) for query in queries}
            if len(queries) != 15 or set(counts.values()) - set(range(10, 16)):
                raise ValueError('El benchmark debe tener 15 consultas y 10-15 relevantes por consulta')
            if not args.frozen:
                print('Benchmark válido pero provisional: revise y apruebe data/qrels.tsv; no se calcularon métricas.')
                return 0
            report = evaluate(queries, qrels, bundle=joblib.load(VECTORS), frozen=True)
            print(f"Evaluación completada: {report['queries']} consultas, {report['configurations']} configuraciones.")
            return 0
        search_loop(args.query, args.field, args.ngram, args.representation)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Problema: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
