
import csv
import hashlib
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collector import CORPUS, ROOT, plain_text, read_corpus

NORMALIZED = ROOT / 'data' / 'news_normalized_corpus.tsv'
VECTORS = ROOT / 'results' / 'vectors.joblib'


STOP_POS = {'ADP', 'CCONJ', 'SCONJ', 'PRON'}
ENGLISH_WORDS = set('the a an of to in on for with from by and or but that this is are was were has have it its as at be not will which their they he she we'.split())


@lru_cache(maxsize=1)
def load_nlp():
    try:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    except OSError as exc:
        raise RuntimeError('Instala el modelo: python -m spacy download en_core_web_sm') from exc
    if nlp.lang != 'en' or not {'tagger', 'lemmatizer'}.issubset(nlp.pipe_names):
        raise ValueError('Se requiere un modelo inglés con POS y lematización')
    return nlp


def clean_text(text):
    text = ''.join(' ' if unicodedata.category(ch).startswith('C') else ch for ch in str(text))
    return plain_text(text)


def normalize_text(text):
    
    doc = load_nlp()(clean_text(text))
    return ' '.join(token.lemma_.lower().strip() for token in doc
                    if not token.is_space and not token.is_punct and token.tag_ != 'POS'
                    and token.lemma_.lower() not in {"'s", '’s'}
                    and any(ch.isalnum() for ch in token.lemma_)
                    and not (token.is_stop and (token.pos_ in STOP_POS
                             or (token.tag_ == 'DT' and token.lower_ in {'a', 'an', 'the'}))))


def verify_english(row):
    
    host = urlsplit(row['url']).hostname
    expected = {'Wikinews': 'en.wikinews.org', 'Global Voices': 'globalvoices.org',
                'EFF': 'www.eff.org'}
    words = re.findall(r'[a-z]+', clean_text(row['title'] + ' ' + row['description/body']).lower())
    hits = sum(word in ENGLISH_WORDS for word in words)
    if host != expected.get(row['source']) or hits < 2 or hits / max(1, len(words)) < .04:
        raise ValueError(f"Idioma inglés no verificado: {row['id']}")


def corpus_digest(rows):
    return hashlib.sha256(repr(rows).encode('utf-8')).hexdigest()


def build_vectors(rows, path=VECTORS):
    
    if not rows:
        raise ValueError('El corpus está vacío')
    spaces = {}
    for field, column in [('title', 'normalized_title'), ('body', 'normalized_body')]:
        for n in (1, 2):
            for representation in ('frequency', 'binary', 'tfidf'):
                options = dict(ngram_range=(n, n), tokenizer=str.split,
                               preprocessor=None, token_pattern=None, lowercase=False)
                vectorizer = (TfidfVectorizer(**options) if representation == 'tfidf'
                              else CountVectorizer(binary=representation == 'binary', **options))
                matrix = vectorizer.fit_transform([row[column] for row in rows]).tocsr()
                spaces[f'{field}_{n}_{representation}'] = {'vectorizer': vectorizer, 'matrix': matrix}
    bundle = {'ids': [row['id'] for row in rows], 'documents': rows,
              'spaces': spaces, 'model': load_nlp().meta['version'],
              'spacy_version': spacy.__version__, 'raw_digest': corpus_digest([
                  {k: v for k, v in row.items() if k not in {'normalized_title', 'normalized_body', 'language'}}
                  for row in rows])}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    joblib.dump(bundle, temp)
    temp.replace(path)
    return bundle


def process_corpus(path=CORPUS, normalized_path=NORMALIZED, vectors_path=VECTORS):
    rows = read_corpus(path)
    if not rows:
        raise ValueError('El corpus está vacío')
    
    for row in rows:
        verify_english(row)
    normalized = [dict(row, language='en', normalized_title=normalize_text(row['title']),
                       normalized_body=normalize_text(row['description/body'])) for row in rows]
    bundle = build_vectors(normalized, vectors_path)
    normalized_path = Path(normalized_path)
    temp = normalized_path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(normalized[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(normalized)
    temp.replace(normalized_path)
    return {key: {'shape': list(value['matrix'].shape),
                  'vocabulary': len(value['vectorizer'].vocabulary_)}
            for key, value in bundle['spaces'].items()}


def search_news(query, field='body', ngram=1, representation='tfidf', top_k=10,
                path=VECTORS, bundle=None):
    if field not in {'title', 'body'} or ngram not in {1, 2} or representation not in {'frequency', 'binary', 'tfidf'} or top_k < 1:
        raise ValueError('Configuración de búsqueda inválida')
    bundle = bundle if bundle is not None else joblib.load(path)
    if bundle['model'] != load_nlp().meta['version'] or bundle['spacy_version'] != spacy.__version__:
        raise ValueError('El modelo cambió; ejecuta process nuevamente')
    space = bundle['spaces'][f'{field}_{ngram}_{representation}']
    vector = space['vectorizer'].transform([normalize_text(query)])
    scores = cosine_similarity(vector, space['matrix']).ravel()
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
    return [dict(bundle['documents'][i], score=float(scores[i])) for i in order]
