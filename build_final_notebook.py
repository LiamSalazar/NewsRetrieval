"""Construye el notebook académico final de NewsRetrieval."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "NewsRetrieval_Pipeline.ipynb"


def md(value):
    return nbf.v4.new_markdown_cell(value.strip() + "\n")


def code(value):
    return nbf.v4.new_code_cell(value.strip() + "\n")


cells = [
md(r"""
# NewsRetrieval — Recuperación de información en noticias

Práctica académica completa sobre un corpus fijo de 2,000 noticias en inglés. Se comparan doce representaciones VSM mediante quince consultas y cinco métricas. La referencia de 303 qrels está congelada para **evaluación analítica delegada**; este alcance no equivale a aprobación humana institucional exhaustiva.
"""),
md(r"""
## 1. Introducción

El sistema combina dos campos (título/cuerpo), unigramas o bigramas y tres ponderaciones (frecuencia, binaria y TF-IDF). Las consultas se normalizan como los documentos y los resultados se ordenan por similitud coseno. El notebook reutiliza el corpus y las matrices persistidas: no descarga noticias ni reconstruye vectores.
"""),
code(r"""
from collections import Counter
from datetime import datetime
from hashlib import sha256
from html import escape
from pathlib import Path
import csv, json, math, platform

import joblib, spacy, sklearn
from IPython.display import HTML, Markdown, SVG, display
from sklearn.metrics.pairwise import cosine_similarity

from collector import FEEDS, read_corpus, validate_corpus
from evaluation import (CONFIGURATIONS, MAIN_CONFIGURATION, METRICS, calculate_metrics,
                        evaluate, load_benchmark, validate_benchmark)
from retrieval import STOP_POS, clean_text, load_nlp, normalize_text, search_news

def find_root(start=Path.cwd()):
    for candidate in (start, *start.parents):
        if (candidate/'data/news_raw_corpus.tsv').exists() and (candidate/'retrieval.py').exists():
            return candidate
    raise FileNotFoundError('No se encontró la raíz del proyecto')

ROOT=find_root(); DATA=ROOT/'data'; RESULTS=ROOT/'results'; VECTORS=RESULTS/'vectors.joblib'

def read_tsv(path, comments=False):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        lines=(line for line in stream if not(comments and line.startswith('#')))
        return list(csv.DictReader(lines, delimiter='\t'))

def hash_file(path):
    value=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()

def table(rows, columns=None, limit=20, title=''):
    rows=list(rows); columns=columns or (list(rows[0]) if rows else []); shown=rows[:limit]
    def fmt(value): return f'{value:.6f}' if isinstance(value,float) else str(value)
    head=''.join(f'<th>{escape(str(column))}</th>' for column in columns)
    body=''.join('<tr>'+''.join(f'<td>{escape(fmt(row.get(column,"")))}</td>' for column in columns)+'</tr>' for row in shown)
    caption=f'<caption><strong>{escape(title)}</strong></caption>' if title else ''
    note=f'<p><em>Se muestran {len(shown)} de {len(rows)} filas.</em></p>' if len(shown)<len(rows) else ''
    display(HTML(f'<table>{caption}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}'))

versions=[{'componente':'Python','versión':platform.python_version()},
          {'componente':'spaCy','versión':spacy.__version__},
          {'componente':'modelo en_core_web_sm','versión':load_nlp().meta['version']},
          {'componente':'scikit-learn','versión':sklearn.__version__},
          {'componente':'joblib','versión':joblib.__version__}]
display(Markdown(f'**Rutas del proyecto:** relativas a `./` — **ejecución:** {datetime.now().isoformat(timespec="seconds")}'))
table(versions,title='Entorno utilizado')
"""),
md(r"""
## 2. Extracción de noticias

La adquisición original utilizó la API de MediaWiki para Wikinews y feeds RSS/Atom para Global Voices y EFF. Se aplicaron validación, deduplicación, atribución y escritura atómica. Aquí se carga el corpus persistido y se muestran registros auténticos.
"""),
code(r"""
raw=read_corpus(DATA/'news_raw_corpus.tsv'); normalized=read_tsv(DATA/'news_normalized_corpus.tsv')
validate_corpus(raw)
assert len(raw)==len(normalized)==2000
assert [row['id'] for row in raw]==[row['id'] for row in normalized]
sources=Counter(row['source'] for row in raw); categories=Counter(row['category'] for row in raw)
dates=[row['date'] for row in raw]
summary=[{'indicador':'Noticias','valor':len(raw)}, {'indicador':'IDs únicos','valor':len({r['id'] for r in raw})},
         {'indicador':'Fuentes','valor':dict(sources)}, {'indicador':'Categorías','valor':dict(categories)},
         {'indicador':'Fechas','valor':f'{min(dates)} — {max(dates)}'}]
table(summary,title='Corpus fijo')
examples=[]
for source in sorted(sources):
    row=next(item for item in raw if item['source']==source)
    examples.append({'id':row['id'],'fuente':source,'categoría':row['category'],'fecha':row['date'],
                     'título':row['title'],'cuerpo':row['description/body'][:300]+'…','autor':row['author'],
                     'licencia':row['license'],'URL':row['url']})
table(examples,limit=3,title='Ejemplos originales y atribución')
table([{'fuente':'Wikinews','mecanismo':'API MediaWiki; texto introductorio de páginas inglesas'},
       {'fuente':'Global Voices','mecanismo':'feeds RSS/Atom temáticos'},
       {'fuente':'EFF','mecanismo':'feed Deeplinks; exclusión de republicaciones de terceros'}],
      title='Procedencia y mecanismo de extracción')
"""),
md(r"""
## 3. Normalización

El proceso real incluye limpieza, tokenización, POS tagging, eliminación selectiva de palabras funcionales, lematización y unión con espacios simples. Título y cuerpo se normalizan por separado.
"""),
code(r"""
sample=next(row for row in raw if len(row['description/body'].split())>=100)
persisted=next(row for row in normalized if row['id']==sample['id'])
doc=load_nlp()(clean_text(sample['title']+' '+sample['description/body']))
def decision(token):
    lemma=token.lemma_.lower().strip()
    if token.is_space:return 'eliminar: espacio'
    if token.is_punct:return 'eliminar: puntuación'
    if token.tag_=='POS' or lemma in {"'s",'’s'}:return 'eliminar: posesivo'
    if not any(ch.isalnum() for ch in lemma):return 'eliminar: no alfanumérico'
    if token.is_stop and token.pos_ in STOP_POS:return 'eliminar: palabra funcional/POS'
    if token.is_stop and token.tag_=='DT' and token.lower_ in {'a','an','the'}:return 'eliminar: artículo'
    return 'conservar'
token_trace=[{'token':t.text,'POS':t.pos_,'tag':t.tag_,'lema':t.lemma_.lower().strip(),'decisión':decision(t)} for t in doc[:50]]
display(Markdown(f"**Noticia:** `{sample['id']}` — {sample['title']}"))
table(token_trace,limit=50,title='Tokenización, POS, lema y filtrado')
assert normalize_text(sample['title'])==persisted['normalized_title']
assert normalize_text(sample['description/body'])==persisted['normalized_body']
table([{'campo':'título','original':sample['title'],'normalizado':persisted['normalized_title']},
       {'campo':'cuerpo','original':sample['description/body'][:450]+'…','normalizado':persisted['normalized_body'][:450]+'…'}],
      title='Texto original y normalizado')
"""),
md(r"""
## 4. Representaciones vectoriales

Se cargan las doce matrices CSR persistidas. Cada configuración posee su propio vectorizador y vocabulario. Los valores siguientes pertenecen a la noticia mostrada en la sección anterior.
"""),
code(r"""
bundle=joblib.load(VECTORS); assert len(bundle['spaces'])==12 and len(bundle['ids'])==2000
sample_index=bundle['ids'].index(sample['id']); spaces=[]
for key,space in bundle['spaces'].items():
    vector=space['matrix'].getrow(sample_index); names=space['vectorizer'].get_feature_names_out()
    pairs=sorted(((names[i],float(v)) for i,v in zip(vector.indices,vector.data)),key=lambda x:(-x[1],x[0]))[:5]
    field,ngram,weight=key.split('_',2)
    spaces.append({'configuración':key,'campo':field,'n-grama':ngram,'ponderación':weight,
                   'dimensiones':str(space['matrix'].shape),'vocabulario':len(names),'no nulos':vector.nnz,
                   'características y valores':'; '.join(f'{term}={value:.4f}' for term,value in pairs)})
table(spaces,limit=12,title='Las doce matrices VSM')

frequency=bundle['spaces']['body_1_frequency']; tfidf=bundle['spaces']['body_1_tfidf']
terms=frequency['vectorizer'].get_feature_names_out(); fv=frequency['matrix'].getrow(sample_index)
tv=tfidf['matrix'].getrow(sample_index); fdict=dict(zip(fv.indices,fv.data)); tdict=dict(zip(tv.indices,tv.data))
illustration=[]
for index,count in sorted(fdict.items(),key=lambda x:(-x[1],terms[x[0]]))[:12]:
    illustration.append({'término':terms[index],'frecuencia':int(count),'TF-IDF':float(tdict.get(index,0.0))})
table(illustration,limit=12,title='Frecuencia y TF-IDF de términos reales del cuerpo')
"""),
md(r"""
## 5. Recuperación paso a paso

Se demuestra una consulta real con la configuración principal: normalización, vectorización, coseno, ordenamiento y Top-10.
"""),
code(r"""
queries=read_tsv(DATA/'queries.tsv'); demo_query=queries[0]; space=bundle['spaces'][MAIN_CONFIGURATION]
normalized_query=normalize_text(demo_query['query']); query_vector=space['vectorizer'].transform([normalized_query])
scores=cosine_similarity(query_vector,space['matrix']).ravel(); order=sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:10]
retrieved=search_news(demo_query['query'],'body',1,'tfidf',bundle=bundle)
assert [row['id'] for row in retrieved]==[bundle['ids'][i] for i in order]
display(Markdown(f"**Consulta original:** {demo_query['query']}  \n**Consulta normalizada:** `{normalized_query}`  \n"
                 f"**Vector:** {query_vector.shape[1]} dimensiones, {query_vector.nnz} valores no nulos.  \n"
                 "**Similitud:** producto coseno entre el vector de consulta y cada fila; orden descendente con desempate estable."))
table([{'rank':rank,'id':row['id'],'título':row['title'],'score':row['score'],'URL':row['url']} for rank,row in enumerate(retrieved,1)],
      limit=10,title='Diez documentos recuperados')
"""),
md(r"""
## 6. Relevancia de los resultados

`show_top10(query_id, configuration)` permite inspeccionar cualquiera de las 180 combinaciones. Distingue grados 0, 1 y 2, muestra el estado de juicio y resume aciertos. La similitud solo determina el orden; nunca la etiqueta.
"""),
code(r"""
judgment_rows=read_tsv(RESULTS/'retrieval_judgments.tsv')
def show_top10(query_id, configuration):
    rows=[row for row in judgment_rows if row['query_id']==query_id and row['configuration']==configuration]
    rows.sort(key=lambda row:int(row['rank']))
    if len(rows)!=10: raise ValueError('La selección no contiene exactamente diez resultados')
    shown=[{'Rank':int(row['rank']),'Documento':f"{row['document_id']} — {row['title']}",
            'Similitud':float(row['score']),'Relevancia':int(row['relevance_grade']),
            '¿Correcto?':'sí' if row['is_relevant']=='1' else 'no','Estado':row['judgment_status']} for row in rows]
    relevant=sum(row['is_relevant']=='1' for row in rows); irrelevant=10-relevant
    table(shown,limit=10,title=f'{query_id} / {configuration}')
    display(Markdown(f'**Resumen:** 10 noticias devueltas; {relevant} relevantes; {irrelevant} irrelevantes; '
                     f'Precision@10 = {relevant}/10 = {relevant/10:.2f}.'))
    return rows

demo_rows=show_top10('Q01',MAIN_CONFIGURATION)
"""),
md(r"""
## 7. Benchmark

Las quince consultas corresponden a cinco necesidades y tres dificultades. Los 303 qrels contienen únicamente positivos 1 y 2. Los negativos confirmados se conservan en la tabla consolidada de adjudicación.

La referencia supera el máximo literal de 15 en comercio, fútbol y vigilancia. Se conserva completa para no ocultar positivos ni reducir artificialmente el denominador de Recall. El núcleo académico es solo expositivo.
"""),
code(r"""
benchmark_queries,qrels=load_benchmark(DATA/'queries.tsv',DATA/'qrels.tsv')
validation=validate_benchmark(benchmark_queries,qrels,raw)
status=read_tsv(DATA/'benchmark_status.tsv'); freeze=json.loads((DATA/'reference_freeze_record.json').read_text(encoding='utf-8'))
query_table=[{**row,'positivos':len(qrels[row['query_id']]),
              'grado 1':sum(v==1 for v in qrels[row['query_id']].values()),
              'grado 2':sum(v==2 for v in qrels[row['query_id']].values())} for row in benchmark_queries]
table(status,title='Estado y alcance de validación')
table(query_table,limit=15,title='Consultas, dificultades y referencia completa')
table([{'control':'qrels congelados','valor':sum(len(v) for v in qrels.values())},
       {'control':'SHA-256 qrels','valor':freeze['frozen_qrels_sha256']},
       {'control':'consultas por dificultad','valor':validation['difficulty_levels']},
       {'control':'máximo académico superado','valor':validation['academic_range_exceeded']}],title='Controles del benchmark')
"""),
md(r"""
## 8. Cálculo de las cinco métricas

Para la combinación demostrativa se reconstruyen las métricas directamente desde los diez grados. Precision usa diez como denominador; Recall usa todos los positivos de la referencia completa; F1 es la media armónica; MRR usa el primer positivo; nDCG emplea ganancia `2^rel - 1` y descuento `log2(rank+1)`.
"""),
code(r"""
demo_grades=[int(row['relevance_grade']) for row in demo_rows]
hits=sum(grade>0 for grade in demo_grades); total=len(qrels['Q01'])
precision=hits/10; recall=hits/total; f1=2*precision*recall/(precision+recall) if precision+recall else 0
first=next((rank for rank,grade in enumerate(demo_grades,1) if grade>0),0); mrr=1/first if first else 0
dcg_parts=[(2**grade-1)/math.log2(rank+1) for rank,grade in enumerate(demo_grades,1)]
ideal=sorted(qrels['Q01'].values(),reverse=True)[:10]
idcg_parts=[(2**grade-1)/math.log2(rank+1) for rank,grade in enumerate(ideal,1)]
ndcg=sum(dcg_parts)/sum(idcg_parts)
library=calculate_metrics([{'id':row['document_id']} for row in demo_rows],qrels['Q01'])
manual={'precision@10':precision,'recall@10':recall,'f1@10':f1,'mrr':mrr,'ndcg@10':ndcg}
assert all(abs(manual[key]-library[key])<1e-12 for key in METRICS)
table([{'métrica':'Precision@10','conteo/fórmula':f'{hits}/10','resultado':precision},
       {'métrica':'Recall@10','conteo/fórmula':f'{hits}/{total}','resultado':recall},
       {'métrica':'F1@10','conteo/fórmula':'2PR/(P+R)','resultado':f1},
       {'métrica':'MRR','conteo/fórmula':f'1/{first}' if first else '0','resultado':mrr},
       {'métrica':'nDCG@10','conteo/fórmula':f'{sum(dcg_parts):.6f}/{sum(idcg_parts):.6f}','resultado':ndcg}],
      title='Derivación desde los diez juicios reales')
table([{'rank':rank,'grado':grade,'ganancia':2**grade-1,'aporte_DCG':dcg_parts[rank-1]} for rank,grade in enumerate(demo_grades,1)],
      limit=10,title='Ganancia graduada y descuento por posición')
"""),
md(r"""
## 9. Evaluación de las 180 combinaciones

La celda ejecuta nuevamente la evaluación congelada, verifica que el hash de qrels no cambie y reconstruye todos los resultados finales. Cada configuración aporta 150 devoluciones.
"""),
code(r"""
qrels_hash_before=hash_file(DATA/'qrels.tsv')
run=evaluate(benchmark_queries,qrels,bundle=bundle,output_dir=RESULTS,frozen=True)
assert hash_file(DATA/'qrels.tsv')==qrels_hash_before
assert run['evaluations']==180 and run['rankings_all']==1800 and run['unjudged']==0
metrics_rows=read_tsv(RESULTS/'metrics_by_query_configuration.tsv'); comparison=read_tsv(RESULTS/'comparison.tsv')
means=[row for row in comparison if row['difficulty_level']=='all']
table(means,limit=12,title='Resultados finales por configuración (15 consultas)')
display(Markdown(f"**Ejecución:** {run['evaluations']} evaluaciones, {run['rankings_all']} devoluciones, "
                 f"{run['zero_similarity_results']} con score cero y {run['unjudged']} sin juicio."))
"""),
md(r"""
## 10. Gráficas comparativas

Las gráficas SVG proceden de la misma ejecución: desempeño de las doce configuraciones, dificultad, precisión/recall, positivos recuperados y resultados con coincidencia léxica nula.
"""),
code(r"""
for filename in ['01_ndcg_by_configuration.svg','02_relevant_returns_by_configuration.svg',
                 '03_zero_similarity_by_configuration.svg','04_ndcg_by_difficulty.svg',
                 '05_precision_recall_by_configuration.svg']:
    display(Markdown(f'**{filename[:-4].replace("_"," ")}**'))
    display(SVG(filename=str(RESULTS/'figures'/filename)))
"""),
md(r"""
## 11. Análisis de errores

La selección primaria utiliza `body_1_tfidf`. Se muestran cinco consultas con mayor y cinco con menor nDCG@10, evidencia observable y comparación con las otras configuraciones. Los indicadores léxicos permiten formular hipótesis, pero no prueban causalidad por sí solos.
"""),
code(r"""
for row in metrics_rows:
    for key in [*METRICS,'score']:
        if key in row and row[key]!='': row[key]=float(row[key])
primary=sorted((row for row in metrics_rows if row['configuration']==MAIN_CONFIGURATION),
               key=lambda row:(row['ndcg@10'],row['mrr'],row['query_id']))
selected=primary[:5]+list(reversed(primary[-5:])); vocabulary=set(bundle['spaces'][MAIN_CONFIGURATION]['vectorizer'].get_feature_names_out())
diagnosis=[]
for metric in selected:
    query=next(row for row in queries if row['query_id']==metric['query_id']); tokens=normalize_text(query['query']).split()
    oov=[token for token in tokens if token not in vocabulary]
    rows=[row for row in judgment_rows if row['query_id']==metric['query_id'] and row['configuration']==MAIN_CONFIGURATION]
    zero=sum(row['zero_similarity']=='1' for row in rows)
    observation=(f"{metric['relevant_retrieved']} positivos en Top-10; cobertura léxica {len(tokens)-len(oov)}/{len(tokens)}; "
                 f"{zero} scores cero"+(f"; OOV: {', '.join(oov)}" if oov else ''))
    hypothesis='Revisar sinónimos, paráfrasis y ambigüedad' if metric['precision@10']==0 or oov else 'La evidencia no permite atribuir una causa única'
    diagnosis.append({'grupo':'peor' if metric in primary[:5] else 'mejor','consulta':metric['query_id'],
                      'nDCG@10':metric['ndcg@10'],'P@10':metric['precision@10'],'R@10':metric['recall@10'],
                      'observación demostrada':observation,'posible explicación':hypothesis,
                      'primer título':rows[0]['title']})
table(diagnosis,limit=10,title='Cinco mejores y cinco peores en la configuración principal')

cross=[]
for metric in selected:
    rows=[row for row in metrics_rows if row['query_id']==metric['query_id']]
    best=max(rows,key=lambda row:(row['ndcg@10'],row['mrr'],row['configuration']))
    worst=min(rows,key=lambda row:(row['ndcg@10'],row['mrr'],row['configuration']))
    cross.append({'consulta':metric['query_id'],'principal nDCG':metric['ndcg@10'],
                  'mejor configuración':best['configuration'],'mejor nDCG':best['ndcg@10'],
                  'peor configuración':worst['configuration'],'peor nDCG':worst['ndcg@10']})
table(cross,limit=10,title='Comparación con las restantes configuraciones')
"""),
md(r"""
## 12. Conclusiones y reproducibilidad

Las conclusiones se derivan de las métricas observadas, sin imponer a priori una ponderación ganadora. El sistema conserva corpus, qrels y matrices mediante hashes; las pruebas comprueban fórmulas e invariantes.
"""),
code(r"""
numeric=[{**row,**{key:float(row[key]) for key in METRICS}} for row in means]
best=[]
for metric in METRICS:
    row=max(numeric,key=lambda item:(item[metric],item['configuration']))
    best.append({'métrica':metric,'mejor configuración':row['configuration'],'valor':row[metric]})
table(best,title='Mejor configuración observada por métrica')
integrity=[{'archivo':str(path.relative_to(ROOT)).replace('\\','/'),'bytes':path.stat().st_size,'SHA-256':hash_file(path)} for path in
           [DATA/'news_raw_corpus.tsv',DATA/'news_normalized_corpus.tsv',VECTORS,DATA/'queries.tsv',DATA/'qrels.tsv',
            RESULTS/'rankings_all_configurations.tsv',RESULTS/'retrieval_judgments.tsv',RESULTS/'metrics_by_query_configuration.tsv']]
table(integrity,limit=20,title='Integridad de artefactos esenciales')
display(Markdown('**Ejecución reproducible:** `python -m unittest -v test_project` y `python main.py --benchmark --frozen`.  \n'
                 '**Limitaciones:** validación analítica delegada; pool no exhaustivo; sensibilidad léxica a sinónimos, paráfrasis, '
                 'ambigüedad, nombres propios y extractos; empates con similitud cero conservados y juzgados.'))
"""),
]

notebook=nbf.v4.new_notebook(cells=cells)
notebook.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info":{"name":"python","version":"3.11"}}
nbf.write(notebook,TARGET)
print(f'Notebook generado: {TARGET} ({len(cells)} celdas)')
