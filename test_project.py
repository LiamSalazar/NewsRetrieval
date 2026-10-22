"""Pruebas SIN Internet del módulo de extracción (unittest estándar)."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collector as c


RSS = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
  <item><title>Prices &amp; wages rise</title>
    <description><![CDATA[<p>Local prices went up this week, according to the report.</p>]]></description>
    <link>https://globalvoices.org/2026/10/test/?utm_source=rss</link>
    <pubDate>Wed, 14 Oct 2026 12:00:00 GMT</pubDate>
    <dc:creator>Original Writer</dc:creator>
  </item>
</channel></rss>'''

WIKITEXT = '''{{date|October 12, 2024}}
{{publish}}
This is a report on a recent sporting event between two national teams.
The players gave several statements after the final whistle of the match.
The referee confirmed that the event had ended with a narrow victory.
== Sources ==
* [https://example.org/source One cited source]
'''


class Response:
    def __init__(self, payload, content=None, code=200):
        self.payload = payload
        self.content = content or b""
        self.status_code = code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise c.requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload


class Client:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        return next(self.answers)


class ExtractionTests(unittest.TestCase):
    def test_rss_and_attribution(self):
        rows = list(c.rss_records(RSS, "finanzas"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Prices & wages rise")
        self.assertEqual(rows[0]["date"], "2026-10-14")
        self.assertEqual(rows[0]["author"], "Original Writer")
        self.assertIn("CC BY", rows[0]["license"])
        self.assertNotIn("utm_source", rows[0]["url"])
        self.assertNotIn("<p>", rows[0]["description/body"])

    def test_atom_feed(self):
        atom = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <title>Technology advances</title><summary>Scientists discovered a new method.</summary>
        <published>2026-10-14T10:00:00Z</published>
        <author><name>Author Name</name></author>
        <link rel="alternate" href="https://globalvoices.org/test" />
        </entry></feed>'''
        rows = list(c.rss_records(atom, "tecnologia"))
        self.assertEqual(rows[0]["author"], "Author Name")

    def test_reject_invalid_article(self):
        self.assertIsNone(c.prepare("Title", "", "Publisher", "finanzas", "2026-10-14",
                                    "https://example.com", "A", "CC", "url"))
        self.assertIsNone(c.prepare("Title", "Body", "Publisher", "finanzas", "",
                                    "https://example.com", "A", "CC", "url"))
        self.assertIsNone(c.prepare("Title", "Body", "Publisher", "finanzas", "2026-10-14",
                                    "javascript:alert(1)", "A", "CC", "url"))

    def test_archive_article_and_license(self):
        page = {"ns": 0, "title": "Match won at home", "revisions":
                [{"slots": {"main": {"content": WIKITEXT}}}]}
        row = c.extract_wikinews(page, "deportes")
        self.assertIsNotNone(row)
        self.assertEqual(row["date"], "2024-10-12")
        self.assertEqual(row["license"], "CC BY 2.5 (Wikinews)")
        self.assertNotIn("One cited source", row["description/body"])

    def test_explicit_archive_license_overrides_date(self):
        page = {"ns": 0, "title": "Match", "revisions":
                [{"slots": {"main": {"content": WIKITEXT + "\n{{Archived-cc-4.0}}"}}}]}
        self.assertEqual(c.extract_wikinews(page, "deportes")["license_url"], c.WIKINEWS_CC40)

    def test_incomplete_tsv_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "raw.tsv"
            file.write_text("\t".join(c.FIELDS) + "\nN1\tTitle\n", encoding="utf-8")
            original = file.read_bytes()
            with self.assertRaises(ValueError):
                c.collect_news(path=file, session=Client([]))
            self.assertEqual(file.read_bytes(), original)

    def test_archive_pagination_and_uniqueness(self):
        listing_1 = {"query": {"categorymembers": [{"title": "Match won at home"}]},
                     "continue": {"cmcontinue": "next", "continue": "-||"}}
        listing_2 = {"query": {"categorymembers": [{"title": "Another match"}]}}
        def item(title):
            return {"ns": 0, "title": title, "revisions":
                    [{"slots": {"main": {"content": WIKITEXT}}}]}
        client = Client([Response(listing_1), Response({"query": {"pages": [item("Match won at home")]}}),
                         Response(listing_2), Response({"query": {"pages": [item("Another match")]}})])
        with patch.object(c.time, "sleep"):
            rows = c.fetch_archive(client, "deportes", 2, set(), [], max_batches=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(client.calls[2][1]["cmcontinue"], "next")

    def test_wikitext_keeps_names_and_removes_media(self):
        text = """{{date|October 12, 2024}}
[[File:Photo.jpg|thumb|Photo by [[User:Photographer|Name]] {{image|Someone}}]]
<gallery>Image:Other.jpg|Another caption</gallery>
The {{w|black box (transportation)|black box}} from {{w|Ethiopian Airlines}}
was recovered. [[Category:Aviation]]
{| class="wikitable"
| medal table
|}
== Sources ==
A source"""
        body = c.wikinews_excerpt(text)
        self.assertIn("black box from Ethiopian Airlines", body)
        for unwanted in ("Photo", "thumb", "Name", "Aviation", "Source", "{{", "[[", "wikitable", "medal table", "Another caption", "Image:"):
            self.assertNotIn(unwanted, body)

    def test_republished_summary_is_not_given_default_license(self):
        rss = RSS.replace(b"</item>", b"<content:encoded xmlns:content='http://purl.org/rss/1.0/modules/content/'>"
                          b"This article is republished with permission.</content:encoded></item>")
        errors = []
        self.assertEqual(list(c.rss_records(rss, "finanzas", errors)), [])
        self.assertIn("derechos", errors[0])

    def test_reject_unpublished_archive_and_invalid_limits(self):
        page = {"ns": 0, "title": "Unpublished", "revisions":
                [{"slots": {"main": {"content": WIKITEXT.replace("{{publish}}", "{{develop}}")}}}]}
        self.assertIsNone(c.extract_wikinews(page, "deportes"))
        page["title"] = "Digest/12October2024"
        page["revisions"][0]["slots"]["main"]["content"] = WIKITEXT
        self.assertIsNone(c.extract_wikinews(page, "deportes"))
        with self.assertRaises(ValueError):
            c.collect_news(max_batches=0)
        self.assertEqual(c.canonical_url("https://[invalid"), "")

    def test_columns_and_tsv(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "raw.tsv"
            row = next(c.rss_records(RSS, "finanzas"))
            c.save_corpus([row], file)
            with file.open(encoding="utf-8", newline="") as f:
                data = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(data[0]["id"], row["id"])
            self.assertEqual(len(data), 1)
            self.assertIn("description/body", data[0])





class NewExtractionTests(unittest.TestCase):
    def test_basic_science_is_not_automatically_technology(self):
        page = {'ns': 0, 'title': 'An anthropologist dies', 'revisions':
                [{'slots': {'main': {'content': WIKITEXT}}}]}
        self.assertIsNone(c.extract_wikinews(page, 'tecnologia'))
        page['title'] = 'A new satellite launches'
        self.assertIsNotNone(c.extract_wikinews(page, 'tecnologia'))

    def test_eff_attribution_and_third_party_exclusion(self):
        rss = RSS.replace(b'https://globalvoices.org/2026/10/test/', b'https://www.eff.org/deeplinks/2026/10/test')
        rss = rss.replace(b'<p>Local', b'<blockquote>Third party words</blockquote><p>Local')
        row = next(c.rss_records(rss, 'tecnologia', source='EFF'))
        self.assertEqual(row['source'], 'EFF')
        self.assertEqual(row['license_url'], c.WIKINEWS_CC40)
        self.assertEqual(row['author'], 'Original Writer')
        self.assertNotIn('Third party', row['description/body'])
        self.assertEqual(list(c.rss_records(rss, 'finanzas', source='EFF')), [])
        self.assertEqual(list(c.rss_records(RSS, 'tecnologia', source='EFF')), [])

    def test_validation_prevents_overwriting_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'raw.tsv'
            row = next(c.rss_records(RSS, 'finanzas'))
            c.save_corpus([row], path)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                c.save_corpus([row, dict(row, id='different')], path)
            self.assertEqual(path.read_bytes(), original)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import retrieval as r
        cls.r = r
        cls.temp = tempfile.TemporaryDirectory()
        cls.raw = Path(cls.temp.name) / 'raw.tsv'
        cls.normalized = Path(cls.temp.name) / 'normalized.tsv'
        cls.vectors = Path(cls.temp.name) / 'vectors.joblib'
        texts = [('NASA launches rockets', 'The engineers were testing rockets and satellites in 2026.'),
                 ('Banks report profits', 'The banks reported higher profits in the financial markets.'),
                 ('Players win match', 'The players were running with their teams and scored 3 goals.')]
        rows = [c.prepare(title, body, 'Global Voices', cat, '2026-10-14',
                          f'https://globalvoices.org/test/{i}', 'Writer', 'CC BY 3.0', c.GV_LICENSE)
                for i, ((title,body),cat) in enumerate(zip(texts,c.CATEGORIES))]
        c.save_corpus(rows, cls.raw)
        cls.report = r.process_corpus(cls.raw, cls.normalized, cls.vectors)
        cls.bundle = r.joblib.load(cls.vectors)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pos_lemmas_and_selective_stopwords(self):
        r = self.r
        text = 'The players were running with their teams and scored 2026 goals. NASA uses GPUs.'
        doc = r.load_nlp()(text)
        self.assertTrue({'DET','ADP','CCONJ','PRON'}.issubset({t.pos_ for t in doc}))
        value = r.normalize_text(text).split()
        for word in ['player','run','team','score','2026','goal','nasa','gpu','be']:
            self.assertIn(word,value)
        for word in ['the','with','their','and']:
            self.assertNotIn(word,value)
        self.assertIn('not',r.normalize_text('The banks are not profitable.').split())
        self.assertNotIn('’s', r.normalize_text('NASA’s rockets are ready.').split())
        self.assertEqual(r.normalize_text(text),r.normalize_text(text))
        self.assertEqual(r.normalize_text('<p>The players\x00 were running.</p><script>bad()</script>'), 'player be run')

    def test_language_validation(self):
        row = dict(self.bundle['documents'][0], title='Los jugadores ganaron',
                   **{'description/body':'Los jugadores ganaron el partido con sus equipos y celebraron la victoria.'})
        with self.assertRaises(ValueError):
            self.r.verify_english(row)

    def test_normalized_ids_metadata_and_raw_preservation(self):
        raw = c.read_corpus(self.raw)
        normalized = c.read_corpus(self.normalized)
        self.assertEqual([row['id'] for row in raw], self.bundle['ids'])
        self.assertEqual([row['id'] for row in raw], [row['id'] for row in normalized])
        self.assertEqual(len(normalized),3)
        for before, after in zip(raw,normalized):
            self.assertTrue(all(after[key] == value for key,value in before.items()))
            self.assertEqual(after['language'],'en')
            self.assertTrue(after['normalized_title'] and after['normalized_body'])
        self.assertEqual(self.r.corpus_digest(raw),self.bundle['raw_digest'])

    def test_twelve_sparse_spaces_and_persistence(self):
        import numpy as np
        from scipy.sparse import issparse
        self.assertEqual(len(self.report),12)
        for key,space in self.bundle['spaces'].items():
            matrix, vectorizer = space['matrix'],space['vectorizer']
            self.assertTrue(issparse(matrix))
            self.assertEqual(matrix.shape,(3,len(vectorizer.vocabulary_)))
            self.assertTrue(np.isfinite(matrix.data).all())
            self.assertTrue((matrix.data >= 0).all())
            if key.endswith('binary'):
                self.assertEqual(set(matrix.data),{1})
            elif key.endswith('frequency'):
                self.assertTrue(np.issubdtype(matrix.dtype,np.integer))
            else:
                self.assertTrue((matrix.data <= 1).all())
                self.assertTrue(np.allclose(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel(),1))
            n = int(key.split('_')[1])
            self.assertEqual(vectorizer.ngram_range,(n,n))
            self.assertTrue(all(len(term.split()) == n for term in vectorizer.vocabulary_))
            column = 'normalized_title' if key.startswith('title') else 'normalized_body'
            transformed = vectorizer.transform([row[column] for row in self.bundle['documents']])
            self.assertEqual((transformed != matrix).nnz,0)

    def test_search_high_partial_zero_and_all_configurations(self):
        r = self.r
        strong = r.search_news('engineers testing rockets satellites',bundle=self.bundle)
        partial = r.search_news('satellites unfamiliarword',bundle=self.bundle)
        self.assertIn('NASA',strong[0]['title'])
        self.assertGreater(strong[0]['score'],partial[0]['score'])
        self.assertGreater(partial[0]['score'],0)
        self.assertEqual(strong, sorted(strong,key=lambda row:-row['score']))
        zero = r.search_news('zzzxxyyqqvv',bundle=self.bundle)
        self.assertEqual([row['score'] for row in zero],[0,0,0])
        self.assertEqual([row['id'] for row in zero],self.bundle['ids'])
        for field in ['title','body']:
            for n in [1,2]:
                for rep in ['frequency','binary','tfidf']:
                    query = 'NASA launches rockets' if field == 'title' else 'engineers testing rockets satellites'
                    rows=r.search_news(query,field,n,rep,path=self.vectors)
                    self.assertEqual(len(rows),3)
                    self.assertEqual(rows[0]['title'],'NASA launches rockets')
                    self.assertGreater(rows[0]['score'],0)


class EvaluationTests(unittest.TestCase):
    def test_known_metrics(self):
        from evaluation import calculate_metrics
        relevant = {'a': 2, 'b': 1, 'c': 2}
        metrics = calculate_metrics([{'id': 'x'}, {'id': 'b'}, {'id': 'a'}], relevant)
        self.assertAlmostEqual(metrics['precision@10'], 0.2)
        self.assertAlmostEqual(metrics['recall@10'], 2 / 3)
        self.assertAlmostEqual(metrics['f1@10'], 2 * 0.2 * (2 / 3) / (0.2 + (2 / 3)))
        self.assertAlmostEqual(metrics['mrr'], 0.5)
        self.assertGreater(metrics['ndcg@10'], 0)

    def test_benchmark_files_are_provisional_and_well_formed(self):
        from evaluation import load_benchmark
        queries, qrels = load_benchmark()
        self.assertEqual(len(queries), 15)
        self.assertEqual({len(qrels[q['query_id']]) for q in queries}, {10})


if __name__ == '__main__':
    unittest.main()
