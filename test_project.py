"""Pruebas SIN Internet del módulo de extracción (unittest estándar)."""
import csv
import tempfile
import unittest
from datetime import datetime, timezone
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

    def test_weekly_dry_run_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "news.tsv"
            now = datetime(2026, 10, 15, tzinfo=timezone.utc)
            # Tres feeds, cada uno devuelve un ejemplo en el mismo URL; sólo se guarda uno.
            with patch.object(c, "FEEDS", {k: [f"https://globalvoices.org/{k}/feed/"] for k in c.CATEGORIES}), \
                 patch.object(c, "LOG", Path(d) / "log.jsonl"):
                dry = c.collect_news("weekly", per_category=1, dry_run=True, path=file,
                                     now=now, session=Client([Response({}, RSS)] * 3))
                self.assertFalse(file.exists())
                self.assertEqual(dry["added"], 1)
                run = c.collect_news("weekly", per_category=1, path=file,
                                     now=now, session=Client([Response({}, RSS)] * 3))
                self.assertEqual(run["added"], 1)
                self.assertEqual(len(c.read_corpus(file)), 1)
                twice = c.collect_news("weekly", per_category=1, path=file,
                                       now=now, session=Client([Response({}, RSS)] * 3))
                self.assertEqual(twice["added"], 0)

    def test_init_uses_archive_for_remaining_quota(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "news.tsv"
            item = c.prepare("Relevant news", "Several sentences of text here", "Wikinews",
                             "finanzas", "2024-10-12", "https://en.wikinews.org/wiki/Example",
                             "Wikinews", "CC BY", c.WIKINEWS_CC25)
            with patch.object(c, "fetch_feeds", return_value=[]), \
                 patch.object(c, "fetch_archive", side_effect=lambda session, category, wanted, ids,
                                                           errors, max_batches=80: [item] if category == "finanzas" else []), \
                 patch.object(c, "LOG", Path(d) / "log.jsonl"):
                result = c.collect_news("init", target=3, path=file, session=Client([]))
            self.assertEqual(result["added"], 1)
            self.assertFalse(result["complete"])
            self.assertEqual(len(c.read_corpus(file)), 1)

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

    def test_weekly_shared_quota_and_next_friday(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "news.tsv"
            log = Path(d) / "log.jsonl"
            def feeds(client, category, errors):
                return [c.prepare("News", "Authentic fixture text", "Global Voices", category,
                                  "2026-10-14", f"https://globalvoices.org/{category}/{i}",
                                  "Writer", "CC BY 3.0", c.GV_LICENSE) for i in range(5)]
            with patch.object(c, "fetch_feeds", side_effect=feeds):
                args = dict(per_category=2, path=file, log_path=log, session=Client([]))
                first = c.collect_news(now=datetime(2026, 10, 15, tzinfo=timezone.utc), **args)
                self.assertEqual(first["added"], 6)
                original = (file.read_bytes(), log.read_bytes())
                again = c.collect_news(now=datetime(2026, 10, 15, tzinfo=timezone.utc), **args)
                self.assertEqual(again["added"], 0)
                self.assertTrue(again["complete"])
                self.assertEqual(original, (file.read_bytes(), log.read_bytes()))
                # Viernes local (16 UTC) abre una cuota nueva; dry-run no la consume.
                dry = c.collect_news(now=datetime(2026, 10, 16, 16, tzinfo=timezone.utc),
                                     dry_run=True, **args)
                self.assertEqual(dry["added"], 6)
                self.assertEqual(original, (file.read_bytes(), log.read_bytes()))
                next_week = c.collect_news(now=datetime(2026, 10, 16, 16, tzinfo=timezone.utc), **args)
                self.assertEqual(next_week["added"], 6)
                self.assertEqual(len(c.read_corpus(file)), 12)

    def test_reject_unpublished_archive_and_invalid_limits(self):
        page = {"ns": 0, "title": "Unpublished", "revisions":
                [{"slots": {"main": {"content": WIKITEXT.replace("{{publish}}", "{{develop}}")}}}]}
        self.assertIsNone(c.extract_wikinews(page, "deportes"))
        page["title"] = "Digest/12October2024"
        page["revisions"][0]["slots"]["main"]["content"] = WIKITEXT
        self.assertIsNone(c.extract_wikinews(page, "deportes"))
        with self.assertRaises(ValueError):
            c.collect_news(dry_run=True, max_batches=0)
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


if __name__ == "__main__":
    unittest.main()
