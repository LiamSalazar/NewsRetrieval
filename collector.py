"""Adquisición académica de noticias: RSS (actual) y Wikinews (histórico).

Solo recupera textos ofrecidos por fuentes con licencias documentadas.
No abre artículos arbitrarios, no sortea paywalls ni fabrica registros.
"""

import csv
import hashlib
import html
import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import quote, urlencode, urlsplit, urlunsplit, parse_qsl

import requests

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "news_raw_corpus.tsv"
LOG = ROOT / "results" / "collection_log.jsonl"
FIELDS = ["id", "title", "description/body", "source", "category", "date",
          "url", "retrieved_at", "author", "license", "license_url"]
CATEGORIES = ("finanzas", "deportes", "tecnologia")

# Tres feeds temáticos publicados en el directorio oficial de Global Voices.
# No garantizan diez noticias nuevas por categoría cada semana.
FEEDS = {
    "finanzas": ["https://globalvoices.org/-/topics/economics-business/feed/"],
    "deportes": ["https://globalvoices.org/-/topics/sport/feed/"],
    "tecnologia": ["https://globalvoices.org/-/topics/technology/feed/"],
}
FEEDS['tecnologia'].append('https://www.eff.org/rss/updates.xml')
# Archivo HISTÓRICO: cerró para nuevas publicaciones en mayo de 2026.
WIKINEWS = {
    "finanzas": "Category:Economy and business",
    "deportes": "Category:Sports",
    "tecnologia": "Category:Science and technology",
}
API = "https://en.wikinews.org/w/api.php"
HEADERS = {"User-Agent": "Global Voices / EFF news collector (academic research;"}
GV_LICENSE = "https://creativecommons.org/licenses/by/3.0/"
# Global Voices dice CC BY; revisar licencia concreta del artículo antes de republicarlo.
WIKINEWS_CC25 = "https://creativecommons.org/licenses/by/2.5/"
WIKINEWS_CC40 = "https://creativecommons.org/licenses/by/4.0/"
_LAST_REQUEST = {}
# La categoría de Wikinews mezcla ciencia básica y tecnología. Exigir evidencia
# textual de tecnología evita clasificar biografías o biología general como tal.
TECH_TERMS = re.compile(r'\b(?:technolog\w*|comput\w*|internet|software|hardware|digital|'
                        r'cyber\w*|artificial intelligence|robot\w*|semiconductor\w*|'
                        r'telecom\w*|smartphone\w*|mobile phone\w*|wi-fi|websit\w*|'
                        r'encrypt\w*|broadband|browser\w*|satellite\w*|spacecraft|'
                        r'spacewalk\w*|space station|space shuttle|rocket\w*|'
                        r'telescope\w*|rover\w*|hoverboard\w*|laser\w*|3d print\w*|'
                        r'nuclear reactor\w*|particle accelerator\w*)\b', re.I)


class TextOnly(HTMLParser):
    """Extrae texto de fragmentos HTML de un RSS, sin imágenes/scripts."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.blocked = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.blocked += 1
        if tag in {"p", "br", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.blocked = max(0, self.blocked - 1)
        if tag in {"p", "div", "li"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.blocked:
            self.parts.append(data)


def plain_text(value):
    parser = TextOnly()
    parser.feed(str(value or ""))
    return " ".join(html.unescape("".join(parser.parts)).split())


def published_date(value):
    """Fecha UTC 'YYYY-MM-DD'; None cuando la fuente no permite determinarla."""
    if not value:
        return None
    value = str(value).strip()
    try:
        dt = parsedate_to_datetime(value)
    except (ValueError, TypeError, IndexError):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(value, "%B %d, %Y")
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.date().isoformat() if 2000 <= dt.year and dt.date() <= datetime.now(timezone.utc).date() else None


def canonical_url(url):
    """Descarta fragmentos y parámetros comunes de seguimiento, sin inventar URL."""
    try:
        parsed = urlsplit(str(url).strip())
    except ValueError:
        return ""
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        return ""
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query)
                       if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}])
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/") or "/", query, ""))


def prepare(title, body, source, category, date, url, author, license_name, license_url):
    title, body = plain_text(title), plain_text(body)
    url = canonical_url(url)
    date = published_date(date)
    if not (title and body and url and date and source and author and license_name
            and canonical_url(license_url) and category in CATEGORIES):
        return None
    uid = "N" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return {"id": uid, "title": title, "description/body": body, "source": source,
            "category": category, "date": date, "url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "author": author, "license": license_name, "license_url": license_url}


def get(session, url, params=None):
    """HTTP limitado; reintenta errores transitorios, nunca elude bloqueos."""
    for attempt in range(3):
        try:
            # El robots.txt de Global Voices pide diez segundos entre accesos.
            host = urlsplit(url).netloc
            if host == 'globalvoices.org' and isinstance(session, requests.Session):
                time.sleep(max(0, 10 - (time.monotonic() - _LAST_REQUEST.get(host, 0))))
                _LAST_REQUEST[host] = time.monotonic()
            response = session.get(url, params=params, timeout=25, headers=HEADERS)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(attempt + 1)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if attempt == 2 or (status is not None and status not in {429, 500, 502, 503, 504}):
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Falló la consulta HTTP")


def xml_value(node, names):
    for name in names:
        child = node.find(name)
        if child is not None and child.text:
            return child.text.strip()
    return ""


def rss_records(xml_content, category, errors=None, source='Global Voices'):
    """RSS/Atom autorizados: conserva autor y rechaza derechos particulares."""
    if source not in {'Global Voices', 'EFF'}:
        raise ValueError('Fuente RSS no autorizada')
    root = ET.fromstring(xml_content)
    dc = "{http://purl.org/dc/elements/1.1/}"
    atom = "{http://www.w3.org/2005/Atom}"
    content = "{http://purl.org/rss/1.0/modules/content/}encoded"
    items = root.findall(".//item") + root.findall(f".//{atom}entry")
    for item in items:
        title = xml_value(item, ["title", f"{atom}title"])
        description = xml_value(item, ["description", f"{atom}summary"])
        author = xml_value(item, [f"{dc}creator", "author", f"{atom}author/{atom}name"])
        date = xml_value(item, ["pubDate", f"{dc}date", f"{atom}published", f"{atom}updated"])
        link = xml_value(item, ["link"])
        if not link:
            link = next((node.get("href", "") for node in item.findall(f"{atom}link")
                         if node.get("rel", "alternate") == "alternate"), "")
        # El contenido completo sirve para detectar republicaciones; no se almacena.
        full = plain_text(xml_value(item, [content, f"{atom}content"]) + ' ' + description)
        full = re.sub(r"Originally published on Global Voices", "", full, flags=re.I)
        rights = xml_value(item, [f"{dc}rights", f"{atom}rights"])
        license_url = GV_LICENSE if source == 'Global Voices' else WIKINEWS_CC40
        if (re.search(r"republished|(?:originally|first) published|all rights reserved", full, re.I)
                or (rights and rights not in {license_url, 'CC BY 3.0' if source == 'Global Voices' else 'CC BY 4.0'})):
            if errors is not None:
                errors.append(f"RSS {category}: omitido por derechos particulares/republicación: {title}")
            continue
        host = urlsplit(canonical_url(link)).netloc
        if host != ('globalvoices.org' if source == 'Global Voices' else 'www.eff.org'):
            continue
        if source == 'EFF':
            if category != 'tecnologia' or '/deeplinks/' not in link or re.search(
                    r'podcast|effector|daf day|organizing in|award winners|welcomes .*board|join us|donate', title, re.I):
                continue
            # No conservar bloques citados de terceros ni imágenes del feed.
            description = re.sub(r'(?is)<blockquote\b[^>]*>.*?</blockquote>', ' ', description)
        record = prepare(title, description, source, category, date, link,
                         plain_text(author), 'CC BY 3.0 (resumen de Global Voices)' if source == 'Global Voices'
                         else 'CC BY 4.0 (texto original EFF; sin bloques citados)', license_url)
        if record:
            yield record


def fetch_feeds(session, category, errors, page=1):
    records = []
    for url in FEEDS[category]:
        try:
            if page > 1:
                if urlsplit(url).netloc != 'globalvoices.org':
                    continue
                url += f'?paged={page}'
            response = get(session, url)
            source = 'EFF' if urlsplit(url).netloc == 'www.eff.org' else 'Global Voices'
            parsed = list(rss_records(response.content, category, errors, source))
            if not parsed:
                errors.append(f"RSS {category}: no ofreció artículos con fecha, URL y texto válidos: {url}")
            records.extend(parsed)
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            errors.append(f"RSS {category}: {url}: {type(exc).__name__}: {exc}")
    return records


def wikinews_excerpt(wikitext):
    """Texto plano introductorio a partir del wikitext (sin medios de terceros).

    Es una limpieza conservadora: no pretende convertir wikitext complejo a HTML.
    """
    # El cuerpo precede a apartados de fuentes/enlaces.
    text = re.split(r"(?im)^==+\s*(?:Sources|External links|Related news)\s*==+", wikitext, maxsplit=1)[0]
    text = re.sub(r"(?is)<gallery\b[^>]*>.*?</gallery\s*>", " ", text)
    text = re.sub(r"(?is)<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/>", " ", text)
    text = re.sub(r"(?s)\{\|.*?\|\}", " ", text)
    # Las plantillas w son enlaces con texto, no metadatos descartables.
    text = re.sub(r"\{\{\s*[wW]\s*\|([^{}]+)\}\}",
                  lambda m: m.group(1).split("|")[-1].strip(), text)
    # Descartar plantillas (a veces anidadas) y metadatos de la página.
    for _ in range(12):
        updated = re.sub(r"\{\{[^{}]*\}\}", " ", text, flags=re.S)
        if updated == text:
            break
        text = updated
    # Quitar medios ANTES de convertir enlaces, incluyendo enlaces en sus pies.
    text = re.sub(r"\[\[(?:File|Image|Category):(?:[^\[\]]|\[\[[^\[\]]*\]\])*\]\]",
                  " ", text, flags=re.I)
    text = re.sub(r"\[\[([^\[\]]+)\]\]",
                  lambda m: m.group(1).split("|")[-1].removeprefix("w:"), text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"(?m)^(?:\||__).*$", " ", text)
    text = re.sub(r"(?m)^={2,}.*?={2,}\s*$", " ", text)
    text = re.sub(r"(?m)^\s*[*#:;]+\s*", " ", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"\[[^\]]+\]", " ", text)
    return plain_text(text)


def extract_wikinews(page, category):
    if page.get("ns") != 0 or page.get("missing") or page.get("title", "").startswith("Digest/"):
        return None
    revs = page.get("revisions") or []
    if not revs:
        return None
    text = revs[0].get("slots", {}).get("main", {}).get("content", "")
    if not text or not re.search(r"\{\{\s*(?:publish|published|archived)\s*[|}]", text, re.I):
        return None
    if re.search(r"\{\{\s*(?:copyvio|copyright violation|fair use)\b", text, re.I):
        return None
    match = re.search(r"\{\{\s*date\s*\|\s*([^|{}]+)", text, re.I)
    if not match:
        return None
    date = published_date(match.group(1))
    if not date:
        return None
    body = wikinews_excerpt(text)
    if category == 'tecnologia' and not TECH_TERMS.search(page['title'] + ' ' + body):
        return None
    # Al menos unas cuantas palabras para que sea una noticia recuperable.
    if len(body.split()) < 18 or any(mark in body for mark in ("{{", "}}", "[[", "]]", "{|", "|}")):
        return None
    title = page["title"]
    url = "https://en.wikinews.org/wiki/" + quote(title.replace(" ", "_"), safe="()/,:-")
    # Una licencia explícita del artículo prevalece sobre la regla por fecha.
    explicit = re.search(r"\{\{\s*Archived-(?:cc-)?(2\.5|4\.0|pd)\s*[|}]", text, re.I)
    version = explicit.group(1).lower() if explicit else None
    if version == "pd" or (not version and date < "2005-09-25"):
        lic, lic_url = "Public domain (Wikinews)", "https://en.wikinews.org/wiki/Wikinews:Copyright"
    elif version == "2.5" or (not version and date < "2024-12-16"):
        lic, lic_url = "CC BY 2.5 (Wikinews)", WIKINEWS_CC25
    else:
        lic, lic_url = "CC BY 4.0 (Wikinews)", WIKINEWS_CC40
    return prepare(title, body, "Wikinews", category, date, url, "Wikinews", lic, lic_url)


def fetch_archive(session, category, wanted, existing_ids, errors, max_batches=80):
    """Consulta páginas públicas de una categoría hasta obtener N noticias únicas.

    Respeta la paginación de MediaWiki; evita consultas por artículo.
    """
    found = []
    cursor = {}
    reason = f"límite de {max_batches} lotes alcanzado"
    for _ in range(max_batches):
        if len(found) >= wanted:
            break
        params = {"action": "query", "list": "categorymembers", "cmtitle": WIKINEWS[category],
                  "cmnamespace": "0", "cmlimit": "50", "format": "json", **cursor}
        try:
            response = get(session, API, params)
            listing = response.json()
            if "error" in listing:
                raise ValueError(listing["error"].get("info", "MediaWiki API error"))
            titles = [p["title"] for p in listing.get("query", {}).get("categorymembers", [])]
            if not titles:
                reason = "categoría agotada"
                break
            detail = get(session, API, {"action": "query", "prop": "revisions",
                                        "titles": "|".join(titles), "rvprop": "content",
                                        "rvslots": "main", "formatversion": "2", "format": "json"}).json()
            if "error" in detail:
                raise ValueError(detail["error"].get("info", "MediaWiki API error"))
            for page in detail.get("query", {}).get("pages", []):
                item = extract_wikinews(page, category)
                if item and item["id"] not in existing_ids:
                    existing_ids.add(item["id"])
                    found.append(item)
                    if len(found) >= wanted:
                        break
            cursor = listing.get("continue") or {}
            if not cursor:
                reason = "categoría agotada"
                break
            time.sleep(0.15)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            break
    if len(found) < wanted:
        errors.append(f"ARCHIVO {category}: {len(found)}/{wanted} noticias nuevas válidas; {reason}.")
    return found


def read_corpus(path=CORPUS):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Columnas inválidas en {path}")
        rows = list(reader)
        for number, row in enumerate(rows, start=2):
            if None in row or any(not row.get(field) for field in FIELDS):
                raise ValueError(f"Registro incompleto en {path}, línea {number}")
        return rows


def save_corpus(rows, path=CORPUS):
    """Escritura atómica: reemplaza el TSV solo cuando está completo."""
    validate_corpus(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.with_suffix('.tsv.bak').exists():
        shutil.copy2(path, path.with_suffix('.tsv.bak'))
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore", lineterminator='\n')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["date"], r["id"]), reverse=True))
    validate_corpus(read_corpus(temp))
    temp.replace(path)


def validate_corpus(rows):
    """Rechaza registros incompletos y duplicados antes del reemplazo atómico."""
    for field in ('id', 'url', 'title', 'description/body'):
        values = [' '.join(row.get(field, '').casefold().split()) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicados en {field}')
    for row in rows:
        if any(not row.get(field) for field in FIELDS) or row['category'] not in CATEGORIES:
            raise ValueError('Registro incompleto o categoría inválida')


def collect_news(mode="weekly", target=2000, per_category=10, days=8, dry_run=False,
                 session=None, path=CORPUS, max_batches=80, now=None, log_path=None):
    """Devuelve informe; no inventa cuotas. init: objetivo total; weekly: novedades.

    session permite pruebas sin tráfico real. dry_run no escribe archivos.
    """
    if mode not in {"init", "weekly"}:
        raise ValueError("mode debe ser init o weekly")
    if target < 1 or not 1 <= per_category <= 10 or days < 1 or max_batches < 1:
        raise ValueError("per_category debe estar entre 1 y 10; target, days y max_batches deben ser positivos")
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo("America/Mexico_City"))
    # Una cuota compartida desde el viernes 00:00 hasta el siguiente jueves.
    period = (local.date() - timedelta(days=(local.weekday() - 4) % 7)).isoformat()
    before = (now - timedelta(days=days)).date().isoformat()
    log_path = log_path or (LOG if path == CORPUS else path.with_suffix(".jsonl"))
    weekly_saved = Counter()
    if mode == "weekly" and log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("mode") == "weekly" and entry.get("period_start") == period:
                weekly_saved.update(entry["per_category_added"])
    current = read_corpus(path)
    validate_corpus(current)
    present = {r["id"] for r in current}
    urls = {canonical_url(r['url']) for r in current}
    titles = {r['title'].casefold() for r in current}
    bodies = {r['description/body'].casefold() for r in current}
    original = Counter(r["category"] for r in current)
    quotas = ({k: target // 3 + (i < target % 3) for i, k in enumerate(CATEGORIES)}
              if mode == "init" else {k: per_category for k in CATEGORIES})
    expected = {c: max(0, quotas[c] - (original[c] if mode == "init" else weekly_saved[c]))
                for c in CATEGORIES}
    errors, added = [], []
    client = session or requests.Session()
    try:
        for category in CATEGORIES:
            needed = expected[category]
            have = 0
            if not needed:
                continue
            feeds = fetch_feeds(client, category, errors)
            if mode == 'init' and needed > len(feeds):
                for page in range(2, min(5, max_batches) + 1):
                    feeds.extend(fetch_feeds(client, category, errors, page=page))
            feeds.sort(key=lambda a: a['date'], reverse=True)
            for item in feeds:
                if have >= needed:
                    break
                if mode == "weekly" and not before <= item["date"] <= now.date().isoformat():
                    continue
                if (item['id'] not in present and canonical_url(item['url']) not in urls
                        and item['title'].casefold() not in titles and item['description/body'].casefold() not in bodies):
                    present.add(item["id"])
                    urls.add(canonical_url(item['url']))
                    titles.add(item['title'].casefold())
                    bodies.add(item['description/body'].casefold())
                    added.append(item)
                    have += 1
            if mode == "init" and have < needed:
                for item in fetch_archive(client, category, needed - have, present, errors,
                                          max_batches=max_batches):
                    if (canonical_url(item['url']) not in urls and item['title'].casefold() not in titles
                            and item['description/body'].casefold() not in bodies):
                        urls.add(canonical_url(item['url']))
                        titles.add(item['title'].casefold())
                        bodies.add(item['description/body'].casefold())
                        added.append(item)
        if not dry_run and added:
            save_corpus(current + added, path)
    finally:
        if session is None:
            client.close()
    counts = Counter(r["category"] for r in added)
    report = {"mode": mode, "dry_run": dry_run, "executed_at": now.isoformat(),
              "already_saved": len(current), "added": len(added),
              "per_category_added": {c: counts[c] for c in CATEGORIES},
              "per_category_expected": expected,
              "per_category_missing": {c: max(0, expected[c] - counts[c]) for c in CATEGORIES},
              "per_source_added": dict(Counter(r["source"] for r in added)),
              "period_start": period if mode == "weekly" else None,
              "weekly_already_added": dict(weekly_saved),
              "total_after": len(current) + (0 if dry_run else len(added)),
              "errors": errors,
              "target_met": all(counts[c] >= expected[c] for c in CATEGORIES),
              "complete": (all(original[c] + counts[c] == quotas[c] for c in CATEGORIES)
                           and len(current) + len(added) == target
                           and len({r['source'] for r in current + added}) >= 3 if mode == 'init'
                           else all(weekly_saved[c] + counts[c] >= 1 for c in CATEGORIES))}
    if not dry_run and added:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report
