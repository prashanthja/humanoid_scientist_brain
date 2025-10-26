"""
InternetRetriever — Phase D / Step 1
------------------------------------
Fetches fresh, real science content from the internet and normalizes it for the KB.

Sources supported (out-of-the-box):
  • arXiv API           (physics, math, cs)
  • Crossref API        (metadata + abstracts where available)
  • RSS feeds           (journals / org feeds)
  • Generic web pages   (best-effort text extraction)

Design:
  - Small, dependency-light; optional extras used if installed.
  - Robust timeouts, graceful fallbacks, and de-duplication by URL hash.
  - Output normalized to: {id, source, title, text, url, tags, timestamp, meta}

Usage:
  retr = InternetRetriever()
  items = retr.retrieve_topics(["physics", "linear algebra"], max_items=100)
  # Optionally run items through SafetyFilter in your pipeline, then KB.store(items)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import datetime as dt
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass, asdict
import os

# --- Optional deps (all gracefully optional) ---
try:
    import requests
except Exception:
    requests = None

try:
    import feedparser  # RSS/Atom
except Exception:
    feedparser = None

try:
    from bs4 import BeautifulSoup  # HTML -> text
except Exception:
    BeautifulSoup = None


# ----------------------------
# Config & small utilities
# ----------------------------

DEFAULT_TIMEOUT = 12  # seconds
ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
USER_AGENT = "HumanoidScientist/1.0 (+https://example.local)"

def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _strip_html(html: str) -> str:
    if not html:
        return ""
    if BeautifulSoup is None:
        # crude fallback
        return _norm_space(re.sub(r"<[^>]+>", " ", html))
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return _norm_space(soup.get_text(" "))

def _requests_get(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    if requests is None:
        return None
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            return resp
        return None
    except Exception:
        return None


# ----------------------------
# Normalized item
# ----------------------------

@dataclass
class InternetItem:
    id: str
    source: str              # "arxiv" | "crossref" | "rss" | "web"
    title: str
    text: str
    url: str
    tags: List[str]
    timestamp: str           # ISO8601
    meta: Dict              # free-form metadata

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure all strings are compact
        d["title"] = _norm_space(d["title"])
        d["text"] = _norm_space(d["text"])
        return d


# ----------------------------
# Source adapters
# ----------------------------

class ArxivSource:
    """Minimal arXiv fetch via Atom API. No external lib required."""
    def fetch(self, query: str, max_items: int = 20) -> List[InternetItem]:
        items: List[InternetItem] = []
        if requests is None:
            return items

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_items,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = _requests_get(ARXIV_ENDPOINT, params=params)
        if resp is None:
            return items

        # Very lightweight parse; arXiv returns Atom XML
        text = resp.text
        # Split entries
        entries = re.split(r"<entry>\s*", text)[1:]
        for raw in entries:
            # title
            m_title = re.search(r"<title>(.*?)</title>", raw, re.S)
            title = _strip_html(m_title.group(1)) if m_title else ""
            # summary
            m_sum = re.search(r"<summary>(.*?)</summary>", raw, re.S)
            summary = _strip_html(m_sum.group(1)) if m_sum else ""
            # link (abs)
            m_link = re.search(r'<link[^>]+href="([^"]+)"[^>]*/?>', raw)
            link = (m_link.group(1) if m_link else "").replace("&amp;", "&")
            # updated/published
            m_upd = re.search(r"<updated>(.*?)</updated>", raw)
            updated = m_upd.group(1) if m_upd else _now_iso()

            if not summary and not title:
                continue

            uid = _hash(f"arxiv::{link or title}")
            items.append(
                InternetItem(
                    id=uid,
                    source="arxiv",
                    title=title or "arXiv paper",
                    text=summary,
                    url=link or "",
                    tags=["arxiv", "paper", query],
                    timestamp=updated,
                    meta={"query": query},
                )
            )
        return items


class CrossrefSource:
    """Crossref JSON API: rich metadata, sometimes abstracts (HTML-ish)."""
    def fetch(self, query: str, max_items: int = 20) -> List[InternetItem]:
        items: List[InternetItem] = []
        if requests is None:
            return items

        params = {
            "query": query,
            "rows": min(max_items, 50),
            "select": "title,URL,abstract,issued,container-title,subject,type",
            "sort": "issued",
            "order": "desc",
        }
        resp = _requests_get(CROSSREF_ENDPOINT, params=params)
        if resp is None:
            return items

        try:
            data = resp.json()
        except Exception:
            return items

        for w in data.get("message", {}).get("items", []):
            title = " / ".join(w.get("title") or []) or "Crossref item"
            url = w.get("URL", "")
            abstract = _strip_html(w.get("abstract") or "")
            if not abstract and not title:
                continue
            issued = w.get("issued", {}).get("date-parts", [[None]])[0][0]
            year = str(issued) if issued else None

            uid = _hash(f"crossref::{url or title}")
            items.append(
                InternetItem(
                    id=uid,
                    source="crossref",
                    title=_norm_space(title),
                    text=_norm_space(abstract or title),
                    url=url,
                    tags=["crossref", w.get("type", "work"), query] + (w.get("subject") or []),
                    timestamp=_now_iso() if not year else f"{year}-01-01T00:00:00Z",
                    meta={"container": w.get("container-title"), "subject": w.get("subject")},
                )
            )
        return items


class RSSSource:
    """Generic RSS/Atom feeds (journals, orgs)."""
    def __init__(self, feeds: Optional[List[str]] = None):
        self.feeds = feeds or [
            # Add more feeds you trust:
            "https://www.nature.com/subjects/physics.rss",
            "https://feeds.aps.org/rss/featured",
        ]

    def fetch(self, query: str, max_items: int = 20) -> List[InternetItem]:
        items: List[InternetItem] = []
        if feedparser is None:
            return items

        per_feed = max(1, max_items // max(1, len(self.feeds)))
        for url in self.feeds:
            try:
                parsed = feedparser.parse(url)
            except Exception:
                continue
            for entry in parsed.entries[:per_feed]:
                title = _norm_space(entry.get("title", ""))
                summary = _strip_html(entry.get("summary", ""))
                link = entry.get("link", "")
                if not (title or summary):
                    continue
                uid = _hash(f"rss::{link or title}")
                timestamp = entry.get("published", "") or _now_iso()
                items.append(
                    InternetItem(
                        id=uid,
                        source="rss",
                        title=title,
                        text=summary or title,
                        url=link,
                        tags=["rss", query],
                        timestamp=timestamp,
                        meta={"feed": url},
                    )
                )
        return items


class GenericWebSource:
    """Best-effort HTML fetch & sanitize for arbitrary URLs."""
    def fetch_urls(self, urls: Iterable[str], tag: str = "web") -> List[InternetItem]:
        items: List[InternetItem] = []
        for u in urls:
            resp = _requests_get(u)
            if resp is None or not resp.text:
                continue
            text = _strip_html(resp.text)
            title = ""
            if BeautifulSoup is not None:
                try:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    title = _norm_space(soup.title.get_text()) if soup.title else ""
                except Exception:
                    pass
            uid = _hash(f"web::{u}")
            items.append(
                InternetItem(
                    id=uid,
                    source="web",
                    title=title or "Web Page",
                    text=text[:5000],  # limit
                    url=u,
                    tags=[tag],
                    timestamp=_now_iso(),
                    meta={},
                )
            )
            time.sleep(0.2)
        return items


# ----------------------------
# InternetRetriever (orchestration)
# ----------------------------

class InternetRetriever:
    """
    Orchestrates multi-source retrieval, normalization, and de-duplication.
    You can pass a SafetyFilter to `retrieve_topics(..., safety=...)`.
    """

    def __init__(self):
        self.arxiv = ArxivSource()
        self.crossref = CrossrefSource()
        self.rss = RSSSource()
        self.web = GenericWebSource()

        # simple in-memory dedupe (you can persist this if you want)
        self._seen: set[str] = set()

    def retrieve_topics(
        self,
        topics: List[str],
        max_items: int = 60,
        per_source_cap: Optional[int] = None,
        safety=None,
        extra_urls: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        For each topic, pull from: arXiv + Crossref + RSS (optionally URLs), dedupe and return dicts.
        Also logs all fetched papers with proper titles & abstracts.
        """
        os.makedirs("logs", exist_ok=True)
        results: List[InternetItem] = []
        cap = per_source_cap or max(10, max_items // 3)

        print(f"🌐 [InternetRetriever] Collecting papers for topics: {', '.join(topics)}")

        for topic in topics:
            print(f"   🔍 Fetching topic: {topic}")
            try:
                a_items = self.arxiv.fetch(topic, max_items=cap)
                c_items = self.crossref.fetch(topic, max_items=cap)
                r_items = self.rss.fetch(topic, max_items=cap)
                results.extend(a_items + c_items + r_items)
                print(f"   ✅ {len(a_items)} arXiv | {len(c_items)} Crossref | {len(r_items)} RSS items")
            except Exception as e:
                print(f"   ⚠️ Error fetching {topic}: {e}")
            time.sleep(0.3)

        # Optional arbitrary URLs (white-listed)
        if extra_urls:
            web_items = self.web.fetch_urls(extra_urls, tag="extra")
            print(f"   🌐 Added {len(web_items)} from custom URLs.")
            results.extend(web_items)

        # De-duplicate by hash
        uniq: List[dict] = []
        for item in results:
            key = item.id
            if key in self._seen:
                continue
            self._seen.add(key)
            clean = item.to_dict()
            if not clean.get("title"):
                clean["title"] = "Untitled Research"
            if not clean.get("text"):
                clean["text"] = clean["title"]
            uniq.append(clean)

        # Optional safety pass
        if safety is not None:
            try:
                before = len(uniq)
                uniq = safety.filter(uniq)
                print(f"   🔒 SafetyFilter removed {before - len(uniq)} unsafe items")
            except Exception as e:
                print(f"   ⚠️ Safety filter error: {e}")

        # ✅ Log results
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = f"logs/retrieved_papers_{ts}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(uniq, f, indent=2, ensure_ascii=False)

        # Print a brief summary
        print(f"📚 Retrieved {len(uniq)} total papers — saved list → {log_path}")
        sample_titles = [u['title'] for u in uniq[:5]]
        for t in sample_titles:
            print(f"   🧾 {t[:100]}")

        return uniq
