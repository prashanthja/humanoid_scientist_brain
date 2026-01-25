# data_pipeline/omni_retriever.py
"""
OmniRetriever — Universal Scientific Data Fetcher
-------------------------------------------------
Fetches from: arXiv, Crossref, Semantic Scholar, Wikipedia
Outputs normalized items:
    {"title", "text", "url", "source", "timestamp"}
Includes:
  - quality gating (drop empty/short text)
  - stronger dedupe (title+text)
  - safe log filenames
"""

from __future__ import annotations
import os
import re
import time
import json
import random
import hashlib
from typing import List, Dict, Any
import requests

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

HEADERS = {"User-Agent": "HumanoidScientist/1.0 (omni retriever)"}


def _clean_text(t: str) -> str:
    return " ".join((t or "").replace("\u00a0", " ").split()).strip()


def _safe_slug(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:max_len] or "topic")


def _dedupe_key(title: str, text: str) -> str:
    base = (title or "").strip().lower() + "\n" + (text or "")[:600].strip().lower()
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()


class OmniRetriever:
    def __init__(self):
        self.topics_seen = set()
        self.last_save_dir = "logs"
        os.makedirs(self.last_save_dir, exist_ok=True)

        # Quality thresholds (tune)
        self.min_text_chars = 250
        self.bad_phrases = [
            "title pending",
            "table of contents",
            "no abstract available",
        ]

    # -----------------------------
    # Fetchers
    # -----------------------------

    def _fetch_arxiv(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            import feedparser
        except ImportError:
            return []

        url = f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={limit}"
        try:
            feed = feedparser.parse(url)
        except Exception:
            return []

        out = []
        for e in getattr(feed, "entries", []) or []:
            out.append({
                "title": _clean_text(e.get("title", "")),
                "text": _clean_text(e.get("summary", "")),
                "url": e.get("id", ""),
                "source": "arxiv",
                "timestamp": e.get("published", ""),
            })
        return out

    def _fetch_crossref(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        url = f"{CROSSREF_API}?query={query}&rows={limit}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            items = (r.json() or {}).get("message", {}).get("items", []) or []
        except Exception:
            return []

        out = []
        for it in items:
            title = ""
            t = it.get("title", [""])
            if isinstance(t, list) and t:
                title = t[0] or ""
            subtitle = it.get("subtitle", [])
            if not isinstance(subtitle, list):
                subtitle = []
            abstract = it.get("abstract", "") or ""
            text = _clean_text(" ".join([*subtitle, abstract]))
            out.append({
                "title": _clean_text(title),
                "text": text,
                "url": it.get("URL", ""),
                "source": "crossref",
                "timestamp": (it.get("issued", {}).get("date-parts", [[None]])[0][0]),
            })
        return out

    def _fetch_semanticscholar(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        url = f"{SEMANTIC_SCHOLAR_API}?query={query}&limit={limit}&fields=title,abstract,url,year"
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            data = (r.json() or {}).get("data", []) or []
        except Exception:
            return []

        out = []
        for d in data:
            out.append({
                "title": _clean_text(d.get("title", "")),
                "text": _clean_text(d.get("abstract", "")),
                "url": d.get("url", ""),
                "source": "semantic_scholar",
                "timestamp": d.get("year", ""),
            })
        return out

    def _fetch_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        try:
            q = query.replace(" ", "_")
            r = requests.get(WIKIPEDIA_API + q, headers=HEADERS, timeout=12)
            if r.status_code != 200:
                return []
            d = r.json() or {}
        except Exception:
            return []

        return [{
            "title": _clean_text(d.get("title", query)),
            "text": _clean_text(d.get("extract", "")),
            "url": (d.get("content_urls", {}) or {}).get("desktop", {}).get("page", ""),
            "source": "wikipedia",
            "timestamp": time.strftime("%Y-%m-%d"),
        }]

    # -----------------------------
    # Unified interface
    # -----------------------------

    def _is_good(self, it: Dict[str, Any]) -> bool:
        title = _clean_text(it.get("title", ""))
        text = _clean_text(it.get("text", ""))

        if not title:
            return False
        if len(text) < self.min_text_chars:
            return False

        low = (title + " " + text).lower()
        if any(p in low for p in self.bad_phrases):
            return False

        # very common garbage if text is basically a title repeat
        if len(text) < 2 * len(title):
            return False

        return True

    def retrieve_universal(self, query: str) -> List[Dict[str, Any]]:
        print(f"🌐 [OmniRetriever] Collecting data for topic: {query}")

        all_items: List[Dict[str, Any]] = []
        all_items += self._fetch_arxiv(query)
        all_items += self._fetch_crossref(query)
        all_items += self._fetch_semanticscholar(query)
        all_items += self._fetch_wikipedia(query)

        # Quality gate
        all_items = [it for it in all_items if self._is_good(it)]

        # Strong dedupe: title + text
        seen = set()
        unique: List[Dict[str, Any]] = []
        for it in all_items:
            k = _dedupe_key(it.get("title", ""), it.get("text", ""))
            if k in seen:
                continue
            seen.add(k)
            unique.append(it)

        print(f"   ✅ Retrieved {len(unique)} quality entries for '{query}'")
        self.topics_seen.add(query)

        # Save logs (safe filename)
        slug = _safe_slug(query)
        path = os.path.join(self.last_save_dir, f"retrieved_universal_{slug}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(unique, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return unique

    def extract_new_topics(self, docs: List[Dict[str, Any]]) -> List[str]:
        topics: List[str] = []
        for d in docs:
            t = (d.get("title", "") or "").lower()
            if not t:
                continue
            for kw in re.findall(r"[a-zA-Z]{4,}", t):
                if kw not in self.topics_seen and kw not in topics:
                    topics.append(kw)
        random.shuffle(topics)
        return topics[:15]
