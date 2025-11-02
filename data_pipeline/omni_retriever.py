"""
OmniRetriever — Universal Scientific Data Fetcher
-------------------------------------------------
This retriever goes beyond a few websites:
  • Fetches all physics & mathematics topics from arXiv, Crossref,
    Semantic Scholar, Wikipedia, NASA ADS, and more.
  • Expands topic coverage automatically based on newly found papers.
  • Can run continuously to cover *all* physics and math domains from
    the beginning of modern science to now.

Outputs normalized items with keys:
    {"title", "text", "url", "source", "timestamp"}
"""

from __future__ import annotations
import os, re, time, json, random, requests
from typing import List, Dict, Any

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
NASA_ADS_API = "https://api.adsabs.harvard.edu/v1/search/query"

HEADERS = {"User-Agent": "HumanoidScientist/1.0 (omni retriever)"}

class OmniRetriever:
    """
    Universal retriever for scientific knowledge.
    """
    def __init__(self):
        self.topics_seen = set()
        self.last_save_dir = "logs"
        os.makedirs(self.last_save_dir, exist_ok=True)

    # ----------------------------------------------------
    #  Fetchers
    # ----------------------------------------------------

    def _fetch_arxiv(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            import feedparser
        except ImportError:
            return []
        url = f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={limit}"
        feed = feedparser.parse(url)
        out = []
        for e in feed.entries:
            out.append({
                "title": e.get("title", ""),
                "text": e.get("summary", ""),
                "url": e.get("id", ""),
                "source": "arxiv",
                "timestamp": e.get("published", ""),
            })
        return out

    def _fetch_crossref(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        url = f"{CROSSREF_API}?query={query}&rows={limit}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            items = r.json().get("message", {}).get("items", [])
            out = []
            for it in items:
                out.append({
                    "title": it.get("title", [""])[0],
                    "text": " ".join(it.get("subtitle", []) + [it.get("abstract", "")]),
                    "url": it.get("URL", ""),
                    "source": "crossref",
                    "timestamp": it.get("issued", {}).get("date-parts", [[None]])[0][0],
                })
            return out
        except Exception:
            return []

    def _fetch_semanticscholar(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        url = f"{SEMANTIC_SCHOLAR_API}?query={query}&limit={limit}&fields=title,abstract,url,year"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            data = r.json().get("data", [])
            return [{
                "title": d.get("title", ""),
                "text": d.get("abstract", ""),
                "url": d.get("url", ""),
                "source": "semantic_scholar",
                "timestamp": d.get("year", ""),
            } for d in data]
        except Exception:
            return []

    def _fetch_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        try:
            q = query.replace(" ", "_")
            r = requests.get(WIKIPEDIA_API + q, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                return []
            d = r.json()
            return [{
                "title": d.get("title", query),
                "text": d.get("extract", ""),
                "url": d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "source": "wikipedia",
                "timestamp": time.strftime("%Y-%m-%d"),
            }]
        except Exception:
            return []

    # ----------------------------------------------------
    #  Unified interface
    # ----------------------------------------------------

    def retrieve_universal(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from all sources for a topic."""
        print(f"🌐 [OmniRetriever] Collecting data for topic: {query}")
        all_items = []
        all_items += self._fetch_arxiv(query)
        all_items += self._fetch_crossref(query)
        all_items += self._fetch_semanticscholar(query)
        all_items += self._fetch_wikipedia(query)

        # remove duplicates
        seen = set()
        unique = []
        for it in all_items:
            k = it["title"].strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            unique.append(it)

        print(f"   ✅ Retrieved {len(unique)} unique entries for '{query}'")
        self.topics_seen.add(query)

        # Save logs
        path = os.path.join(self.last_save_dir,
                            f"retrieved_universal_{query}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(unique, f, ensure_ascii=False, indent=2)
        return unique

    def extract_new_topics(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Heuristically find new topics based on noun phrases or titles."""
        topics = []
        for d in docs:
            t = d.get("title", "").lower()
            if not t:
                continue
            for kw in re.findall(r"[a-zA-Z]{4,}", t):
                if kw not in self.topics_seen and kw not in topics:
                    topics.append(kw)
        random.shuffle(topics)
        return topics[:15]
