# retrieval/bm25.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import re

_WORD_RE = re.compile(r"\b[a-zA-Z0-9_+\-]{2,}\b")

def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())

@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    doc_freq: Dict[str, int] | None = None
    postings: Dict[str, List[Tuple[int, int]]] | None = None  # term -> [(doc_id, tf)]
    doc_len: List[int] | None = None
    n_docs: int = 0
    avgdl: float = 0.0

    def build(self, docs: List[str]) -> "BM25Index":
        self.doc_freq = {}
        self.postings = {}
        self.doc_len = []
        self.n_docs = len(docs)

        for doc_id, text in enumerate(docs):
            toks = tokenize(text)
            self.doc_len.append(len(toks))
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            # doc freq
            for term in tf.keys():
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

            # postings with term frequency
            for term, f in tf.items():
                self.postings.setdefault(term, []).append((doc_id, f))

        self.avgdl = (sum(self.doc_len) / max(1, self.n_docs)) if self.n_docs else 0.0
        return self

    def idf(self, term: str) -> float:
        df = (self.doc_freq or {}).get(term, 0)
        if df == 0:
            return 0.0
        # Okapi BM25 idf (+1 keeps it stable/positive-ish)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score_query(self, query: str) -> Dict[int, float]:
        q_terms = tokenize(query)
        if not q_terms or self.n_docs == 0:
            return {}

        if not self.postings or not self.doc_len:
            return {}

        scores: Dict[int, float] = {}
        for term in q_terms:
            plist = self.postings.get(term)
            if not plist:
                continue

            idf = self.idf(term)
            for doc_id, tf in plist:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-8, self.avgdl)))
                s = idf * (tf * (self.k1 + 1.0)) / max(1e-8, denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + s

        return scores
