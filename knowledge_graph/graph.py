# knowledge_graph/graph.py
"""
Knowledge Graph (Phase B – Step 2)
Robust extraction + co-occurrence fallback so we always get edges.
No external NLP deps.
"""

import re
import json
from collections import defaultdict, Counter
from typing import Dict, List

REL_VERBS = [
    # causal / process
    "causes", "cause", "requires", "require", "produces", "produce",
    "forms", "form", "contains", "contain", "transfers", "transfer",
    "converts", "convert", "leads to", "lead to", "results in", "result in",
    # definitional / equality
    "equals", "equal", "is", "are", "defined as", "is the study of", "relates to", "related to",
    "implies", "imply", "correlates with", "correlate with"
]

STOPWORDS = set("""
a an the of to in on at for from by with as about into through over during including
until against among throughout despite toward upon concerning
is are was were be been being
""".split())

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\+\*/\^]*")

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-\+\*/\^=]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _keep_token(tok: str) -> bool:
    if not tok: return False
    if tok in STOPWORDS: return False
    if tok.isdigit(): return False
    return True


class KnowledgeGraph:
    def __init__(self):
        # subj -> relation -> set(objects)
        self.graph = defaultdict(lambda: defaultdict(set))
        self._edge_count = 0

    # ---------- Core ops ----------
    def add_relation(self, subj: str, rel: str, obj: str):
        subj, rel, obj = subj.strip(), rel.strip(), obj.strip()
        if not subj or not rel or not obj: return
        if subj == obj: return
        self.graph[subj][rel].add(obj)

    def stats(self) -> Dict[str, int]:
        edges = sum(len(v) for rels in self.graph.values() for v in rels.values())
        return {"nodes": len(self.graph), "edges": edges}

    def all_concepts(self) -> List[str]:
        return list(self.graph.keys())

    def get_relations(self, concept: str) -> Dict[str, List[str]]:
        rels = self.graph.get(concept, {})
        return {r: sorted(list(objs)) for r, objs in rels.items()}

    # ---------- Extraction ----------
    def _extract_explicit_relations(self, text: str) -> int:
        """Extract explicit pattern-based relations, return count of edges added."""
        added = 0
        t = _normalize(text)

        # 1) relation verbs: <phrase> <rel> <phrase>
        # allow multiword phrases on both sides
        rel_group = "|".join(map(re.escape, REL_VERBS))
        pat_rel = re.compile(
            rf"\b([a-z0-9][a-z0-9\s\-]{{1,40}}?)\s+(?:{rel_group})\s+([a-z0-9][a-z0-9\s\-]{{1,40}}?)\b"
        )
        for m in pat_rel.finditer(t):
            full = m.group(0)
            # find which rel matched
            rel_match = None
            for rv in REL_VERBS:
                if f" {rv} " in f" {full} ":
                    rel_match = rv
                    break
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            if subj and obj and rel_match:
                self.add_relation(subj, rel_match, obj)
                added += 1

        # 2) equality / equations: x = y z ...  (e.g., f = m a  → f equals m*a)
        pat_eq = re.compile(r"\b([a-z])\s*=\s*([a-z0-9\*\s\+\-\/\^]+)\b")
        for lhs, rhs in pat_eq.findall(t):
            rhs_norm = re.sub(r"\s+", "*", rhs.strip())
            if lhs and rhs_norm:
                self.add_relation(lhs, "equals", rhs_norm)
                added += 1

        # 3) "X of Y" → (“law of motion”, “part of system”)
        pat_of = re.compile(r"\b([a-z0-9][a-z0-9\s\-]{1,40})\s+of\s+([a-z0-9][a-z0-9\s\-]{1,40})\b")
        for a, b in pat_of.findall(t):
            a, b = a.strip(), b.strip()
            if a and b and a != b:
                self.add_relation(a, "of", b)
                added += 1

        return added

    def _cooccurrence_edges(self, text: str, window: int = 6, max_pairs: int = 12) -> int:
        """
        Fallback: connect informative tokens that co-occur in a sliding window.
        Prevents empty graphs when explicit patterns aren't found.
        """
        t = _normalize(text)
        toks = [tok for tok in TOKEN_RE.findall(t) if _keep_token(tok)]
        if len(toks) < 2:
            return 0

        added = 0
        for i in range(len(toks)):
            a = toks[i]
            # connect with tokens within the window
            for j in range(i+1, min(i+window, len(toks))):
                b = toks[j]
                if a == b: continue
                self.add_relation(a, "co-occurs-with", b)
                added += 1
                if added >= max_pairs:
                    return added
        return added

    def build_from_corpus(self, texts: List[str]):
        total_added = 0
        for txt in texts:
            if not txt: continue
            added = self._extract_explicit_relations(txt)
            if added == 0:
                # fallback to ensure we get *some* structure
                added += self._cooccurrence_edges(txt)
            total_added += added
        s = self.stats()
        print(f"🔗 KG built: {s['nodes']} concepts, {s['edges']} edges (added {total_added} edges this pass).")

    # ---------- Persistence ----------
    def save(self, path: str):
        serial = {s: {r: list(objs) for r, objs in rels.items()} for s, rels in self.graph.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serial, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = defaultdict(lambda: defaultdict(set))
        for s, rels in data.items():
            for r, objs in rels.items():
                self.graph[s][r].update(objs)

    # ---------- Preview ----------
    def visualize(self, max_lines: int = 30):
        print("\n🕸️ Knowledge Graph Preview:")
        printed = 0
        for subj, rels in self.graph.items():
            for rel, objs in rels.items():
                for obj in objs:
                    print(f"  {subj} --{rel}--> {obj}")
                    printed += 1
                    if printed >= max_lines:
                        print("  ... (truncated)")
                        return
