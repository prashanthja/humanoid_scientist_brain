# knowledge_graph/graph.py
"""
Knowledge Graph (Phase B – Step 2)
- Robust, lowercase-friendly extraction (no external NLP deps)
- Handles "X is the study of Y", equations (f = m a), relation verbs,
  "X of Y", and a co-occurrence fallback.
- Persists to disk.
"""

import re
import json
from collections import defaultdict
from typing import Dict, List

REL_VERBS = [
    # causal / process
    "causes", "cause", "requires", "require", "produces", "produce",
    "forms", "form", "contains", "contain", "transfers", "transfer",
    "converts", "convert", "leads to", "lead to", "results in", "result in",
    # definitional / equality
    "equals", "equal", "is", "are", "defined as", "relates to", "related to",
    "implies", "imply", "correlates with", "correlate with"
]

STOPWORDS = set("""
a an the of to in on at for from by with as about into through over during including
until against among throughout despite toward upon concerning
is are was were be been being this that these those and or nor but so yet if then than
it its their his her our your my we you they i who whom whose which what when where why how
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

    # ---------------- Core ops ----------------
    def add_relation(self, subj: str, rel: str, obj: str):
        subj, rel, obj = subj.strip(), rel.strip(), obj.strip()
        if not subj or not rel or not obj: return
        if subj in STOPWORDS or obj in STOPWORDS: return
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

    # ------------- Extraction -------------
    def _extract_study_of(self, t: str) -> int:
        """
        Capture: "<X> is the study of <Y...>"
        -> relation: study_of
        """
        added = 0
        # allow a short phrase for X and a longer phrase for Y
        pat = re.compile(r"\b([a-z0-9][a-z0-9\s\-]{1,40})\s+is\s+the\s+study\s+of\s+([a-z0-9][a-z0-9\s\-\ ,]{1,120})\b")
        for subj, rhs in pat.findall(t):
            subj = subj.strip()
            # split rhs into salient tokens/phrases separated by commas or "and"
            parts = re.split(r"[,\;]| and ", rhs)
            for p in parts:
                p = " ".join(w for w in p.split() if w not in STOPWORDS)
                p = p.strip()
                if len(p) > 1:
                    self.add_relation(subj, "study_of", p)
                    added += 1
        return added

    def _extract_relation_verbs(self, t: str) -> int:
        """
        "<phrase> <rel_verb> <phrase>"
        """
        added = 0
        rel_group = "|".join(map(re.escape, REL_VERBS))
        pat = re.compile(
            rf"\b([a-z0-9][a-z0-9\s\-]{{1,40}}?)\s+(?:{rel_group})\s+([a-z0-9][a-z0-9\s\-]{{1,80}}?)\b"
        )
        for m in pat.finditer(t):
            full = m.group(0)
            # locate the rel verb actually present in the full span
            rel_match = None
            for rv in REL_VERBS:
                if f" {rv} " in f" {full} ":
                    rel_match = rv
                    break
            subj = " ".join(w for w in m.group(1).split() if w not in STOPWORDS).strip()
            obj  = " ".join(w for w in m.group(2).split() if w not in STOPWORDS).strip()
            if subj and obj and rel_match:
                # collapse trivial leftovers like single letters unless eq
                if len(subj) == 1 and rel_match != "equals": continue
                if len(obj) == 1 and rel_match != "equals": continue
                self.add_relation(subj, rel_match, obj)
                added += 1
        return added

    def _extract_equations(self, t: str) -> int:
        """
        equations like: f = m a  -> f equals m*a
        """
        added = 0
        pat = re.compile(r"\b([a-z])\s*=\s*([a-z0-9\*\s\+\-\/\^]+)\b")
        for lhs, rhs in pat.findall(t):
            rhs_norm = re.sub(r"\s+", "*", rhs.strip())
            if lhs and rhs_norm:
                self.add_relation(lhs, "equals", rhs_norm)
                added += 1
        return added

    def _extract_of_relation(self, t: str) -> int:
        """
        "X of Y" -> (X, of, Y)  (e.g., "law of motion")
        """
        added = 0
        pat = re.compile(r"\b([a-z0-9][a-z0-9\s\-]{1,40})\s+of\s+([a-z0-9][a-z0-9\s\-]{1,60})\b")
        for a, b in pat.findall(t):
            a = " ".join(w for w in a.split() if w not in STOPWORDS).strip()
            b = " ".join(w for w in b.split() if w not in STOPWORDS).strip()
            if a and b and a != b:
                self.add_relation(a, "of", b)
                added += 1
        return added

    def _cooccurrence_edges(self, t: str, window: int = 6, max_pairs: int = 12) -> int:
        """
        Fallback: connect informative tokens that co-occur in a sliding window.
        Ensures we always get some structure even when explicit patterns fail.
        """
        added = 0
        toks = [tok for tok in TOKEN_RE.findall(t) if _keep_token(tok)]
        if len(toks) < 2:
            return 0
        for i in range(len(toks)):
            a = toks[i]
            for j in range(i+1, min(i+window, len(toks))):
                b = toks[j]
                if a == b: continue
                self.add_relation(a, "co-occurs-with", b)
                added += 1
                if added >= max_pairs:
                    return added
        return added

    def extract_relations_from_text(self, text: str) -> int:
        if not text:
            return 0
        t = _normalize(text)
        total = 0
        total += self._extract_study_of(t)
        total += self._extract_equations(t)
        total += self._extract_relation_verbs(t)
        total += self._extract_of_relation(t)
        if total == 0:
            total += self._cooccurrence_edges(t)
        return total

    def build_from_corpus(self, texts: List[str]):
        added = 0
        for txt in texts:
            added += self.extract_relations_from_text(txt)
        s = self.stats()
        print(f"🔗 KG built: {s['nodes']} concepts, {s['edges']} edges (added {added} edges this pass).")

    # ------------- Persistence -------------
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
