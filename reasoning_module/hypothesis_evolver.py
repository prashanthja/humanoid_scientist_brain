# reasoning_module/hypothesis_evolver.py
"""
Phase C – Step 3: Hypothesis Evolution Engine
- Tracks hypotheses over cycles (memory)
- Merges near-duplicates (fuzzy)
- Mutates promising ideas (bridge/specialize/generalize/paraphrase)
- Scores/elevates ideas through stages: seed → candidate → maturing → theory
- Exports evolution state for dashboard

Outputs:
  data/hypothesis_evolution.json      # current state snapshot
  outputs/evolved_hypotheses.jsonl    # append-only evolution log
"""

from __future__ import annotations
import json, os, re, time
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

EVOLUTION_STATE_PATH = "data/hypothesis_evolution.json"
EVOLUTION_LOG_PATH   = "outputs/evolved_hypotheses.jsonl"

# --- tiny utils --------------------------------------------------------------

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-_,:;]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _tokens(s: str) -> set:
    return set(_norm(s).split())

def _jaccard(a: str, b: str) -> float:
    A, B = _tokens(a), _tokens(b)
    if not A or not B: return 0.0
    return len(A & B) / float(len(A | B))

def _ensure_dirs():
    os.makedirs(os.path.dirname(EVOLUTION_STATE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EVOLUTION_LOG_PATH),   exist_ok=True)

# --- main engine -------------------------------------------------------------

class HypothesisEvolver:
    """
    Keeps a registry of hypotheses and pushes them along maturity stages.
    Works with raw generator output and/or validator output.
    """
    def __init__(self, kg=None, encoder=None, kb=None):
        self.kg = kg
        self.encoder = encoder
        self.kb = kb
        self.registry: Dict[str, Dict[str, Any]] = {}   # key = canonical text
        self.cycle = 0
        _ensure_dirs()
        self._load_state()

    # ---------- public API ---------------------------------------------------

    def step(self, hypotheses: List[Dict[str, Any]], cycle: int) -> List[Dict[str, Any]]:
        """
        Ingest hypotheses for this cycle, merge, score, mutate, and advance stages.
        Returns top evolved candidates this cycle (for display/dashboard).
        """
        self.cycle = cycle
        if not hypotheses:
            return []

        # 1) normalize + merge new into registry
        for h in hypotheses:
            self._ingest(h)

        # 2) evolve: score, stage transitions, optional mutations
        evolved_now: List[Dict[str, Any]] = []
        for key, node in list(self.registry.items()):
            score = self._score(node)
            node["last_score"] = score
            node["cycles_seen"] += 1
            node["last_cycle"] = cycle

            prev_stage = node["stage"]
            node["stage"] = self._advance_stage(node, score)

            # generate mutations for promising candidates
            muts = []
            if node["stage"] in ("candidate", "maturing") and score >= 0.6:
                muts.extend(self._mutate(node["text"]))

            for mtxt in muts:
                evolved_now.append(self._ingest({"hypothesis": mtxt, "type": "mutation", "score": score * 0.95}))

            # keep short history
            node.setdefault("history", []).append({
                "cycle": cycle, "score": score, "stage": node["stage"]
            })
            node["history"] = node["history"][-20:]

            # Append event to log (only if stage changed or new)
            if prev_stage != node["stage"]:
                self._log_event({
                    "ts": time.time(), "cycle": cycle, "event": "stage_change",
                    "from": prev_stage, "to": node["stage"], "hypothesis": node["text"],
                    "score": round(score, 3)
                })

        # 3) small prune to keep registry lean
        self._prune()

        # 4) persist
        self._save_state()

        # return a small set of the most interesting right now
        return self.top_k(10)

    def top_k(self, k: int = 10) -> List[Dict[str, Any]]:
        items = sorted(self.registry.values(), key=lambda n: n.get("last_score", 0.0), reverse=True)
        return items[:k]

    def export_for_dashboard(self) -> Dict[str, Any]:
        stages = defaultdict(int)
        for n in self.registry.values():
            stages[n["stage"]] += 1

        return {
            "evolution_cycle": self.cycle,
            "counts_by_stage": dict(stages),
            "top_hypotheses": [
                {
                    "text": n["text"],
                    "stage": n["stage"],
                    "score": round(n.get("last_score", 0.0), 3),
                    "cycles_seen": n.get("cycles_seen", 0),
                }
                for n in self.top_k(8)
            ],
        }

    # ---------- internal helpers --------------------------------------------

    def _ingest(self, h: Dict[str, Any]) -> Dict[str, Any]:
        txt = h.get("hypothesis") or h.get("text") or ""
        txt = txt.strip()
        if not txt:
            return {"text": ""}

        # fuzzy merge target
        best_key, best_sim = None, 0.0
        for k, node in self.registry.items():
            sim = _jaccard(k, txt)
            if sim > best_sim:
                best_key, best_sim = k, sim

        if best_key and best_sim >= 0.7:
            # merge into existing
            node = self.registry[best_key]
            node["mentions"] += 1
            node["sources"].append({
                "cycle": self.cycle,
                "raw": h,
            })
            # keep slightly better score
            base = node.get("gen_score", 0.0)
            node["gen_score"] = max(base, float(h.get("score", 0.0)))
            return node
        else:
            key = txt
            node = {
                "text": txt,
                "stage": "seed",
                "gen_score": float(h.get("score", 0.0)),
                "last_score": 0.0,
                "cycles_seen": 0,
                "mentions": 1,
                "last_cycle": self.cycle,
                "sources": [{"cycle": self.cycle, "raw": h}],
                "history": [],
            }
            self.registry[key] = node
            self._log_event({"ts": time.time(), "cycle": self.cycle, "event": "ingest", "hypothesis": txt})
            return node

    def _score(self, node: Dict[str, Any]) -> float:
        """
        Combine generator score + validation stats if present in sources.
        """
        gen = float(node.get("gen_score", 0.0))
        # aggregate validation metrics when available
        sup = cons = conf = pers = 0.0
        n_val = 0
        for s in node.get("sources", []):
            raw = s.get("raw", {})
            if "support" in raw or "consistency" in raw or "confidence" in raw or "persistence" in raw:
                sup += float(raw.get("support", 0.0))
                cons += float(raw.get("consistency", 0.0))
                conf += float(raw.get("confidence", 0.0))
                pers += float(raw.get("persistence", 0.0))
                n_val += 1
        if n_val:
            sup /= n_val; cons /= n_val; conf /= n_val; pers /= n_val

        # heuristic blend
        # more weight to support/consistency/persistence if present
        if n_val:
            score = 0.15*gen + 0.30*sup + 0.25*cons + 0.15*conf + 0.15*pers
        else:
            score = gen
        return max(0.0, min(1.0, score))

    def _advance_stage(self, node: Dict[str, Any], score: float) -> str:
        stage = node["stage"]
        seen  = node.get("cycles_seen", 0)
        mentions = node.get("mentions", 1)

        # simple rules:
        if stage == "seed" and (score >= 0.45 and mentions >= 1):
            return "candidate"
        if stage == "candidate" and (score >= 0.6 and seen >= 2):
            return "maturing"
        if stage == "maturing" and (score >= 0.75 and seen >= 4):
            return "theory"
        return stage

    def _mutate(self, text: str) -> List[str]:
        """
        Deterministic, lightweight mutations to explore neighborhood.
        """
        t = _norm(text)
        muts = set()

        # 1) paraphrase (very light)
        muts.add(text.replace(" --related_to--> ", " relates to ").replace("--", "—"))

        # 2) specialize: add a context cue if missing
        for ctx in ["in condensed matter", "in quantum regimes", "for non-linear systems"]:
            if ctx not in t:
                muts.add(f"{text} ({ctx})")
                break

        # 3) generalize: remove trailing qualifiers like parentheses
        g = re.sub(r"\s*\([^)]*\)$", "", text).strip()
        if g and g != text:
            muts.add(g)

        # 4) bridge: try to connect two entities if pattern is present
        parts = re.split(r"\s--[a-z_]+-->\s", text)
        if len(parts) == 2:
            a, b = parts
            muts.add(f"{b} --related_to--> {a}")

        return list(muts)[:3]

    def _prune(self, max_keep: int = 400):
        """
        Keep registry bounded; drop low-score stale seeds.
        """
        if len(self.registry) <= max_keep:
            return
        items = sorted(self.registry.items(), key=lambda kv: kv[1].get("last_score", 0.0))
        drop = len(self.registry) - max_keep
        for k, _ in items[:drop]:
            self._log_event({"ts": time.time(), "cycle": self.cycle, "event": "prune", "hypothesis": k})
            self.registry.pop(k, None)

    # ---------- persistence --------------------------------------------------

    def _load_state(self):
        try:
            if os.path.exists(EVOLUTION_STATE_PATH):
                with open(EVOLUTION_STATE_PATH, "r") as f:
                    data = json.load(f)
                self.registry = {k: v for k, v in data.get("registry", {}).items()}
                self.cycle = int(data.get("cycle", 0))
        except Exception:
            # start clean on any error
            self.registry, self.cycle = {}, 0

    def _save_state(self):
        snap = {
            "cycle": self.cycle,
            "registry": self.registry,
            "ts": time.time(),
        }
        with open(EVOLUTION_STATE_PATH, "w") as f:
            json.dump(snap, f, indent=2)

    def _log_event(self, event: Dict[str, Any]):
        with open(EVOLUTION_LOG_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
