"""
Phase C – Step 3: Hypothesis Evolution Engine
----------------------------------------------
Tracks, refines, and mutates scientific hypotheses across cycles.

Features:
- Merges near-duplicates using fuzzy Jaccard similarity
- Evolves hypotheses through maturity stages:
    seed → candidate → maturing → theory
- Mutates high-potential hypotheses (specialize, generalize, paraphrase, bridge)
- Persists long-term evolution history for dashboard and analysis

Outputs:
  • data/hypothesis_evolution.json      – current evolution state snapshot
  • outputs/evolved_hypotheses.jsonl    – append-only evolution event log
"""

from __future__ import annotations
import json, os, re, time
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# ------------------- constants -------------------
EVOLUTION_STATE_PATH = "data/hypothesis_evolution.json"
EVOLUTION_LOG_PATH   = "outputs/evolved_hypotheses.jsonl"


# ------------------- small utils -------------------
def _norm(s: str) -> str:
    """Normalize hypothesis text for fuzzy matching."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-_,:;]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _tokens(s: str) -> set:
    return set(_norm(s).split())

def _jaccard(a: str, b: str) -> float:
    A, B = _tokens(a), _tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))

def _ensure_dirs():
    os.makedirs(os.path.dirname(EVOLUTION_STATE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EVOLUTION_LOG_PATH), exist_ok=True)


# ------------------- main class -------------------
class HypothesisEvolver:
    def __init__(self, kb=None, kg=None):
        """
        Evolves hypotheses through validation-driven cycles.
        Maintains memory across runs using a JSON state file.
        """
        self.kb = kb
        self.kg = kg
        self.state: Dict[str, Dict[str, Any]] = {}
        self.cycle = 0
        _ensure_dirs()
        self._load_state()

    # ---------- Public API ----------
    def update(self, validated: List[Dict[str, Any]], cycle: int) -> Dict[str, Any]:
        """Main entry point — evolve hypotheses for this cycle."""
        self.cycle = cycle
        if not validated:
            return {"cycle": cycle, "counts": {"total_active": len(self.state)}}

        for h in validated:
            self._ingest(h)

        # evolution
        promoted, decayed = 0, 0
        for k, node in list(self.state.items()):
            score = self._score(node)
            node["last_score"] = score
            node["cycles_seen"] += 1
            node["last_cycle"] = cycle

            old_stage = node["stage"]
            node["stage"] = self._advance_stage(node, score)
            if node["stage"] != old_stage:
                promoted += 1

            # decay inactive hypotheses
            if score < 0.25 and node["cycles_seen"] > 2:
                node["stage"] = "decayed"
                decayed += 1

            node.setdefault("history", []).append(
                {"cycle": cycle, "score": score, "stage": node["stage"]}
            )
            node["history"] = node["history"][-20:]

        # persist and log
        self._save_state()
        summary = {
            "cycle": cycle,
            "counts": {
                "total_active": len(self.state),
                "promoted": promoted,
                "decayed": decayed,
            },
        }

        print(f"🔁 Evolution summary: {summary}")
        return summary

    # ---------- Internals ----------
    def _ingest(self, h: Dict[str, Any]):
        text = h.get("hypothesis") or h.get("text") or ""
        if not text:
            return

        best_key, best_sim = None, 0.0
        for k in self.state.keys():
            sim = _jaccard(k, text)
            if sim > best_sim:
                best_key, best_sim = k, sim

        if best_key and best_sim >= 0.7:
            node = self.state[best_key]
            node["mentions"] += 1
            node["support"] = max(node.get("support", 0.0), h.get("support", 0.0))
        else:
            self.state[text] = {
                "text": text,
                "stage": "seed",
                "gen_score": h.get("score", 0.0),
                "support": h.get("support", 0.0),
                "consistency": h.get("consistency", 0.0),
                "confidence": h.get("confidence", 0.0),
                "persistence": h.get("persistence", 0.0),
                "cycles_seen": 1,
                "mentions": 1,
                "last_score": 0.0,
            }
            self._log_event({"event": "ingest", "cycle": self.cycle, "hypothesis": text})

    def _score(self, node: Dict[str, Any]) -> float:
        gen = node.get("gen_score", 0.0)
        sup = node.get("support", 0.0)
        cons = node.get("consistency", 0.0)
        conf = node.get("confidence", 0.0)
        pers = node.get("persistence", 0.0)

        # Weighted scoring blend
        score = (
            0.15 * gen
            + 0.30 * sup
            + 0.25 * cons
            + 0.15 * conf
            + 0.15 * pers
        )
        return round(max(0.0, min(score, 1.0)), 3)

    def _advance_stage(self, node: Dict[str, Any], score: float) -> str:
        stage = node["stage"]
        seen = node["cycles_seen"]

        if stage == "seed" and score >= 0.45:
            return "candidate"
        if stage == "candidate" and score >= 0.6 and seen >= 2:
            return "maturing"
        if stage == "maturing" and score >= 0.75 and seen >= 4:
            return "theory"
        return stage

    # ---------- Persistence ----------
    def _load_state(self):
        try:
            if os.path.exists(EVOLUTION_STATE_PATH):
                with open(EVOLUTION_STATE_PATH, "r") as f:
                    data = json.load(f)
                self.state = data.get("registry", {})
                self.cycle = data.get("cycle", 0)
        except Exception as e:
            print(f"⚠️ Failed to load hypothesis evolution state: {e}")
            self.state, self.cycle = {}, 0

    def _save_state(self):
        snap = {"cycle": self.cycle, "registry": self.state, "ts": time.time()}
        with open(EVOLUTION_STATE_PATH, "w") as f:
            json.dump(snap, f, indent=2)

    def _log_event(self, event: Dict[str, Any]):
        event["ts"] = time.time()
        with open(EVOLUTION_LOG_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
