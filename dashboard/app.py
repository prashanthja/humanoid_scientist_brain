# dashboard/app.py
from flask import Flask, render_template, jsonify, request
import json, os, glob, time, sys, threading

app  = Flask(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REPORTS_DIR = os.path.join(ROOT, "outputs", "discovery_reports")
KG_PATH     = os.path.join(ROOT, "knowledge_graph", "graph.json")
HYP_JSONL   = os.path.join(ROOT, "outputs", "hypotheses.jsonl")
CHUNK_DB    = os.path.join(ROOT, "knowledge_base", "knowledge.db")

_sim_cache: dict = {}
_sim_lock = threading.Lock()


def _read_json(path, fallback=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback if fallback is not None else {}


def _chunk_count():
    try:
        import sqlite3
        conn = sqlite3.connect(CHUNK_DB)
        n = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        conn.close()
        return n
    except Exception:
        return 0


def _load_reports(limit=20):
    reports = []
    try:
        files = sorted(glob.glob(os.path.join(REPORTS_DIR,"*.json")),
                       key=os.path.getmtime, reverse=True)
        for f in files[:limit]:
            try:
                r = _read_json(f)
                r["_file"] = os.path.basename(f)
                reports.append(r)
            except Exception:
                pass
    except Exception:
        pass
    return reports


def _load_hypotheses(limit=30):
    hyps, seen = [], set()
    try:
        if not os.path.exists(HYP_JSONL): return hyps
        with open(HYP_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                h = json.loads(line.strip())
                k = (h.get("hypothesis","") or "").strip().lower()
                if k and k not in seen:
                    seen.add(k); hyps.append(h)
                    if len(hyps) >= limit: break
            except Exception:
                pass
    except Exception:
        pass
    return hyps


def _kg_stats():
    data = _read_json(KG_PATH, {})
    if not isinstance(data, dict):
        return {"nodes": 0, "edges": 0, "relations": []}
    nodes, edges, relations = set(data.keys()), 0, []
    for subj, rels in data.items():
        if not isinstance(rels, dict): continue
        for rel, objs in rels.items():
            if not isinstance(objs, list): objs = [objs]
            for obj in objs:
                nodes.add(str(obj)); edges += 1
                relations.append({"subject": subj, "relation": rel, "object": str(obj)})
    return {"nodes": len(nodes), "edges": edges, "relations": relations[:120]}


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    reports = _load_reports(50)
    kg      = _kg_stats()
    verdicts = {"supported":0,"partially_supported":0,"inconclusive":0,"contradicted":0}
    for r in reports:
        v = r.get("proposal_verdict","unknown")
        if v in verdicts: verdicts[v] += 1
    confs = [float(r.get("proposal_confidence",0)) for r in reports]
    return jsonify({
        "chunk_count":    _chunk_count(),
        "report_count":   len(reports),
        "kg_nodes":       kg["nodes"],
        "kg_edges":       kg["edges"],
        "verdicts":       verdicts,
        "avg_confidence": round(sum(confs)/len(confs),3) if confs else 0,
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/reports")
def api_reports():
    reports = _load_reports(20)
    return jsonify({"reports": [{
        "query":             r.get("query",""),
        "verdict":           r.get("proposal_verdict","unknown"),
        "confidence":        round(float(r.get("proposal_confidence",0)),3),
        "domain":            r.get("domain","unknown"),
        "evidence_count":    r.get("evidence_count",0),
        "supported_count":   r.get("supported_count",0),
        "contradicted_count":r.get("contradicted_count",0),
        "knowledge_gaps":    r.get("knowledge_gaps",[]),
        "timestamp":         r.get("timestamp",""),
        "file":              r.get("_file",""),
    } for r in reports]})


@app.route("/api/report/<filename>")
def api_report_detail(filename):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error":"not found"}), 404
    return jsonify(_read_json(path, {}))


@app.route("/api/kg")
def api_kg():
    return jsonify(_kg_stats())


@app.route("/api/hypotheses")
def api_hypotheses():
    hyps = _load_hypotheses(30)
    return jsonify({"hypotheses": hyps, "count": len(hyps)})


# ── SWMS ─────────────────────────────────────────────────

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        body  = request.get_json(force=True) or {}
        query = (body.get("query") or "How will transformer efficiency research evolve?").strip()

        with _sim_lock:
            if query in _sim_cache:
                return jsonify(_sim_cache[query])

        from knowledge_graph.graph import KnowledgeGraph
        from simulation_module.swms import SWMS

        kg = KnowledgeGraph()
        kg.load()

        hypotheses = []
        try:
            from reasoning_module.hypothesis_generator import HypothesisGenerator
            hypotheses = HypothesisGenerator(kg=kg).generate(top_n=10)
        except Exception:
            pass

        swms   = SWMS(kg=kg, n_simulations=300, n_steps=8)
        result = swms.simulate(hypotheses=hypotheses, query=query, save=False)

        slim = {
            "query":               result.get("query", query),
            "avg_field_score":     result.get("avg_field_score", 0),
            "summary":             result.get("summary", ""),
            "focus_nodes":         result.get("focus_nodes", []),
            "dominant_methods":    result.get("dominant_methods", [])[:6],
            "rising_nodes":        result.get("rising_nodes", [])[:5],
            "contradiction_risks": result.get("contradiction_risks", [])[:4],
            "field_trajectory":    result.get("field_trajectory", []),
            "roadmap":             result.get("roadmap", []),
            "best_experiments":    result.get("best_experiments", [])[:5],
            "hypothesis_outcomes": result.get("hypothesis_outcomes", [])[:5],
            "n_simulations":       result.get("n_simulations", 0),
            "data_source":         result.get("data_source", "domain"),
        }

        with _sim_lock:
            _sim_cache[query] = slim

        return jsonify(slim)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/simulate/clear_cache", methods=["POST"])
def api_clear_sim_cache():
    with _sim_lock:
        _sim_cache.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)