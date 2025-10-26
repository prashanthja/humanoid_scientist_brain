from flask import Flask, render_template, jsonify, send_from_directory
import json, os, time

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dashboard_state.json")
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "..", "visualization", "graphs")
HIST_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "training_history.jsonl")
VALIDATED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "validated_hypotheses.json")

def _read_json_safely(path, fallback):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/state")
def state():
    """Serve the latest dashboard state + latest KG visualization"""
    state = _read_json_safely(DATA_PATH, {})
    # Find latest graph image
    latest_graph = None
    try:
        graphs = [f for f in os.listdir(GRAPH_DIR) if f.endswith(".png")]
        if graphs:
            graphs.sort(key=lambda f: os.path.getmtime(os.path.join(GRAPH_DIR, f)), reverse=True)
            latest_graph = "/graphs/" + graphs[0]
    except Exception:
        latest_graph = None

    state["latest_graph"] = latest_graph
    state["timestamp_readable"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(state)

@app.route("/history")
def history():
    """
    Return parsed per-epoch metrics from logs/training_history.jsonl as:
    {
      "epochs": [1,2,...],
      "loss": [...],
      "similarity": [...],
      "accuracy": [...],
      "vocab": [...],
      "device": "cpu|cuda|mps"
    }
    """
    epochs, loss, sim, acc, vocab = [], [], [], [], []
    device = None
    if os.path.exists(HIST_PATH):
        try:
            with open(HIST_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    epochs.append(rec.get("epoch"))
                    loss.append(rec.get("loss"))
                    sim.append(rec.get("similarity"))
                    acc.append(rec.get("accuracy"))
                    vocab.append(rec.get("vocab_size"))
                    device = rec.get("device", device)
        except Exception:
            pass
    return jsonify({
        "epochs": epochs,
        "loss": loss,
        "similarity": sim,
        "accuracy": acc,
        "vocab": vocab,
        "device": device
    })

@app.route("/validated")
def validated():
    """
    Summarize validated hypotheses:
    {
      "total": N,
      "promoted": K,
      "avg_confidence": x.x,
      "items": [... up to 50 latest ...]
    }
    """
    data = _read_json_safely(VALIDATED_PATH, {"hypotheses": {}})
    # file format: {"hypotheses": {"<hypo_str>":{"ema":..,"count":..}, ...}}
    # You may also enrich this later with full rows; for now, show summary.
    items = []
    try:
        # If HypothesisValidator.validate() returns rows, you can store them separately.
        # Here we compute a compact summary from the EMA persistence store.
        for h, rec in data.get("hypotheses", {}).items():
            items.append({
                "hypothesis": h,
                "persistence": round(rec.get("ema", 0.0), 3),
                "observations": rec.get("count", 0)
            })
        items.sort(key=lambda x: x["persistence"], reverse=True)
    except Exception:
        items = []

    # If you want “promoted” counts, you can compute from a separate log later.
    # For now, estimate promoted as persistence >= 0.7
    promoted = sum(1 for x in items if x["persistence"] >= 0.7)
    avg_conf = round(sum(x["persistence"] for x in items) / len(items), 3) if items else 0.0

    return jsonify({
        "total": len(items),
        "promoted": promoted,
        "avg_confidence": avg_conf,
        "items": items[:50]
    })

@app.route("/graphs/<path:filename>")
def serve_graph(filename):
    """Serve graph images dynamically"""
    return send_from_directory(GRAPH_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
