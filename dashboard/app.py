# dashboard/app.py
from flask import Flask, jsonify, render_template, send_from_directory
import os, json

app = Flask(__name__, template_folder="templates", static_folder="static")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STATE_FILE = os.path.join(DATA_DIR, "dashboard_state.json")
EVOLVE_FILE = os.path.join(DATA_DIR, "evolution_state.json")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state")
def api_state():
    try:
        return jsonify(json.load(open(STATE_FILE)))
    except Exception:
        return jsonify({"error": "state unavailable"}), 200

@app.route("/api/evolution")
def api_evolution():
    try:
        if os.path.exists(EVOLVE_FILE):
            return jsonify(json.load(open(EVOLVE_FILE)))
        return jsonify({"records": {}, "summary": {}})
    except Exception:
        return jsonify({"records": {}, "summary": {}})

@app.route("/graphs/<path:fname>")
def graphs(fname):
    # serve saved KG images
    graph_dir = os.path.join(os.path.dirname(__file__), "..", "visualization", "graphs")
    return send_from_directory(graph_dir, fname)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
