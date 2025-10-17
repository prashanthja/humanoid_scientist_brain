from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/system_state.json")

def load_state():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    return {"cycle": 0, "hypotheses": [], "metrics": {}, "kg_stats": {}}

@app.route("/")
def index():
    state = load_state()
    return render_template("index.html", state=state)

@app.route("/api/state")
def api_state():
    return jsonify(load_state())

if __name__ == "__main__":
    app.run(debug=True, port=5000)
