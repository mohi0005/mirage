"""
app.py — Flask application for LeakGuard.

Routes:
    GET  /              → Main UI
    GET  /api/health    → Health check
    POST /api/review    → Analyze code for ML pipeline issues
    POST /api/prove     → Run the "Prove it" leaky vs. correct experiment
    GET  /metrics       → Evaluation metrics page
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory

from analyzer.parser import parse_source
from analyzer.flow import FlowTracker
from analyzer.rules import run_all_rules
from analyzer.report import build_report
from llm import enhance_findings
from prove import run_prove_experiment

# ─── App setup ───────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024  # 200 KB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Vulnerable example for the "Load Example" button ────────────────

VULNERABLE_EXAMPLE = '''\
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("patient_data.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Preprocess — but BEFORE splitting!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # BUG: fit on ALL data

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2
)

# Train and evaluate
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
'''


# ─── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "LeakGuard",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
    })


@app.route("/api/review", methods=["POST"])
def review():
    """Analyze code for ML pipeline issues.

    Accepts:
        - JSON body: {"code": "...python code..."}
        - Multipart file upload: .py or .ipynb file

    Returns:
        JSON report with score, grade, summary, and findings.
    """
    try:
        code = None
        filename = "<paste>"

        # Handle file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename:
                filename = file.filename
                code = file.read().decode("utf-8", errors="replace")
        # Handle JSON body
        elif request.is_json:
            data = request.get_json(silent=True)
            if data and "code" in data:
                code = data["code"]
        # Handle form data
        elif request.form.get("code"):
            code = request.form["code"]

        if not code or not code.strip():
            return jsonify({"error": "No code provided. Send JSON {code} or upload a file."}), 400

        if len(code) > 200 * 1024:
            return jsonify({"error": "File too large. Maximum size is 200 KB."}), 400

        # ─── Analysis pipeline ───
        # 1. Parse
        source, cell_map, tree = parse_source(code, filename)

        if tree is None:
            return jsonify({
                "error": "Could not parse the code. Check for syntax errors.",
                "score": None,
                "grade": None,
                "findings": [],
            }), 400

        # 2. Track variable flow
        tracker = FlowTracker()
        tracker.analyze(tree)

        # 3. Run detection rules
        findings = run_all_rules(tree, source, tracker, cell_map)

        # 4. Build report
        report = build_report(findings, cell_map)

        # 5. Enhance with LLM (if available)
        report["findings"], llm_used = enhance_findings(report["findings"])
        report["meta"] = {
            "llm_used": llm_used,
            "filename": filename,
            "is_notebook": filename.endswith(".ipynb"),
        }

        return jsonify(report)

    except Exception as e:
        logger.exception("Error during code review")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/prove", methods=["POST"])
def prove():
    """Run the 'Prove it' experiment."""
    try:
        result = run_prove_experiment()
        return jsonify(result)
    except Exception as e:
        logger.exception("Error during prove experiment")
        return jsonify({"error": f"Experiment failed: {str(e)}"}), 500


@app.route("/api/example")
def example():
    """Return the vulnerable example code."""
    return jsonify({"code": VULNERABLE_EXAMPLE})


@app.route("/metrics")
def metrics():
    """Render the evaluation metrics page."""
    results_path = Path(__file__).parent / "eval" / "results.json"
    results = None
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
        except Exception:
            pass
    return render_template("metrics.html", results=results)


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
