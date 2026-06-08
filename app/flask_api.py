"""Small Flask API for serving AQI forecasts and model metadata.

Run:
    flask --app app.flask_api run --host 0.0.0.0 --port 8000

The API reads the same feature store, model registry artifacts, and prediction
snapshot as the Streamlit dashboard. It is intentionally lightweight so it can
run in local demos, containers, or a serverless web target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.feature_store import get_feature_store  # noqa: E402

app = Flask(__name__)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _records_from_predictions_store() -> dict:
    df = get_feature_store().read(CONFIG.backend["predictions_group"])
    if df.empty:
        return {}
    df = df.sort_values("event_date").tail(CONFIG.horizon)
    records = df.to_dict(orient="records")
    for rec in records:
        if hasattr(rec.get("event_date"), "strftime"):
            rec["event_date"] = rec["event_date"].strftime("%Y-%m-%d")
    return {
        "city": CONFIG.city,
        "generated_at": records[-1].get("predicted_at") if records else None,
        "best_model": None,
        "predictions": records,
    }


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "city": CONFIG.city,
        "backend": CONFIG.resolved_backend_mode(),
        "horizon_days": CONFIG.horizon,
    })


@app.get("/forecast")
@app.get("/predictions")
def predictions():
    payload = _read_json(CONFIG.local_dir / "latest_predictions.json")
    if not payload:
        payload = _records_from_predictions_store()
    if not payload:
        return jsonify({
            "error": "No predictions found. Run python -m pipelines.inference_pipeline."
        }), 404
    return jsonify(payload)


@app.get("/metrics")
def metrics():
    payload = _read_json(CONFIG.models_dir / "metrics.json")
    if not payload:
        from src.model_registry import get_model_registry

        bundle = get_model_registry().load()
        payload = bundle.get("metrics", {}) if bundle else {}
    if not payload:
        return jsonify({
            "error": "No metrics found. Run python -m pipelines.training_pipeline."
        }), 404
    return jsonify(payload)


@app.get("/history/latest")
def latest_history():
    n = max(1, min(int(request.args.get("n", 30)), 365))
    df = get_feature_store().read(CONFIG.backend["feature_group"])
    if df.empty:
        return jsonify({
            "error": "No feature history found. Run the backfill or sample-data pipeline."
        }), 404
    keep = [
        "event_date", "city", "aqi", "temperature", "humidity",
        "wind_speed", "pressure", "precipitation",
    ]
    cols = [c for c in keep if c in df.columns]
    recent = df.sort_values("event_date").tail(n)[cols]
    rows = recent.where(pd.notna(recent), None).to_dict(orient="records")
    return jsonify({"city": CONFIG.city, "records": rows})


@app.post("/run-inference")
def run_inference():
    from pipelines.inference_pipeline import run

    result = run()
    return jsonify(result)
