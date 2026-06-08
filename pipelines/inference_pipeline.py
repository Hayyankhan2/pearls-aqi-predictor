"""Inference (batch prediction) pipeline.

Loads the registered model + feature list, pulls the next 3 days of forecast
weather, builds the matching feature vector for each future day (using the last
observed AQI as a persistence signal), predicts AQI, attaches category + health
guidance + hazard alerts, and writes the predictions back to the feature store /
a JSON the dashboard reads.

Run:  python -m pipelines.inference_pipeline
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.aqi import categorize, is_hazardous
from src.config import CONFIG
from src.data_sources import fetch_forecast_weather
from src.feature_engineering import (
    FEATURE_COLUMNS, add_time_features, add_weather_features, latest_known_state,
)
from src.feature_store import get_feature_store
from src.model_registry import get_model_registry
from src.utils import get_logger

log = get_logger("inference")


def _build_future_features(forecast_weather: pd.DataFrame, last_state: dict) -> pd.DataFrame:
    """Assemble the model's feature columns for each forecast day."""
    df = forecast_weather.copy()
    df = add_time_features(df)
    df = add_weather_features(df)
    # AQI dynamics are unknown for the future -> seed with the last observed AQI
    # (persistence). This is a standard, explainable baseline signal.
    last_aqi = float(last_state.get("aqi", last_state.get("aqi_lag_1", 100)))
    df["aqi_lag_1"] = last_aqi
    df["aqi_lag_2"] = float(last_state.get("aqi_lag_1", last_aqi))
    df["aqi_lag_3"] = float(last_state.get("aqi_lag_2", last_aqi))
    df["aqi_roll_3"] = float(last_state.get("aqi_roll_3", last_aqi))
    df["aqi_roll_7"] = float(last_state.get("aqi_roll_7", last_aqi))
    df["aqi_change_rate"] = 0.0
    for col in FEATURE_COLUMNS:
        if col not in df:
            df[col] = 0.0
    return df


def _fallback_forecast(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build a naive weather 'forecast' from the mean of the last 7 observed days.

    Keeps the offline demo fully functional when no live forecast is reachable.
    """
    recent = history.sort_values("event_date").tail(7)
    cols = ["temperature", "humidity", "wind_speed", "wind_direction",
            "pressure", "precipitation"]
    means = {c: float(recent[c].mean()) for c in cols if c in recent}
    last_date = pd.to_datetime(recent["event_date"].iloc[-1])
    rows = []
    for i in range(1, horizon + 1):
        row = dict(means)
        row["date"] = last_date + pd.Timedelta(days=i)
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> dict:
    registry = get_model_registry()
    bundle = registry.load()
    if not bundle:
        raise RuntimeError("No model found. Run: python -m pipelines.training_pipeline")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    store = get_feature_store()
    history = store.read(CONFIG.backend["feature_group"])
    if history.empty:
        raise RuntimeError("Feature store empty. Run the backfill pipeline first.")
    last_state = latest_known_state(history)

    horizon = CONFIG.horizon
    log.info("Fetching %d-day weather forecast for %s", horizon, CONFIG.city)
    try:
        fc_weather = fetch_forecast_weather(days=horizon).head(horizon)
        if fc_weather.empty:
            raise RuntimeError("empty forecast")
    except Exception as exc:
        # Offline / API-down fallback: roll recent observed weather forward.
        log.warning("Forecast fetch failed (%s); using recent-climatology fallback.", exc)
        fc_weather = _fallback_forecast(history, horizon)

    future = _build_future_features(fc_weather, last_state)
    for col in feature_columns:
        if col not in future:
            future[col] = 0.0
    preds = np.asarray(model.predict(future[feature_columns]))
    preds = np.clip(preds, 0, 500)

    records = []
    threshold = CONFIG.hazardous_threshold
    for date, aqi in zip(pd.to_datetime(future["date"]), preds):
        cat = categorize(aqi)
        records.append({
            "event_date": date.strftime("%Y-%m-%d"),
            "predicted_aqi": round(float(aqi), 1),
            "category": cat.name,
            "color": cat.color,
            "emoji": cat.emoji,
            "health": cat.health,
            "hazardous": bool(is_hazardous(aqi, threshold)),
            "city": CONFIG.city,
            "predicted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        })

    pred_df = pd.DataFrame(records)
    store.insert(pred_df, group=CONFIG.backend["predictions_group"], primary_key="event_date")

    # Dashboard-friendly JSON snapshot.
    out_path = CONFIG.local_dir / "latest_predictions.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "city": CONFIG.city,
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "best_model": bundle.get("metrics", {}).get("best_model"),
            "predictions": records,
        }, fh, indent=2)

    alerts = [r for r in records if r["hazardous"]]
    if alerts:
        log.warning("HAZARDOUS AQI ALERT for %d day(s): %s", len(alerts),
                    ", ".join(f"{a['event_date']}={a['predicted_aqi']}" for a in alerts))
    log.info("Wrote %d predictions to %s", len(records), out_path.name)
    for r in records:
        log.info("  %s -> AQI %.0f (%s)", r["event_date"], r["predicted_aqi"], r["category"])
    return {"predictions": records, "alerts": alerts}


if __name__ == "__main__":
    run()
