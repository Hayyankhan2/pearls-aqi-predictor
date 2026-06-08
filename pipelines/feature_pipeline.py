"""Hourly feature pipeline.

Fetches the most recent observed weather + AQI, recomputes features over a short
recent window (so lag/rolling features are correct), and upserts the newest rows
into the feature store. Designed to be cheap and idempotent so it can run every
hour on GitHub Actions.

Run:  python -m pipelines.feature_pipeline
"""
from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.data_sources import aqicn_current, fetch_history
from src.feature_engineering import build_feature_table
from src.feature_store import get_feature_store
from src.utils import get_logger

log = get_logger("feature_pipeline")

# Recompute over a trailing window so lags/rolling means are accurate.
LOOKBACK_DAYS = 20


def run() -> pd.DataFrame:
    end = pd.Timestamp.now(tz=CONFIG.timezone).normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    log.info("Feature pipeline tick for %s (%s -> %s)", CONFIG.city,
             start.date(), end.date())

    raw = fetch_history(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # Enrich the latest row with the live official AQICN reading when available.
    live = aqicn_current()
    if live and live.get("aqi") is not None and not raw.empty:
        log.info("Live AQICN reading: AQI=%s", live["aqi"])
        raw.loc[raw.index[-1], "aqi"] = live["aqi"]
        if live.get("pm2_5") is not None:
            raw.loc[raw.index[-1], "pm2_5"] = live["pm2_5"]

    features = build_feature_table(raw)
    # Only upsert the trailing few days (the rest already exist).
    recent = features.tail(5)

    store = get_feature_store()
    store.insert(recent, group=CONFIG.backend["feature_group"], primary_key="event_date")
    log.info("Upserted %d recent feature rows.", len(recent))
    return recent


if __name__ == "__main__":
    run()
