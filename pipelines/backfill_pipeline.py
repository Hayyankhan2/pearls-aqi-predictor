"""Historical data backfill.

Runs the feature pipeline over a long historical window once, to create the
training dataset. After this, the hourly feature pipeline only appends new days.

Run:  python -m pipelines.backfill_pipeline
      python -m pipelines.backfill_pipeline --start 2021-01-01
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import CONFIG
from src.data_sources import fetch_history
from src.feature_engineering import build_feature_table
from src.feature_store import get_feature_store
from src.utils import get_logger

log = get_logger("backfill")


def run(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    start = start or CONFIG.data_sources.get("backfill_start_date", "2022-01-01")
    end = end or (
        pd.Timestamp.now(tz=CONFIG.timezone).normalize() - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    log.info("Backfilling %s from %s to %s", CONFIG.city, start, end)

    raw = fetch_history(start, end)
    features = build_feature_table(raw)

    store = get_feature_store()
    group = CONFIG.backend["feature_group"]
    store.insert(features, group=group, primary_key="event_date")
    log.info("Backfill complete: %d feature rows stored in '%s'.", len(features), group)
    return features


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backfill historical AQI features.")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD start date")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD end date")
    args = ap.parse_args()
    run(args.start, args.end)
