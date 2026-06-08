"""Unit tests for the core, deterministic logic (no network needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi import categorize, is_hazardous, pm25_to_aqi  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS, TARGET, build_feature_table, training_frame,
)


def test_pm25_to_aqi_breakpoints():
    # 12.0 ug/m3 sits at the top of the 'Good' band -> AQI 50.
    assert pm25_to_aqi(12.0) == 50
    assert pm25_to_aqi(0.0) == 0
    assert pm25_to_aqi(35.5) == 101  # bottom of 'Unhealthy for Sensitive'
    assert pm25_to_aqi(-5) is None


def test_categories_and_alerts():
    assert categorize(30).name == "Good"
    assert categorize(175).name == "Unhealthy"
    assert is_hazardous(160, threshold=150) is True
    assert is_hazardous(120, threshold=150) is False


def _toy_raw(n=60):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": dates,
        "temperature": rng.normal(25, 3, n),
        "humidity": rng.uniform(30, 80, n),
        "wind_speed": rng.uniform(1, 6, n),
        "wind_direction": rng.uniform(0, 360, n),
        "pressure": rng.normal(1010, 3, n),
        "precipitation": rng.gamma(0.3, 2, n),
        "aqi": rng.uniform(50, 200, n),
    })


def test_feature_table_has_all_columns():
    feats = build_feature_table(_toy_raw())
    for col in FEATURE_COLUMNS + [TARGET, "event_date"]:
        assert col in feats.columns, f"missing {col}"


def test_training_frame_drops_nans():
    feats = build_feature_table(_toy_raw())
    clean = training_frame(feats)
    assert not clean[FEATURE_COLUMNS + [TARGET]].isna().any().any()
    assert len(clean) > 0
