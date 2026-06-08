"""Feature engineering.

Turns the raw daily weather + pollutant table into the feature matrix used for
training and inference. Features fall into three families:

  1. Time-based   : day-of-year sin/cos, month, day-of-week, weekend, season.
  2. Weather      : temperature, humidity, wind, pressure, precipitation +
                    derived interactions (stagnation index, temp-humidity).
  3. AQI dynamics : prior observed AQI lag values, rolling means, and the
                    AQI *change rate* (yesterday vs the day before).

Each historical row models AQI for that same calendar date from same-date
weather plus AQI values observed before that date. For a 3-day forecast we
apply the same model day-by-day using the forecast weather for each future day
plus the latest observed AQI as a persistence signal - simple to explain and
robust, which matters for a live viva demo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 3]
ROLLING_WINDOWS = [3, 7]

# The exact, ordered list of columns the model consumes. Persisted alongside
# the model so inference always builds the matrix in the same order.
FEATURE_COLUMNS = [
    # time
    "month", "day_of_week", "day_of_year", "is_weekend",
    "doy_sin", "doy_cos", "season",
    # weather
    "temperature", "humidity", "wind_speed", "wind_direction",
    "pressure", "precipitation",
    # derived weather
    "stagnation_index", "temp_humidity",
    # aqi dynamics
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_roll_3", "aqi_roll_7", "aqi_change_rate",
]
TARGET = "target_aqi"


def _season(month: int) -> int:
    # 0 winter, 1 spring, 2 summer, 3 autumn (northern hemisphere)
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[int(month)]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["month"] = d.dt.month
    df["day_of_week"] = d.dt.dayofweek
    df["day_of_year"] = d.dt.dayofyear
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["season"] = df["month"].apply(_season)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Air stagnation rises when wind is low and pressure is high (poor dispersion).
    df["stagnation_index"] = df["pressure"] / (df["wind_speed"].clip(lower=0.1) + 1.0)
    df["temp_humidity"] = df["temperature"] * df["humidity"] / 100.0
    return df


def add_aqi_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    for lag in LAGS:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"aqi_roll_{w}"] = df["aqi"].shift(1).rolling(w, min_periods=1).mean()
    # AQI change rate: relative change from the previous day.
    df["aqi_change_rate"] = (df["aqi"].shift(1) - df["aqi"].shift(2)) / (
        df["aqi"].shift(2).abs() + 1.0
    )
    return df


def build_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Full feature table from a raw daily weather+AQI frame. Includes target."""
    df = raw.sort_values("date").reset_index(drop=True)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_aqi_dynamics(df)
    # Supervised target = AQI for this date. Weather is same-date; AQI dynamics
    # only use values from previous dates, so there is no target leakage.
    df[TARGET] = df["aqi"]
    # A stable string primary key for the feature store.
    df["event_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def training_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaNs in features/target (warm-up lags + final target row)."""
    cols = FEATURE_COLUMNS + [TARGET]
    out = features.dropna(subset=cols).reset_index(drop=True)
    return out


def latest_known_state(features: pd.DataFrame) -> dict:
    """Return the most recent fully-populated feature row as a dict (for inference)."""
    valid = features.dropna(subset=[c for c in FEATURE_COLUMNS if c.startswith("aqi_")])
    row = valid.iloc[-1]
    return row.to_dict()
