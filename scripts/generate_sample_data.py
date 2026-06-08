"""Generate realistic synthetic data so the whole project runs OFFLINE.

No API keys, no network: this seeds the local feature store with ~3 years of
plausible Karachi daily weather + AQI, where AQI is driven by the same physical
relationships the model is meant to learn (low wind + high pressure + winter
=> worse air). This makes the offline demo behave like the real thing and lets
the training pipeline actually achieve good metrics.

For real data, skip this and run:  python -m pipelines.backfill_pipeline

Run:  python -m scripts.generate_sample_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.feature_engineering import build_feature_table
from src.feature_store import get_feature_store
from src.utils import get_logger

log = get_logger("sample_data")
RNG = np.random.default_rng(42)


def _synthesize(days: int = 1100) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").normalize()
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    doy = dates.dayofyear.to_numpy()

    # Seasonal weather (Karachi-like): hot summers, mild winters.
    temp = 27 + 7 * np.sin(2 * np.pi * (doy - 110) / 365) + RNG.normal(0, 2, len(dates))
    humidity = 55 + 20 * np.sin(2 * np.pi * (doy - 200) / 365) + RNG.normal(0, 6, len(dates))
    humidity = np.clip(humidity, 15, 95)
    wind = np.clip(3.0 + 1.5 * np.sin(2 * np.pi * (doy - 150) / 365)
                   + RNG.normal(0, 0.8, len(dates)), 0.3, 9)
    pressure = 1009 + 6 * np.cos(2 * np.pi * (doy - 10) / 365) + RNG.normal(0, 2, len(dates))
    precip = np.clip(RNG.gamma(0.3, 3, len(dates)) * (humidity > 70), 0, 60)
    wind_dir = RNG.uniform(0, 360, len(dates))

    # AQI driven by stagnation (low wind, high pressure), winter inversions,
    # dryness, and rain wash-out. Add autocorrelation for realism.
    winter = np.cos(2 * np.pi * (doy - 10) / 365)            # peaks in winter
    base = (90
            + 45 * np.clip(winter, 0, 1)                     # winter smog
            + 60 / (wind + 0.5)                              # stagnation
            + 0.25 * (pressure - 1009)
            - 0.4 * (humidity - 55)
            - 1.5 * precip)
    noise = RNG.normal(0, 6, len(dates))
    aqi = np.zeros(len(dates))
    aqi[0] = max(40, base[0])
    for i in range(1, len(dates)):
        aqi[i] = 0.70 * base[i] + 0.30 * aqi[i - 1] + noise[i]
    aqi = np.clip(aqi, 20, 480)
    pm25 = np.clip(aqi * 0.5 + RNG.normal(0, 3, len(dates)), 5, 350)

    return pd.DataFrame({
        "date": dates, "temperature": temp.round(1), "humidity": humidity.round(1),
        "wind_speed": wind.round(2), "wind_direction": wind_dir.round(0),
        "pressure": pressure.round(1), "precipitation": precip.round(1),
        "pm2_5": pm25.round(1), "aqi": aqi.round(0), "city": CONFIG.city,
    })


def run(days: int = 1100) -> pd.DataFrame:
    raw = _synthesize(days)
    features = build_feature_table(raw)
    store = get_feature_store()
    store.insert(features, group=CONFIG.backend["feature_group"], primary_key="event_date")
    log.info("Seeded %d synthetic feature rows for %s into the feature store.",
             len(features), CONFIG.city)
    return features


if __name__ == "__main__":
    run()
