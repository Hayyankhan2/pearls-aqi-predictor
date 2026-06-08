"""Raw data acquisition layer.

Fetches weather + pollutant data from external APIs. Three providers are
supported and combined per the config:

  * Open-Meteo  (default) - free, no key. Provides historical archives AND a
                            3-day forecast for both weather and pollutants,
                            which is exactly what a multi-day AQI forecast needs.
  * AQICN                  - live, official city AQI reading (needs a free token).
  * OpenWeather            - alternate weather + air-pollution provider (free key).

Every function returns a tidy pandas DataFrame with a ``date`` column (daily)
so the rest of the pipeline is provider-agnostic.
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from .aqi import pm25_to_aqi
from .config import CONFIG
from .utils import get_logger

log = get_logger("data_sources")

_OPEN_METEO_WEATHER_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_WEATHER_FORECAST = "https://api.open-meteo.com/v1/forecast"
_OPEN_METEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
_AQICN_FEED = "https://api.waqi.info/feed"
_OPENWEATHER_AQ = "https://api.openweathermap.org/data/2.5/air_pollution"

_DAILY_WEATHER = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "surface_pressure_mean",
    "precipitation_sum",
]
_HOURLY_WEATHER = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",
]
_HOURLY_AQ = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
              "sulphur_dioxide", "ozone", "us_aqi"]


def _get(url: str, params: dict, timeout: int = 60) -> dict:
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 2:
                break
            time.sleep(1.5 * (attempt + 1))
    raise last_exc


# ============================================================
# Open-Meteo
# ============================================================

def openmeteo_weather(start: str, end: str, forecast: bool = False,
                      forecast_days: int = 3) -> pd.DataFrame:
    """Daily weather aggregates from Open-Meteo (archive or forecast)."""
    if forecast:
        params = {
            "latitude": CONFIG.latitude, "longitude": CONFIG.longitude,
            "daily": ",".join(_DAILY_WEATHER),
            "timezone": CONFIG.timezone, "forecast_days": forecast_days,
            "wind_speed_unit": "ms",
        }
        data = _get(_OPEN_METEO_WEATHER_FORECAST, params)
    else:
        params = {
            "latitude": CONFIG.latitude, "longitude": CONFIG.longitude,
            "start_date": start, "end_date": end,
            "daily": ",".join(_DAILY_WEATHER),
            "timezone": CONFIG.timezone, "wind_speed_unit": "ms",
        }
        data = _get(_OPEN_METEO_WEATHER_ARCHIVE, params)

    d = data["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "temperature": d["temperature_2m_mean"],
        "humidity": d["relative_humidity_2m_mean"],
        "wind_speed": d["wind_speed_10m_max"],
        "wind_direction": d["wind_direction_10m_dominant"],
        "pressure": d["surface_pressure_mean"],
        "precipitation": d["precipitation_sum"],
    })
    return df


def openmeteo_air_quality(start: str, end: str, forecast: bool = False,
                          forecast_days: int = 3) -> pd.DataFrame:
    """Daily pollutant aggregates + US AQI from Open-Meteo air-quality API."""
    params = {
        "latitude": CONFIG.latitude, "longitude": CONFIG.longitude,
        "hourly": ",".join(_HOURLY_AQ), "timezone": CONFIG.timezone,
    }
    if forecast:
        params["forecast_days"] = forecast_days
    else:
        params["start_date"] = start
        params["end_date"] = end
    data = _get(_OPEN_METEO_AQ, params)

    h = data["hourly"]
    hourly = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    for col in _HOURLY_AQ:
        hourly[col] = h.get(col)
    hourly["date"] = hourly["time"].dt.normalize()
    daily = hourly.groupby("date", as_index=False).mean(numeric_only=True)
    daily = daily.drop(columns=["time"], errors="ignore")
    daily = daily.rename(columns={
        "pm2_5": "pm2_5", "pm10": "pm10", "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2",
        "ozone": "o3", "us_aqi": "aqi",
    })
    # Fall back to EPA conversion if the provider AQI is missing.
    if "aqi" not in daily or daily["aqi"].isna().all():
        daily["aqi"] = daily["pm2_5"].apply(pm25_to_aqi)
    return daily


# ============================================================
# AQICN (live official reading)
# ============================================================

def aqicn_current() -> Optional[dict]:
    """Live AQI for the configured station. Returns None if unavailable."""
    token = CONFIG.secrets.aqicn_token
    if not token:
        log.warning("AQICN_TOKEN not set - skipping live AQICN reading.")
        return None
    url = f"{_AQICN_FEED}/{CONFIG.aqicn_station}/"
    try:
        data = _get(url, {"token": token})
        if data.get("status") != "ok":
            log.warning("AQICN returned status=%s", data.get("status"))
            return None
        d = data["data"]
        iaqi = d.get("iaqi", {})
        return {
            "aqi": d.get("aqi"),
            "pm2_5": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "time": d.get("time", {}).get("s"),
        }
    except Exception as exc:  # pragma: no cover - network
        log.warning("AQICN fetch failed: %s", exc)
        return None


# ============================================================
# OpenWeather (alternate provider)
# ============================================================

def openweather_air_pollution(start_ts: int, end_ts: int) -> pd.DataFrame:
    """Historical pollutant data from OpenWeather. Returns daily means."""
    key = CONFIG.secrets.openweather_api_key
    if not key:
        log.warning("OPENWEATHER_API_KEY not set - skipping OpenWeather.")
        return pd.DataFrame()
    params = {
        "lat": CONFIG.latitude, "lon": CONFIG.longitude,
        "start": start_ts, "end": end_ts, "appid": key,
    }
    data = _get(f"{_OPENWEATHER_AQ}/history", params)
    rows = []
    for item in data.get("list", []):
        comp = item.get("components", {})
        rows.append({
            "time": pd.to_datetime(item["dt"], unit="s"),
            "pm2_5": comp.get("pm2_5"), "pm10": comp.get("pm10"),
            "co": comp.get("co"), "no2": comp.get("no2"),
            "so2": comp.get("so2"), "o3": comp.get("o3"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = df["time"].dt.normalize()
    daily = df.groupby("date", as_index=False).mean(numeric_only=True)
    daily["aqi"] = daily["pm2_5"].apply(pm25_to_aqi)
    return daily.drop(columns=["time"], errors="ignore")


# ============================================================
# Unified fetch helpers used by the pipelines
# ============================================================

def fetch_history(start: str, end: str) -> pd.DataFrame:
    """Merge historical weather + pollutants into one daily table."""
    log.info("Fetching historical weather %s -> %s", start, end)
    weather = openmeteo_weather(start, end)
    log.info("Fetching historical air quality %s -> %s", start, end)
    aq = openmeteo_air_quality(start, end)
    merged = pd.merge(weather, aq, on="date", how="inner").sort_values("date")
    merged["city"] = CONFIG.city
    log.info("Merged history: %d daily rows", len(merged))
    return merged.reset_index(drop=True)


def fetch_forecast_weather(days: int = 3) -> pd.DataFrame:
    """Weather forecast for the next ``days`` days (drives the AQI prediction)."""
    df = openmeteo_weather("", "", forecast=True, forecast_days=days + 1)
    today = pd.Timestamp.now(tz=CONFIG.timezone).normalize().tz_localize(None)
    future = df[pd.to_datetime(df["date"]) > today].head(days).reset_index(drop=True)
    return future if len(future) >= days else df.tail(days).reset_index(drop=True)


def fetch_latest_observation() -> pd.DataFrame:
    """Most recent complete day of observed weather + AQI (for the feature pipeline)."""
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=7)
    df = fetch_history(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return df.tail(1).reset_index(drop=True)
