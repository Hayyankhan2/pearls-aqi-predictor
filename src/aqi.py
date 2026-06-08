"""US EPA Air Quality Index helpers.

Converts a PM2.5 concentration (ug/m3) into the US AQI using the official EPA
breakpoint table, and maps an AQI value to its category, colour and health
guidance. Keeping this in one place makes the dashboard and alerts consistent.

Reference: https://www.airnow.gov/aqi/aqi-calculator/
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# (C_low, C_high, I_low, I_high) for PM2.5 (24h avg, ug/m3) -> US AQI
_PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


@dataclass(frozen=True)
class AqiCategory:
    name: str
    color: str          # hex, used by the dashboard
    emoji: str
    health: str         # public-health style guidance


_CATEGORIES = [
    (0, 50, AqiCategory("Good", "#009966", "\U0001F642",
                        "Air quality is satisfactory and poses little or no risk.")),
    (51, 100, AqiCategory("Moderate", "#FFDE33", "\U0001F610",
                          "Acceptable; unusually sensitive people should limit prolonged outdoor exertion.")),
    (101, 150, AqiCategory("Unhealthy for Sensitive Groups", "#FF9933", "\U0001F637",
                           "Children, elderly and people with respiratory issues should reduce outdoor activity.")),
    (151, 200, AqiCategory("Unhealthy", "#CC0033", "\U0001F634",
                           "Everyone may begin to feel effects; sensitive groups should avoid outdoor exertion.")),
    (201, 300, AqiCategory("Very Unhealthy", "#660099", "\u2620\uFE0F",
                           "Health alert: everyone should avoid outdoor activity and wear a mask outside.")),
    (301, 500, AqiCategory("Hazardous", "#7E0023", "\u26A0\uFE0F",
                           "Emergency conditions; stay indoors with air filtration if possible.")),
]


def pm25_to_aqi(pm25: float) -> Optional[float]:
    """Convert a PM2.5 concentration (ug/m3) to US AQI. Returns None if invalid."""
    if pm25 is None:
        return None
    try:
        c = float(pm25)
    except (TypeError, ValueError):
        return None
    if c < 0:
        return None
    c = min(c, 500.4)  # clamp to top of scale
    for c_low, c_high, i_low, i_high in _PM25_BREAKPOINTS:
        if c_low <= c <= c_high:
            return round((i_high - i_low) / (c_high - c_low) * (c - c_low) + i_low)
    return 500.0


def categorize(aqi: float) -> AqiCategory:
    """Map an AQI value to its EPA category."""
    if aqi is None:
        return _CATEGORIES[0][2]
    a = max(0, min(float(aqi), 500))
    for low, high, cat in _CATEGORIES:
        if low <= a <= high:
            return cat
    return _CATEGORIES[-1][2]


def is_hazardous(aqi: float, threshold: float = 150) -> bool:
    """True if AQI is at/above the alert threshold (default: 'Unhealthy')."""
    try:
        return float(aqi) >= float(threshold)
    except (TypeError, ValueError):
        return False


def all_categories() -> List[AqiCategory]:
    return [cat for _, _, cat in _CATEGORIES]
