"""Central configuration loader.

Loads ``config.yaml`` and overlays environment variables (from a local ``.env``
if present). Everything in the codebase imports ``CONFIG`` / ``get_config()``
from here so there is a single source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

try:  # optional - only used for local dev convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# Repo root = parent of the ``src`` directory.
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class Secrets:
    """API keys / tokens read from the environment (never hard-coded)."""

    hopsworks_api_key: str = ""
    hopsworks_project: str = ""
    aqicn_token: str = ""
    openweather_api_key: str = ""

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(
            hopsworks_api_key=os.getenv("HOPSWORKS_API_KEY", ""),
            hopsworks_project=os.getenv("HOPSWORKS_PROJECT", ""),
            aqicn_token=os.getenv("AQICN_TOKEN", ""),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
        )


@dataclass
class Config:
    raw: Dict[str, Any]
    secrets: Secrets = field(default_factory=Secrets.from_env)

    # ----- convenience accessors -----
    @property
    def city(self) -> str:
        return self.raw["location"]["city"]

    @property
    def latitude(self) -> float:
        return float(self.raw["location"]["latitude"])

    @property
    def longitude(self) -> float:
        return float(self.raw["location"]["longitude"])

    @property
    def timezone(self) -> str:
        return self.raw["location"]["timezone"]

    @property
    def aqicn_station(self) -> str:
        return self.raw["location"].get("aqicn_station", self.city.lower())

    @property
    def horizon(self) -> int:
        return int(self.raw["project"]["forecast_horizon_days"])

    @property
    def data_sources(self) -> Dict[str, Any]:
        return self.raw["data_sources"]

    @property
    def backend(self) -> Dict[str, Any]:
        return self.raw["backend"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def hazardous_threshold(self) -> float:
        return float(self.training.get("hazardous_aqi_threshold", 150))

    @property
    def hopsworks_project(self) -> str:
        return (
            self.secrets.hopsworks_project
            or self.backend.get("hopsworks_project", "")
        )

    def path(self, *parts: str) -> Path:
        """Resolve a path relative to the repo root, creating parents."""
        p = ROOT.joinpath(*parts)
        return p

    @property
    def local_dir(self) -> Path:
        d = ROOT / self.backend.get("local_dir", "data")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def models_dir(self) -> Path:
        d = ROOT / self.backend.get("models_dir", "models")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolved_backend_mode(self) -> str:
        """Decide whether to use Hopsworks or the local store."""
        mode = self.backend.get("mode", "auto")
        if mode == "auto":
            return "hopsworks" if self.secrets.hopsworks_api_key else "local"
        return mode


@lru_cache(maxsize=1)
def get_config() -> Config:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)


# Eagerly available singleton for ergonomic imports.
CONFIG = get_config()
