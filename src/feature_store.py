"""Feature store abstraction.

The project codes against a tiny ``FeatureStore`` interface with two backends:

  * ``HopsworksFeatureStore`` - the real serverless feature store. Used when a
    ``HOPSWORKS_API_KEY`` is configured.
  * ``LocalFeatureStore``     - parquet files on disk. Zero signup, so the whole
    project runs end-to-end on a laptop / in a viva. This is the graceful
    fallback and what powers the offline demo.

Both expose the same ``insert`` / ``read`` / ``read_latest`` methods, so the
pipelines never care which one is active.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CONFIG
from .utils import get_logger

log = get_logger("feature_store")


class FeatureStore(abc.ABC):
    @abc.abstractmethod
    def insert(self, df: pd.DataFrame, group: str, primary_key: str) -> None: ...

    @abc.abstractmethod
    def read(self, group: str) -> pd.DataFrame: ...

    def read_latest(self, group: str, n: int = 1) -> pd.DataFrame:
        df = self.read(group)
        if df.empty:
            return df
        return df.sort_values("event_date").tail(n).reset_index(drop=True)


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


class LocalFeatureStore(FeatureStore):
    """On-disk store. Uses parquet when an engine is available, else CSV.

    The CSV fallback means the project runs even on a minimal Python install
    (no pyarrow) - handy for a constrained demo machine.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base = Path(base_dir or CONFIG.local_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.ext = "parquet" if _parquet_available() else "csv"

    def _path(self, group: str) -> Path:
        # Prefer an existing file in either format, else use the preferred ext.
        for ext in ("parquet", "csv"):
            p = self.base / f"{group}.{ext}"
            if p.exists():
                return p
        return self.base / f"{group}.{self.ext}"

    def _read_file(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _write_file(self, df: pd.DataFrame, path: Path) -> None:
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

    def insert(self, df: pd.DataFrame, group: str, primary_key: str = "event_date") -> None:
        path = self._path(group)
        if path.exists():
            existing = self._read_file(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[primary_key], keep="last")
        else:
            combined = df.copy()
        combined = combined.sort_values(primary_key).reset_index(drop=True)
        self._write_file(combined, path)
        log.info("[local] wrote %d rows to %s", len(combined), path.name)

    def read(self, group: str) -> pd.DataFrame:
        path = self._path(group)
        if not path.exists():
            return pd.DataFrame()
        return self._read_file(path)


class HopsworksFeatureStore(FeatureStore):
    """Hopsworks-backed feature store using feature groups."""

    def __init__(self):
        import hopsworks  # imported lazily so local mode needs no install

        self._project = hopsworks.login(
            api_key_value=CONFIG.secrets.hopsworks_api_key,
            project=CONFIG.hopsworks_project or None,
        )
        self._fs = self._project.get_feature_store()
        self._version = int(CONFIG.backend.get("feature_group_version", 1))
        log.info("Connected to Hopsworks project '%s'", self._project.name)

    def _feature_group(self, group: str, primary_key: str):
        return self._fs.get_or_create_feature_group(
            name=group,
            version=self._version,
            primary_key=[primary_key],
            event_time="event_date",
            description=f"Pearls AQI Predictor - {group}",
            online_enabled=False,
        )

    def insert(self, df: pd.DataFrame, group: str, primary_key: str = "event_date") -> None:
        fg = self._feature_group(group, primary_key)
        payload = df.copy()
        if "event_date" in payload:
            payload["event_date"] = pd.to_datetime(payload["event_date"])
        fg.insert(payload, write_options={"wait_for_job": True})
        log.info("[hopsworks] inserted %d rows into '%s'", len(df), group)

    def read(self, group: str) -> pd.DataFrame:
        try:
            fg = self._fs.get_feature_group(
                name=group, version=int(CONFIG.backend.get("feature_group_version", 1))
            )
            return fg.read()
        except Exception as exc:  # pragma: no cover - network
            log.warning("Hopsworks read failed for '%s': %s", group, exc)
            return pd.DataFrame()


def get_feature_store() -> FeatureStore:
    """Factory: pick the backend based on config + available credentials."""
    mode = CONFIG.resolved_backend_mode()
    if mode == "hopsworks":
        try:
            return HopsworksFeatureStore()
        except Exception as exc:
            log.warning("Falling back to LocalFeatureStore (Hopsworks error: %s)", exc)
            return LocalFeatureStore()
    return LocalFeatureStore()
