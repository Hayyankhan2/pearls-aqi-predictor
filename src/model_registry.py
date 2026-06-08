"""Model registry abstraction.

Mirrors the feature store: a ``HopsworksModelRegistry`` for the serverless setup
and a ``LocalModelRegistry`` (joblib + JSON metrics on disk) so the project runs
without any account. The training pipeline ``save``s the best model + metadata;
the inference pipeline and dashboard ``load`` it back.

The payload we persist:
  * model.joblib        - the fitted estimator (or a Keras wrapper)
  * feature_columns.json- ordered feature list (so inference matches training)
  * metrics.json        - RMSE / MAE / R2 for every candidate + the winner
  * scaler.joblib       - StandardScaler used by linear / NN models
"""
from __future__ import annotations

import abc
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from .config import CONFIG
from .utils import get_logger

log = get_logger("model_registry")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


class ModelRegistry(abc.ABC):
    @abc.abstractmethod
    def save(self, payload: Dict[str, Any], metrics: Dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def load(self) -> Optional[Dict[str, Any]]: ...


class LocalModelRegistry(ModelRegistry):
    def __init__(self, models_dir: Optional[Path] = None):
        self.dir = Path(models_dir or CONFIG.models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, payload: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        joblib.dump(payload["model"], self.dir / "model.joblib")
        if payload.get("scaler") is not None:
            joblib.dump(payload["scaler"], self.dir / "scaler.joblib")
        elif (self.dir / "scaler.joblib").exists():
            (self.dir / "scaler.joblib").unlink()
        with open(self.dir / "feature_columns.json", "w", encoding="utf-8") as fh:
            json.dump(payload["feature_columns"], fh, indent=2)
        with open(self.dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(_json_safe(metrics), fh, indent=2)
        log.info("[local] saved model '%s' to %s", metrics.get("best_model"), self.dir)

    def load(self) -> Optional[Dict[str, Any]]:
        model_path = self.dir / "model.joblib"
        if not model_path.exists():
            return None
        scaler_path = self.dir / "scaler.joblib"
        with open(self.dir / "feature_columns.json", encoding="utf-8") as fh:
            feature_columns = json.load(fh)
        metrics = {}
        if (self.dir / "metrics.json").exists():
            with open(self.dir / "metrics.json", encoding="utf-8") as fh:
                metrics = json.load(fh)
        return {
            "model": joblib.load(model_path),
            "scaler": joblib.load(scaler_path) if scaler_path.exists() else None,
            "feature_columns": feature_columns,
            "metrics": metrics,
        }


class HopsworksModelRegistry(ModelRegistry):
    """Stores the same artifacts in the Hopsworks Model Registry."""

    def __init__(self):
        import hopsworks

        self._project = hopsworks.login(
            api_key_value=CONFIG.secrets.hopsworks_api_key,
            project=CONFIG.hopsworks_project or None,
        )
        self._mr = self._project.get_model_registry()
        self._name = CONFIG.backend.get("model_name", "aqi_predictor")
        # Local staging dir to assemble artifacts before upload.
        self._local = LocalModelRegistry()

    def save(self, payload: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        # Stage artifacts locally, then upload the directory to the registry.
        self._local.save(payload, metrics)
        eval_metrics = {
            k: float(metrics["models"][metrics["best_model"]][k])
            for k in ("rmse", "mae", "r2")
        }
        model = self._mr.python.create_model(
            name=self._name,
            metrics=eval_metrics,
            description=f"AQI 3-day predictor for {CONFIG.city} ({metrics.get('best_model')})",
        )
        model.save(str(self._local.dir))
        log.info("[hopsworks] registered model '%s' v%s", self._name, model.version)

    def load(self) -> Optional[Dict[str, Any]]:
        try:
            model = self._mr.get_best_model(self._name, "rmse", "min")
            download_dir = Path(model.download())
            return LocalModelRegistry(download_dir).load()
        except Exception as exc:  # pragma: no cover - network
            log.warning("Hopsworks model load failed: %s; trying local.", exc)
            return LocalModelRegistry().load()


def get_model_registry() -> ModelRegistry:
    mode = CONFIG.resolved_backend_mode()
    if mode == "hopsworks":
        try:
            return HopsworksModelRegistry()
        except Exception as exc:
            if CONFIG.backend.get("mode", "auto") == "auto":
                log.warning("Falling back to LocalModelRegistry (Hopsworks error: %s)", exc)
                return LocalModelRegistry()
            raise RuntimeError(
                "Hopsworks model registry is required but could not be initialized. "
                "Set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT in this environment."
            ) from exc
    return LocalModelRegistry()
