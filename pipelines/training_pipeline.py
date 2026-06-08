"""Training pipeline.

Reads the feature table from the feature store and trains several candidate
models, from a simple statistical baseline up to a deep network:

  * Ridge Regression            (linear baseline, scaled)
  * Random Forest               (bagged trees)
  * XGBoost / LightGBM          (gradient boosting; optional)
  * TensorFlow MLP              (small dense net; optional)
  * Weighted ensemble           (inverse-RMSE weighted blend of the above)

Evaluation uses a chronological holdout (last N days) PLUS TimeSeriesSplit
cross-validation, reporting RMSE / MAE / R2 for each. The best model by holdout
RMSE is registered in the model registry, and SHAP feature importances are
computed for the best tree model.

Run:  python -m pipelines.training_pipeline
      python -m pipelines.training_pipeline --quick   (skip TF + SHAP, faster)
"""
from __future__ import annotations

import argparse
import json
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.feature_engineering import FEATURE_COLUMNS, TARGET, training_frame
from src.feature_store import get_feature_store
from src.model_registry import get_model_registry
from src.utils import get_logger

log = get_logger("training")
SEED = int(CONFIG.training.get("random_seed", 42))
np.random.seed(SEED)


def _metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_candidates(quick: bool):
    """Return {name: estimator}. Boosting/TF added only if importable + enabled."""
    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=SEED)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            n_jobs=-1, random_state=SEED,
        ),
    }

    if CONFIG.training.get("enable_boosting", True):
        try:
            import xgboost as xgb
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1,
            )
        except Exception as exc:
            log.warning("XGBoost unavailable: %s", exc)
        try:
            import lightgbm as lgb
            models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1,
                verbose=-1,
            )
        except Exception as exc:
            log.warning("LightGBM unavailable: %s", exc)

    if not quick and CONFIG.training.get("enable_tensorflow", True):
        tf_model = _build_tf_model()
        if tf_model is not None:
            models["TensorFlow MLP"] = tf_model
    return models


def _build_tf_model():
    """A small Keras MLP wrapped to look like an sklearn regressor."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        log.warning("TensorFlow unavailable, skipping deep model: %s", exc)
        return None

    class KerasMLP:
        """Minimal sklearn-style wrapper (fit/predict) around a Keras MLP."""

        def __init__(self, n_features: int):
            self.scaler = StandardScaler()
            tf.random.set_seed(SEED)
            self.model = keras.Sequential([
                keras.layers.Input(shape=(n_features,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1),
            ])
            self.model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

        def fit(self, X, y):
            Xs = self.scaler.fit_transform(np.asarray(X))
            self.model.fit(Xs, np.asarray(y), epochs=80, batch_size=32,
                           verbose=0, validation_split=0.1)
            return self

        def predict(self, X):
            Xs = self.scaler.transform(np.asarray(X))
            return self.model.predict(Xs, verbose=0).ravel()

    return KerasMLP(len(FEATURE_COLUMNS))


def _cross_val_rmse(model, X, y, splits: int) -> float:
    if len(X) < 4:
        return float("nan")
    splits = min(splits, max(2, len(X) - 2))
    tscv = TimeSeriesSplit(n_splits=splits)
    scores = []
    for tr, va in tscv.split(X):
        try:
            m = _clone_like(model)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[va])
            scores.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
        except Exception:
            return float("nan")
    return float(np.mean(scores)) if scores else float("nan")


def _clone_like(model):
    """Clone sklearn estimators; rebuild the Keras wrapper fresh."""
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        return model.__class__(len(FEATURE_COLUMNS))  # KerasMLP


def _can_register_model(name: str, model) -> bool:
    """Return True when the fitted model can be persisted by the registry."""
    try:
        pickle.dumps(model)
        return True
    except Exception as exc:
        log.warning("%s trained but is not registry-serializable: %s", name, exc)
        return False


def run(quick: bool = False, auto_backfill: bool = False) -> dict:
    store = get_feature_store()
    raw = store.read(CONFIG.backend["feature_group"])
    if raw.empty:
        data = pd.DataFrame()
    else:
        data = training_frame(raw).sort_values("event_date").reset_index(drop=True)

    min_rows = int(CONFIG.training.get("test_size_days", 60)) + 40
    if auto_backfill and len(data) < min_rows:
        log.info(
            "Feature store has %d trainable rows; running historical backfill first.",
            len(data),
        )
        from pipelines.backfill_pipeline import run as backfill_run

        backfill_run()
        raw = store.read(CONFIG.backend["feature_group"])
        data = training_frame(raw).sort_values("event_date").reset_index(drop=True)

    if data.empty:
        raise RuntimeError(
            "Feature store is empty. Run the backfill pipeline first: "
            "python -m pipelines.backfill_pipeline"
        )
    log.info("Training on %d rows, %d features", len(data), len(FEATURE_COLUMNS))

    X, y = data[FEATURE_COLUMNS], data[TARGET]
    test_days = int(CONFIG.training.get("test_size_days", 60))
    split = max(len(data) - test_days, int(len(data) * 0.6))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    candidates = _build_candidates(quick)
    results: Dict[str, Dict] = {}
    fitted = {}
    holdout_preds = {}

    for name, model in candidates.items():
        log.info("Training %s ...", name)
        model.fit(X_tr, y_tr)
        pred = np.asarray(model.predict(X_te))
        m = _metrics(y_te, pred)
        m["cv_rmse"] = _cross_val_rmse(model, X_tr, y_tr,
                                       int(CONFIG.training.get("cv_splits", 5)))
        results[name] = m
        fitted[name] = model
        holdout_preds[name] = pred
        log.info("  %s -> RMSE=%.2f MAE=%.2f R2=%.3f",
                 name, m["rmse"], m["mae"], m["r2"])

    # ---- Weighted ensemble (inverse-RMSE weights over base learners) ----
    base = {n: p for n, p in holdout_preds.items() if n != "TensorFlow MLP" or not quick}
    weights = {n: 1.0 / max(results[n]["rmse"], 1e-6) for n in base}
    wsum = sum(weights.values())
    ens_pred = sum(weights[n] / wsum * holdout_preds[n] for n in base)
    results["Weighted Ensemble"] = _metrics(y_te, ens_pred)
    results["Weighted Ensemble"]["cv_rmse"] = float("nan")
    results["Weighted Ensemble"]["_weights"] = {n: round(weights[n] / wsum, 3) for n in base}
    log.info("  Weighted Ensemble -> RMSE=%.2f R2=%.3f",
             results["Weighted Ensemble"]["rmse"], results["Weighted Ensemble"]["r2"])

    # ---- Pick the winner (lowest holdout RMSE among serializable single models) ----
    ranked_single = sorted(fitted, key=lambda n: results[n]["rmse"])
    best_by_rmse = ranked_single[0]
    best_name = next((n for n in ranked_single if _can_register_model(n, fitted[n])), None)
    if best_name is None:
        raise RuntimeError("No trained model could be serialized for the model registry.")
    if best_name != best_by_rmse:
        log.warning(
            "Lowest-RMSE model was %s, but registering %s because it can be persisted.",
            best_by_rmse, best_name,
        )
    best_model = fitted[best_name]
    log.info("Best registered model: %s (RMSE=%.2f)", best_name, results[best_name]["rmse"])

    # ---- SHAP feature importance for the best tree model ----
    shap_importance = _shap_importance(best_name, best_model, X_te, quick)

    # ---- Persist comparison + winner ----
    comparison = pd.DataFrame(
        {n: {k: results[n][k] for k in ("rmse", "mae", "r2", "cv_rmse")}
         for n in results}
    ).T.reset_index().rename(columns={"index": "model"})
    comparison.to_csv(CONFIG.models_dir / "model_comparison.csv", index=False)

    metrics_payload = {
        "city": CONFIG.city,
        "best_model": best_name,
        "best_model_by_rmse": best_by_rmse,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "models": {n: {k: results[n][k] for k in ("rmse", "mae", "r2", "cv_rmse")}
                   for n in results},
        "ensemble_weights": results["Weighted Ensemble"].get("_weights", {}),
        "shap_importance": shap_importance,
    }

    registry = get_model_registry()
    scaler = None  # tree models need none; Ridge/TF carry their own internally
    registry.save(
        payload={"model": best_model, "scaler": scaler,
                 "feature_columns": FEATURE_COLUMNS},
        metrics=metrics_payload,
    )
    log.info("Training complete. Winner: %s", best_name)
    return metrics_payload


def _shap_importance(name, model, X_te, quick):
    if quick or name in ("Ridge Regression", "TensorFlow MLP"):
        return {}
    try:
        import shap
        sample = X_te.iloc[: min(200, len(X_te))]
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(sample)
        mean_abs = np.abs(vals).mean(axis=0)
        imp = {f: float(v) for f, v in zip(FEATURE_COLUMNS, mean_abs)}
        return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
    except Exception as exc:
        log.warning("SHAP importance skipped: %s", exc)
        return {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train + register AQI models.")
    ap.add_argument("--quick", action="store_true", help="Skip TF + SHAP for speed")
    ap.add_argument(
        "--auto-backfill",
        action="store_true",
        help="Run historical backfill automatically if the feature store is empty",
    )
    args = ap.parse_args()
    out = run(quick=args.quick, auto_backfill=args.auto_backfill)
    print(json.dumps({"best_model": out["best_model"],
                      "metrics": out["models"][out["best_model"]]}, indent=2))
