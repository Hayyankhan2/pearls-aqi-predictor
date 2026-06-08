"""Exploratory Data Analysis.

Generates a set of figures from the feature store into ``figures/`` to support
the report and viva: AQI over time, monthly seasonality, AQI-vs-weather
relationships, and a correlation heatmap. Run after a backfill.

Run:  python -m notebooks.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CONFIG
from src.feature_store import get_feature_store
from src.utils import get_logger

log = get_logger("eda")
FIG = CONFIG.path("figures")
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def run() -> None:
    df = get_feature_store().read(CONFIG.backend["feature_group"])
    if df.empty:
        raise RuntimeError("No data. Run: python -m pipelines.backfill_pipeline")
    df["date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("date")

    # 1) AQI over time
    plt.figure(figsize=(12, 4))
    plt.plot(df["date"], df["aqi"], lw=1)
    plt.axhline(CONFIG.hazardous_threshold, color="red", ls="--", label="Hazard")
    plt.title(f"{CONFIG.city} - Daily AQI over time"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "01_aqi_timeseries.png", dpi=120); plt.close()

    # 2) Monthly seasonality
    df["month"] = df["date"].dt.month
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="month", y="aqi")
    plt.title(f"{CONFIG.city} - AQI seasonality by month")
    plt.tight_layout(); plt.savefig(FIG / "02_seasonality.png", dpi=120); plt.close()

    # 3) AQI vs weather drivers
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["wind_speed", "temperature", "humidity"]):
        if col in df:
            sns.scatterplot(data=df, x=col, y="aqi", alpha=0.4, ax=ax)
            ax.set_title(f"AQI vs {col}")
    plt.tight_layout(); plt.savefig(FIG / "03_aqi_vs_weather.png", dpi=120); plt.close()

    # 4) Correlation heatmap
    num = df.select_dtypes("number").drop(columns=["month"], errors="ignore")
    plt.figure(figsize=(11, 9))
    sns.heatmap(num.corr(numeric_only=True), cmap="coolwarm", center=0,
                square=False, cbar_kws={"shrink": 0.6})
    plt.title("Feature correlation heatmap")
    plt.tight_layout(); plt.savefig(FIG / "04_correlation_heatmap.png", dpi=120); plt.close()

    log.info("Saved 4 EDA figures to %s", FIG)


if __name__ == "__main__":
    run()
