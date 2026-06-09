"""Pearls AQI Predictor - Streamlit dashboard (polished UI).

Reads the registered model, the feature-store history, and the latest batch
predictions, then renders a modern operational dashboard:
  * a hero header with live backend/model status
  * a KPI strip (latest AQI, live AQICN, best model, history depth)
  * a Forecast tab: hazard alert, colour-coded 3-day cards, current-condition gauges
  * a Trends & EDA tab: history line + seasonality / weather / distribution
  * a Model evidence tab: RMSE/MAE/R2 leaderboard + SHAP feature importance
  * an About tab: AQI glossary, architecture, disclaimer

The data-loading logic is intentionally unchanged from the working app; only the
presentation layer is upgraded. Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Bring Streamlit Cloud secrets into the environment (no-op locally) ------
for _secret_name in (
    "HOPSWORKS_API_KEY",
    "HOPSWORKS_PROJECT",
    "AQICN_TOKEN",
    "OPENWEATHER_API_KEY",
):
    try:
        if _secret_name in st.secrets and not os.getenv(_secret_name):
            os.environ[_secret_name] = str(st.secrets[_secret_name])
    except Exception:
        pass

# Make ``src`` importable when run via ``streamlit run``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi import all_categories, categorize  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.data_sources import aqicn_current  # noqa: E402

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="\U0001F32B\uFE0F",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- Styling -------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px; }
    .hero {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 52%, #8b5cf6 100%);
        border-radius: 22px; padding: 30px 34px; color: #fff; margin-bottom: 18px;
        box-shadow: 0 14px 34px rgba(99,102,241,.28);
    }
    .hero h1 { font-size: 2.05rem; font-weight: 800; margin: 0; letter-spacing:-.5px; }
    .hero p  { opacity: .94; margin: 8px 0 0; font-size: 1rem; }
    .badge {
        display:inline-block; padding:5px 13px; border-radius:999px; font-size:.78rem;
        font-weight:600; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.38);
        margin:12px 8px 0 0;
    }
    .kpi {
        background:#fff; border:1px solid #eef0f5; border-radius:16px; padding:16px 18px;
        box-shadow:0 2px 12px rgba(15,23,42,.05);
    }
    .kpi .label { color:#64748b; font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
    .kpi .value { font-size:1.7rem; font-weight:800; color:#0f172a; line-height:1.1; margin-top:5px; }
    .kpi .sub   { color:#94a3b8; font-size:.76rem; margin-top:3px; }
    .fcard { border-radius:18px; padding:18px 18px 16px; box-shadow:0 8px 20px rgba(15,23,42,.14); height:100%; }
    .fcard .d { font-size:.84rem; font-weight:700; opacity:.92; }
    .fcard .v { font-size:2.7rem; font-weight:800; line-height:1; margin:8px 0 4px; }
    .fcard .c { font-size:.92rem; font-weight:700; }
    .fcard .h { font-size:.78rem; opacity:.95; margin-top:9px; line-height:1.38; }
    .alert {
        background:linear-gradient(135deg,#ef4444,#b91c1c); color:#fff; padding:15px 20px;
        border-radius:14px; font-weight:600; margin:4px 0 16px; box-shadow:0 8px 20px rgba(239,68,68,.3);
    }
    .section-title { font-size:1.12rem; font-weight:700; color:#0f172a; margin:20px 0 10px; }
    #MainMenu, footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_data(ttl=600)
def load_predictions() -> dict:
    path = CONFIG.local_dir / "latest_predictions.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from src.feature_store import get_feature_store

    df = get_feature_store().read(CONFIG.backend["predictions_group"])
    if df.empty:
        return {}
    df = df.sort_values("event_date").tail(CONFIG.horizon)
    records = df.to_dict(orient="records")
    for rec in records:
        if hasattr(rec.get("event_date"), "strftime"):
            rec["event_date"] = rec["event_date"].strftime("%Y-%m-%d")
    return {
        "city": CONFIG.city,
        "generated_at": records[-1].get("predicted_at") if records else None,
        "best_model": None,
        "predictions": records,
    }


@st.cache_data(ttl=600)
def load_history() -> pd.DataFrame:
    from src.feature_store import get_feature_store

    df = get_feature_store().read(CONFIG.backend["feature_group"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["event_date"])
    return df


@st.cache_data(ttl=600)
def load_metrics() -> dict:
    path = CONFIG.models_dir / "metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        from src.model_registry import get_model_registry

        bundle = get_model_registry().load()
        return bundle.get("metrics", {}) if bundle else {}
    except Exception:
        return {}


def gauge(value: float, title: str) -> go.Figure:
    cat = categorize(value)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": cat.color},
                "steps": [
                    {"range": [0, 50], "color": "#d8f3dc"},
                    {"range": [50, 100], "color": "#fff3b0"},
                    {"range": [100, 150], "color": "#ffd6a5"},
                    {"range": [150, 200], "color": "#ffadad"},
                    {"range": [200, 300], "color": "#e0c3fc"},
                    {"range": [300, 500], "color": "#d4a373"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=46, b=10, l=20, r=20))
    return fig


# ----------------------------- Load data -----------------------------------
backend = CONFIG.resolved_backend_mode()
metrics = load_metrics()
preds = load_predictions()
history = load_history()
best = metrics.get("best_model") or "\u2014"

latest_aqi = None
if not history.empty and "aqi" in history:
    latest_row = history.sort_values("date").iloc[-1]
    try:
        latest_aqi = float(latest_row.get("aqi"))
    except (TypeError, ValueError):
        latest_aqi = None

live = aqicn_current()
live_aqi = float(live["aqi"]) if (live and live.get("aqi") is not None) else None
n_points = 0 if history.empty else len(history)

# ----------------------------- Hero ----------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>\U0001F32B\uFE0F Pearls AQI Predictor</h1>
      <p>3-day air-quality forecast for <b>{CONFIG.city}</b> &mdash; an automated, serverless ML pipeline.</p>
      <div>
        <span class="badge">Backend: {backend}</span>
        <span class="badge">Horizon: {CONFIG.horizon} days</span>
        <span class="badge">Active model: {best}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- KPI strip -----------------------------------
k1, k2, k3, k4 = st.columns(4)


def _kpi(col, label: str, value: str, sub: str) -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


_kpi(k1, "Latest observed AQI",
     f"{latest_aqi:.0f}" if latest_aqi is not None else "\u2014",
     categorize(latest_aqi).name if latest_aqi is not None else "no data yet")
_kpi(k2, "Live AQICN",
     f"{live_aqi:.0f}" if live_aqi is not None else "\u2014",
     "real-time reading" if live_aqi is not None else "set AQICN_TOKEN")
_kpi(k3, "Best model", best,
     f"{len(metrics.get('models', {}))} models compared" if metrics.get("models") else "train to populate")
_kpi(k4, "History depth", f"{n_points}", "daily records")

st.write("")

# ----------------------------- Tabs ----------------------------------------
t_forecast, t_trends, t_model, t_about = st.tabs(
    ["\U0001F52E Forecast", "\U0001F4C8 Trends & EDA", "\U0001F3C6 Model evidence", "\u2139\uFE0F About"]
)

# ------ Forecast tab ------
with t_forecast:
    records = preds.get("predictions", [])
    alerts = [r for r in records if r.get("hazardous")]
    if alerts:
        days = ", ".join(f"{a['event_date']} (AQI {a['predicted_aqi']:.0f})" for a in alerts)
        st.markdown(
            f'<div class="alert">\u26A0\uFE0F Hazardous air quality predicted on {days}. '
            "Limit outdoor exposure and consider a mask or air purifier.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="section-title">Next {CONFIG.horizon}-day forecast</div>', unsafe_allow_html=True)
    if records:
        cols = st.columns(len(records))
        for col, rec in zip(cols, records):
            aqi_val = float(rec["predicted_aqi"])
            cat = categorize(aqi_val)
            light = rec["category"] in {"Good", "Moderate", "Unhealthy for Sensitive Groups"}
            txt = "#111827" if light else "#ffffff"
            col.markdown(
                f'<div class="fcard" style="background:{cat.color};color:{txt};">'
                f'<div class="d">{rec["event_date"]}</div>'
                f'<div class="v">{aqi_val:.0f}</div>'
                f'<div class="c">{rec.get("emoji", "")} {rec["category"]}</div>'
                f'<div class="h">{rec.get("health", "")}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No predictions yet. Run: python -m pipelines.inference_pipeline")

    st.markdown('<div class="section-title">Current conditions</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    if latest_aqi is not None:
        g1.plotly_chart(gauge(latest_aqi, "Latest observed AQI"), use_container_width=True)
    else:
        g1.info("No history yet. Run the feature/backfill pipeline.")
    if live_aqi is not None:
        g2.plotly_chart(gauge(live_aqi, "Live AQICN reading"), use_container_width=True)
    else:
        g2.info("Live AQICN reading unavailable (set AQICN_TOKEN to enable).")

# ------ Trends & EDA tab ------
with t_trends:
    if history.empty:
        st.info("No history loaded yet.")
    else:
        st.markdown('<div class="section-title">Historical AQI trend</div>', unsafe_allow_html=True)
        fig = px.line(history.sort_values("date"), x="date", y="aqi",
                      labels={"aqi": "AQI", "date": "Date"})
        fig.add_hline(y=CONFIG.hazardous_threshold, line_dash="dash", line_color="red",
                      annotation_text="Hazard threshold")
        fig.update_layout(height=340, margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Seasonality", "AQI vs weather", "Distribution"])
        with tab1:
            h = history.copy()
            h["month"] = h["date"].dt.month_name().str[:3]
            order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            monthly = h.groupby("month")["aqi"].mean().reindex(order).reset_index()
            st.plotly_chart(px.bar(monthly, x="month", y="aqi", title="Average AQI by month"),
                            use_container_width=True)
        with tab2:
            if "wind_speed" in history:
                st.plotly_chart(
                    px.scatter(history, x="wind_speed", y="aqi",
                               trendline="ols" if _has_statsmodels() else None,
                               labels={"wind_speed": "Wind speed (m/s)"},
                               title="AQI vs wind speed (dispersion effect)"),
                    use_container_width=True)
            else:
                st.info("Wind data not present in history.")
        with tab3:
            st.plotly_chart(px.histogram(history, x="aqi", nbins=40, title="AQI distribution"),
                            use_container_width=True)

# ------ Model evidence tab ------
with t_model:
    if metrics.get("models"):
        st.markdown('<div class="section-title">Model leaderboard (held-out test set)</div>',
                    unsafe_allow_html=True)
        rows = [{"Model": n, "RMSE": round(m["rmse"], 2), "MAE": round(m["mae"], 2),
                 "R\u00b2": round(m["r2"], 3)} for n, m in metrics["models"].items()]
        board = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
        st.dataframe(board, use_container_width=True, hide_index=True)
        st.caption("RMSE / MAE: average prediction error in AQI points (lower is better). "
                   "R\u00b2: variance explained (closer to 1 is better).")
    else:
        st.info("Train the model to populate the leaderboard: python -m pipelines.training_pipeline")

    if metrics.get("shap_importance"):
        st.markdown('<div class="section-title">Why the model predicts what it does (SHAP)</div>',
                    unsafe_allow_html=True)
        imp = metrics["shap_importance"]
        idf = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())}).head(12)
        st.plotly_chart(
            px.bar(idf.sort_values("importance"), x="importance", y="feature", orientation="h",
                   title="Mean |SHAP value| (feature impact)"),
            use_container_width=True)

# ------ About tab ------
with t_about:
    st.markdown('<div class="section-title">AQI scale & health guidance</div>', unsafe_allow_html=True)
    for cat in all_categories():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:5px 0;">'
            f'<span style="width:14px;height:14px;border-radius:4px;background:{cat.color};'
            f'display:inline-block;"></span><b>{cat.emoji} {cat.name}</b> &mdash; '
            f'<span style="color:#475569;">{cat.health}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        "**How it works:** External APIs &rarr; feature pipeline &rarr; feature store &rarr; "
        "training pipeline &rarr; model registry &rarr; batch inference &rarr; this dashboard. "
        "The hourly feature pipeline and daily training pipeline run on GitHub Actions, so the "
        "system is fully serverless."
    )
    st.markdown(
        "**Glossary:** **AQI** = US EPA Air Quality Index (0-500). **RMSE/MAE** = average "
        "prediction error (lower is better). **R\u00b2** = variance explained (closer to 1 is "
        "better). **SHAP** = how much each feature pushes a prediction up or down."
    )
    st.caption("For education/demonstration only. Forecasts are estimates and are not a "
               "substitute for official public-health air-quality advisories.")
