"""Pearls AQI Predictor - Streamlit dashboard.

Reads the registered model, the feature store history, and the latest batch
predictions, then renders:
  * current air quality + a live AQICN reading (if a token is configured)
  * the 3-day AQI forecast as colour-coded cards with health guidance
  * hazardous-AQI alerts
  * historical AQI trend + EDA charts
  * model leaderboard (RMSE / MAE / R2)
  * SHAP feature-importance explanation
  * a glossary for non-technical viewers

Run:  streamlit run app/streamlit_app.py
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

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="\U0001F32B\uFE0F", layout="wide")


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
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title},
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
    ))
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
    return fig


# ----------------------------- Sidebar -----------------------------
st.sidebar.title("\U0001F32B\uFE0F Pearls AQI Predictor")
st.sidebar.write(f"**City:** {CONFIG.city}")
st.sidebar.write(f"**Horizon:** next {CONFIG.horizon} days")
st.sidebar.write(f"**Backend:** {CONFIG.resolved_backend_mode()}")
metrics = load_metrics()
if metrics:
    st.sidebar.success(f"Active model: {metrics.get('best_model', 'n/a')}")
st.sidebar.caption("Serverless ML pipeline: feature store -> training -> "
                   "model registry -> batch inference -> this dashboard.")

preds = load_predictions()
history = load_history()

# ----------------------------- Header ------------------------------
st.title("Air Quality Forecast")
st.caption(f"3-day AQI outlook for {CONFIG.city} | powered by an automated, "
           "serverless ML pipeline")

# Live / latest observed AQI
col_now, col_live = st.columns([1, 1])
with col_now:
    if not history.empty:
        latest = history.sort_values("date").iloc[-1]
        st.plotly_chart(gauge(float(latest.get("aqi", 0)),
                              f"Latest observed AQI ({latest['date'].date()})"),
                        width="stretch")
with col_live:
    live = aqicn_current()
    if live and live.get("aqi") is not None:
        st.plotly_chart(gauge(float(live["aqi"]), "Live AQICN reading"),
                        width="stretch")
    else:
        st.info("Live AQICN reading unavailable (set AQICN_TOKEN to enable). "
                "Showing model + history below.")

# ----------------------------- Alerts ------------------------------
records = preds.get("predictions", [])
alerts = [r for r in records if r.get("hazardous")]
if alerts:
    days = ", ".join(f"{a['event_date']} (AQI {a['predicted_aqi']:.0f})" for a in alerts)
    st.error(f"\u26A0\uFE0F Hazardous air quality predicted on: {days}. "
             "Limit outdoor exposure and consider a mask / air purifier.")

# ----------------------- 3-day forecast cards ----------------------
st.subheader(f"\U0001F4C5 Next {CONFIG.horizon}-day forecast")
if records:
    cols = st.columns(len(records))
    for col, rec in zip(cols, records):
        with col:
            text_color = (
                "#111827"
                if rec["category"] in {"Good", "Moderate", "Unhealthy for Sensitive Groups"}
                else "white"
            )
            st.markdown(
                f"<div style='background:{rec['color']};padding:18px;border-radius:8px;"
                f"color:{text_color};text-align:center'>"
                f"<div style='font-size:15px'>{rec['event_date']}</div>"
                f"<div style='font-size:40px;font-weight:700'>{rec['predicted_aqi']:.0f}</div>"
                f"<div style='font-size:15px'>{rec['emoji']} {rec['category']}</div>"
                f"</div>", unsafe_allow_html=True)
            st.caption(rec["health"])
else:
    st.warning("No predictions yet. Run: python -m pipelines.inference_pipeline")

# ----------------------- Historical trend / EDA --------------------
if not history.empty:
    st.subheader("\U0001F4C8 Historical AQI trend")
    fig = px.line(history.sort_values("date"), x="date", y="aqi",
                  labels={"aqi": "AQI", "date": "Date"})
    fig.add_hline(y=CONFIG.hazardous_threshold, line_dash="dash", line_color="red",
                  annotation_text="Hazard threshold")
    st.plotly_chart(fig, width="stretch")

    tab1, tab2, tab3 = st.tabs(["Seasonality", "AQI vs weather", "Distribution"])
    with tab1:
        h = history.copy()
        h["month"] = h["date"].dt.month_name().str[:3]
        order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                 "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = h.groupby("month")["aqi"].mean().reindex(order).reset_index()
        st.plotly_chart(px.bar(monthly, x="month", y="aqi",
                               title="Average AQI by month"), width="stretch")
    with tab2:
        if "wind_speed" in history:
            st.plotly_chart(px.scatter(history, x="wind_speed", y="aqi",
                            trendline="ols" if _has_statsmodels() else None,
                            labels={"wind_speed": "Wind speed (m/s)"},
                            title="AQI vs wind speed (dispersion effect)"),
                            width="stretch")
    with tab3:
        st.plotly_chart(px.histogram(history, x="aqi", nbins=40,
                        title="AQI distribution"), width="stretch")

# ----------------------- Model leaderboard -------------------------
if metrics.get("models"):
    st.subheader("\U0001F3C6 Model leaderboard (held-out test set)")
    rows = [{"Model": n, "RMSE": round(m["rmse"], 2), "MAE": round(m["mae"], 2),
             "R\u00b2": round(m["r2"], 3)} for n, m in metrics["models"].items()]
    board = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    st.dataframe(board, width="stretch", hide_index=True)

# ----------------------- SHAP explanation --------------------------
if metrics.get("shap_importance"):
    st.subheader("\U0001F50D Why the model predicts what it does (SHAP)")
    imp = metrics["shap_importance"]
    idf = pd.DataFrame({"feature": list(imp.keys()),
                        "importance": list(imp.values())}).head(12)
    st.plotly_chart(px.bar(idf.sort_values("importance"), x="importance", y="feature",
                    orientation="h", title="Mean |SHAP value| (feature impact)"),
                    width="stretch")

# ----------------------------- Glossary ----------------------------
with st.expander("\U0001F4D6 Glossary & AQI scale"):
    for cat in all_categories():
        st.markdown(f"<span style='color:{cat.color};font-weight:700'>"
                    f"{cat.emoji} {cat.name}</span> &mdash; {cat.health}",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**AQI** = US EPA Air Quality Index (0-500). **RMSE/MAE** = average "
                "prediction error (lower is better). **R\u00b2** = variance explained "
                "(closer to 1 is better). **SHAP** = how much each feature pushes a "
                "prediction up or down.")
