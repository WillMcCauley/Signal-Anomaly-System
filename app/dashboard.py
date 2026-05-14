from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import METRICS_DIR, PROCESSED_DIR
from src.inference import latest_alert_for_unit, load_artifacts, score_frame


st.set_page_config(page_title="Signal Anomaly Decision Support", layout="wide")
st.title("Signal Anomaly Detection and Decision Support System")

test_path = PROCESSED_DIR / "test.csv"
metrics_path = METRICS_DIR / "metrics.json"

if not test_path.exists():
    st.error("Run `python -m src.evaluate --synthetic` from the project root before opening the dashboard.")
    st.stop()

raw = pd.read_csv(test_path)
artifacts = load_artifacts()
unit_ids = sorted(raw["unit_id"].unique())
selected_unit = st.sidebar.selectbox("Unit", unit_ids)
unit_raw = raw[raw["unit_id"] == selected_unit].sort_values("cycle")
scored = score_frame(unit_raw, artifacts)
alert = latest_alert_for_unit(unit_raw, artifacts)

metric_cols = st.columns(4)
metric_cols[0].metric("Severity", alert.severity.upper())
metric_cols[1].metric("Anomaly score", f"{alert.anomaly_score:.2f}")
metric_cols[2].metric("Failure probability", f"{alert.probability_failure:.2f}")
metric_cols[3].metric("Recommended action", alert.recommended_action)

st.subheader("Alert Interpretation")
st.write(alert.explanation)

fig = go.Figure()
fig.add_trace(go.Scatter(x=scored["cycle"], y=scored["sensor_2"], name="sensor_2", yaxis="y1"))
fig.add_trace(go.Scatter(x=scored["cycle"], y=scored["sensor_11"], name="sensor_11", yaxis="y1"))
fig.add_trace(
    go.Scatter(
        x=scored["cycle"],
        y=scored["anomaly_score"],
        name="anomaly score",
        yaxis="y2",
        line={"color": "crimson"},
    )
)
fig.update_layout(
    height=520,
    xaxis={"title": "Cycle"},
    yaxis={"title": "Sensor value"},
    yaxis2={"title": "Risk score", "overlaying": "y", "side": "right", "range": [0, 1]},
    legend={"orientation": "h"},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Contributing Signals")
st.dataframe(pd.DataFrame({"signal": alert.contributing_signals}), use_container_width=True, hide_index=True)

if metrics_path.exists():
    st.subheader("Evaluation Summary")
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    baseline = metrics["baseline_isolation_forest"]
    st.json(baseline)
