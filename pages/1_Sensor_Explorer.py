"""
pages/1_Sensor_Explorer.py
==========================
Interactive sensor time-series explorer.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from src.data_loader import (
    get_machine_ids, get_machine_sequence,
    available_sensor_cols, dataset_summary, LABEL_COL, CYCLE_COL
)
from src.preprocessing import compute_sensor_stats, compute_correlation
from src.visualizations import sensor_timeseries, correlation_heatmap, THEME
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sensor Explorer", page_icon="🔬", layout="wide")

# -- Shared CSS (minimal duplicate from app.py)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0f1923;color:#d0dde8;}
.stApp{background:#0f1923;}
[data-testid="stSidebar"]{background:#0d1620;border-right:1px solid #1e2d3d;}
[data-testid="stMetric"]{background:#162032;border-radius:10px;padding:12px 16px;border:1px solid #1e2d3d;}
[data-testid="stMetricValue"]{color:#4c8ef7;font-size:1.4rem;}
h1,h2,h3{color:#e0ecff;}
h1{border-bottom:2px solid #1e2d3d;padding-bottom:8px;}
</style>""", unsafe_allow_html=True)

st.markdown("# 🔬 Sensor Explorer")
st.markdown("Explore raw sensor measurements, distributions, and correlations.")
st.divider()

df = st.session_state.get("df")
if df is None:
    from src.data_loader import load_builtin_dataset
    df = load_builtin_dataset()
    st.session_state["df"] = df

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
machine_ids  = get_machine_ids(df)
sensor_cols  = available_sensor_cols(df)

c1, c2 = st.columns([1, 3])
with c1:
    sel_machine = st.selectbox("Select Machine", machine_ids,
                               index=(machine_ids.index(st.session_state.get("selected_machine", machine_ids[0]))
                                      if st.session_state.get("selected_machine") in machine_ids else 0))
    st.session_state["selected_machine"] = sel_machine

    sel_sensors = st.multiselect("Sensor Channels", sensor_cols, default=sensor_cols)
    if not sel_sensors:
        sel_sensors = sensor_cols

with c2:
    seq = get_machine_sequence(df, sel_machine)
    min_c, max_c = int(seq[CYCLE_COL].min()), int(seq[CYCLE_COL].max())
    if max_c - min_c > 1:
        cycle_range = st.slider("Cycle Range", min_c, max_c, (min_c, max_c))
        seq = seq[(seq[CYCLE_COL] >= cycle_range[0]) & (seq[CYCLE_COL] <= cycle_range[1])]

# ---------------------------------------------------------------------------
# Missing data info
# ---------------------------------------------------------------------------
missing = seq[sel_sensors].isnull().sum()
if missing.sum() > 0:
    st.warning(f"⚠️ Missing values found: {missing[missing > 0].to_dict()}")

# ---------------------------------------------------------------------------
# Time-series chart
# ---------------------------------------------------------------------------
st.markdown("## 📈 Time-Series Plots")
hmm       = st.session_state.get("hmm")
prep      = st.session_state.get("preprocessor")
interp    = st.session_state.get("state_interp")

state_seq_plot  = None
state_colors_p  = None
state_labels_p  = None

if hmm and prep and interp and st.session_state.get("model_trained"):
    try:
        X_s = prep.transform_sequence(get_machine_sequence(df, sel_machine))
        full_seq = get_machine_sequence(df, sel_machine)
        full_seq = full_seq[(full_seq[CYCLE_COL] >= seq[CYCLE_COL].min()) &
                            (full_seq[CYCLE_COL] <= seq[CYCLE_COL].max())]
        # Re-predict for the full machine then slice
        X_full = prep.transform_sequence(get_machine_sequence(df, sel_machine))
        states_full = hmm.predict_states(X_full)
        # Slice to cycle range
        cycle_mask = (get_machine_sequence(df, sel_machine)[CYCLE_COL] >= seq[CYCLE_COL].min()) & \
                     (get_machine_sequence(df, sel_machine)[CYCLE_COL] <= seq[CYCLE_COL].max())
        state_seq_plot = states_full[cycle_mask.values]
        state_colors_p = interp["state_colors"]
        state_labels_p = interp["state_labels"]
    except Exception:
        pass

fig_ts = sensor_timeseries(
    seq, sel_sensors,
    state_sequence=state_seq_plot,
    state_colors=state_colors_p,
    state_labels=state_labels_p,
    title=f"Machine {sel_machine} — Sensor Readings",
)
st.plotly_chart(fig_ts, use_container_width=True)

if state_seq_plot is not None:
    st.caption("Coloured background shading shows the HMM-inferred hidden state at each cycle.")
else:
    st.caption("Train the HMM model (Model Configuration page) to overlay hidden-state shading.")

# ---------------------------------------------------------------------------
# Statistics table
# ---------------------------------------------------------------------------
st.divider()
st.markdown("## 📊 Statistical Summary")
stats = compute_sensor_stats(seq, sel_sensors)
st.dataframe(
    stats.style.format({c: "{:.4f}" for c in stats.columns if c != "sensor"}),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------
st.divider()
st.markdown("## 📉 Sensor Distributions")
dcols = st.columns(min(len(sel_sensors), 4))
for i, col in enumerate(sel_sensors):
    with dcols[i % len(dcols)]:
        fig_hist = px.histogram(
            seq, x=col, nbins=30,
            color_discrete_sequence=[THEME["accent"]],
            title=col,
        )
        fig_hist.update_layout(
            paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["panel"],
            font=dict(color=THEME["text"]), margin=dict(l=20, r=20, t=40, b=20),
            height=230,
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------
if len(sel_sensors) > 1:
    st.divider()
    st.markdown("## 🔗 Sensor Correlation Matrix")
    corr = compute_correlation(df[df.get(df.columns[0]) == sel_machine]
                               if False else df, sel_sensors)
    # Use all machines for correlation to get more data points
    corr = compute_correlation(df, sel_sensors)
    fig_corr = correlation_heatmap(corr, title="Sensor Pearson Correlation (all machines)")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Correlation is computed across all machines and all cycles for statistical robustness.")

# ---------------------------------------------------------------------------
# Ground-truth stage (if available)
# ---------------------------------------------------------------------------
if LABEL_COL in seq.columns:
    st.divider()
    st.markdown("## 🏷️ Ground-Truth Health Stages")
    stage_map = {0: "Healthy", 1: "Degrading", 2: "Critical"}
    stage_colors = {"Healthy": "#2ecc71", "Degrading": "#f39c12", "Critical": "#e74c3c"}

    seq2 = seq.copy()
    seq2["stage_name"] = seq2[LABEL_COL].map(stage_map)
    counts = seq2["stage_name"].value_counts().reset_index()
    counts.columns = ["Stage", "Cycles"]

    fig_gt = go.Figure(go.Bar(
        x=counts["Stage"], y=counts["Cycles"],
        marker_color=[stage_colors.get(s, "#888") for s in counts["Stage"]],
    ))
    fig_gt.update_layout(
        title="Cycles per Ground-Truth Stage",
        paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"]), height=300,
        xaxis_title="Stage", yaxis_title="Cycles",
    )
    st.plotly_chart(fig_gt, use_container_width=True)
    st.info("ℹ️ Ground-truth labels are from the **synthetic data generator** for evaluation only. "
            "The HMM is trained on sensor observations, not on these labels.")
