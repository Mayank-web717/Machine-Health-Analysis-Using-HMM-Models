"""
pages/3_Machine_Health.py
==========================
Per-machine degradation analysis and alert system.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import (
    get_machine_ids, get_machine_sequence, available_sensor_cols,
    CYCLE_COL, LABEL_COL
)
from src.state_interpretation import compute_alert_level
from src.visualizations import (
    sensor_timeseries, posterior_heatmap, state_timeline, THEME
)

st.set_page_config(page_title="Machine Health", page_icon="📉", layout="wide")
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

st.markdown("# 📉 Machine Health Analysis")
st.divider()

df     = st.session_state.get("df")
hmm    = st.session_state.get("hmm")
prep   = st.session_state.get("preprocessor")
interp = st.session_state.get("state_interp")

if df is None:
    from src.data_loader import load_builtin_dataset
    df = load_builtin_dataset()
    st.session_state["df"] = df
if not st.session_state.get("model_trained"):
    st.info("Train the HMM first (Model Configuration page)."); st.stop()

machine_ids = get_machine_ids(df)
default_idx = machine_ids.index(st.session_state.get("selected_machine", machine_ids[0])) \
              if st.session_state.get("selected_machine") in machine_ids else 0
sel_machine = st.selectbox("Select Machine", machine_ids, index=default_idx)
st.session_state["selected_machine"] = sel_machine

seq    = get_machine_sequence(df, sel_machine)
cycles = seq[CYCLE_COL].values
X_seq  = prep.transform_sequence(seq)

states_seq = hmm.predict_states(X_seq)
posteriors = hmm.predict_proba(X_seq)

n_states     = hmm.n_states
state_labels = interp["state_labels"]
state_colors = interp["state_colors"]

# ---------------------------------------------------------------------------
# Alert section
# ---------------------------------------------------------------------------
threshold = st.slider("Alert threshold (critical-state probability)", 0.1, 0.9, 0.5, 0.05)
alert = compute_alert_level(posteriors, interp, critical_threshold=threshold)

alert_bg = {"CRITICAL": "#4a1a1a", "WARNING": "#4a3a1a", "NORMAL": "#1a4a2e"}
alert_border = {"CRITICAL": "#e74c3c", "WARNING": "#f39c12", "NORMAL": "#2ecc71"}
level = alert["alert_label"]

st.markdown(f"""
<div style="background:{alert_bg.get(level,'#162032')};
            border-left:4px solid {alert_border.get(level,'#4c8ef7')};
            border-radius:8px;padding:16px 20px;margin-bottom:20px;">
  <div style="font-size:1.4rem;font-weight:700">{alert['alert_color']} — {level}</div>
  <div style="color:#7fa0c0;margin-top:6px">
    Machine {sel_machine} | P({alert['critical_state_label']}) at latest cycle:
    <strong style="color:#e0ecff">{alert['latest_critical_prob']:.1%}</strong>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("How is the alert computed?"):
    st.markdown(alert["explanation"])

st.divider()

# ---------------------------------------------------------------------------
# Combined sensor + state chart
# ---------------------------------------------------------------------------
st.markdown("## 📊 Sensor Trends with Inferred States")
sensor_cols = prep.feature_cols
fig_ts = sensor_timeseries(
    seq, sensor_cols,
    state_sequence=states_seq,
    state_colors=state_colors,
    state_labels=state_labels,
    title=f"Machine {sel_machine} — Sensor Readings & HMM States",
)
st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------------------
# State timeline
# ---------------------------------------------------------------------------
st.markdown("## 🎨 State Timeline")
fig_tl = state_timeline(cycles, states_seq, state_colors, state_labels)
st.plotly_chart(fig_tl, use_container_width=True)

# ---------------------------------------------------------------------------
# Critical-state probability trend
# ---------------------------------------------------------------------------
st.markdown(f"## 📈 P({alert['critical_state_label']}) Over Time")
crit_idx  = alert["critical_state_idx"]
crit_prob = posteriors[:, crit_idx]

fig_cp = go.Figure()
fig_cp.add_trace(go.Scatter(
    x=cycles, y=crit_prob,
    mode="lines", name=f"P({alert['critical_state_label']})",
    line=dict(color=THEME["danger"], width=2),
    fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
))
fig_cp.add_hline(y=threshold, line_dash="dash",
                  line_color=THEME["warning"],
                  annotation_text=f"Alert threshold ({threshold:.0%})",
                  annotation_position="top right")
fig_cp.update_layout(
    paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["panel"],
    font=dict(color=THEME["text"]),
    xaxis_title="Cycle", yaxis_title="Probability",
    yaxis_range=[0, 1], height=280,
    title=dict(text=f"Probability of '{alert['critical_state_label']}' State Over Time", x=0.5),
)
st.plotly_chart(fig_cp, use_container_width=True)

# ---------------------------------------------------------------------------
# State change events
# ---------------------------------------------------------------------------
st.markdown("## ⚡ Notable State Transitions")
events = []
for t in range(1, len(states_seq)):
    if states_seq[t] != states_seq[t - 1]:
        events.append({
            "Cycle": int(cycles[t]),
            "From":  state_labels.get(int(states_seq[t-1]), f"State {states_seq[t-1]}"),
            "To":    state_labels.get(int(states_seq[t]),   f"State {states_seq[t]}"),
            f"P({alert['critical_state_label']})": f"{posteriors[t, crit_idx]:.3f}",
        })

if events:
    st.dataframe(pd.DataFrame(events), use_container_width=True)
else:
    st.info("No state transitions detected for this machine.")

# ---------------------------------------------------------------------------
# Posterior heatmap
# ---------------------------------------------------------------------------
st.divider()
st.markdown("## 🌡️ Posterior Probabilities Heatmap")
fig_ph = posterior_heatmap(posteriors, state_labels, cycles)
st.plotly_chart(fig_ph, use_container_width=True)

# ---------------------------------------------------------------------------
# Ground-truth comparison (if available)
# ---------------------------------------------------------------------------
if LABEL_COL in seq.columns:
    st.divider()
    st.markdown("## 🏷️ HMM States vs Ground-Truth Labels")
    stage_map = {0: "Healthy", 1: "Degrading", 2: "Critical"}
    gt_labels = seq[LABEL_COL].map(stage_map).values
    inferred  = np.array([state_labels.get(s, f"State {s}") for s in states_seq])

    fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Inferred HMM State", "Ground-Truth Stage"),
                              vertical_spacing=0.1)

    state_num = {v: k for k, v in state_labels.items()}
    stage_num = {"Healthy": 0, "Degrading": 1, "Critical": 2}

    fig_comp.add_trace(go.Scatter(
        x=cycles, y=[list(state_labels.values()).index(s) if s in state_labels.values() else 0
                     for s in inferred],
        mode="lines+markers", name="HMM Inferred",
        marker=dict(color=[state_colors.get(s, "#888") for s in states_seq], size=5),
        line=dict(color=THEME["accent"]),
    ), row=1, col=1)

    gt_num_arr = [stage_num.get(g, -1) for g in gt_labels]
    gt_color_map = {"Healthy": "#2ecc71", "Degrading": "#f39c12", "Critical": "#e74c3c"}
    fig_comp.add_trace(go.Scatter(
        x=cycles, y=gt_num_arr,
        mode="lines+markers", name="Ground Truth",
        marker=dict(color=[gt_color_map.get(g, "#888") for g in gt_labels], size=5),
        line=dict(color=THEME["warning"]),
    ), row=2, col=1)

    fig_comp.update_layout(
        paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"]), height=400,
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.info(
        "Ground-truth labels are from the synthetic data generator. "
        "The HMM is trained **without** these labels; any agreement is evidence "
        "of meaningful state discovery."
    )
