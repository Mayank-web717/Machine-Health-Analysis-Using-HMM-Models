"""
app.py  —  Predictive Machine Health Monitoring Using HMM
==========================================================
Streamlit multi-page application entry point (Overview page).
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# ---------------------------------------------------------------------------
# Page configuration (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Health Monitoring — HMM",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import (
    load_builtin_dataset,
    load_uploaded_csv,
    get_machine_ids,
    get_machine_sequence,
    available_sensor_cols,
    dataset_summary,
    SENSOR_COLS,
    LABEL_COL,
    ID_COL,
    CYCLE_COL,
)
from src.preprocessing import (
    SensorPreprocessor,
    split_machine_ids,
    prepare_sequences,
)
from src.hmm_model import MachineHealthHMM
from src.state_interpretation import interpret_states, compute_alert_level
from src.visualizations import (
    sensor_timeseries,
    state_timeline,
    posterior_heatmap,
    THEME,
)

# ---------------------------------------------------------------------------
# CSS — dark industrial theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1923;
    color: #d0dde8;
  }
  .stApp { background-color: #0f1923; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0d1620;
    border-right: 1px solid #1e2d3d;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background-color: #162032;
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid #1e2d3d;
  }
  [data-testid="stMetricValue"] { color: #4c8ef7; font-size: 1.6rem; }
  [data-testid="stMetricLabel"] { color: #7fa0c0; font-size: 0.78rem; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color: #0d1620; }
  .stTabs [data-baseweb="tab"] { color: #7fa0c0; }
  .stTabs [aria-selected="true"] { color: #4c8ef7; border-bottom: 2px solid #4c8ef7; }

  /* Select boxes and sliders */
  .stSelectbox select, .stMultiSelect { background-color: #162032 !important; color: #d0dde8 !important; }

  /* Buttons */
  .stButton > button {
    background-color: #4c8ef7;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
  }
  .stButton > button:hover { background-color: #3a78e0; }

  /* Info / warning boxes */
  .stAlert { border-radius: 8px; }

  /* Section headings */
  h1, h2, h3 { color: #e0ecff; }
  h1 { border-bottom: 2px solid #1e2d3d; padding-bottom: 8px; }

  /* Divider */
  hr { border-color: #1e2d3d; }

  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
  }
  .badge-green  { background:#1a4a2e; color:#2ecc71; }
  .badge-amber  { background:#4a3a1a; color:#f39c12; }
  .badge-red    { background:#4a1a1a; color:#e74c3c; }
  .badge-blue   { background:#1a2a4a; color:#4c8ef7; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session-state initialisation
# ===========================================================================
def _init_session():
    defaults = {
        "dataset_source": "builtin",
        "df":             None,
        "preprocessor":   None,
        "hmm":            None,
        "train_ids":      None,
        "val_ids":        None,
        "test_ids":       None,
        "n_states":       3,
        "feature_cols":   None,
        "n_iter":         200,
        "tol":            1e-4,
        "random_seed":    42,
        "n_init":         3,
        "covariance_type": "diag",
        "model_trained":  False,
        "state_interp":   None,
        "selected_machine": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ===========================================================================
# Sidebar — global controls
# ===========================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Machine Health HMM")
        st.markdown("*Unit 4 — Hidden Markov Models*")
        st.divider()

        # -- Dataset source
        st.markdown("### 📂 Dataset")
        source = st.radio(
            "Source",
            options=["Built-in synthetic", "Upload CSV"],
            index=0 if st.session_state["dataset_source"] == "builtin" else 1,
        )

        if source == "Built-in synthetic":
            st.session_state["dataset_source"] = "builtin"
            if st.button("Load / Reload Dataset", use_container_width=True):
                with st.spinner("Generating synthetic dataset …"):
                    st.session_state["df"] = load_builtin_dataset()
                    st.session_state["model_trained"] = False
                    st.session_state["hmm"] = None
                st.success("Dataset loaded!")

        else:
            st.session_state["dataset_source"] = "upload"
            uploaded = st.file_uploader("Upload machine sensor CSV", type=["csv"])
            if uploaded:
                df_up, err = load_uploaded_csv(uploaded)
                if err:
                    st.error(f"❌ {err}")
                else:
                    st.session_state["df"] = df_up
                    st.session_state["model_trained"] = False
                    st.session_state["hmm"] = None
                    st.success("CSV loaded successfully!")

        # -- Auto-load on first run
        if st.session_state["df"] is None:
            with st.spinner("Loading built-in dataset …"):
                st.session_state["df"] = load_builtin_dataset()

        df = st.session_state["df"]

        # -- Machine selector
        if df is not None:
            machine_ids = get_machine_ids(df)
            sel = st.selectbox(
                "Selected Machine",
                machine_ids,
                index=0,
                key="selected_machine_select",
            )
            st.session_state["selected_machine"] = sel

        st.divider()

        # -- Model config summary
        st.markdown("### 🤖 Model")
        st.markdown(f"Hidden states: **{st.session_state['n_states']}**")
        status_color = "badge-green" if st.session_state["model_trained"] else "badge-amber"
        status_text  = "Trained" if st.session_state["model_trained"] else "Not trained"
        st.markdown(
            f'<span class="badge {status_color}">{status_text}</span>',
            unsafe_allow_html=True,
        )

        if st.session_state["model_trained"] and st.session_state.get("state_interp"):
            conf = st.session_state["state_interp"].get("confidence", "?")
            st.markdown(f"Interpretation confidence: **{conf}**")

        st.divider()
        st.caption("B.Tech CS — ML Project\nUnit 4: Hidden Markov Models")


render_sidebar()


# ===========================================================================
# Main page — Overview
# ===========================================================================
st.markdown("# ⚙️ Predictive Machine Health Monitoring")
st.markdown("**Using Hidden Markov Models** &nbsp;|&nbsp; Unit 4 ML Project")
st.divider()

df = st.session_state["df"]

if df is None:
    st.info("👈 Load a dataset from the sidebar to begin.")
    st.stop()

summary = dataset_summary(df)
mid     = st.session_state.get("selected_machine") or get_machine_ids(df)[0]
seq     = get_machine_sequence(df, mid)

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏭 Machines",      summary["n_machines"])
c2.metric("📊 Observations",  summary["n_observations"])
c3.metric("🔬 Sensor Features", summary["n_sensor_cols"])
c4.metric("🔷 Hidden States",  st.session_state["n_states"])
c5.metric("🔄 Model Status",   "Trained" if st.session_state["model_trained"] else "Untrained")

st.divider()

# ---------------------------------------------------------------------------
# Project summary
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("## 🎯 Project Objective")
    st.markdown("""
This system demonstrates **predictive machine health monitoring** using a
**Gaussian Hidden Markov Model (HMM)**.  Sensor readings from industrial machines
(temperature, vibration, pressure, rotation speed) are analysed to:

1. **Infer hidden operating states** (Healthy → Degrading → Critical)
2. **Calculate posterior probabilities** of each state at every cycle
3. **Detect transitions** between operating regimes
4. **Visualise degradation trajectories** over time

> ⚠️ **Transparency note:** This project uses a **synthetic dataset** for
> demonstration.  All inferences are probabilistic; no real equipment failure
> predictions are made.
    """)

    # -- Dataset info
    st.markdown("## 📋 Dataset Overview")
    cols = st.columns(3)
    cols[0].metric("Machines", summary["n_machines"])
    cols[1].metric("Min sequence", f"{summary['seq_lengths_min']} cycles")
    cols[2].metric("Max sequence", f"{summary['seq_lengths_max']} cycles")

    if summary["missing_values"] > 0:
        st.warning(f"⚠️ {summary['missing_values']} missing values detected.")
    else:
        st.success("✅ No missing values.")

    if summary["has_labels"]:
        st.info("ℹ️ Ground-truth health-stage labels are available for supervised evaluation.")

with col_right:
    st.markdown("## 📡 Selected Machine Preview")
    st.markdown(f"**Machine ID:** {mid} &nbsp;|&nbsp; **{len(seq)} cycles**")

    sensor_cols = available_sensor_cols(df)[:2]
    if sensor_cols:
        fig = sensor_timeseries(seq, sensor_cols, title="")
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Current state summary (if model trained)
# ---------------------------------------------------------------------------
if st.session_state["model_trained"]:
    hmm   = st.session_state["hmm"]
    prep  = st.session_state["preprocessor"]
    interp = st.session_state["state_interp"]

    X_seq = prep.transform_sequence(seq)
    states_seq = hmm.predict_states(X_seq)
    post_seq   = hmm.predict_proba(X_seq)

    latest_state  = int(states_seq[-1])
    latest_label  = interp["state_labels"].get(latest_state, f"State {latest_state}")
    latest_color  = interp["state_colors"].get(latest_state, "#888")
    alert         = compute_alert_level(post_seq, interp)

    st.markdown("## 🔍 Current Machine Status")
    a1, a2, a3 = st.columns(3)
    a1.metric("Latest Inferred State", latest_label)
    a2.metric("Alert Level", alert["alert_label"])
    a3.metric(f"P({alert['critical_state_label']})", f"{alert['latest_critical_prob']:.1%}")

    with st.expander("ℹ️ How the alert is computed"):
        st.markdown(alert["explanation"])
else:
    st.markdown("## 🔍 Current Machine Status")
    st.info("💡 The HMM model has not been trained yet. Click below to quickly train the default 3-state HMM, or configure custom hyperparameters in **Model Configuration**.")
    train_c1, train_c2 = st.columns([1, 3])
    with train_c1:
        if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
            with st.spinner("Training 3-state Gaussian HMM (Baum-Welch EM) …"):
                sensor_cols = available_sensor_cols(df)
                train_ids, val_ids, test_ids = split_machine_ids(get_machine_ids(df), random_seed=42)
                prep = SensorPreprocessor(sensor_cols)
                prep.fit(df, train_ids)
                X_tr, len_tr = prepare_sequences(df, prep, train_ids)
                hmm_model = MachineHealthHMM(n_states=3, covariance_type="diag", n_iter=100, n_init=2, random_seed=42)
                hmm_model.fit(X_tr, len_tr)
                interp = interpret_states(hmm_model.n_states, hmm_model.means, sensor_cols)
                st.session_state["hmm"] = hmm_model
                st.session_state["preprocessor"] = prep
                st.session_state["state_interp"] = interp
                st.session_state["model_trained"] = True
                st.session_state["train_ids"] = train_ids
                st.session_state["val_ids"] = val_ids
                st.session_state["test_ids"] = test_ids
                st.session_state["feature_cols"] = sensor_cols
                st.rerun()
    with train_c2:
        st.caption("Fits a 3-state Gaussian HMM with EM across 7 training machines, automatically ranking hidden states into Healthy, Degrading, and Critical.")

st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------
st.markdown("## 🗺️ Navigation Guide")
pages = [
    ("🔬", "Sensor Explorer",       "Interactive time-series charts and statistical summaries."),
    ("🔷", "HMM State Inference",   "Inferred hidden states, posterior probabilities, transition matrix."),
    ("📉", "Machine Health",        "Per-machine degradation timeline and alert indicator."),
    ("📊", "Model Evaluation",      "Accuracy, confusion matrix, AIC/BIC comparison."),
    ("🎓", "Algorithm Playground",  "Step-by-step Forward, Backward, Viterbi on small examples."),
    ("⚙️", "Model Configuration",   "Tune HMM parameters and retrain the model."),
]

cols = st.columns(3)
for i, (icon, name, desc) in enumerate(pages):
    with cols[i % 3]:
        st.markdown(f"""
<div style="background:#162032;border:1px solid #1e2d3d;border-radius:10px;padding:14px;margin-bottom:12px;">
  <div style="font-size:1.4rem">{icon}</div>
  <div style="font-weight:600;color:#e0ecff;margin-top:4px">{name}</div>
  <div style="color:#7fa0c0;font-size:0.82rem;margin-top:4px">{desc}</div>
</div>
""", unsafe_allow_html=True)
