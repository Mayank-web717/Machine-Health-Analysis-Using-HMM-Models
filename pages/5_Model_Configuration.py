"""
pages/6_Model_Configuration.py
================================
HMM training configuration and retraining controls.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import pandas as pd

from src.data_loader import (
    get_machine_ids, available_sensor_cols, dataset_summary
)
from src.preprocessing import (
    SensorPreprocessor, split_machine_ids, prepare_sequences
)
from src.hmm_model import MachineHealthHMM
from src.state_interpretation import interpret_states
from src.visualizations import (
    emission_means_chart, transition_matrix_heatmap,
    loglikelihood_convergence, THEME
)

st.set_page_config(page_title="Model Configuration", page_icon="⚙️", layout="wide")
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

st.markdown("# ⚙️ Model Configuration")
st.markdown("Configure and train the Gaussian HMM on the loaded sensor data.")
st.divider()

df = st.session_state.get("df")
if df is None:
    from src.data_loader import load_builtin_dataset
    df = load_builtin_dataset()
    st.session_state["df"] = df

machine_ids  = get_machine_ids(df)
sensor_cols  = available_sensor_cols(df)
summary      = dataset_summary(df)

# ---------------------------------------------------------------------------
# Configuration controls
# ---------------------------------------------------------------------------
st.markdown("## 🔧 Training Parameters")

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    n_states = st.slider(
        "Number of Hidden States",
        min_value=2, max_value=6,
        value=st.session_state.get("n_states", 3),
        help="2 = Healthy/Degraded; 3 = Healthy/Degrading/Critical; 4+ = finer granularity",
    )
    st.session_state["n_states"] = n_states

    n_iter = st.slider(
        "Max Baum-Welch Iterations", 50, 500,
        value=st.session_state.get("n_iter", 200),
    )
    st.session_state["n_iter"] = n_iter

    n_init = st.slider(
        "Random Initializations", 1, 5,
        value=st.session_state.get("n_init", 3),
        help="More initializations → more robust, but slower.",
    )
    st.session_state["n_init"] = n_init

with col_cfg2:
    tol = st.select_slider(
        "Convergence Tolerance",
        options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=st.session_state.get("tol", 1e-4),
        format_func=lambda x: f"{x:.0e}",
    )
    st.session_state["tol"] = tol

    cov_type = st.selectbox(
        "Covariance Type",
        ["diag", "full", "spherical", "tied"],
        index=0,
        help=(
            "diag=independent per feature (fast); full=correlated (rich, needs more data); "
            "spherical=scalar variance; tied=shared covariance."
        ),
    )
    st.session_state["covariance_type"] = cov_type

    random_seed = st.number_input(
        "Random Seed", 0, 9999,
        value=st.session_state.get("random_seed", 42),
    )
    st.session_state["random_seed"] = random_seed

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
st.markdown("## 🔬 Sensor Features")
sel_features = st.multiselect(
    "Features to include in HMM training",
    sensor_cols,
    default=st.session_state.get("feature_cols", sensor_cols) or sensor_cols,
)
if not sel_features:
    st.error("Select at least one sensor feature."); st.stop()
st.session_state["feature_cols"] = sel_features

# ---------------------------------------------------------------------------
# Train / test split display
# ---------------------------------------------------------------------------
st.markdown("## 🗂️ Dataset Split")
train_ids, val_ids, test_ids = split_machine_ids(
    machine_ids, train_frac=0.70, val_frac=0.15, random_seed=random_seed
)
st.session_state["train_ids"] = train_ids
st.session_state["val_ids"]   = val_ids
st.session_state["test_ids"]  = test_ids

sc1, sc2, sc3 = st.columns(3)
sc1.metric("Train machines",      len(train_ids),  help=str(train_ids))
sc2.metric("Validation machines", len(val_ids),    help=str(val_ids))
sc3.metric("Test machines",       len(test_ids),   help=str(test_ids))
st.caption(
    "Machines are split at the machine level to prevent leakage across sequences. "
    "Scaling is fitted on training machines only."
)

# ---------------------------------------------------------------------------
# TRAIN button
# ---------------------------------------------------------------------------
st.divider()
if st.button("🚀 Train Model", use_container_width=True, type="primary"):
    with st.spinner("Fitting StandardScaler on training sequences …"):
        prep = SensorPreprocessor(feature_cols=sel_features)
        try:
            prep.fit(df, train_ids)
        except ValueError as e:
            st.error(f"Preprocessing failed: {e}"); st.stop()

    with st.spinner("Preparing training sequences …"):
        try:
            X_train, lengths_train = prepare_sequences(df, prep, train_ids)
        except ValueError as e:
            st.error(f"Sequence preparation failed: {e}"); st.stop()

    n_obs  = len(X_train)
    n_feat = len(sel_features)
    min_obs_needed = n_states * n_feat * 5
    if n_obs < min_obs_needed:
        st.warning(
            f"Only {n_obs} training observations for {n_states} states and {n_feat} features. "
            f"Recommend at least {min_obs_needed}. Consider using fewer states or more machines."
        )

    hmm_model = MachineHealthHMM(
        n_states        = n_states,
        covariance_type = cov_type,
        n_iter          = n_iter,
        tol             = tol,
        n_init          = n_init,
        random_seed     = random_seed,
    )

    t0 = time.time()
    progress_bar = st.progress(0, text="Training HMM (Baum-Welch EM) …")
    with st.spinner(f"Training {n_init} initialization(s) …"):
        try:
            hmm_model.fit(X_train, lengths_train)
        except Exception as e:
            st.error(f"Training failed: {e}"); st.stop()
    progress_bar.progress(100, text="Training complete!")
    elapsed = time.time() - t0

    # Compute state interpretation using means
    interp = interpret_states(
        n_states    = n_states,
        means       = hmm_model.means,
        feature_cols= sel_features,
    )

    # Store everything in session state
    st.session_state["hmm"]          = hmm_model
    st.session_state["preprocessor"] = prep
    st.session_state["state_interp"] = interp
    st.session_state["model_trained"]= True

    st.success(f"✅ Model trained in {elapsed:.1f}s!")

    # ---------------------------------------------------------------------------
    # Post-training display
    # ---------------------------------------------------------------------------
    st.divider()
    st.markdown("## 📊 Training Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Log-Likelihood", f"{hmm_model.best_log_prob:.2f}")
    m2.metric("Converged", "Yes" if hmm_model.converged else "No (reached max iter)")
    m3.metric("Training Observations", n_obs)
    m4.metric("Interpretation Confidence", interp["confidence"])

    tab_em, tab_trans, tab_interp = st.tabs(["Emission Means", "Transition Matrix", "Interpretation"])

    with tab_em:
        fig_em = emission_means_chart(hmm_model.means, sel_features, interp["state_labels"])
        st.plotly_chart(fig_em, use_container_width=True)
        st.markdown("### Feature Statistics used for Scaling")
        st.dataframe(prep.feature_stats(), use_container_width=True)

    with tab_trans:
        fig_tr = transition_matrix_heatmap(hmm_model.transmat, interp["state_labels"])
        st.plotly_chart(fig_tr, use_container_width=True)

        st.markdown("### Initial State Probabilities (π)")
        pi_df = pd.DataFrame({
            "State": [interp["state_labels"].get(i, f"State {i}") for i in range(n_states)],
            "π (initial prob)": hmm_model.startprob,
        })
        st.dataframe(pi_df.style.format({"π (initial prob)": "{:.4f}"}), use_container_width=True)

    with tab_interp:
        st.markdown(interp["explanation"])

    # Log-likelihood chart
    st.markdown("## 📈 Log-Likelihood per Initialization")
    fig_ll = loglikelihood_convergence(hmm_model.training_log_probs)
    st.plotly_chart(fig_ll, use_container_width=True)

# ---------------------------------------------------------------------------
# Show current model state if already trained
# ---------------------------------------------------------------------------
elif st.session_state.get("model_trained"):
    hmm_model = st.session_state["hmm"]
    interp    = st.session_state["state_interp"]
    prep      = st.session_state["preprocessor"]

    st.info("Model is already trained. Edit parameters above and click **Train Model** to retrain.")
    st.divider()
    st.markdown("## Current Model Parameters")

    m1, m2, m3 = st.columns(3)
    m1.metric("Log-Likelihood", f"{hmm_model.best_log_prob:.2f}")
    m2.metric("Converged",      "Yes" if hmm_model.converged else "No")
    m3.metric("States",         hmm_model.n_states)

    fig_em = emission_means_chart(
        hmm_model.means, prep.feature_cols, interp["state_labels"]
    )
    st.plotly_chart(fig_em, use_container_width=True)

    fig_tr = transition_matrix_heatmap(hmm_model.transmat, interp["state_labels"])
    st.plotly_chart(fig_tr, use_container_width=True)
