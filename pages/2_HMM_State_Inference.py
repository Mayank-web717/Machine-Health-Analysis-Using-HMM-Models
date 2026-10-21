"""
pages/2_HMM_State_Inference.py
================================
Displays HMM hidden-state inference results.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data_loader import (
    get_machine_ids, get_machine_sequence,
    available_sensor_cols, CYCLE_COL, LABEL_COL
)
from src.evaluation import transition_counts, state_dwell_times, unsupervised_evaluation
from src.visualizations import (
    state_timeline, posterior_heatmap, transition_matrix_heatmap,
    emission_means_chart, state_occupancy_pie, THEME
)
import plotly.graph_objects as go

st.set_page_config(page_title="HMM State Inference", page_icon="🔷", layout="wide")

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

st.markdown("# 🔷 HMM State Inference")
st.divider()

# ---------------------------------------------------------------------------
# Guard: model must be trained
# ---------------------------------------------------------------------------
df   = st.session_state.get("df")
hmm  = st.session_state.get("hmm")
prep = st.session_state.get("preprocessor")
interp = st.session_state.get("state_interp")

if df is None:
    from src.data_loader import load_builtin_dataset
    df = load_builtin_dataset()
    st.session_state["df"] = df

if not st.session_state.get("model_trained"):
    st.info("⚙️ The HMM has not been trained yet. Go to the **Model Configuration** page and click **Train Model**.")
    st.stop()

# ---------------------------------------------------------------------------
# Machine selector
# ---------------------------------------------------------------------------
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

n_states      = hmm.n_states
state_labels  = interp["state_labels"]
state_colors  = interp["state_colors"]

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cycles",   len(cycles))
c2.metric("Hidden States",  n_states)

latest_state = int(states_seq[-1])
c3.metric("Latest State", state_labels.get(latest_state, f"State {latest_state}"))
c4.metric("Avg Posterior Entropy",
          f"{float(-np.sum(posteriors * np.log(posteriors + 1e-12), axis=1).mean()):.3f}")

st.divider()

# ---------------------------------------------------------------------------
# State timeline
# ---------------------------------------------------------------------------
st.markdown("## 🎨 Hidden-State Timeline")
fig_tl = state_timeline(cycles, states_seq, state_colors, state_labels,
                         title=f"Machine {sel_machine} — Inferred State Sequence")
st.plotly_chart(fig_tl, use_container_width=True)

# ---------------------------------------------------------------------------
# Posterior heatmap
# ---------------------------------------------------------------------------
st.markdown("## 🌡️ Posterior State Probabilities")
fig_ph = posterior_heatmap(posteriors, state_labels, cycles,
                            title=f"Machine {sel_machine} — P(State | Observations)")
st.plotly_chart(fig_ph, use_container_width=True)

with st.expander("What does this heatmap show?"):
    st.markdown("""
The **posterior probability** γ_t(i) = P(S_t = i | O₁…O_T, λ) is computed using the
**Forward-Backward algorithm**.

- Each row is a hidden state; each column is a time step (cycle).
- Brighter colour → higher probability that the machine was in that state at that cycle.
- The machine is most likely in the state with the brightest cell at each time step.
- If two rows have similar brightness, the model is uncertain about which state the
  machine is in.
    """)

st.divider()

# ---------------------------------------------------------------------------
# Emission means
# ---------------------------------------------------------------------------
st.markdown("## 📊 Emission Means per State")
sensor_cols = prep.feature_cols
fig_em = emission_means_chart(hmm.means, sensor_cols, state_labels)
st.plotly_chart(fig_em, use_container_width=True)
st.caption("Values are in **standardised units** (z-scores relative to training data mean/std).")

st.divider()

# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------
col_tm, col_oc = st.columns(2)

with col_tm:
    st.markdown("## 🔄 Learned Transition Matrix")
    fig_trans = transition_matrix_heatmap(hmm.transmat, state_labels)
    st.plotly_chart(fig_trans, use_container_width=True)
    with st.expander("Interpretation"):
        st.markdown("""
The **transition matrix A** where A[i,j] = P(next state = j | current state = i).

- Values along the **diagonal** indicate self-loops (staying in the same state).
- High diagonal values → states are persistent.
- Off-diagonal values → probability of transitioning to a different state.
        """)

with col_oc:
    st.markdown("## 🥧 State Occupancy")
    occ = {s: {"count": int((states_seq == s).sum()),
               "percent": 100.0 * (states_seq == s).sum() / max(1, len(states_seq))}
           for s in range(n_states)}
    fig_pie = state_occupancy_pie(occ, state_labels, state_colors)
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Observed transition counts
# ---------------------------------------------------------------------------
st.markdown("## 🔢 Observed Transition Counts")
tc  = transition_counts(states_seq, n_states)
tc_df = pd.DataFrame(
    tc,
    index=[f"From: {state_labels.get(i, f'State {i}')}" for i in range(n_states)],
    columns=[f"To: {state_labels.get(j, f'State {j}')}" for j in range(n_states)],
)
st.dataframe(tc_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Dwell times
# ---------------------------------------------------------------------------
st.markdown("## ⏱️ State Dwell Times")
dwell = state_dwell_times(states_seq, n_states)
dwell_rows = []
for s in range(n_states):
    d = dwell[s]
    dwell_rows.append({
        "State": state_labels.get(s, f"State {s}"),
        "Episodes": d["n_episodes"],
        "Mean Dwell (cycles)": f"{d['mean_dwell']:.1f}",
        "Max Dwell (cycles)":  d["max_dwell"],
    })
st.dataframe(pd.DataFrame(dwell_rows), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# State interpretation explanation
# ---------------------------------------------------------------------------
st.markdown("## 🧠 State Interpretation")
st.markdown(interp["explanation"])
st.info(
    "HMM hidden states do **not** automatically correspond to named health stages. "
    "The labels above are assigned by examining emission means and sensor trends. "
    "If sensor statistics are similar across states, the labels may be unreliable."
)
