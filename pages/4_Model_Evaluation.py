"""
pages/4_Model_Evaluation.py
============================
Supervised and unsupervised model evaluation.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data_loader import (
    get_machine_ids, get_machine_sequence, LABEL_COL, ID_COL
)
from src.preprocessing import prepare_sequences
from src.hmm_model import compare_n_states
from src.evaluation import (
    supervised_evaluation, unsupervised_evaluation,
    transition_counts, state_dwell_times
)
from src.visualizations import (
    confusion_matrix_chart, aic_bic_chart,
    loglikelihood_convergence, THEME
)
import plotly.graph_objects as go

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
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

st.markdown("# 📊 Model Evaluation")
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

n_states     = hmm.n_states
state_labels = interp["state_labels"]

# ---------------------------------------------------------------------------
# Section 1: Training log-likelihood
# ---------------------------------------------------------------------------
st.markdown("## 📈 Training Log-Likelihood (per Initialization)")
fig_ll = loglikelihood_convergence(hmm.training_log_probs)
st.plotly_chart(fig_ll, use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("Best Log-Likelihood",  f"{hmm.best_log_prob:.2f}")
m2.metric("Converged?",           "Yes" if hmm.converged else "No")
m3.metric("Initializations",      len(hmm.training_log_probs))

st.info(
    "Baum-Welch (EM) may converge to a **local optimum**. "
    "Multiple random initializations are run and the best is kept. "
    "The log-likelihood chart shows the final training score per run."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Unsupervised metrics on eval sequences
# ---------------------------------------------------------------------------
st.markdown("## 🔍 Unsupervised Evaluation (Test Sequences)")

test_ids = st.session_state.get("test_ids", [])
val_ids  = st.session_state.get("val_ids",  [])
eval_ids = test_ids if test_ids else val_ids

if eval_ids:
    try:
        X_eval, lengths_eval = prepare_sequences(df, prep, eval_ids)
        eval_ll = hmm.score(X_eval, lengths_eval)

        # gather all states and posteriors for eval sequences
        all_states = hmm.predict_states(X_eval, lengths_eval)
        all_post   = hmm.predict_proba(X_eval,  lengths_eval)

        # Number of free params (diag covariance)
        nf = len(prep.feature_cols)
        k  = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * nf

        unsup = unsupervised_evaluation(
            log_likelihood=eval_ll,
            n_params=k,
            n_observations=len(X_eval),
            state_sequence=all_states,
            n_states=n_states,
            posteriors=all_post,
        )

        ua, ub, uc, ud = st.columns(4)
        ua.metric("Test Log-Likelihood", f"{unsup['log_likelihood']:.2f}")
        ub.metric("AIC",  f"{unsup['aic']:.1f}")
        uc.metric("BIC",  f"{unsup['bic']:.1f}")
        ud.metric("Mean Posterior Entropy",
                  f"{unsup['mean_posterior_entropy']:.4f}" if unsup['mean_posterior_entropy'] else "N/A")

        st.markdown("### State Occupancy on Evaluation Set")
        occ_rows = []
        for s, d in unsup["state_occupancy"].items():
            occ_rows.append({
                "State": state_labels.get(s, f"State {s}"),
                "Count": d["count"],
                "Percent": f"{d['percent']:.1f}%",
            })
        st.dataframe(pd.DataFrame(occ_rows), use_container_width=True)

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
else:
    st.warning("No evaluation sequences available. "
               "Ensure at least 3 machines are in the dataset for a train/val/test split.")

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Supervised metrics (if labels available)
# ---------------------------------------------------------------------------
has_labels = LABEL_COL in df.columns

if has_labels:
    st.markdown("## 🏷️ Supervised Evaluation (Ground-Truth Labels)")
    st.info(
        "Labels come from the **synthetic data generator** and are used for evaluation only. "
        "The HMM is trained without them. The best permutation of HMM state indices is found "
        "to match ground-truth labels before computing accuracy."
    )

    eval_ids_sup = eval_ids if eval_ids else get_machine_ids(df)[:2]
    all_pred, all_gt = [], []
    for mid in eval_ids_sup:
        seq = get_machine_sequence(df, mid)
        if LABEL_COL not in seq.columns:
            continue
        X_s = prep.transform_sequence(seq)
        pred_s = hmm.predict_states(X_s)
        gt_s   = prep.get_labels(seq)
        if gt_s is not None:
            all_pred.append(pred_s)
            all_gt.append(gt_s)

    if all_pred:
        all_pred_arr = np.concatenate(all_pred)
        all_gt_arr   = np.concatenate(all_gt)

        sup_result = supervised_evaluation(
            all_pred_arr, all_gt_arr, n_states,
            state_labels=state_labels,
        )

        if "error" in sup_result:
            st.error(sup_result["error"])
        else:
            sa, sb, sc = st.columns(3)
            sa.metric("Accuracy (best permutation)", f"{sup_result['accuracy']:.1%}")
            sb.metric("Samples evaluated", sup_result["n_samples"])
            sc.metric("GT labels",
                      ", ".join(str(x) for x in sup_result["unique_gt_labels"]))

            # F1 per class
            ul = sup_result["unique_gt_labels"]
            stage_map = {0: "Healthy", 1: "Degrading", 2: "Critical"}
            f1_rows = []
            for idx, lbl in enumerate(ul):
                f1_rows.append({
                    "Stage": stage_map.get(lbl, str(lbl)),
                    "Precision": f"{sup_result['precision'][idx]:.3f}",
                    "Recall":    f"{sup_result['recall'][idx]:.3f}",
                    "F1-Score":  f"{sup_result['f1_score'][idx]:.3f}",
                    "Support":   sup_result["support"][idx],
                })
            st.dataframe(pd.DataFrame(f1_rows), use_container_width=True)

            # Confusion matrix
            cm_labels = [stage_map.get(l, str(l)) for l in ul]
            fig_cm = confusion_matrix_chart(
                sup_result["confusion_matrix"], cm_labels,
                title="Confusion Matrix (best label permutation)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            with st.expander("Note on supervised accuracy"):
                st.markdown(sup_result["note"])
    else:
        st.warning("No ground-truth labels found in evaluation sequences.")

else:
    st.markdown("## 🔎 No Ground-Truth Labels Available")
    st.info(
        "The uploaded dataset does not include a 'health_stage' column. "
        "Only unsupervised metrics are reported. "
        "Do **not** interpret the unsupervised log-likelihood as a classification accuracy."
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Number-of-states comparison
# ---------------------------------------------------------------------------
st.markdown("## 🔢 Model Comparison: Number of Hidden States")
st.caption("Compare HMMs with different state counts using AIC / BIC (lower = better).")

with st.expander("Run state-count comparison (may take 1–2 minutes)"):
    min_s = st.slider("Min states", 2, 4, 2)
    max_s = st.slider("Max states", 3, 6, 4)
    if max_s < min_s:
        max_s = min_s + 1

    run_compare = st.button("Run Comparison", use_container_width=True)
    if run_compare:
        train_ids = st.session_state.get("train_ids", get_machine_ids(df)[:6])
        try:
            X_tr, len_tr = prepare_sequences(df, prep, train_ids)
            with st.spinner("Training models …"):
                compare_res = compare_n_states(
                    X_tr, len_tr,
                    state_range=list(range(min_s, max_s + 1)),
                    n_features=len(prep.feature_cols),
                    n_iter=100, n_init=2,
                    random_seed=st.session_state.get("random_seed", 42),
                )

            # Show table
            table_rows = []
            for r in compare_res:
                table_rows.append({
                    "States": r["n_states"],
                    "Log-Likelihood": f"{r['log_likelihood']:.2f}",
                    "AIC":  f"{r['aic']:.1f}" if not np.isnan(r["aic"]) else "N/A",
                    "BIC":  f"{r['bic']:.1f}" if not np.isnan(r["bic"]) else "N/A",
                    "Converged": "✅" if r["converged"] else "⚠️",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            fig_cmp = aic_bic_chart(compare_res)
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.markdown("""
**Interpretation:**
- **Lower AIC / BIC** indicates a better trade-off between fit and complexity.
- Both AIC and BIC penalise additional parameters; BIC penalises more strongly.
- The optimal number of states is where AIC or BIC is minimised.
- Note: HMMs may converge to local optima; results may vary between runs.
            """)
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
