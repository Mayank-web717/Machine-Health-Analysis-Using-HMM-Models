"""
pages/5_Algorithm_Playground.py
================================
Interactive educational HMM algorithm explorer.
Demonstrates: Forward, Backward, Forward-Backward, Viterbi, Scaling, Baum-Welch step.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.hmm_algorithms import (
    forward_algorithm, forward_algorithm_scaled, backward_algorithm_scaled,
    forward_backward, viterbi, marginal_decoding, baum_welch_one_step,
    underflow_demo, validate_hmm_params, EXAMPLE_HMMS,
)
from src.visualizations import (
    forward_table_chart, viterbi_delta_chart, scaling_factors_chart, THEME
)

st.set_page_config(page_title="Algorithm Playground", page_icon="🎓", layout="wide")
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
.math-box{background:#162032;border:1px solid #1e2d3d;border-radius:8px;padding:14px;font-family:monospace;font-size:0.85rem;}
</style>""", unsafe_allow_html=True)

st.markdown("# 🎓 Algorithm Playground")
st.markdown("Interact with core HMM algorithms step by step. "
            "All computations use a **discrete-emission HMM** for pedagogical clarity.")
st.divider()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_theory, tab_forward, tab_backward, tab_viterbi, tab_scaling, tab_baumwelch = st.tabs([
    "📚 HMM Theory",
    "→ Forward Algorithm",
    "← Backward & F-B",
    "⚡ Viterbi",
    "🔢 Scaling Factors",
    "🔄 Baum-Welch Step",
])


# ===========================================================================
# TAB 1 — Theory
# ===========================================================================
with tab_theory:
    st.markdown("## 🔍 Hidden Markov Model — Fundamentals")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### What is an HMM?
A **Hidden Markov Model** is a probabilistic model for sequential data where:
- The system moves through a sequence of **hidden states** S₁, S₂, … S_T
- At each time step t, a **visible observation** O_t is emitted
- The hidden states follow the **Markov property**: the next state depends only on the current state

### The Markov Property
> P(S_{t+1} | S_t, S_{t-1}, …, S₁) = P(S_{t+1} | S_t)

The future is independent of the past given the present — this is the fundamental assumption.

### Three HMM Parameters (λ = {π, A, B})

| Symbol | Name | Meaning |
|--------|------|---------|
| **π** | Initial probability | π_i = P(S₁ = i) |
| **A** | Transition matrix | A_{ij} = P(S_{t+1}=j \\| S_t=i) |
| **B** | Emission matrix | B_{ik} = P(O_t=k \\| S_t=i) |
        """)

    with c2:
        st.markdown("""
### Three Core Problems

| Problem | Algorithm | Description |
|---------|-----------|-------------|
| **Evaluation** | Forward | P(O \\| λ) — likelihood of observations |
| **Decoding** | Viterbi | Most likely hidden-state sequence |
| **Learning** | Baum-Welch | Estimate λ from observations |

### Gaussian HMM (continuous)
When observations are **continuous** (e.g., sensor readings), the emission
probability B is replaced by a **Gaussian distribution**:

    P(O_t | S_t=i) = 𝒩(O_t ; μ_i, Σ_i)

where μ_i and Σ_i are the mean vector and covariance matrix for state i.
Learned via **Baum-Welch EM**.

### Sum-Product / Forward-Backward
The Forward-Backward algorithm is an instance of **sum-product dynamic
programming** on the HMM factor graph.  It sums over all possible hidden-state
paths using the factored conditional independence structure, giving exact
marginal posteriors in O(T·N²) time.
        """)

    st.divider()
    st.markdown("### 📐 Mathematical Notation")
    st.markdown("""
```
λ = (π, A, B)          — model parameters

π_i  = P(S_1 = i)                                          — initial state probability

A_{ij} = P(S_{t+1} = j | S_t = i)                         — transition probability
         Row sums: Σ_j A_{ij} = 1  ∀i

B_{ik} = P(O_t = k | S_t = i)      [discrete]              — emission probability
         Row sums: Σ_k B_{ik} = 1  ∀i

α_t(i) = P(O_1…O_t, S_t=i | λ)                            — forward probability
α_1(i) = π_i · B_{i,O_1}
α_{t+1}(j) = [Σ_i α_t(i) · A_{ij}] · B_{j,O_{t+1}}

β_t(i) = P(O_{t+1}…O_T | S_t=i, λ)                       — backward probability
β_T(i) = 1
β_t(i) = Σ_j A_{ij} · B_{j,O_{t+1}} · β_{t+1}(j)

γ_t(i) = P(S_t=i | O, λ) = α_t(i)·β_t(i) / P(O|λ)       — posterior probability

δ_t(j) = max_{S1…St-1} P(S1…St=j, O1…Ot | λ)             — Viterbi score
δ_1(j) = π_j · B_{j,O_1}
δ_t(j) = max_i[δ_{t-1}(i) · A_{ij}] · B_{j,O_t}
```
    """)

    st.divider()
    st.markdown("### 🔄 Linear Dynamical Systems vs HMMs")
    st.markdown("""
| Feature | HMM | Linear Dynamical System (Kalman Filter) |
|---------|-----|-----------------------------------------|
| Hidden state | Discrete | Continuous |
| Emission | Discrete or Gaussian | Gaussian |
| Transition | Discrete Markov | Linear: s_{t+1} = As_t + noise |
| Inference | Forward-Backward | Kalman smoothing |
| Learning | Baum-Welch (EM) | EM / maximum likelihood |
| Tractability | Exact, O(T·N²) | Exact (Gaussian) |

An HMM is preferred when the hidden process has **distinct operating regimes**
(e.g., Healthy / Degrading / Critical), while an LDS models smooth continuous
latent dynamics.
    """)


# ===========================================================================
# Helper: example HMM selector + parameter editor
# ===========================================================================
def _select_example_hmm(prefix=""):
    """Returns (pi, A, B, state_names, obs_names, description)"""
    example_name = st.selectbox(
        "Select Example HMM", list(EXAMPLE_HMMS.keys()), key=f"{prefix}_ex"
    )
    ex = EXAMPLE_HMMS[example_name]
    st.caption(ex["description"])

    with st.expander("Edit HMM Parameters"):
        N = ex["pi"].shape[0]
        M = ex["B"].shape[1]

        st.markdown("**Initial probabilities π**")
        pi_vals = []
        cols_pi = st.columns(N)
        for i, col in enumerate(cols_pi):
            v = col.number_input(f"π[{ex['state_names'][i]}]",
                                  min_value=0.0, max_value=1.0,
                                  value=float(ex["pi"][i]),
                                  step=0.01, key=f"{prefix}_pi_{i}")
            pi_vals.append(v)
        pi = np.array(pi_vals)

        st.markdown("**Transition matrix A** (row i = from state i)")
        A_vals = []
        for i in range(N):
            row = []
            cols_a = st.columns(N)
            for j, col in enumerate(cols_a):
                v = col.number_input(f"A[{ex['state_names'][i]}→{ex['state_names'][j]}]",
                                      min_value=0.0, max_value=1.0,
                                      value=float(ex["A"][i, j]),
                                      step=0.01, key=f"{prefix}_A_{i}_{j}")
                row.append(v)
            A_vals.append(row)
        A = np.array(A_vals)

        st.markdown("**Emission matrix B** (row i = state i, col k = observation k)")
        B_vals = []
        for i in range(N):
            row = []
            cols_b = st.columns(M)
            for k, col in enumerate(cols_b):
                v = col.number_input(f"B[{ex['state_names'][i]}|{ex['obs_names'][k]}]",
                                      min_value=0.0, max_value=1.0,
                                      value=float(ex["B"][i, k]),
                                      step=0.01, key=f"{prefix}_B_{i}_{k}")
                row.append(v)
            B_vals.append(row)
        B = np.array(B_vals)

    # Validate
    err = validate_hmm_params(pi, A, B)
    if err:
        st.error(f"❌ Parameter error: {err}")
        # Normalise as fallback
        pi = pi / (pi.sum() + 1e-12)
        for i in range(N):
            A[i] = A[i] / (A[i].sum() + 1e-12)
        for i in range(N):
            B[i] = B[i] / (B[i].sum() + 1e-12)

    return pi, A, B, ex["state_names"], ex["obs_names"]


def _select_obs_sequence(obs_names, prefix="", default_len=6):
    st.markdown("**Observation Sequence**")
    col_gen, col_manual = st.columns([1, 2])
    with col_gen:
        gen_len = st.number_input("Generate length", 3, 20, default_len, key=f"{prefix}_glen")
        if st.button("Generate Random Sequence", key=f"{prefix}_gen"):
            rng = np.random.default_rng()
            obs = rng.integers(0, len(obs_names), size=gen_len).tolist()
            st.session_state[f"{prefix}_obs_str"] = " ".join(str(o) for o in obs)
    with col_manual:
        default_obs = st.session_state.get(f"{prefix}_obs_str",
                                           " ".join(str(i % len(obs_names))
                                                    for i in range(default_len)))
        obs_str = st.text_input(
            f"Indices (0–{len(obs_names)-1}) space-separated",
            value=default_obs,
            key=f"{prefix}_obs_input",
        )
        st.session_state[f"{prefix}_obs_str"] = obs_str
        st.caption("Observation names: " + ", ".join(f"{i}={n}" for i, n in enumerate(obs_names)))
    try:
        obs = [int(x) for x in obs_str.split()]
        if any(o < 0 or o >= len(obs_names) for o in obs):
            st.error(f"Observations must be in range [0, {len(obs_names)-1}].")
            obs = [0] * default_len
    except Exception:
        st.error("Please enter space-separated integers.")
        obs = [0] * default_len
    return obs


# ===========================================================================
# TAB 2 — Forward Algorithm
# ===========================================================================
with tab_forward:
    st.markdown("## → Forward Algorithm")
    st.markdown("""
The **forward probability** α_t(i) = P(O₁…O_t, S_t=i | λ) accumulates the joint
probability of seeing observations up to time t **and** being in state i at time t.

**Recursion:**
- α₁(i) = π_i · B[i, O₁]
- α_{t+1}(j) = [Σᵢ α_t(i) · A[i,j]] · B[j, O_{t+1}]
- P(O|λ) = Σᵢ α_T(i)
    """)

    pi_f, A_f, B_f, snames_f, onames_f = _select_example_hmm("fwd")
    obs_f = _select_obs_sequence(onames_f, "fwd", 5)

    if st.button("Run Forward Algorithm", key="run_fwd"):
        alpha_u, lp_u = forward_algorithm(pi_f, A_f, B_f, obs_f)
        alpha_s, c_s, lp_s = forward_algorithm_scaled(pi_f, A_f, B_f, obs_f)

        st.markdown("### Unscaled Forward Probabilities (α)")
        fig_u = forward_table_chart(alpha_u, snames_f, onames_f, obs_f, scaled=False,
                                    title="Unscaled α table")
        st.plotly_chart(fig_u, use_container_width=True)

        st.markdown("### Scaled Forward Probabilities (α̂)")
        fig_s = forward_table_chart(alpha_s, snames_f, onames_f, obs_f, scaled=True,
                                    title="Scaled α̂ table")
        st.plotly_chart(fig_s, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("log P(O|λ) — unscaled", f"{lp_u:.6f}" if lp_u > -np.inf else "−∞ (underflow!)")
        c2.metric("log P(O|λ) — scaled",   f"{lp_s:.6f}")

        if np.isinf(lp_u) and not np.isinf(lp_s):
            st.warning("⚠️ Unscaled computation produced −∞ (numerical underflow). "
                       "The scaled version avoids this problem.")

        with st.expander("Step-by-step values"):
            T = len(obs_f)
            N = pi_f.shape[0]
            for t in range(T):
                obs_name = onames_f[obs_f[t]]
                st.markdown(f"**t={t+1}, O={obs_name}**")
                row_data = {snames_f[i]: [f"{alpha_u[t,i]:.6e}", f"{alpha_s[t,i]:.6e}"]
                            for i in range(N)}
                df_step = pd.DataFrame(row_data, index=["α (unscaled)", "α̂ (scaled)"])
                st.dataframe(df_step, use_container_width=True)


# ===========================================================================
# TAB 3 — Backward & Forward-Backward
# ===========================================================================
with tab_backward:
    st.markdown("## ← Backward Algorithm & Forward-Backward Inference")
    st.markdown("""
The **backward probability** β_t(i) = P(O_{t+1}…O_T | S_t=i, λ) — the probability
of future observations given the current state.

**Recursion (from T backward to 1):**
- β_T(i) = 1
- β_t(i) = Σⱼ A[i,j] · B[j, O_{t+1}] · β_{t+1}(j)

**Posterior (from Forward-Backward):**
- γ_t(i) = α_t(i) · β_t(i) / P(O|λ)

This is the probability that the machine was in state i at cycle t, given **all** observations.

The forward-backward algorithm is an example of the **sum-product algorithm** on a
linear chain factor graph, computing exact marginals in O(T·N²).
    """)

    pi_b, A_b, B_b, snames_b, onames_b = _select_example_hmm("bwd")
    obs_b = _select_obs_sequence(onames_b, "bwd", 5)

    if st.button("Run Forward-Backward", key="run_fb"):
        result = forward_backward(pi_b, A_b, B_b, obs_b)

        alpha_h = result["alpha_hat"]
        beta_h  = result["beta_hat"]
        gamma   = result["gamma"]
        c_fb    = result["scaling_c"]

        st.markdown("### γ — Posterior State Probabilities")
        cycles_b = list(range(1, len(obs_b) + 1))
        fig_gamma = forward_table_chart(gamma, snames_b, onames_b, obs_b, scaled=True,
                                        title="γ_t(i) = P(S_t=i | O, λ)")
        st.plotly_chart(fig_gamma, use_container_width=True)

        st.metric("log P(O|λ)", f"{result['log_prob']:.6f}")

        # Marginal decoding
        marginal = marginal_decoding(gamma)
        ml_labels = [snames_b[s] for s in marginal]
        df_dec = pd.DataFrame({
            "Cycle":          list(range(1, len(obs_b)+1)),
            "Observation":    [onames_b[o] for o in obs_b],
            "Marginal State": ml_labels,
        })
        st.markdown("### Marginal Posterior Decoding")
        st.dataframe(df_dec, use_container_width=True)
        st.caption(
            "Marginal decoding picks argmax γ_t independently at each time step. "
            "It may produce paths that violate transition structure; Viterbi finds "
            "the globally most likely path."
        )

        with st.expander("Show α̂ and β̂ tables"):
            st.markdown("**Scaled Forward (α̂)**")
            fig_a = forward_table_chart(alpha_h, snames_b, onames_b, obs_b, scaled=True,
                                        title="Scaled α̂")
            st.plotly_chart(fig_a, use_container_width=True)

            st.markdown("**Scaled Backward (β̂)**")
            fig_bk = forward_table_chart(beta_h, snames_b, onames_b, obs_b, scaled=True,
                                         title="Scaled β̂")
            st.plotly_chart(fig_bk, use_container_width=True)

        with st.expander("Sum-Product dynamic programming explanation"):
            st.markdown("""
The forward-backward algorithm computes exact **marginal posteriors** on a
**linear chain factor graph** using the **sum-product message-passing** rule:

- **Forward pass (α):** Each node sends a message to the right summing over its
  own states (× transition × emission).
- **Backward pass (β):** Each node sends a message to the left in the same way.
- **Marginal posterior:** Product of incoming messages from both directions,
  normalised.

This is O(T · N²) — polynomial in both sequence length and number of states.
It avoids the exponential cost of enumerating all N^T possible state sequences.
            """)


# ===========================================================================
# TAB 4 — Viterbi
# ===========================================================================
with tab_viterbi:
    st.markdown("## ⚡ Viterbi Algorithm")
    st.markdown("""
The **Viterbi algorithm** finds the single **most likely complete hidden-state sequence**:

    S* = argmax_S P(S | O, λ)

It uses **max-product dynamic programming** in log space:

- δ_t(j) = max over S₁…S_t of log P(S₁…S_t=j, O₁…O_t | λ)
- δ₁(j) = log π_j + log B[j, O₁]
- δ_t(j) = max_i [δ_{t-1}(i) + log A[i,j]] + log B[j, O_t]
- Backtracking: ψ_t(j) = argmax_i [δ_{t-1}(i) + log A[i,j]]

**Key difference from marginal decoding:**
Viterbi finds the globally optimal path; marginal decoding maximises each step independently.
    """)

    pi_v, A_v, B_v, snames_v, onames_v = _select_example_hmm("vit")
    obs_v = _select_obs_sequence(onames_v, "vit", 6)

    if st.button("Run Viterbi", key="run_vit"):
        vit_result  = viterbi(pi_v, A_v, B_v, obs_v)
        fb_result   = forward_backward(pi_v, A_v, B_v, obs_v)
        marg        = marginal_decoding(fb_result["gamma"])

        best_path   = vit_result["best_path"]
        delta       = vit_result["delta"]
        psi         = vit_result["psi"]

        st.metric("Viterbi log P(S*|O,λ)", f"{vit_result['log_prob']:.4f}")

        st.markdown("### Viterbi DP Table (δ — log scores)")
        fig_delta = viterbi_delta_chart(delta, snames_v, onames_v, obs_v)
        st.plotly_chart(fig_delta, use_container_width=True)

        st.markdown("### Backpointer Table (ψ)")
        psi_df = pd.DataFrame(
            psi,
            columns=[f"State {j}: {snames_v[j]}" for j in range(len(snames_v))],
            index=[f"t={t+1}" for t in range(len(obs_v))],
        )
        st.dataframe(psi_df.applymap(lambda x: snames_v[x]), use_container_width=True)

        st.markdown("### Decoding Comparison")
        path_df = pd.DataFrame({
            "Cycle":          list(range(1, len(obs_v)+1)),
            "Observation":    [onames_v[o] for o in obs_v],
            "Viterbi Path":   [snames_v[s] for s in best_path],
            "Marginal Decoding": [snames_v[s] for s in marg],
            "Same?":          ["✅" if best_path[t] == marg[t] else "❌" for t in range(len(obs_v))],
        })
        st.dataframe(path_df, use_container_width=True)

        n_diff = sum(1 for t in range(len(obs_v)) if best_path[t] != marg[t])
        if n_diff > 0:
            st.info(f"Viterbi and marginal decoding **disagree on {n_diff} time steps**. "
                    "This shows that locally optimal decisions (marginal) ≠ globally optimal path (Viterbi).")
        else:
            st.success("Both algorithms agree on this sequence!")


# ===========================================================================
# TAB 5 — Scaling Factors
# ===========================================================================
with tab_scaling:
    st.markdown("## 🔢 Numerical Scaling Factors")
    st.markdown("""
### Why Do We Need Scaling?

The forward probability α_t(i) is a **product of many small numbers**:

    α_t(i) ~ (small emission prob)^t

For T=100 time steps with emission prob ~ 0.3:  0.3^100 ≈ 5 × 10⁻⁵³ → **underflow to 0**

### Rabiner Scaling
At each time step t, we normalise α by a scaling factor c_t:

    c_t = 1 / Σᵢ α_t(i)
    α̂_t(i) = α_t(i) × c_t

Then:  log P(O|λ) = -Σ_t log c_t

This keeps all intermediate values in a numerically safe range.

### Alternative: Log-Space
Work entirely in log-space using log-sum-exp:

    log α_t(j) = log_sum_exp_i [ log α_{t-1}(i) + log A[i,j] ] + log B[j, O_t]
    """)

    pi_sc, A_sc, B_sc, snames_sc, onames_sc = _select_example_hmm("sc")

    seq_len = st.slider("Sequence length (to observe underflow)", 5, 50, 20, key="sc_len")
    if st.button("Run Underflow Demonstration", key="run_sc"):
        rng = np.random.default_rng(42)
        obs_sc = rng.integers(0, len(onames_sc), size=seq_len).tolist()

        uf_result = underflow_demo(pi_sc, A_sc, B_sc, obs_sc)

        c1, c2, c3 = st.columns(3)
        c1.metric("log P (scaled)",   f"{uf_result['log_prob_scaled']:.4f}")
        lp_u = uf_result["log_prob_unscaled"]
        c2.metric("log P (unscaled)", f"{lp_u:.4f}" if lp_u > -1e300 else "−∞ (underflow!)")
        c3.metric("Underflow occurred?", "Yes ⚠️" if uf_result["underflow_occurred"] else "No ✅")

        st.markdown("### Scaling Factors c_t")
        fig_sc = scaling_factors_chart(uf_result)
        st.plotly_chart(fig_sc, use_container_width=True)

        with st.expander("Compare unscaled vs scaled values"):
            df_cmp = pd.DataFrame({
                "t": list(range(seq_len)),
                "c_t (scaling)": uf_result["scaling_c"],
                "α_unscaled sum": uf_result["unscaled_alpha"].sum(axis=1),
                "α_scaled sum":   uf_result["scaled_alpha"].sum(axis=1),
            })
            st.dataframe(df_cmp.style.format({"c_t (scaling)": "{:.4e}",
                                               "α_unscaled sum": "{:.4e}",
                                               "α_scaled sum":   "{:.4f}"}),
                         use_container_width=True)


# ===========================================================================
# TAB 6 — Baum-Welch Step
# ===========================================================================
with tab_baumwelch:
    st.markdown("## 🔄 Baum-Welch EM (One Iteration)")
    st.markdown("""
**Baum-Welch** is the **Expectation-Maximisation (EM)** algorithm for HMMs.

### E-Step
Compute posterior probabilities using the current parameters (Forward-Backward):
- γ_t(i) = P(S_t=i | O, λ)
- ξ_t(i,j) = P(S_t=i, S_{t+1}=j | O, λ)

### M-Step
Re-estimate parameters to maximise expected log-likelihood:

    π*_i = γ_1(i)

    A*_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)

    B*_{ik} = Σ_{t: O_t=k} γ_t(i) / Σ_t γ_t(i)

This is repeated until convergence (increase in log-likelihood < tolerance).

> ⚠️ Baum-Welch converges to a **local maximum** of the likelihood.
> Multiple random initialisations improve robustness.
    """)

    pi_bw, A_bw, B_bw, snames_bw, onames_bw = _select_example_hmm("bw")
    obs_bw = _select_obs_sequence(onames_bw, "bw", 8)

    if st.button("Run One Baum-Welch Step", key="run_bw"):
        ll_before = forward_backward(pi_bw, A_bw, B_bw, obs_bw)["log_prob"]
        pi_new, A_new, B_new = baum_welch_one_step(pi_bw, A_bw, B_bw, obs_bw)
        ll_after  = forward_backward(pi_new, A_new, B_new, obs_bw)["log_prob"]

        c1, c2, c3 = st.columns(3)
        c1.metric("log P(O|λ) before", f"{ll_before:.6f}")
        c2.metric("log P(O|λ) after",  f"{ll_after:.6f}")
        delta_ll = ll_after - ll_before
        c3.metric("Δ log-likelihood", f"{delta_ll:+.6f}",
                  delta_color="normal" if delta_ll >= 0 else "inverse")

        if delta_ll >= -1e-8:
            st.success("✅ Log-likelihood did not decrease (EM guarantee: likelihood is non-decreasing).")
        else:
            st.warning("⚠️ Log-likelihood decreased slightly — this can happen due to numerical precision.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### π (initial probabilities)")
            pi_df = pd.DataFrame({
                "State": snames_bw,
                "Before": pi_bw,
                "After":  pi_new,
                "Δ":      pi_new - pi_bw,
            })
            st.dataframe(pi_df.style.format({"Before": "{:.4f}", "After": "{:.4f}", "Δ": "{:+.4f}"}),
                         use_container_width=True)

            st.markdown("#### A (transition matrix)")
            for i, sname in enumerate(snames_bw):
                st.markdown(f"Row {sname}")
                row_df = pd.DataFrame({
                    "To State": snames_bw,
                    "Before":   A_bw[i],
                    "After":    A_new[i],
                    "Δ":        A_new[i] - A_bw[i],
                })
                st.dataframe(row_df.style.format({"Before": "{:.4f}", "After": "{:.4f}", "Δ": "{:+.4f}"}),
                             use_container_width=True)

        with col2:
            st.markdown("#### B (emission matrix)")
            for i, sname in enumerate(snames_bw):
                st.markdown(f"State {sname}")
                b_df = pd.DataFrame({
                    "Observation": onames_bw,
                    "Before":      B_bw[i],
                    "After":       B_new[i],
                    "Δ":           B_new[i] - B_bw[i],
                })
                st.dataframe(b_df.style.format({"Before": "{:.4f}", "After": "{:.4f}", "Δ": "{:+.4f}"}),
                             use_container_width=True)
