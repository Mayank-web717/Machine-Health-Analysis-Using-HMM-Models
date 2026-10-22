# Predictive Machine Health Monitoring Using Hidden Markov Models

> **B.Tech Computer Science — Machine Learning Project**  
> **Unit 4: Hidden Markov Models**

---

## Abstract

This project implements a complete **Predictive Machine Health Monitoring** system using **Gaussian Hidden Markov Models (HMMs)**. The system analyses sequential sensor readings from industrial machines (temperature, vibration, pressure, rotation speed) to infer hidden operating states, detect degradation patterns, and visualise health trajectories over time.

The project is built as a fully interactive **Streamlit dashboard** with implementations of all core HMM algorithms: Forward, Backward, Forward-Backward, Viterbi, Baum-Welch EM, and Scaling Factors — both as educational tools and as the production inference engine.

> ⚠️ **Transparency:** All results use a synthetic dataset. No real equipment failure predictions are made.

---

## Problem Statement

Industrial machines fail over time. Traditional threshold-based monitoring misses gradual degradation because individual sensor readings remain within bounds even as the machine transitions through distinct operating regimes.

**HMMs model this directly**: the machine's operating condition is a _hidden_ state that probabilistically generates sensor observations. By learning these hidden states from data, the model can:
1. Infer the most likely current state (Viterbi decoding)
2. Compute the probability of each state at each time step (Forward-Backward)
3. Detect when the machine transitions to more critical states

---

## Project Objectives

| # | Objective |
|---|-----------|
| 1 | Load and preprocess sequential machine sensor data |
| 2 | Train a Gaussian HMM using Baum-Welch EM |
| 3 | Infer hidden operating states via Viterbi decoding |
| 4 | Compute posterior state probabilities via Forward-Backward |
| 5 | Interpret and label inferred states from emission statistics |
| 6 | Visualise degradation trajectories and state transitions |
| 7 | Evaluate with AIC/BIC (unsupervised) and accuracy/F1 (supervised) |
| 8 | Provide interactive algorithm demonstrations for all Unit 4 algorithms |

---

## System Architecture

```
HMM Unit 4/
├── app.py                    ← Streamlit entry point (Overview page)
├── requirements.txt
├── README.md                 ← This file (quick-start + theory)
├── PROJECT_GUIDE.md          ← Full guide: every page, algorithm, dataset
├── data/
│   ├── sample_sensor_data.csv   ← Auto-generated on first run
│   └── README.md
├── src/
│   ├── data_loader.py        ← Dataset loading, synthetic generation, validation
│   ├── preprocessing.py      ← StandardScaler, train/val/test split, sequence prep
│   ├── hmm_model.py          ← Pure NumPy/SciPy Gaussian HMM (no C++ compiler needed)
│   ├── hmm_algorithms.py     ← Educational: Forward, Backward, FB, Viterbi, BW-step
│   ├── state_interpretation.py  ← Transparent state labeling from emission means
│   ├── evaluation.py         ← Supervised/unsupervised metrics, dwell times
│   └── visualizations.py     ← Plotly chart factory (dark navy theme)
├── pages/
│   ├── 1_Sensor_Explorer.py
│   ├── 2_HMM_State_Inference.py
│   ├── 3_Machine_Health.py
│   ├── 4_Model_Evaluation.py
│   └── 5_Model_Configuration.py
└── tests/
    ├── test_data_loader.py
    ├── test_hmm_algorithms.py
    └── test_hmm_model.py
```

---

## HMM Theory

### Model Parameters: λ = {π, A, B}

| Symbol | Name | Definition |
|--------|------|-----------|
| **N** | Number of states | Configurable (2–6) |
| **M** | Number of observations | Sensor features |
| **π** | Initial probability | π_i = P(S₁ = i) |
| **A** | Transition matrix | A_{ij} = P(S_{t+1}=j \| S_t=i) |
| **B** | Emission (Gaussian) | P(O_t \| S_t=i) = 𝒩(O_t; μ_i, Σ_i) |

### The Markov Property

```
P(S_{t+1} | S_t, S_{t-1}, ..., S₁) = P(S_{t+1} | S_t)
```

The future state depends only on the present, not the history.

### Forward Algorithm

```
α₁(i) = π_i · B(O₁ | i)
α_{t+1}(j) = [Σᵢ α_t(i) · A_{ij}] · B(O_{t+1} | j)
P(O | λ) = Σᵢ α_T(i)
```

### Backward Algorithm

```
β_T(i) = 1
β_t(i) = Σⱼ A_{ij} · B(O_{t+1} | j) · β_{t+1}(j)
```

### Posterior (Forward-Backward)

```
γ_t(i) = P(S_t=i | O, λ) = α_t(i) · β_t(i) / P(O | λ)
```

### Viterbi (Max-Product in Log Space)

```
δ₁(j) = log π_j + log B(O₁ | j)
δ_t(j) = max_i [δ_{t-1}(i) + log A_{ij}] + log B(O_t | j)
ψ_t(j) = argmax_i [δ_{t-1}(i) + log A_{ij}]
```

### Scaling Factors (Numerical Stability)

To prevent numerical underflow in long sequences:
```
c_t = Σᵢ α_t(i)          ← normalising factor at step t
α̂_t(i) = α_t(i) / c_t   ← scaled forward probability
log P(O | λ) = Σ_t log c_t
```

Without scaling, α_t(i) ≈ (0.3)^100 ≈ 5×10⁻⁵³ → underflows to 0.  
Scaling keeps all values in [0, 1] at every step.

### Baum-Welch EM

**E-Step:** Compute γ_t(i) and ξ_t(i,j) using Forward-Backward.

**M-Step:**
```
π*_i  = γ₁(i)
A*_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
μ*_i  = Σ_t γ_t(i) · O_t / Σ_t γ_t(i)   [Gaussian emission mean]
Σ*_i  = Σ_t γ_t(i)·(O_t−μ*_i)(O_t−μ*_i)ᵀ / Σ_t γ_t(i)
```

Iterates until `|ΔLL| < tolerance`. Guaranteed non-decreasing log-likelihood per step.

---

## Algorithm Playground (Reference)

The following interactive demonstrations are documented here as reference. They were previously available as a dedicated Streamlit page and can be run directly from `src/hmm_algorithms.py`.

### Example HMMs Included

| Name | States | Observations | Description |
|------|--------|--------------|-------------|
| Weather-Activity (2-state) | Sunny, Rainy | Walk, Shop, Clean | Classic Rabiner 1989 |
| Machine Health (3-state) | Healthy, Degrading, Critical | Low, Medium, High vibration | Project domain |
| Coin-Bias (2-state) | Fair, Loaded | Heads, Tails | Simplest illustrative HMM |

### Forward Algorithm Demo

Given HMM parameters λ = (π, A, B) and an observation sequence O:

```python
from src.hmm_algorithms import forward_algorithm, forward_algorithm_scaled

# Unscaled (may underflow for long sequences)
alpha, log_prob = forward_algorithm(pi, A, B, obs)

# Scaled (numerically stable)
alpha_hat, c, log_prob = forward_algorithm_scaled(pi, A, B, obs)
```

**Output:** α table (T × N), log P(O|λ)

### Backward & Forward-Backward Demo

```python
from src.hmm_algorithms import forward_backward

result = forward_backward(pi, A, B, obs)
# result["gamma"]   — (T, N) posterior state probabilities
# result["xi"]      — (T-1, N, N) joint transition posteriors
# result["log_prob"] — log P(O|λ)
```

**Marginal Posterior Decoding** (picks argmax γ_t independently per step):
```python
from src.hmm_algorithms import marginal_decoding
best_states = marginal_decoding(gamma)
```

### Viterbi Demo

```python
from src.hmm_algorithms import viterbi

result = viterbi(pi, A, B, obs)
# result["best_path"]  — most likely state sequence
# result["delta"]      — (T, N) DP log-score table
# result["psi"]        — (T, N) backpointer table
# result["log_prob"]   — log P(S* | O, λ)
```

**Viterbi vs Marginal Decoding:**
- Viterbi finds the globally most-likely path (joint maximum)
- Marginal decoding maximises each step independently (marginal maximum)
- They can disagree — marginal may produce transitions with near-zero probability in A

### Scaling Factors Demo

```python
from src.hmm_algorithms import underflow_demo

result = underflow_demo(pi, A, B, long_obs)
# result["underflow_occurred"]  — bool
# result["log_prob_scaled"]     — correct
# result["log_prob_unscaled"]   — may be -inf
# result["scaling_c"]           — c_t values over time
```

### Baum-Welch One-Step Demo

```python
from src.hmm_algorithms import baum_welch_one_step, forward_backward

ll_before = forward_backward(pi, A, B, obs)["log_prob"]
pi_new, A_new, B_new = baum_welch_one_step(pi, A, B, obs)
ll_after  = forward_backward(pi_new, A_new, B_new, obs)["log_prob"]
# EM guarantee: ll_after >= ll_before (up to numerical precision)
```

### Mathematical Summary Table

```
Algorithm          | Problem       | Complexity  | Output
-------------------|---------------|-------------|---------------------------
Forward (unscaled) | Evaluation    | O(T·N²)     | α, log P(O|λ)
Forward (scaled)   | Evaluation    | O(T·N²)     | α̂, c, log P(O|λ)
Backward           | (helper)      | O(T·N²)     | β̂
Forward-Backward   | Smoothing     | O(T·N²)     | γ, ξ, log P(O|λ)
Viterbi            | Decoding      | O(T·N²)     | best path, δ, ψ
Baum-Welch         | Learning      | O(I·T·N²)   | updated λ = (π, A, B/μ/Σ)
```

---

## Dataset

### Synthetic Dataset (Built-in)
- **10 machines**, each with 80–200 operating cycles
- **4 sensor channels**: temperature, vibration, pressure, rotation_speed
- **Ground-truth labels** (0=Healthy, 1=Degrading, 2=Critical) — used for evaluation only
- Generated reproducibly with `numpy.random.default_rng(42)`
- Physics-inspired: temperature and vibration increase as health degrades; pressure and rotation_speed decrease

### Real Dataset (Optional)
The project accepts the **NASA CMAPSS Turbofan Degradation Dataset**:
- URL: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- The CSV upload feature accepts any properly formatted sensor CSV

---

## Evaluation Methodology

### Unsupervised (no labels)
- **Log-likelihood**: P(O | λ) on held-out test sequences
- **AIC**: -2ℓ + 2k (penalises complexity)
- **BIC**: -2ℓ + k·log(n) (stronger penalty)
- **State occupancy and dwell times**

### Supervised (ground-truth labels available)
- **Best permutation accuracy**: Find the mapping of HMM state indices to GT labels that maximises accuracy
- **Precision, Recall, F1** per health stage
- **Confusion matrix**

> ⚠️ Data leakage prevention: Scaler is fitted only on training machines. Evaluation uses separate held-out machines.

---

## Installation and Execution (Windows PowerShell)

```powershell
# 1. Navigate to project directory
cd "d:\HMM Unit 4"

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# If execution policy blocks activation:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the application
streamlit run app.py

# 6. Run unit tests
python -m pytest tests/ -v
```

The application will open at http://localhost:8501

---

## Limitations and Future Work

| Limitation | Notes |
|-----------|-------|
| Local optima | Baum-Welch may not find the global optimum; mitigated with multiple initializations |
| State labeling | State ↔ health-stage mapping is heuristic (based on emission means) |
| No RUL prediction | Remaining Useful Life estimation requires additional regression modeling |
| Synthetic data | Real industrial datasets may have different statistical properties |
| Single covariance type | Full covariance would capture feature correlations but needs more data |

**Future extensions:**
- Integration with real NASA CMAPSS data
- Remaining Useful Life (RUL) regression using HMM posterior as input
- Online/streaming state inference for live monitoring
- Comparison with LSTM-based approaches

---

## Further Reading

- [Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf)
- [Viterbi, A.J. (1967). Error bounds for convolutional codes](https://ieeexplore.ieee.org/document/1054010)
- [Dempster et al. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm](https://www.jstor.org/stable/2984875)
