# PROJECT GUIDE — Predictive Machine Health Monitoring Using HMMs

> **B.Tech Computer Science — Machine Learning, Unit 4: Hidden Markov Models**  
> Complete documentation of every page, feature, algorithm, and design decision.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [How to Run](#how-to-run)
3. [Dashboard Overview Page](#1-dashboard-overview-page)
4. [Page 1 — Sensor Explorer](#2-page-1--sensor-explorer)
5. [Page 2 — HMM State Inference](#3-page-2--hmm-state-inference)
6. [Page 3 — Machine Health Analysis](#4-page-3--machine-health-analysis)
7. [Page 4 — Model Evaluation](#5-page-4--model-evaluation)
8. [Page 5 — Model Configuration](#6-page-5--model-configuration)
9. [Source Modules Explained](#7-source-modules-explained)
10. [Algorithm Reference](#8-algorithm-reference)
11. [Unit Tests](#9-unit-tests)
12. [Viva Preparation](#10-viva-preparation)

---

## What This Project Does

This is an end-to-end, fully working **Machine Learning system** built for a B.Tech ML course project on **Hidden Markov Models**. The system:

1. **Generates** a synthetic dataset of 10 industrial machines with 4 sensor channels each — temperature, vibration, pressure, and rotation speed — simulating physics-inspired degradation from a Healthy → Degrading → Critical progression.

2. **Trains** a Gaussian HMM using the Baum-Welch Expectation-Maximisation algorithm on the sensor sequences. The model discovers hidden operating states from the raw sensor data **without using any labels**.

3. **Infers** the most likely hidden state at every point in time for any machine — using the Viterbi algorithm for the best sequence and the Forward-Backward algorithm for posterior probabilities.

4. **Interprets** hidden states by comparing their emission means with known sensor degradation trends (temperature and vibration should rise with degradation; pressure and rotation speed should fall). States are automatically ranked and labelled as Healthy, Degrading, or Critical.

5. **Visualises** sensor trends, state timelines, posterior heatmaps, transition matrices, and critical-state probability over time — all on an interactive dark-themed Streamlit dashboard.

6. **Evaluates** the model using both unsupervised metrics (AIC, BIC, log-likelihood on held-out machines) and supervised metrics (accuracy, F1, confusion matrix by comparing against synthetic ground-truth labels).

---

## How to Run

### Requirements
- Python 3.9+
- Windows / macOS / Linux
- No C++ compiler needed (pure NumPy/SciPy HMM)

### Steps (Windows PowerShell)

```powershell
cd "d:\HMM Unit 4"
python -m venv .venv
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser  # only once
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**

### Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
# Expected: 73 passed
```

---

## 1. Dashboard Overview Page

**File:** [`app.py`](app.py)  
**URL:** http://localhost:8501 (root page)

### What you see

The Overview page is the landing page of the application. It auto-loads the built-in synthetic dataset when the app starts.

#### Top Metrics Row (5 cards)

| Metric | Description |
|--------|-------------|
| 🏭 Machines | Total number of machines in the dataset (10) |
| 📊 Observations | Total sensor readings across all machines |
| 🔬 Sensor Features | Number of sensor channels (4) |
| 🔷 Hidden States | Current HMM state count (configurable 2–6, default 3) |
| 🔄 Model Status | Whether the HMM has been trained (Trained / Untrained) |

#### Sidebar Controls

The sidebar is present on every page and contains:

- **Dataset Source** — toggle between "Built-in synthetic" (auto-generated) and "Upload CSV"
- **Load / Reload Dataset** button — generates a fresh synthetic dataset
- **CSV Uploader** — accepts any properly formatted machine sensor CSV
- **Selected Machine** — dropdown to pick which machine to analyse on sensor/health pages
- **Model Summary** — shows current hidden state count, Trained/Not Trained badge, and interpretation confidence

#### Project Objective Section

Explains the four core tasks the HMM solves:
1. Infer hidden operating states (Viterbi decoding)
2. Calculate posterior probabilities at every cycle (Forward-Backward)
3. Detect regime transitions
4. Visualise degradation trajectories

#### Selected Machine Preview

Shows a quick two-sensor time-series preview of the currently selected machine.

#### Current Machine Status (after training)

Once the model is trained, this section shows:
- Latest inferred state (e.g., "Healthy")
- Alert level (NORMAL / WARNING / CRITICAL)
- P(Critical) at the last cycle

If the model is not yet trained, a **🚀 Train Model Now** button is shown that trains a default 3-state HMM in one click (≈ 6–10 seconds).

#### Navigation Guide

Six cards link to the six analysis pages, each with a description of what it shows.

---

## 2. Page 1 — Sensor Explorer

**File:** [`pages/1_Sensor_Explorer.py`](pages/1_Sensor_Explorer.py)  
**Sidebar link:** 🔬 Sensor Explorer  

### Purpose

Explore the raw sensor data **before** any model is involved. This page lets you interactively inspect time-series shapes, statistical distributions, and inter-sensor correlations.

### Controls

| Control | Description |
|---------|-------------|
| Machine selector | Choose which of the 10 machines to view |
| Sensor Channels multiselect | Pick which sensors to display (default: all 4) |
| Cycle Range slider | Zoom into a specific range of operating cycles |

### Sections

#### 📈 Time-Series Plots

A Plotly multi-trace line chart showing the selected sensor channels over operating cycles. If the HMM model is already trained, **background shading** is overlaid using the Viterbi state sequence — green for Healthy, amber for Degrading, red for Critical.

#### 📊 Statistical Summary

A pandas table showing for each sensor: `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`. Useful for understanding sensor value ranges and variability.

#### 📉 Sensor Distributions

One histogram per selected sensor (up to 4 per row), showing the frequency distribution of readings. The synthetic data has Gaussian-like distributions per state, which merge into multi-modal distributions when plotted across all health stages.

#### 🔗 Sensor Correlation Matrix

A Pearson correlation heatmap computed across all machines and cycles. Shows how sensors co-vary:
- Temperature and vibration are positively correlated (both rise during degradation)
- Pressure and rotation_speed are negatively correlated with temperature/vibration

#### 🏷️ Ground-Truth Health Stages

If the dataset includes ground-truth labels (0=Healthy, 1=Degrading, 2=Critical), a bar chart shows how many cycles fall in each stage for the selected machine.

---

## 3. Page 2 — HMM State Inference

**File:** [`pages/2_HMM_State_Inference.py`](pages/2_HMM_State_Inference.py)  
**Requires:** Model trained  

### Purpose

The core inference page. Shows everything the trained HMM infers about a machine's hidden operating state sequence.

### Top Metrics (4 cards)

| Metric | Description |
|--------|-------------|
| Total Cycles | Length of the selected machine's sequence |
| Hidden States | Number of HMM states (N) |
| Latest State | Most recently inferred state label |
| Avg Posterior Entropy | Mean entropy of γ_t across all cycles. Low entropy = confident model |

### Sections

#### 🎨 Hidden-State Timeline

A step-line plot where each point in time is coloured by the Viterbi-decoded hidden state. Shows when the model believes the machine transitioned between operating regimes.

#### 🌡️ Posterior State Probabilities Heatmap

A heatmap where rows = hidden states, columns = cycles, and colour intensity = γ_t(i) (Forward-Backward posterior probability). Reading this:
- Brighter cell = higher probability the machine was in that state at that cycle
- If two rows are equally bright, the model is uncertain
- If one row dominates across many cycles, the model is confident

#### 📊 Emission Means per State

A grouped bar chart showing the learned mean μ_i for each sensor, per hidden state. The values are in standardised units (z-scores). States with higher temperature/vibration means are ranked as more degraded.

#### 🔄 Learned Transition Matrix

An annotated heatmap of A[i,j] = P(next state = j | current state = i). Key interpretations:
- High diagonal → states are "sticky" (machine stays in a state for many cycles before transitioning)
- Off-diagonal probability → rate of regime change

#### 🥧 State Occupancy Pie

What fraction of cycles the machine spent in each state.

#### 🔢 Observed Transition Counts

Raw count of how many times the Viterbi sequence moved from state i to state j.

#### ⏱️ State Dwell Times

Per state: how many separate episodes, mean dwell time (cycles), and max dwell time. A degrading machine would show decreasing mean dwell in Healthy and increasing dwell in Critical.

#### 🧠 State Interpretation

A textual explanation of how the state labels were assigned — which features drove the ranking, and how confident the assignment is.

---

## 4. Page 3 — Machine Health Analysis

**File:** [`pages/3_Machine_Health.py`](pages/3_Machine_Health.py)  
**Requires:** Model trained  

### Purpose

Per-machine **health monitoring dashboard** with an alert system. This is the operational page — the one a plant engineer would look at to decide whether to schedule maintenance.

### Alert Threshold Slider

A slider (0.1 to 0.9) controlling what P(Critical) triggers a WARNING or CRITICAL alert. Adjusting this trades off false positives against false negatives.

### Alert Banner

A colour-coded status box at the top showing:
- 🟢 **NORMAL** — P(Critical) is below the threshold
- 🟡 **WARNING** — P(Critical) is above the threshold
- 🔴 **CRITICAL** — latest state is classified as Critical

Clicking "How is the alert computed?" explains the logic in plain text.

### Sections

#### 📊 Sensor Trends with Inferred States

The full multi-sensor line chart with HMM state shading. Lets you visually confirm that state transitions align with visible changes in sensor behaviour.

#### 🎨 State Timeline

Same as on the State Inference page — a step-line of the Viterbi-decoded state sequence.

#### 📈 P(Critical) Over Time

A filled area chart showing the posterior probability of the Critical state at every cycle, with a horizontal dashed line at the alert threshold. This is the most important chart for predictive maintenance — a rising P(Critical) curve indicates an approaching failure regime.

#### ⚡ Notable State Transitions

A table listing every cycle at which the Viterbi state changed, showing the "From" and "To" states and the P(Critical) at that moment. These are potential maintenance trigger events.

#### 🌡️ Posterior Probabilities Heatmap

Same as on the State Inference page, providing a full view of uncertainty over time.

#### 🏷️ HMM States vs Ground-Truth Labels

A two-subplot comparison showing:
- Top: inferred Viterbi state sequence (numeric)
- Bottom: ground-truth health stage from the synthetic data generator

If the HMM has discovered meaningful structure, the two sequences should broadly agree. Note: the model was trained without any labels, so agreement is evidence of valid unsupervised state discovery.

---

## 5. Page 4 — Model Evaluation

**File:** [`pages/4_Model_Evaluation.py`](pages/4_Model_Evaluation.py)  
**Requires:** Model trained  

### Purpose

Rigorous quantitative evaluation of the trained HMM using both unsupervised and supervised metrics.

### Section 1 — Training Log-Likelihood per Initialization

A bar chart showing the final log-likelihood achieved by each of the N random initializations (default N=3). The best run is selected and stored. This demonstrates why multiple restarts are important for escaping local optima.

**Three metrics shown:**
- Best Log-Likelihood
- Converged? (whether the EM tolerance was met before max iterations)
- Number of Initializations

### Section 2 — Unsupervised Evaluation (Test Sequences)

Uses the 15% test-split machines (those not seen during training or scaling). Computes:

| Metric | Meaning |
|--------|---------|
| Test Log-Likelihood | log P(O_test \| λ) — how well the model explains unseen data |
| AIC | Akaike Information Criterion: -2·LL + 2·k. Lower is better. |
| BIC | Bayesian Information Criterion: -2·LL + k·log(n). Lower, more conservative. |
| Mean Posterior Entropy | Average uncertainty of γ_t. Low = confident model. |

Also shows a **State Occupancy** table for the evaluation set.

### Section 3 — Supervised Evaluation (Ground-Truth Labels)

*Only shown when ground-truth labels are available (synthetic dataset has them).*

**Key caveat:** The HMM is trained without labels. State index 0 doesn't necessarily correspond to Healthy. The evaluation tries all N! permutations of state index ↔ label and picks the one that maximises accuracy.

Shows:
- Accuracy (best permutation)
- F1-Score per health stage (Healthy, Degrading, Critical)
- Precision and Recall per stage
- **Confusion Matrix** — a heatmap showing where predictions match or differ from ground-truth

### Section 4 — Model Comparison: Number of Hidden States

An expandable section that trains multiple HMMs (e.g., 2, 3, 4 states) and compares their AIC/BIC. This is how you choose the optimal number of states:

- Too few states: model underfits, poor log-likelihood
- Too many states: AIC/BIC penalise the extra parameters
- The state count with minimum AIC/BIC is the best trade-off

---

## 6. Page 5 — Model Configuration

**File:** [`pages/5_Model_Configuration.py`](pages/5_Model_Configuration.py)  *(was page 6 before Algorithm Playground removal)*  

### Purpose

Configure all HMM hyperparameters and train or retrain the model.

### Training Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Number of Hidden States | 2–6 | 3 | More states = finer granularity, but needs more data |
| Max Baum-Welch Iterations | 50–500 | 200 | More iterations = longer training, better convergence |
| Random Initializations | 1–5 | 3 | More = more robust against local optima, slower |
| Convergence Tolerance | 1e-6 to 1e-2 | 1e-4 | Smaller = more precise, slower |
| Covariance Type | diag, full, spherical, tied | diag | See below |
| Random Seed | 0–9999 | 42 | Reproducibility |

#### Covariance Types Explained

| Type | Description | When to use |
|------|-------------|-------------|
| `diag` | Each state has its own diagonal covariance — features are treated as independent | Default; fast, works well |
| `full` | Each state has a full covariance matrix — captures inter-sensor correlations | Better fit but needs much more data |
| `spherical` | One scalar variance per state | Fastest, least flexible |
| `tied` | All states share the same covariance matrix | When you expect similar spread across states |

### Sensor Features Multiselect

Choose which sensor channels to include in the HMM training. Removing a sensor that is constant or uncorrelated may improve training stability.

### Dataset Split Display

Shows how the 10 machines are split:
- **70% Training** (~7 machines): HMM EM training + StandardScaler fitting
- **15% Validation** (~1–2 machines): Can be used for interim evaluation
- **15% Test** (~1–2 machines): Held out for final evaluation only

Important: Splitting is at the **machine level**, not at the cycle level, preventing data leakage.

### Train Model Button

On clicking 🚀 Train Model:
1. Fits a `StandardScaler` on the training sequences
2. Stacks all training sequences into a single array (with lengths tracked)
3. Runs Baum-Welch EM for `n_init` random starts, keeps the best
4. Runs `interpret_states()` to auto-label states from emission means
5. Stores everything in `st.session_state` for all pages to access

Post-training, shows:
- Best Log-Likelihood
- Convergence status
- Training observations count
- Interpretation confidence
- Emission Means chart (bar chart per state per sensor)
- Transition Matrix heatmap
- Initial State Probabilities π
- Log-Likelihood per initialization bar chart

---

## 7. Source Modules Explained

### `src/data_loader.py`

**What it does:**
- `generate_synthetic_dataset(n_machines, seed)` — creates a DataFrame with 10 machines, each transitioning through Healthy → Degrading → Critical operating phases with physics-inspired sensor noise
- `load_builtin_dataset()` — loads the synthetic dataset (creates and saves `data/sample_sensor_data.csv` if not present)
- `load_uploaded_csv(file)` — validates and loads a user-supplied CSV
- `_validate_dataframe(df)` — checks required columns, non-empty, correct types
- `get_machine_ids(df)` — returns sorted list of machine IDs
- `get_machine_sequence(df, machine_id)` — returns cycles in order for one machine
- `available_sensor_cols(df)` — returns sensor column names
- `dataset_summary(df)` — returns dict with counts, sequence lengths, etc.

**Constants:**
- `ID_COL = "machine_id"`
- `CYCLE_COL = "cycle"`
- `LABEL_COL = "health_stage"` (0=Healthy, 1=Degrading, 2=Critical)
- `SENSOR_COLS = ["temperature", "vibration", "pressure", "rotation_speed"]`

**Synthetic dataset physics:**
- Health stage transitions: 0 → 1 → 2 as cycles increase (with random transition points)
- Temperature: rises by ~0.1 per degradation level + Gaussian noise (σ=0.5)
- Vibration: rises by ~0.15 per degradation level + Gaussian noise (σ=0.4)
- Pressure: falls by ~0.1 per degradation level + Gaussian noise (σ=0.3)
- Rotation speed: falls by ~0.08 per degradation level + Gaussian noise (σ=0.2)

---

### `src/preprocessing.py`

**`SensorPreprocessor`** — wraps `sklearn.StandardScaler` for sequences.

Key methods:
- `fit(df, machine_ids)` — fits scaler on observations from given machines only
- `transform_sequence(seq_df)` — transforms a single machine's DataFrame into scaled numpy array
- `get_labels(seq_df)` — extracts health_stage labels as integer array
- `feature_stats()` — returns DataFrame of mean/std per feature (from training data)

**`split_machine_ids(ids, train_frac, val_frac, random_seed)`** — deterministic machine-level split.

**`prepare_sequences(df, prep, machine_ids)`** — stacks multiple machines' sequences into one (X, lengths) pair for hmmlearn-style APIs.

**`compute_sensor_stats(seq_df, cols)`** — returns statistics table for the Sensor Explorer.

**`compute_correlation(df, cols)`** — Pearson correlation matrix across all observations.

---

### `src/hmm_model.py`

**`MachineHealthHMM`** — a pure NumPy/SciPy Gaussian HMM. No C++ compiler required.

**Architecture:**
- Uses log-space numerics throughout (log-sum-exp, no underflow)
- Supports `diag`, `full`, `spherical`, and `tied` covariance types
- Multi-start training: runs EM from `n_init` random initializations, keeps the best

**Key methods:**

| Method | Description |
|--------|-------------|
| `fit(X, lengths)` | Train with Baum-Welch EM on stacked sequence array |
| `predict_states(X, lengths)` | Viterbi decoding → most likely state sequence |
| `predict_proba(X, lengths)` | Forward-Backward → posterior γ_t matrix |
| `score(X, lengths)` | log P(O \| λ) using forward algorithm |
| `aic(n_obs)` | Akaike Information Criterion |
| `bic(n_obs)` | Bayesian Information Criterion |

**Stored parameters after `fit()`:**

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `startprob` | (N,) | Learned initial distribution π |
| `transmat` | (N, N) | Learned transition matrix A |
| `means` | (N, D) | Learned emission means μ per state |
| `covars` | varies | Learned emission covariances Σ |
| `best_log_prob` | float | Log-likelihood of best initialization |
| `converged` | bool | Whether EM met the tolerance |
| `training_log_probs` | list | Final LL from each initialization |

**`compare_n_states(X, lengths, state_range, ...)`** — trains multiple HMMs with different state counts and returns AIC/BIC for each.

---

### `src/hmm_algorithms.py`

Educational implementations of all core HMM algorithms for discrete-emission HMMs.

| Function | Inputs | Outputs |
|----------|--------|---------|
| `forward_algorithm(pi, A, B, obs)` | HMM params + obs | α table, log P(O\|λ) |
| `forward_algorithm_scaled(pi, A, B, obs)` | same | α̂ table, c vector, log P(O\|λ) |
| `backward_algorithm_scaled(A, B, obs, c)` | same + c | β̂ table |
| `forward_backward(pi, A, B, obs)` | HMM params + obs | α̂, β̂, γ, ξ, c, log P(O\|λ) |
| `viterbi(pi, A, B, obs)` | HMM params + obs | best_path, δ table, ψ table, log_prob |
| `marginal_decoding(gamma)` | γ matrix | state sequence (argmax per step) |
| `baum_welch_one_step(pi, A, B, obs)` | HMM params + obs | updated pi, A, B |
| `underflow_demo(pi, A, B, obs)` | HMM params + obs | comparison dict with underflow flag |
| `validate_hmm_params(pi, A, B)` | HMM params | error string or None |

`EXAMPLE_HMMS` — a dict of 3 pre-built example HMMs (Weather, Machine Health, Coin-Bias).

---

### `src/state_interpretation.py`

**`interpret_states(n_states, means, feature_cols, posteriors, labels)`**

Given the learned emission means, assigns labels to hidden states by:
1. Computing a **degradation score** for each state: features like temperature and vibration that rise with degradation contribute positively; features like pressure and rotation_speed that fall contribute negatively.
2. Ranking states from lowest to highest degradation score.
3. Mapping ranks to labels: lowest → Healthy, middle → Degrading, highest → Critical.
4. Computing a **confidence** score: "High" if sensor signals are clear, "Medium" or "Low" if sensor statistics overlap between states.

Returns a dict with:
- `state_labels` — `{state_idx: label_str}`
- `state_colors` — `{state_idx: hex_color_str}`
- `ranking` — list of state indices from healthiest to most critical
- `confidence` — "High" / "Medium" / "Low"
- `explanation` — human-readable markdown string describing the assignment

**`compute_alert_level(posteriors, interp, critical_threshold)`**

Uses the latest posterior γ_T to assign an alert level:
- `latest_critical_prob` = γ_T[critical_state_idx]
- If > critical_threshold → CRITICAL alert
- If > critical_threshold / 2 → WARNING alert
- Otherwise → NORMAL

Returns a dict with `alert_label`, `alert_color`, `critical_state_label`, `latest_critical_prob`, `explanation`.

---

### `src/evaluation.py`

**`supervised_evaluation(pred_states, gt_labels, n_states, state_labels)`**

Tries all `n_states!` permutations of state index ↔ ground-truth label mappings. Picks the one that maximises accuracy, then computes:
- Accuracy, Precision, Recall, F1 per class
- Confusion matrix (as numpy array)

**`unsupervised_evaluation(log_likelihood, n_params, n_observations, state_sequence, n_states, posteriors)`**

Returns dict with:
- `log_likelihood`, `aic`, `bic`
- `state_occupancy` — count and percent for each state
- `mean_posterior_entropy` — average of -Σᵢ γ_t(i) · log γ_t(i) across all t

**`transition_counts(state_sequence, n_states)`** — counts observed i→j transitions.

**`state_dwell_times(state_sequence, n_states)`** — per-state episode count, mean and max dwell.

---

### `src/visualizations.py`

All charts use a consistent dark navy theme (`THEME` dict):

| Key | Value | Description |
|-----|-------|-------------|
| `bg` | `#0f1923` | App background |
| `panel` | `#162032` | Card/chart background |
| `border` | `#1e2d3d` | Border lines |
| `accent` | `#4c8ef7` | Blue — primary colour |
| `warning` | `#f39c12` | Amber — degrading state |
| `danger` | `#e74c3c` | Red — critical state |
| `success` | `#2ecc71` | Green — healthy state |
| `text` | `#d0dde8` | Body text |

**Chart functions:**

| Function | Chart type | Used on |
|----------|-----------|---------|
| `sensor_timeseries(seq, cols, state_sequence, ...)` | Multi-trace line | Pages 1, 3 |
| `state_timeline(cycles, states, colors, labels)` | Step-line | Pages 2, 3 |
| `posterior_heatmap(posteriors, labels, cycles)` | Heatmap | Pages 2, 3 |
| `transition_matrix_heatmap(transmat, labels)` | Annotated heatmap | Pages 2, 5 |
| `emission_means_chart(means, cols, labels)` | Grouped bar | Pages 2, 5 |
| `state_occupancy_pie(occ, labels, colors)` | Pie chart | Page 2 |
| `confusion_matrix_chart(cm, labels, title)` | Annotated heatmap | Page 4 |
| `aic_bic_chart(compare_results)` | Grouped bar | Page 4 |
| `loglikelihood_convergence(log_probs)` | Bar chart | Pages 4, 5 |
| `forward_table_chart(alpha, ...)` | Annotated heatmap | (was Algo Playground) |
| `viterbi_delta_chart(delta, ...)` | Annotated heatmap | (was Algo Playground) |
| `scaling_factors_chart(underflow_result)` | Bar + line | (was Algo Playground) |
| `correlation_heatmap(corr, title)` | Annotated heatmap | Page 1 |

---

## 8. Algorithm Reference

### Three Core HMM Problems

| Problem | Question | Algorithm |
|---------|----------|-----------|
| **Evaluation** | What is P(O \| λ)? | Forward |
| **Decoding** | What is the most likely state sequence? | Viterbi |
| **Learning** | What parameters λ best explain O? | Baum-Welch EM |

### Forward Algorithm

**Input:** λ = (π, A, B), observation sequence O = (O₁, …, O_T)  
**Output:** α_t(i) = P(O₁…O_t, S_t=i \| λ), total P(O\|λ)

```
Initialisation:   α₁(i) = π_i · B[i, O₁]

Recursion:        α_{t+1}(j) = [Σᵢ α_t(i) · A[i,j]] · B[j, O_{t+1}]
                  (for t = 1 to T−1)

Termination:      P(O|λ) = Σᵢ α_T(i)
```

**Complexity:** O(T · N²)

**Numerical issue:** For T = 100, N = 3, α_T ≈ 10⁻⁵³ → underflows to 0. Use scaled version.

### Forward Algorithm (Scaled)

```
c_t     = Σᵢ α_t(i)          (sum before normalisation)
α̂_t(i) = α_t(i) / c_t       (normalised)
log P(O|λ) = Σ_t log(c_t)   (log-likelihood from scale factors)
```

All α̂_t values remain in [0, 1] regardless of sequence length.

### Backward Algorithm (Scaled)

```
Initialisation:   β̂_T(i) = 1 / c_T

Recursion:        β̂_t(i) = [Σⱼ A[i,j] · B[j, O_{t+1}] · β̂_{t+1}(j)] / c_t
                  (for t = T−1 down to 1)
```

### Forward-Backward (Posterior Inference)

```
γ_t(i) = α̂_t(i) · β̂_t(i) / Σᵢ [α̂_t(i) · β̂_t(i)]     (posterior state prob)

ξ_t(i,j) = α̂_t(i) · A[i,j] · B[j, O_{t+1}] · β̂_{t+1}(j)  /  Σ_{i,j}(...)
           (joint transition posterior)
```

Both are computed by `forward_backward()` in `src/hmm_algorithms.py`.

### Viterbi Algorithm

**Input:** same as Forward  
**Output:** S* = argmax_S P(S \| O, λ), DP tables δ and ψ

```
Initialisation:   δ₁(j) = log π_j + log B[j, O₁]
                  ψ₁(j) = 0

Recursion:        δ_t(j) = max_i [δ_{t-1}(i) + log A[i,j]] + log B[j, O_t]
                  ψ_t(j) = argmax_i [δ_{t-1}(i) + log A[i,j]]

Backtracking:     S*_T = argmax_j δ_T(j)
                  S*_t = ψ_{t+1}(S*_{t+1})   for t = T−1 downto 1
```

**Complexity:** O(T · N²) — same as Forward.

### Baum-Welch EM

**Goal:** Estimate λ = (π, A, μ, Σ) that maximises P(O \| λ) given unlabelled observations O.

**E-Step:**
```
Run Forward-Backward to compute γ_t(i) and ξ_t(i,j)
```

**M-Step (re-estimation):**
```
π*_i      = γ_1(i)

A*[i,j]   = Σ_{t=1}^{T-1} ξ_t(i,j) / Σ_{t=1}^{T-1} γ_t(i)

μ*_i      = Σ_t γ_t(i) · O_t / Σ_t γ_t(i)

Σ*_i      = Σ_t γ_t(i) · (O_t − μ*_i)(O_t − μ*_i)ᵀ / Σ_t γ_t(i)
```

**Properties:**
- Monotonically non-decreasing log-likelihood per EM step
- Converges to a **local** optimum (not global)
- Multiple random restarts mitigate this

### AIC and BIC

Given:
- LL = log P(O \| λ̂)
- k = number of free parameters
- n = number of observations

```
AIC = -2·LL + 2·k
BIC = -2·LL + k·log(n)
```

Lower values are better. BIC penalises complexity more strongly for large n. Used to select the optimal number of hidden states.

**Free parameters for Gaussian HMM with N states, D features, diagonal covariance:**
```
k = (N-1)          ← initial probabilities π (N values, sum-to-1 constraint removes 1)
  + N·(N-1)        ← transition matrix A (each row sums to 1)
  + N·D            ← emission means μ
  + N·D            ← emission variances (diagonal Σ)
```

---

## 9. Unit Tests

Tests are in [`tests/`](tests/) and run with `pytest tests/ -v`.

**73 tests, expected all passing.**

### `tests/test_data_loader.py` — 18 tests

| Class | Tests |
|-------|-------|
| `TestSyntheticDataset` | Shape, columns, no-NaN, reproducibility, seed independence, cycle ordering, health progression |
| `TestDatasetSummary` | Summary dict keys, n_machines count |
| `TestValidation` | Valid df passes, empty df fails, missing columns fail, too-few-rows fails |
| `TestCSVUpload` | Valid CSV accepted, invalid CSV rejected |

### `tests/test_hmm_algorithms.py` — 34 tests

| Class | Tests |
|-------|-------|
| `TestValidation` | Valid params pass, bad sums rejected, shape mismatches rejected |
| `TestForwardAlgorithm` | Shape, finite log_prob, non-negative α, initial step matches π·B |
| `TestForwardAlgorithmScaled` | Shape, rows sum to 1, log_prob matches unscaled |
| `TestBackwardAlgorithm` | Shape, non-zero final step |
| `TestForwardBackward` | γ sums to 1, correct shape, ξ shape, finite log_prob, γ in [0,1] |
| `TestViterbi` | Path length, valid states, finite log_prob, δ shape, Viterbi ≤ Forward log_prob |
| `TestMarginalDecoding` | Length, valid states |
| `TestBaumWelchStep` | Log-likelihood non-decreasing, π sums to 1, A rows sum to 1, B rows sum to 1 |
| `TestUnderflowDemo` | Keys present, scaled log_prob finite |
| `TestExampleHMMs` | All examples valid params, forward-backward runs on all |

### `tests/test_hmm_model.py` — 21 tests

| Class | Tests |
|-------|-------|
| `TestMachineHealthHMM` | Is fitted, finite best_log_prob, startprob sums to 1, transmat rows sum, means shape, covars shape, predict states valid, predict_proba shape, posteriors sum to 1, posteriors in [0,1], unfitted model raises |
| `TestMultiInit` | n_init log_probs recorded, best_log_prob equals max |
| `TestStateInterpretation` | Returns all required keys, all states have labels, ranking is permutation |
| `TestAlertLevel` | Alert dict keys, prob in [0,1] |
| `TestCompareNStates` | Returns list, result keys, log_likelihood finite |

---

## 10. Viva Preparation

Full viva questions and answers are in [`docs/viva_questions.md`](docs/viva_questions.md).

### Key questions

**Q: What is a Hidden Markov Model?**  
An HMM is a probabilistic model for sequential data where the system transitions between hidden states following the Markov property, and at each time step emits an observable output. Defined by λ = (π, A, B): initial probabilities, transition matrix, emission probabilities.

**Q: Why use an HMM for machine health monitoring?**  
Machine operating conditions (Healthy, Degrading, Critical) are discrete hidden regimes that generate observable sensor readings. HMMs model exactly this — discrete hidden states with continuous (Gaussian) emissions, learned from unlabelled time-series data.

**Q: What is the Markov property?**  
P(S_{t+1} \| S_t, S_{t-1}, …, S₁) = P(S_{t+1} \| S_t). The future state depends only on the present state, not the entire history.

**Q: What is the Forward Algorithm for?**  
Evaluating the model — computing P(O \| λ), the probability that the observed sequence was generated by this model. Used in Baum-Welch EM during the E-step.

**Q: Why does the Forward Algorithm suffer underflow?**  
At each step, forward probabilities are multiplied by small emission and transition probabilities. Over T=100 steps with values ≈ 0.3, α_T ≈ 0.3^100 ≈ 5×10⁻⁵³, which underflows to 0 in floating point.

**Q: How do scaling factors solve underflow?**  
At each step, divide α_t by its sum (c_t). All values stay in [0,1]. Recover log P(O\|λ) = Σ_t log(c_t). This is Rabiner scaling.

**Q: What is the Viterbi Algorithm?**  
Max-product dynamic programming to find S* = argmax_S P(S \| O, λ). Uses log-space recursion (δ table) and backpointers (ψ table). Time complexity O(T·N²).

**Q: How is Viterbi different from Marginal (Forward-Backward) decoding?**  
Viterbi finds the globally most likely path (joint maximum). Marginal decoding picks argmax γ_t independently at each step (marginal maxima). They can disagree — marginal may suggest transitions that have near-zero probability in A.

**Q: What is Baum-Welch?**  
The EM algorithm applied to HMMs. E-step: compute γ and ξ posteriors. M-step: re-estimate all parameters. Guaranteed non-decreasing log-likelihood. Converges to a local optimum.

**Q: How do you select the number of hidden states?**  
Train HMMs with different N and compare AIC and BIC. The N with minimum AIC/BIC gives the best trade-off between model fit and complexity.

**Q: What are AIC and BIC?**  
Information criteria for model selection. AIC = -2LL + 2k; BIC = -2LL + k·log(n). Lower is better. BIC penalises more strongly for more observations.

**Q: How are HMM states labelled as Healthy/Degrading/Critical?**  
By examining the learned emission means. Features known to increase with degradation (temperature, vibration) and decrease with degradation (pressure, rotation_speed) are used to compute a degradation score per state. States are ranked and mapped to labels.

**Q: What is data leakage and how is it prevented?**  
Data leakage occurs when information from the test set contaminates training. This project prevents it by splitting at the machine level (not cycle level) and fitting the StandardScaler only on training machines.

**Q: What is the posterior entropy and what does it tell you?**  
Entropy of γ_t = -Σᵢ γ_t(i) · log γ_t(i). High entropy = model is uncertain which state the machine is in at time t. Low entropy = model is confident.

**Q: What is the difference between supervised and unsupervised evaluation in this project?**  
Unsupervised: log-likelihood, AIC, BIC on held-out machines — no labels needed. Supervised: accuracy, F1, confusion matrix by comparing inferred states to ground-truth health_stage labels. The HMM is trained without labels; supervised evaluation shows whether the discovered states correspond to the true health stages.
