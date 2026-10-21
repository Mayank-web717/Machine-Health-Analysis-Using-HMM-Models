# Viva Questions: Predictive Machine Health Monitoring Using HMMs

> **B.Tech CS — ML Unit 4: Hidden Markov Models**
> Prepared for project viva examination

---

## Q1. What is a Hidden Markov Model and how does it differ from a regular Markov chain?

**Answer:**
A **Markov chain** has a set of states that are directly observable. At each time step, the system transitions from one state to another with known probabilities.

A **Hidden Markov Model (HMM)** has two layers:
- **Hidden states** S₁, S₂, … S_T that are *not directly observable*
- **Observations** O₁, O₂, … O_T that are generated from the hidden states

The key difference: in an HMM, we only see the observations and must *infer* the hidden states using probabilistic reasoning. In machine health monitoring, the "hidden states" are operating conditions (Healthy/Degrading/Critical) that we cannot observe directly — we only see the sensor readings.

---

## Q2. State the Markov property and explain why it is important in HMMs.

**Answer:**
The **Markov property** states:

> P(S_{t+1} | S_t, S_{t-1}, …, S₁) = P(S_{t+1} | S_t)

The next state depends *only* on the current state, not on the entire history. This is also called the **first-order Markov assumption**.

**Importance:**
1. It makes the model **computationally tractable** — the joint probability P(S₁…S_T, O₁…O_T) can be decomposed as a product of local factors.
2. It enables **efficient algorithms** (Forward, Viterbi) that run in O(T·N²) instead of O(N^T).
3. It is a reasonable approximation for many real-world sequential processes.

---

## Q3. What are the three parameters of an HMM? Write the mathematical notation.

**Answer:**
An HMM is defined by **λ = {π, A, B}**:

| Parameter | Symbol | Definition |
|-----------|--------|-----------|
| Initial probabilities | **π** | π_i = P(S₁ = i), Σᵢ π_i = 1 |
| Transition matrix | **A** | A_{ij} = P(S_{t+1}=j \| S_t=i), Σⱼ A_{ij} = 1 ∀i |
| Emission distribution | **B** | B_{ik} = P(O_t=k \| S_t=i) (discrete) or 𝒩(O_t; μᵢ, Σᵢ) (Gaussian) |

In this project, we use a **Gaussian HMM** where:
- B is replaced by Gaussian emission: P(O_t | S_t=i) = 𝒩(O_t; μᵢ, diagΣᵢ)
- μᵢ = mean sensor vector for state i
- Σᵢ = diagonal covariance for state i

---

## Q4. Explain the Forward Algorithm. What does α_t(i) represent?

**Answer:**
The **forward probability** α_t(i) = P(O₁, O₂, … O_t, S_t=i | λ) represents the joint probability of:
1. Seeing observations O₁ through O_t
2. Being in hidden state i at time t

**Recursion:**
```
α₁(i) = π_i · B(O₁ | i)                              [Initialization]
α_{t+1}(j) = [Σᵢ α_t(i) · A_{ij}] · B(O_{t+1} | j)  [Recursion]
P(O | λ) = Σᵢ α_T(i)                                  [Termination]
```

**Complexity:** O(T·N²) — polynomial, not exponential.

The Forward algorithm solves the **evaluation problem**: given a model λ, what is the probability of the observation sequence?

---

## Q5. Explain the Backward Algorithm. What does β_t(i) represent?

**Answer:**
The **backward probability** β_t(i) = P(O_{t+1}, O_{t+2}, … O_T | S_t=i, λ) represents the probability of all *future* observations given that we are in state i at time t.

**Recursion (from T back to 1):**
```
β_T(i) = 1                                                          [Initialization]
β_t(i) = Σⱼ A_{ij} · B(O_{t+1} | j) · β_{t+1}(j)   [Recursion]
```

**Combined with Forward:**
```
γ_t(i) = P(S_t=i | O, λ) = α_t(i) · β_t(i) / P(O|λ)
```

The backward probabilities are used in both the **Forward-Backward algorithm** (computing posteriors) and the **Baum-Welch** M-step.

---

## Q6. What is the Forward-Backward algorithm? How does it compute posterior state probabilities?

**Answer:**
The **Forward-Backward algorithm** computes the **posterior probability** γ_t(i) = P(S_t=i | O₁…O_T, λ) — the probability that the system was in state i at time t given the *entire* observation sequence.

**Steps:**
1. **Forward pass:** Compute α_t(i) from t=1 to T (left to right)
2. **Backward pass:** Compute β_t(i) from t=T to 1 (right to left)
3. **Combine:** γ_t(i) = α_t(i) · β_t(i) / Σⱼ α_t(j) · β_t(j)

**Key insight:** α accounts for all past evidence; β accounts for all future evidence. Their product gives the probability given all evidence.

This is an instance of the **sum-product algorithm** on the HMM's factor graph, computing exact marginals efficiently.

---

## Q7. What is the sum-product algorithm and how does it relate to HMM inference?

**Answer:**
The **sum-product algorithm** is a general message-passing algorithm for computing marginal probabilities on factor graphs. It works by passing "messages" between variable nodes and factor nodes.

**In HMMs:**
- The HMM has a **linear chain factor graph** — states S₁…S_T connected in sequence
- The **forward pass** sends messages from left to right (α messages)
- The **backward pass** sends messages from right to left (β messages)
- The **marginal posterior** γ_t(i) is the product of the two incoming messages at each node

The forward-backward algorithm IS the sum-product algorithm applied to the HMM chain, giving exact marginals in O(T·N²) by exploiting the Markov independence structure.

---

## Q8. Why do we need scaling factors in the Forward algorithm?

**Answer:**
**Problem:** The forward probability α_t(i) is a product of many probabilities:
- At each step, α is multiplied by transition and emission probabilities (typically < 1)
- For T=100 steps with emission prob ≈ 0.3: 0.3^100 ≈ 5×10⁻⁵³
- Floating-point underflow to 0.0 — we lose all information!

**Solution — Rabiner scaling:**
At each time step t, we normalise α by a scaling factor c_t:
```
c_t = 1 / Σᵢ α_t(i)
α̂_t(i) = c_t · α_t(i)                      [normalised, sums to 1]
log P(O|λ) = -Σ_t log c_t
```

**Alternative:** Work entirely in **log-space** using log-sum-exp:
```
log α_t(j) = log_sum_exp_i[log α_{t-1}(i) + log A_{ij}] + log B[j, O_t]
```

This project implements both approaches for educational comparison.

---

## Q9. Explain the Viterbi algorithm. How does it differ from the Forward-Backward algorithm?

**Answer:**
The **Viterbi algorithm** finds the **single most likely complete hidden-state sequence**:
```
S* = argmax P(S₁…S_T | O₁…O_T, λ)
```

**DP table (log-space):**
```
δ₁(j) = log π_j + log B(O₁ | j)
δ_t(j) = max_i [δ_{t-1}(i) + log A_{ij}] + log B(O_t | j)
ψ_t(j) = argmax_i [δ_{t-1}(i) + log A_{ij}]   [backpointer]
```
Backtrack from argmax δ_T to recover S*.

**Differences from Forward-Backward:**

| Aspect | Forward-Backward | Viterbi |
|--------|-----------------|---------|
| Operation | Sum (marginalise) | Max (optimise) |
| Output | γ_t(i) for each (t,i) | One complete state path |
| Type | Sum-product | Max-product |
| Produces | Local posteriors | Global optimum path |

Both run in O(T·N²). The Viterbi path may disagree with marginal decoding (argmax γ_t) because marginal decoding is locally optimal but not globally.

---

## Q10. Explain Baum-Welch. Why is it called an EM algorithm?

**Answer:**
**Baum-Welch** is the **Expectation-Maximisation (EM)** algorithm for learning HMM parameters from unlabelled observation sequences.

**E-Step (Expectation):**
Given current parameters λ, compute the expected sufficient statistics:
- γ_t(i) = P(S_t=i | O, λ)         [posterior state probs]
- ξ_t(i,j) = P(S_t=i, S_{t+1}=j | O, λ)  [joint transition posteriors]

**M-Step (Maximisation):**
Re-estimate parameters to maximise expected log-likelihood:
```
π*_i = γ₁(i)
A*_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
μ*_i = Σ_t γ_t(i)·O_t / Σ_t γ_t(i)        [Gaussian mean]
Σ*_i = Σ_t γ_t(i)·(O_t-μ*_i)(O_t-μ*_i)ᵀ / Σ_t γ_t(i)
```

**Why EM?** Because the log-likelihood is intractable to maximise directly (hidden states are unobserved). EM introduces the hidden states as latent variables and alternates between computing their expected values (E-step) and maximising parameters (M-step).

**Guarantees:** Log-likelihood is non-decreasing per iteration. Convergence to a *local* maximum (not global).

---

## Q11. What are the emission distributions in this project? Why Gaussian?

**Answer:**
This project uses a **Gaussian (Normal) emission distribution**:

```
P(O_t | S_t = i) = 𝒩(O_t; μᵢ, diagΣᵢ)
```

Where:
- O_t ∈ ℝⁿ is the sensor reading vector (temperature, vibration, etc.)
- μᵢ = mean sensor readings for state i
- diagΣᵢ = diagonal covariance (independent features per state)

**Why Gaussian?**
1. Sensor readings are **continuous-valued** — not discrete
2. Physical sensor noise is well-modelled by Gaussian distributions
3. The Gaussian family is closed under the M-step update — parameters have closed-form solutions
4. Diagonal covariance is computationally efficient and avoids overfitting with limited data

For discrete observations (e.g., in the Algorithm Playground), we use a **categorical emission** B[i,k] = P(observation k | state i).

---

## Q12. How do you interpret HMM hidden states? Are they automatically "Healthy/Degrading/Critical"?

**Answer:**
**No** — HMM hidden states are **purely mathematical constructs**. The state numbered "0" has no inherent meaning; it is simply the state that best explains a cluster of observations.

**Interpretation requires a post-hoc analysis:**
1. Examine the **emission means** μᵢ for each state
2. States with higher temperature and vibration and lower pressure → likely more degraded
3. Rank states by a **degradation score** derived from emission means
4. Assign labels (Healthy/Degrading/Critical) based on this ranking

**Caveats:**
- If emission means are similar across states, labels are unreliable (low confidence)
- The same physical condition may be split across multiple states if N is too large
- Labels are *interpretations*, not ground truth — always display confidence level

This project implements a transparent `interpret_states()` function that computes confidence and explains its reasoning.

---

## Q13. How do you evaluate an HMM model? What metrics are appropriate?

**Answer:**

**Unsupervised evaluation (no labels):**
- **Log-likelihood** P(O|λ) on held-out test sequences — higher is better
- **AIC** = -2·ℓ + 2·k — penalises over-parameterisation (k = number of free parameters)
- **BIC** = -2·ℓ + k·log(n) — stronger penalty, preferred for model selection
- **State occupancy** — all states should be used; empty states suggest N is too large
- **Convergence** — did Baum-Welch converge within the iteration limit?

**Supervised evaluation (with ground-truth labels):**
- **Best-permutation accuracy** — try all permutations of state indices, report best accuracy
- **F1-score, Precision, Recall** per health stage
- **Confusion matrix**

**Important:** Never report classification accuracy when there are no ground-truth labels.

---

## Q14. What is data leakage and how is it prevented in this project?

**Answer:**
**Data leakage** occurs when information from the test/evaluation set contaminates the training process, leading to inflated performance metrics.

**Leakage risks in time-series / HMM projects:**
1. **Scaler leakage**: Fitting StandardScaler on all data (including test) before splitting
2. **Sequence mixing**: Treating the last observation of Machine 1 and first of Machine 2 as adjacent
3. **Label leakage**: Training the HMM using ground-truth health labels

**Prevention in this project:**
1. `SensorPreprocessor.fit()` is called **only on training machine IDs**
2. `prepare_sequences()` keeps machines separate using `lengths` parameter — hmmlearn-style concatenation with explicit lengths prevents cross-machine transitions
3. The HMM is trained on raw sensor observations only — ground-truth labels are only used in the post-hoc evaluation step

---

## Q15. What is a Linear Dynamical System? How does it differ from an HMM?

**Answer:**
A **Linear Dynamical System (LDS)**, also known as a Kalman filter model, models continuous latent dynamics:

```
s_{t+1} = A·s_t + ε_t,    ε_t ~ 𝒩(0, Q)    [state transition]
O_t      = C·s_t + ν_t,    ν_t ~ 𝒩(0, R)    [observation]
```

**Key differences:**

| Feature | HMM | LDS (Kalman) |
|---------|-----|-------------|
| Hidden state type | **Discrete** (finite set) | **Continuous** (real vector) |
| Transition | Markov: discrete probability table | Linear Gaussian dynamics |
| Inference | Forward-Backward (exact, O(T·N²)) | Kalman smoothing (exact, Gaussian) |
| Learning | Baum-Welch (EM) | EM or maximum likelihood |
| Best for | **Distinct operating regimes** | Smooth continuous evolution |

**For machine health monitoring:**
- HMM is better when machines clearly switch between discrete health stages
- LDS is better when degradation is a smooth continuous process
- In practice, HMMs are more interpretable for anomaly detection tasks

---

## End-to-End Pipeline Summary

1. **Data Loading**: Generate or load sensor CSV → validate columns and types
2. **Preprocessing**: Fit StandardScaler on training machines; transform all sequences
3. **Training**: Run Baum-Welch EM (multiple initializations); select best by log-likelihood
4. **Inference**: Apply Viterbi (most likely states) and Forward-Backward (posteriors)
5. **Interpretation**: Map state indices → health labels using emission means ranking
6. **Evaluation**: AIC/BIC (all data), accuracy/F1 (supervised, on held-out test machines)
7. **Alerting**: Compute P(critical state) at latest time step; compare to threshold
8. **Visualization**: Dashboard shows all results interactively
