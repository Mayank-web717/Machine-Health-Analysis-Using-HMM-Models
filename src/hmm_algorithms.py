"""
hmm_algorithms.py
=================
Educational implementations of core HMM algorithms for a DISCRETE-emission HMM.

These implementations are for Unit 4 pedagogy — they show the mathematics
step by step.  The main application uses hmmlearn's Gaussian HMM; these
implementations illustrate the algorithms on small discrete examples.

Algorithms implemented:
  - Forward algorithm  (with and without scaling)
  - Backward algorithm
  - Forward-backward (posterior probabilities)
  - Viterbi decoding
  - Baum-Welch EM (one iteration, for illustration)
  - Sum-product dynamic programming explanation
  - Scaling factors demonstration

All functions operate on NumPy arrays and return intermediate tables so
the Streamlit UI can display them to the student.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ===========================================================================
# Helper: validate HMM parameters
# ===========================================================================

def validate_hmm_params(pi: np.ndarray, A: np.ndarray, B: np.ndarray) -> Optional[str]:
    """
    Validate HMM parameters.  Return an error message or None.

    Parameters
    ----------
    pi : (N,) initial state probabilities
    A  : (N, N) transition matrix
    B  : (N, M) emission matrix
    """
    N = pi.shape[0]
    if A.shape != (N, N):
        return f"Transition matrix A must be {N}×{N}, got {A.shape}."
    if B.shape[0] != N:
        return f"Emission matrix B must have {N} rows (one per state), got {B.shape[0]}."

    tol = 1e-6
    if abs(pi.sum() - 1.0) > tol:
        return f"Initial probabilities π must sum to 1.0 (got {pi.sum():.6f})."
    if np.any(pi < 0):
        return "Initial probabilities π must all be ≥ 0."
    for i in range(N):
        if abs(A[i].sum() - 1.0) > tol:
            return f"Row {i} of transition matrix A must sum to 1.0 (got {A[i].sum():.6f})."
    for i in range(N):
        if abs(B[i].sum() - 1.0) > tol:
            return f"Row {i} of emission matrix B must sum to 1.0 (got {B[i].sum():.6f})."

    return None


# ===========================================================================
# Forward algorithm  (unscaled)
# ===========================================================================

def forward_algorithm(
    pi: np.ndarray,
    A:  np.ndarray,
    B:  np.ndarray,
    obs: List[int],
) -> Tuple[np.ndarray, float]:
    """
    Standard (unscaled) forward algorithm.

    Parameters
    ----------
    pi  : (N,) initial state probabilities
    A   : (N, N) row-stochastic transition matrix
    B   : (N, M) row-stochastic emission matrix
    obs : list of T integer observation indices

    Returns
    -------
    alpha : (T, N) forward probabilities
    log_prob : log P(O | λ)   (may be -inf for long sequences → underflow)
    """
    N = pi.shape[0]
    T = len(obs)
    alpha = np.zeros((T, N))

    # Initialisation
    alpha[0] = pi * B[:, obs[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.dot(alpha[t - 1], A[:, j]) * B[j, obs[t]]

    prob = alpha[-1].sum()
    log_prob = np.log(prob) if prob > 0 else -np.inf
    return alpha, log_prob


# ===========================================================================
# Forward algorithm  (scaled)
# ===========================================================================

def forward_algorithm_scaled(
    pi: np.ndarray,
    A:  np.ndarray,
    B:  np.ndarray,
    obs: List[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Numerically stable forward algorithm using Rabiner-style scaling factors.

    At each time step t, alpha_hat[t] is normalised so the rows sum to 1.
    The normalising factor c[t] = sum(raw[t]).
    log P(O|λ) = sum(log c[t]).

    Returns
    -------
    alpha_hat : (T, N) scaled forward probabilities
    c         : (T,)  scaling factors
    log_prob  : log P(O | λ)
    """
    N = pi.shape[0]
    T = len(obs)
    alpha_hat = np.zeros((T, N))
    c         = np.zeros(T)

    # Initialisation
    raw = pi * B[:, obs[0]]
    c[0] = raw.sum()
    alpha_hat[0] = raw / (c[0] + 1e-300)

    # Recursion
    for t in range(1, T):
        raw = np.zeros(N)
        for j in range(N):
            raw[j] = np.dot(alpha_hat[t - 1], A[:, j]) * B[j, obs[t]]
        c[t] = raw.sum()
        alpha_hat[t] = raw / (c[t] + 1e-300)

    # log P(O|λ) = Σ log c[t]   where c[t] = sum(raw)
    log_prob = float(np.sum(np.log(c + 1e-300)))
    return alpha_hat, c, log_prob


# ===========================================================================
# Backward algorithm  (scaled)
# ===========================================================================

def backward_algorithm_scaled(
    A:  np.ndarray,
    B:  np.ndarray,
    obs: List[int],
    c:  np.ndarray,
) -> np.ndarray:
    """
    Scaled backward algorithm.

    Parameters
    ----------
    A, B, obs : same as forward_algorithm_scaled
    c         : scaling factors from forward_algorithm_scaled

    Returns
    -------
    beta_hat : (T, N) scaled backward probabilities
    """
    N = A.shape[0]
    T = len(obs)
    beta_hat = np.zeros((T, N))

    # Initialisation
    beta_hat[T - 1] = 1.0 / (c[T - 1] + 1e-300)

    # Recursion (backwards)
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta_hat[t, i] = np.dot(A[i], B[:, obs[t + 1]] * beta_hat[t + 1])
        beta_hat[t] /= (c[t] + 1e-300)

    return beta_hat


# ===========================================================================
# Posterior state probabilities  (forward-backward)
# ===========================================================================

def forward_backward(
    pi:  np.ndarray,
    A:   np.ndarray,
    B:   np.ndarray,
    obs: List[int],
) -> Dict:
    """
    Full forward-backward inference.

    Returns a dict containing:
      alpha_hat  : (T, N) scaled forward probabilities
      beta_hat   : (T, N) scaled backward probabilities
      gamma      : (T, N) posterior state probabilities P(S_t=i | O, λ)
      xi         : (T-1, N, N) transition posteriors P(S_t=i, S_{t+1}=j | O, λ)
      scaling_c  : (T,) scaling factors
      log_prob   : log P(O | λ)
    """
    alpha_hat, c, log_prob = forward_algorithm_scaled(pi, A, B, obs)
    beta_hat               = backward_algorithm_scaled(A, B, obs, c)

    T, N = alpha_hat.shape

    # Posterior  γ_t(i) = α̂_t(i) * β̂_t(i)  (unnormalised; normalise per row)
    gamma_raw = alpha_hat * beta_hat
    row_sums  = gamma_raw.sum(axis=1, keepdims=True)
    gamma     = gamma_raw / (row_sums + 1e-300)

    # Joint transition posteriors  ξ_t(i,j)
    xi = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        numer = np.outer(alpha_hat[t], beta_hat[t + 1]) * A * B[:, obs[t + 1]]
        xi[t] = numer / (numer.sum() + 1e-300)

    return {
        "alpha_hat": alpha_hat,
        "beta_hat":  beta_hat,
        "gamma":     gamma,
        "xi":        xi,
        "scaling_c": c,
        "log_prob":  log_prob,
    }


# ===========================================================================
# Viterbi algorithm
# ===========================================================================

def viterbi(
    pi:  np.ndarray,
    A:   np.ndarray,
    B:   np.ndarray,
    obs: List[int],
) -> Dict:
    """
    Viterbi decoding in log-space (numerically stable).

    Returns
    -------
    dict with:
      best_path  : list[int] most-likely hidden-state sequence
      delta      : (T, N) log-probability DP table
      psi        : (T, N) backpointer table
      log_prob   : log-probability of the best path
    """
    N = pi.shape[0]
    T = len(obs)

    log_pi = np.log(pi + 1e-300)
    log_A  = np.log(A  + 1e-300)
    log_B  = np.log(B  + 1e-300)

    delta = np.full((T, N), -np.inf)
    psi   = np.zeros((T, N), dtype=int)

    # Initialisation
    delta[0] = log_pi + log_B[:, obs[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            scores       = delta[t - 1] + log_A[:, j]
            psi[t, j]    = int(np.argmax(scores))
            delta[t, j]  = scores[psi[t, j]] + log_B[j, obs[t]]

    # Backtracking
    best_path    = [0] * T
    best_path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]

    return {
        "best_path": best_path,
        "delta":     delta,
        "psi":       psi,
        "log_prob":  float(delta[T - 1, best_path[T - 1]]),
    }


# ===========================================================================
# Marginal posterior decoding  (comparison with Viterbi)
# ===========================================================================

def marginal_decoding(gamma: np.ndarray) -> List[int]:
    """
    Marginal posterior decoding: argmax_i γ_t(i) for each t.
    May produce state sequences that violate the transition structure.
    """
    return gamma.argmax(axis=1).tolist()


# ===========================================================================
# Numerical underflow demonstration
# ===========================================================================

def underflow_demo(
    pi: np.ndarray,
    A:  np.ndarray,
    B:  np.ndarray,
    obs: List[int],
) -> Dict:
    """
    Compare unscaled vs. scaled forward probabilities to demonstrate underflow.

    Returns a dict with:
      unscaled_alpha : (T, N) may contain zeros due to underflow
      scaled_alpha   : (T, N) numerically stable
      scaling_c      : (T,)  scaling factors
      log_prob_scaled: float
      log_prob_unscaled: float or -inf
      underflow_occurred: bool
    """
    alpha_unscaled, log_unscaled = forward_algorithm(pi, A, B, obs)
    alpha_scaled, c, log_scaled  = forward_algorithm_scaled(pi, A, B, obs)

    underflow = bool(np.any(alpha_unscaled == 0.0) and len(obs) > 5)

    return {
        "unscaled_alpha":   alpha_unscaled,
        "scaled_alpha":     alpha_scaled,
        "scaling_c":        c,
        "log_prob_scaled":  log_scaled,
        "log_prob_unscaled": log_unscaled,
        "underflow_occurred": underflow,
    }


# ===========================================================================
# Baum-Welch  (one illustrative M-step re-estimation, educational)
# ===========================================================================

def baum_welch_one_step(
    pi:  np.ndarray,
    A:   np.ndarray,
    B:   np.ndarray,
    obs: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one E-step + one M-step of Baum-Welch to illustrate parameter re-estimation.
    Returns updated (pi_new, A_new, B_new).
    """
    result = forward_backward(pi, A, B, obs)
    gamma  = result["gamma"]
    xi     = result["xi"]

    N = pi.shape[0]
    M = B.shape[1]
    T = len(obs)

    # M-step: re-estimate π
    pi_new = gamma[0]

    # M-step: re-estimate A
    A_new = np.zeros((N, N))
    for i in range(N):
        denom = xi[:, i, :].sum()
        for j in range(N):
            A_new[i, j] = xi[:, i, j].sum() / (denom + 1e-300)

    # M-step: re-estimate B
    B_new = np.zeros((N, M))
    for j in range(N):
        denom = gamma[:, j].sum()
        for k in range(M):
            indicator = np.array([1.0 if obs[t] == k else 0.0 for t in range(T)])
            B_new[j, k] = (gamma[:, j] * indicator).sum() / (denom + 1e-300)

    return pi_new, A_new, B_new


# ===========================================================================
# Default small example HMM  (for Algorithm Playground)
# ===========================================================================

EXAMPLE_HMMS = {
    "Weather-Activity (2 states)": {
        "description": (
            "2 hidden states (Sunny / Rainy), 3 observations (Walk / Shop / Clean). "
            "Classic Rabiner 1989 example."
        ),
        "state_names": ["Sunny", "Rainy"],
        "obs_names":   ["Walk", "Shop", "Clean"],
        "pi": np.array([0.6, 0.4]),
        "A":  np.array([[0.7, 0.3],
                        [0.4, 0.6]]),
        "B":  np.array([[0.5, 0.4, 0.1],
                        [0.1, 0.3, 0.6]]),
    },
    "Machine Health (3 states)": {
        "description": (
            "3 hidden states (Healthy / Degrading / Critical), "
            "4 discrete observations (Low / Medium / High / Very-High sensor reading)."
        ),
        "state_names": ["Healthy", "Degrading", "Critical"],
        "obs_names":   ["Low", "Medium", "High", "Very-High"],
        "pi": np.array([0.8, 0.15, 0.05]),
        "A":  np.array([[0.85, 0.13, 0.02],
                        [0.05, 0.80, 0.15],
                        [0.01, 0.09, 0.90]]),
        "B":  np.array([[0.60, 0.30, 0.08, 0.02],
                        [0.15, 0.45, 0.30, 0.10],
                        [0.02, 0.10, 0.38, 0.50]]),
    },
    "Coin Flip (2 states)": {
        "description": (
            "2 hidden states (Fair / Biased coin), 2 observations (Heads / Tails). "
            "Simple bimodal example."
        ),
        "state_names": ["Fair", "Biased"],
        "obs_names":   ["Heads", "Tails"],
        "pi": np.array([0.5, 0.5]),
        "A":  np.array([[0.9, 0.1],
                        [0.2, 0.8]]),
        "B":  np.array([[0.5, 0.5],
                        [0.8, 0.2]]),
    },
}
