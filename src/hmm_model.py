"""
hmm_model.py
============
Pure-NumPy/SciPy Gaussian HMM implementation.
No C extensions required — works on any Python version.

Implements:
  - Gaussian emission probabilities (diagonal covariance)
  - Baum-Welch EM training with log-space numerics
  - Viterbi decoding
  - Forward-Backward posterior probabilities
  - Multi-initialization for robustness
  - AIC / BIC model selection
"""

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp
from typing import Optional, Tuple, List, Dict, Any
import warnings


# ---------------------------------------------------------------------------
# Numerically stable log-sum-exp (delegate to scipy)
# ---------------------------------------------------------------------------

def _logsumexp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    """Stable log-sum-exp via scipy."""
    return _scipy_logsumexp(np.asarray(a, dtype=np.float64), axis=axis, keepdims=keepdims)


# ---------------------------------------------------------------------------
# Gaussian emission log-probability
# ---------------------------------------------------------------------------

def _log_gaussian_emission(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """
    Compute log P(x | state i) for all observations and all states.

    Parameters
    ----------
    X       : (T, n_features)
    means   : (n_states, n_features)
    covars  : (n_states, n_features)  — diagonal variances

    Returns
    -------
    log_emit : (T, n_states)
    """
    T, n_feat   = X.shape
    n_states    = means.shape[0]
    log_emit    = np.zeros((T, n_states))

    for s in range(n_states):
        mu    = means[s]               # (n_features,)
        var   = covars[s]              # (n_features,)  — diagonal variances
        var   = np.maximum(var, 1e-6)  # numerical floor
        diff  = X - mu                 # (T, n_features)
        # log 𝒩(x; μ, diag(σ²)) = -½ Σ_k [(x_k-μ_k)²/σ_k² + log(2π σ_k²)]
        log_emit[:, s] = -0.5 * np.sum(
            diff ** 2 / var + np.log(2 * np.pi * var), axis=1
        )

    return log_emit


# ---------------------------------------------------------------------------
# Forward algorithm (log-space)
# ---------------------------------------------------------------------------

def _forward_log(log_pi: np.ndarray, log_A: np.ndarray, log_emit: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Log-space forward algorithm for one sequence.

    Returns
    -------
    log_alpha : (T, N)
    log_prob  : total log-likelihood
    """
    T, N = log_emit.shape
    log_alpha = np.full((T, N), -np.inf)

    log_alpha[0] = log_pi + log_emit[0]

    for t in range(1, T):
        # log_alpha[t-1, i] + log_A[i, j] → (N, N) → logsumexp over i → (N,)
        log_alpha[t] = _logsumexp(
            log_alpha[t-1, :, None] + log_A,   # broadcast (N,1)+(N,N) = (N,N)
            axis=0
        ) + log_emit[t]

    log_prob = _logsumexp(log_alpha[-1])
    return log_alpha, float(log_prob)


# ---------------------------------------------------------------------------
# Backward algorithm (log-space)
# ---------------------------------------------------------------------------

def _backward_log(log_A: np.ndarray, log_emit: np.ndarray) -> np.ndarray:
    """
    Log-space backward algorithm for one sequence.

    Returns
    -------
    log_beta : (T, N)
    """
    T, N = log_emit.shape
    log_beta = np.full((T, N), -np.inf)
    log_beta[T-1] = 0.0   # β_T(i) = 1  →  log β_T(i) = 0

    for t in range(T-2, -1, -1):
        # log_A[i,j] + log_emit[t+1,j] + log_beta[t+1,j] → logsumexp over j → (N,)
        log_beta[t] = _logsumexp(
            log_A + log_emit[t+1] + log_beta[t+1],   # (N,N) + (N,) + (N,)
            axis=1
        )

    return log_beta


# ---------------------------------------------------------------------------
# E-step: compute posteriors
# ---------------------------------------------------------------------------

def _estep(
    log_pi: np.ndarray, log_A: np.ndarray, log_emit: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    E-step: compute γ (state posteriors) and ξ (transition posteriors).

    Returns
    -------
    gamma   : (T, N) posterior state probs
    xi_sum  : (N, N) summed transition posteriors Σ_t ξ_t(i,j)
    log_prob: float
    """
    T, N = log_emit.shape
    log_alpha, log_prob = _forward_log(log_pi, log_A, log_emit)
    log_beta             = _backward_log(log_A, log_emit)

    # γ_t(i) = α_t(i) β_t(i) / P(O|λ)
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    # ξ summed over t = 1..T-1 — vectorised (N,N) at each step
    # log_xi_t[i,j] = log_alpha[t,i] + log_A[i,j] + log_emit[t+1,j] + log_beta[t+1,j]
    # Shape: (N,1) + (N,N) + (1,N) + (1,N) = (N,N)
    xi_sum = np.full((N, N), -np.inf)
    for t in range(T-1):
        log_xi_t = (log_alpha[t, :, None] +          # (N, 1)
                    log_A +                            # (N, N)
                    log_emit[t+1, None, :] +           # (1, N)
                    log_beta[t+1, None, :])            # (1, N)
        log_xi_t -= _logsumexp(log_xi_t.ravel())
        xi_sum = np.logaddexp(xi_sum, log_xi_t)

    xi_sum = np.exp(xi_sum)
    return gamma, xi_sum, float(log_prob)


# ---------------------------------------------------------------------------
# M-step: re-estimate parameters
# ---------------------------------------------------------------------------

def _mstep(
    gammas: List[np.ndarray],    # one γ per sequence
    xi_sums: List[np.ndarray],   # one (N,N) xi_sum per sequence
    Xs: List[np.ndarray],        # raw data per sequence
    n_states: int,
    n_features: int,
    min_covar: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step combining all sequences.

    Returns
    -------
    new_pi, new_A, new_means, new_covars
    """
    N, F = n_states, n_features

    # π: average of γ_1(i) across sequences
    new_pi = np.zeros(N)
    for gamma in gammas:
        new_pi += gamma[0]
    new_pi /= (new_pi.sum() + 1e-300)

    # A: sum of xi_sums / sum of γ_{0..T-2}
    xi_total = np.zeros((N, N))
    gamma_total_tm1 = np.zeros(N)   # Σ_t γ_t(i) for t < T
    for gamma, xi_sum in zip(gammas, xi_sums):
        xi_total       += xi_sum
        gamma_total_tm1 += gamma[:-1].sum(axis=0)

    new_A = xi_total / (gamma_total_tm1[:, None] + 1e-300)
    # Normalise rows
    row_sums = new_A.sum(axis=1, keepdims=True)
    new_A /= (row_sums + 1e-300)

    # means and covars
    new_means = np.zeros((N, F))
    new_covars = np.zeros((N, F))
    gamma_total = np.zeros(N)
    for gamma, X in zip(gammas, Xs):
        for i in range(N):
            g = gamma[:, i]
            new_means[i]  += (g[:, None] * X).sum(axis=0)
            gamma_total[i] += g.sum()

    for i in range(N):
        new_means[i] /= (gamma_total[i] + 1e-300)

    for gamma, X in zip(gammas, Xs):
        for i in range(N):
            g    = gamma[:, i]
            diff = X - new_means[i]
            new_covars[i] += (g[:, None] * diff**2).sum(axis=0)

    for i in range(N):
        new_covars[i] /= (gamma_total[i] + 1e-300)
        new_covars[i] = np.maximum(new_covars[i], min_covar)

    return new_pi, new_A, new_means, new_covars


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _initialise_params(
    X: np.ndarray,
    lengths: List[int],
    n_states: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialise HMM parameters using k-means-style random cluster assignment.
    """
    T, F = X.shape

    # π: uniform
    pi = np.ones(n_states) / n_states

    # A: near-diagonal with small off-diagonal mass
    A = np.full((n_states, n_states), 0.05 / max(1, n_states - 1))
    np.fill_diagonal(A, 0.80)
    A /= A.sum(axis=1, keepdims=True)

    # means: random subset of data points
    idx   = rng.choice(T, size=n_states, replace=False)
    means = X[idx] + rng.normal(0, 0.1, size=(n_states, F))

    # covars: global variance
    global_var = np.var(X, axis=0) + 1e-3
    covars     = np.tile(global_var, (n_states, 1))

    return pi, A, means, covars


# ===========================================================================
# Main Gaussian HMM class
# ===========================================================================

class MachineHealthHMM:
    """
    Gaussian Hidden Markov Model for machine health monitoring.

    Uses diagonal covariance Gaussian emissions.
    Training via Baum-Welch EM in log-space — no C extensions required.

    Parameters
    ----------
    n_states : int        Number of hidden states (2–6)
    n_iter   : int        Max Baum-Welch iterations
    tol      : float      Convergence tolerance
    n_init   : int        Number of random restarts
    random_seed : int     For reproducibility
    covariance_type : str Currently only 'diag' supported in pure implementation
    """

    def __init__(
        self,
        n_states:        int   = 3,
        covariance_type: str   = "diag",
        n_iter:          int   = 200,
        tol:             float = 1e-4,
        n_init:          int   = 3,
        random_seed:     int   = 42,
    ):
        self.n_states         = n_states
        self.covariance_type  = covariance_type
        self.n_iter           = n_iter
        self.tol              = tol
        self.n_init           = n_init
        self.random_seed      = random_seed

        # Learned parameters
        self._pi:     Optional[np.ndarray] = None
        self._A:      Optional[np.ndarray] = None
        self._means:  Optional[np.ndarray] = None
        self._covars: Optional[np.ndarray] = None

        self._best_log_prob:    float = -np.inf
        self._converged:        bool  = False
        self.training_log_probs: List[float] = []

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, lengths: List[int]) -> "MachineHealthHMM":
        """Train with multiple random initializations; keep best."""
        Xs = _split_sequences(X, lengths)
        best_pi, best_A, best_means, best_covars = None, None, None, None
        self.training_log_probs = []

        for init_idx in range(self.n_init):
            rng = np.random.default_rng(self.random_seed + init_idx * 13)
            pi, A, means, covars = _initialise_params(X, lengths, self.n_states, rng)
            pi, A, means, covars, ll, converged = self._baum_welch(
                Xs, pi, A, means, covars
            )
            self.training_log_probs.append(float(ll))
            if ll > self._best_log_prob:
                self._best_log_prob = ll
                self._converged     = converged
                best_pi, best_A, best_means, best_covars = pi, A, means, covars

        self._pi     = best_pi
        self._A      = best_A
        self._means  = best_means
        self._covars = best_covars
        return self

    # ------------------------------------------------------------------
    def _baum_welch(self, Xs, pi, A, means, covars):
        """Run EM until convergence or max iterations."""
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.n_iter):
            gammas, xi_sums, lls = [], [], []
            log_pi = np.log(pi + 1e-300)
            log_A  = np.log(A  + 1e-300)

            for X_s in Xs:
                log_emit = _log_gaussian_emission(X_s, means, covars)
                gamma, xi_sum, ll = _estep(log_pi, log_A, log_emit)
                gammas.append(gamma)
                xi_sums.append(xi_sum)
                lls.append(ll)

            total_ll = sum(lls)

            # M-step
            pi, A, means, covars = _mstep(
                gammas, xi_sums, Xs, self.n_states, Xs[0].shape[1]
            )

            improvement = total_ll - prev_ll
            if 0 <= improvement < self.tol:
                converged = True
                break
            prev_ll = total_ll

        return pi, A, means, covars, prev_ll, converged

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, lengths: List[int]) -> float:
        """Total log-likelihood of observation sequences."""
        self._check_fitted()
        Xs     = _split_sequences(X, lengths)
        log_pi = np.log(self._pi + 1e-300)
        log_A  = np.log(self._A  + 1e-300)
        total  = 0.0
        for X_s in Xs:
            log_emit = _log_gaussian_emission(X_s, self._means, self._covars)
            _, ll    = _forward_log(log_pi, log_A, log_emit)
            total   += ll
        return total

    # ------------------------------------------------------------------
    def predict_states(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> np.ndarray:
        """Viterbi decoding: most-likely hidden-state sequence."""
        self._check_fitted()
        ln = lengths if lengths is not None else [len(X)]
        Xs = _split_sequences(X, ln)
        result = []
        log_pi = np.log(self._pi + 1e-300)
        log_A  = np.log(self._A  + 1e-300)
        for X_s in Xs:
            log_emit = _log_gaussian_emission(X_s, self._means, self._covars)
            states   = _viterbi(log_pi, log_A, log_emit)
            result.extend(states)
        return np.array(result, dtype=int)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> np.ndarray:
        """Posterior state probabilities via Forward-Backward."""
        self._check_fitted()
        ln = lengths if lengths is not None else [len(X)]
        Xs = _split_sequences(X, ln)
        result = []
        log_pi = np.log(self._pi + 1e-300)
        log_A  = np.log(self._A  + 1e-300)
        for X_s in Xs:
            log_emit = _log_gaussian_emission(X_s, self._means, self._covars)
            gamma, _, _ = _estep(log_pi, log_A, log_emit)
            result.append(gamma)
        return np.vstack(result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def startprob(self)  -> np.ndarray: self._check_fitted(); return self._pi
    @property
    def transmat(self)   -> np.ndarray: self._check_fitted(); return self._A
    @property
    def means(self)      -> np.ndarray: self._check_fitted(); return self._means
    @property
    def covars(self)     -> np.ndarray: self._check_fitted(); return self._covars
    @property
    def converged(self)  -> bool:       return self._converged
    @property
    def best_log_prob(self) -> float:   return self._best_log_prob
    @property
    def is_fitted(self)  -> bool:       return self._pi is not None

    def get_params(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in
                ["n_states", "covariance_type", "n_iter", "tol", "n_init", "random_seed"]}

    def _check_fitted(self):
        if self._pi is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")


# ---------------------------------------------------------------------------
# Viterbi (log-space)
# ---------------------------------------------------------------------------

def _viterbi(log_pi: np.ndarray, log_A: np.ndarray, log_emit: np.ndarray) -> List[int]:
    T, N = log_emit.shape
    delta = np.full((T, N), -np.inf)
    psi   = np.zeros((T, N), dtype=int)

    delta[0] = log_pi + log_emit[0]

    for t in range(1, T):
        for j in range(N):
            scores     = delta[t-1] + log_A[:, j]
            psi[t, j]  = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + log_emit[t, j]

    path    = [0] * T
    path[T-1] = int(np.argmax(delta[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path


# ---------------------------------------------------------------------------
# Sequence splitting utility
# ---------------------------------------------------------------------------

def _split_sequences(X: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    """Split concatenated X back into individual sequences."""
    seqs = []
    start = 0
    for l in lengths:
        seqs.append(X[start : start + l])
        start += l
    return seqs


# ===========================================================================
# Multi-state experiment
# ===========================================================================

def compare_n_states(
    X: np.ndarray,
    lengths: List[int],
    state_range: List[int],
    n_features: int,
    covariance_type: str = "diag",
    n_iter: int = 150,
    n_init: int = 2,
    random_seed: int = 42,
) -> List[Dict]:
    n_obs = len(X)
    results = []

    for n_s in state_range:
        hmm = MachineHealthHMM(
            n_states=n_s, covariance_type=covariance_type,
            n_iter=n_iter, n_init=n_init, random_seed=random_seed,
        )
        try:
            hmm.fit(X, lengths)
            ll = hmm.best_log_prob
            # Parameters: π (n_s-1), A: n_s*(n_s-1), μ: n_s*n_f, σ²: n_s*n_f
            k  = (n_s - 1) + n_s * (n_s - 1) + 2 * n_s * n_features
            aic = -2 * ll + 2 * k
            bic = -2 * ll + k * np.log(max(1, n_obs))
            results.append({
                "n_states":       n_s,
                "log_likelihood": ll,
                "aic":            aic,
                "bic":            bic,
                "converged":      hmm.converged,
                "n_params":       k,
            })
        except Exception as exc:
            results.append({
                "n_states": n_s,
                "log_likelihood": float("nan"),
                "aic":  float("nan"),
                "bic":  float("nan"),
                "converged": False,
                "error": str(exc),
            })

    return results
