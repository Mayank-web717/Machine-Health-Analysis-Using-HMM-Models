"""
evaluation.py
=============
Model evaluation for the Machine Health HMM project.

Handles both:
  - Supervised evaluation (when ground-truth health-stage labels exist)
  - Unsupervised evaluation (log-likelihood, BIC, AIC, state occupancy)

Data-leakage prevention: evaluation sequences are kept separate from
training sequences throughout.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from itertools import permutations


# ---------------------------------------------------------------------------
# Supervised metrics (when ground-truth labels are available)
# ---------------------------------------------------------------------------

def best_label_mapping(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    n_states: int,
) -> Tuple[Dict[int, int], float]:
    """
    Find the permutation of predicted state indices that maximises accuracy
    against ground-truth labels.

    Necessary because HMM hidden states are arbitrary (state 0 ≠ Healthy).

    Returns
    -------
    mapping : dict {predicted_state → ground_truth_label}
    best_accuracy : float
    """
    unique_gt = np.unique(ground_truth)
    n_gt = len(unique_gt)

    best_acc = -1.0
    best_perm = list(range(n_states))

    # Only permute if n_states ≤ 6 (otherwise too many combinations)
    if n_states <= 6:
        state_indices = list(range(n_states))
        for perm in permutations(state_indices):
            # Build a mapping: predicted state i → gt label perm[i]
            mapped = np.array([perm[p] for p in predicted])
            # Only compare valid gt labels
            mask = np.isin(ground_truth, unique_gt)
            acc  = accuracy_score(ground_truth[mask], mapped[mask])
            if acc > best_acc:
                best_acc  = acc
                best_perm = list(perm)
    else:
        # Greedy majority-vote assignment
        for i in range(n_states):
            mask = predicted == i
            if mask.sum() > 0:
                best_perm[i] = int(np.bincount(ground_truth[mask]).argmax())

    mapping = {i: best_perm[i] for i in range(n_states)}
    return mapping, best_acc


def supervised_evaluation(
    predicted_states: np.ndarray,
    ground_truth:     np.ndarray,
    n_states:         int,
    state_labels:     Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Full supervised evaluation: accuracy, precision, recall, F1, confusion matrix.

    Parameters
    ----------
    predicted_states : HMM Viterbi state sequence
    ground_truth     : integer health-stage labels (0, 1, 2 …)
    n_states         : number of HMM states
    state_labels     : optional {state_idx: label_str} from interpret_states

    Returns
    -------
    dict with all metrics and the best label mapping.
    """
    # Remove observations with unknown ground-truth
    mask = ground_truth >= 0
    pred = predicted_states[mask]
    gt   = ground_truth[mask]

    if len(pred) == 0:
        return {"error": "No valid ground-truth labels found."}

    mapping, acc = best_label_mapping(pred, gt, n_states)
    mapped_pred = np.array([mapping[p] for p in pred])

    target_names = (
        [state_labels.get(i, f"State {i}") for i in sorted(mapping.values())]
        if state_labels else None
    )

    unique_labels = sorted(np.unique(gt).tolist())
    prec, rec, f1, sup = precision_recall_fscore_support(
        gt, mapped_pred, labels=unique_labels, zero_division=0
    )
    cm = confusion_matrix(gt, mapped_pred, labels=unique_labels)

    return {
        "accuracy":       float(acc),
        "precision":      prec.tolist(),
        "recall":         rec.tolist(),
        "f1_score":       f1.tolist(),
        "support":        sup.tolist(),
        "confusion_matrix": cm.tolist(),
        "label_mapping":  mapping,
        "unique_gt_labels": unique_labels,
        "n_samples":      int(len(gt)),
        "note": (
            "Accuracy is computed after finding the best permutation of HMM state "
            "indices that matches the ground-truth labels.  Because states are "
            "unsupervised, this mapping is a post-hoc approximation."
        ),
    }


# ---------------------------------------------------------------------------
# Unsupervised metrics
# ---------------------------------------------------------------------------

def unsupervised_evaluation(
    log_likelihood:   float,
    n_params:         int,
    n_observations:   int,
    state_sequence:   np.ndarray,
    n_states:         int,
    posteriors:       Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute unsupervised model quality metrics.

    Returns
    -------
    dict with log_likelihood, AIC, BIC, state_occupancy, state_entropy
    """
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(max(1, n_observations))

    # State occupancy
    occupancy = {}
    for s in range(n_states):
        count = int((state_sequence == s).sum())
        occupancy[s] = {
            "count":   count,
            "percent": 100.0 * count / max(1, len(state_sequence)),
        }

    # Mean posterior entropy (higher → model is uncertain about states)
    entropy = None
    if posteriors is not None:
        eps = 1e-12
        H = -np.sum(posteriors * np.log(posteriors + eps), axis=1)
        entropy = float(H.mean())

    return {
        "log_likelihood":  float(log_likelihood),
        "aic":             float(aic),
        "bic":             float(bic),
        "n_params":        n_params,
        "n_observations":  n_observations,
        "state_occupancy": occupancy,
        "mean_posterior_entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Transition matrix analysis
# ---------------------------------------------------------------------------

def transition_counts(state_sequence: np.ndarray, n_states: int) -> np.ndarray:
    """Count observed state-to-state transitions."""
    counts = np.zeros((n_states, n_states), dtype=int)
    for t in range(len(state_sequence) - 1):
        i = state_sequence[t]
        j = state_sequence[t + 1]
        counts[i, j] += 1
    return counts


def state_dwell_times(state_sequence: np.ndarray, n_states: int) -> Dict:
    """
    Compute dwell-time statistics for each state.
    Returns dict {state_idx: {mean_dwell, max_dwell, n_episodes}}.
    """
    results = {s: {"mean_dwell": 0.0, "max_dwell": 0, "n_episodes": 0} for s in range(n_states)}
    if len(state_sequence) == 0:
        return results

    current = state_sequence[0]
    run = 1
    dwells = {s: [] for s in range(n_states)}

    for t in range(1, len(state_sequence)):
        if state_sequence[t] == current:
            run += 1
        else:
            dwells[current].append(run)
            current = state_sequence[t]
            run = 1
    dwells[current].append(run)

    for s in range(n_states):
        if dwells[s]:
            results[s] = {
                "mean_dwell": float(np.mean(dwells[s])),
                "max_dwell":  int(np.max(dwells[s])),
                "n_episodes": len(dwells[s]),
            }

    return results
