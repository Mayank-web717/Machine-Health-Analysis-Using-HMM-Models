"""
state_interpretation.py
=======================
Builds a transparent, data-driven mapping from HMM hidden-state indices
to human-readable labels such as 'Healthy', 'Degrading', 'Critical'.

The mapping is based on the learned emission means and degradation trends
in the data, NOT on state index numbers.

If the data does not support reliable health-stage labelling, neutral labels
(State 0, State 1 …) are used instead.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEGRADATION_FEATURES = ["temperature", "vibration"]   # tend to rise with degradation
RECOVERY_FEATURES    = ["pressure", "rotation_speed"] # tend to fall with degradation

STAGE_COLORS = {
    "Healthy":   "#2ecc71",   # green
    "Degrading": "#f39c12",   # amber
    "Critical":  "#e74c3c",   # red
    "Unknown":   "#95a5a6",   # grey
}

NEUTRAL_COLOR_PALETTE = [
    "#4c8ef7", "#f7a64c", "#e74c3c", "#2ecc71",
    "#9b59b6", "#1abc9c", "#e67e22",
]


# ---------------------------------------------------------------------------
# Core interpretation logic
# ---------------------------------------------------------------------------

def interpret_states(
    n_states: int,
    means: np.ndarray,             # (n_states, n_features)
    feature_cols: List[str],
    posteriors: Optional[np.ndarray] = None,  # (T, n_states) for a single machine
    labels: Optional[np.ndarray] = None,      # (T,) ground-truth labels if available
) -> Dict:
    """
    Produce a state-interpretation map.

    Strategy
    --------
    1. Build a "degradation score" for each state from emission means:
       - features that should increase with degradation contribute positively
       - features that should decrease contribute negatively
    2. Rank states by degradation score.
    3. If n_states == 3, map the ranks to Healthy / Degrading / Critical.
       If n_states == 2, map to Healthy / Degraded.
       Otherwise, use neutral State k labels.
    4. Assign confidence (high / medium / low) based on score spread.

    Returns
    -------
    dict with keys:
      state_labels  : {state_idx: label_string}
      state_colors  : {state_idx: hex_color}
      state_scores  : {state_idx: float}
      confidence    : str  ('high' | 'medium' | 'low')
      explanation   : str  human-readable reasoning
      ranking       : list of state indices ordered healthy → critical
    """
    scores = _compute_degradation_scores(means, feature_cols)

    ranked = sorted(range(n_states), key=lambda i: scores[i])  # healthy first

    labels_map = {}
    colors_map = {}

    if n_states == 2:
        names  = ["Healthy", "Degraded"]
        colors = [STAGE_COLORS["Healthy"], STAGE_COLORS["Critical"]]
    elif n_states == 3:
        names  = ["Healthy", "Degrading", "Critical"]
        colors = [STAGE_COLORS["Healthy"], STAGE_COLORS["Degrading"], STAGE_COLORS["Critical"]]
    else:
        names  = [f"State {k}" for k in range(n_states)]
        colors = [NEUTRAL_COLOR_PALETTE[k % len(NEUTRAL_COLOR_PALETTE)] for k in range(n_states)]

    for rank, state_idx in enumerate(ranked):
        labels_map[state_idx] = names[rank]
        colors_map[state_idx] = colors[rank]

    score_spread = max(scores) - min(scores)
    if score_spread > 1.0:
        confidence = "high"
    elif score_spread > 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    explanation = _build_explanation(ranked, labels_map, scores, feature_cols, means, confidence)

    return {
        "state_labels": labels_map,
        "state_colors": colors_map,
        "state_scores": {i: float(scores[i]) for i in range(n_states)},
        "confidence":   confidence,
        "explanation":  explanation,
        "ranking":      ranked,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_degradation_scores(means: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """
    Compute a scalar degradation score per state.
    Higher score → more degraded.

    We normalise each feature across states to [0, 1], then:
      +1 weight for features expected to increase with degradation
      -1 weight for features expected to decrease
    """
    n_states, n_features = means.shape
    scores = np.zeros(n_states)

    for f_idx, col in enumerate(feature_cols):
        col_vals = means[:, f_idx]
        col_min  = col_vals.min()
        col_max  = col_vals.max()
        rng      = col_max - col_min

        if rng < 1e-9:
            # Constant across states → no information
            continue

        normed = (col_vals - col_min) / rng  # [0, 1]

        # Determine expected direction
        col_lower = col.lower()
        if any(k in col_lower for k in DEGRADATION_FEATURES):
            scores += normed          # increases → more degraded
        elif any(k in col_lower for k in RECOVERY_FEATURES):
            scores -= normed          # decreases → more degraded
        else:
            # Unknown feature: assume increases → degraded
            scores += normed * 0.5    # lower weight for ambiguous features

    return scores


def _build_explanation(
    ranked: List[int],
    labels: Dict,
    scores: np.ndarray,
    feature_cols: List[str],
    means: np.ndarray,
    confidence: str,
) -> str:
    lines = [
        f"**State interpretation confidence: {confidence.upper()}**\n",
        "States were ranked by a degradation score computed from emission means:\n",
    ]
    for rank, idx in enumerate(ranked):
        row = f"- **{labels[idx]}** (State {idx}): degradation score = {scores[idx]:.3f}"
        if len(feature_cols) >= 2:
            row += f" | mean {feature_cols[0]}={means[idx, 0]:.3f}"
            row += f", {feature_cols[1]}={means[idx, 1]:.3f}"
        lines.append(row)

    if confidence == "low":
        lines.append(
            "\n⚠️ Low confidence: emission means are very similar across states. "
            "Treat labels as approximate; the model may not distinguish health stages clearly."
        )
    elif confidence == "medium":
        lines.append(
            "\n📊 Medium confidence: some separation exists between state means."
        )
    else:
        lines.append(
            "\n✅ High confidence: emission means show clear separation; "
            "labels are likely meaningful."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------

def compute_alert_level(
    posteriors: np.ndarray,       # (T, n_states) for the selected machine
    state_interp: Dict,
    critical_threshold: float = 0.5,
) -> Dict:
    """
    Derive a simple alert level from posterior probabilities.

    Rules
    -----
    - Find which state index corresponds to the most degraded label.
    - If the latest posterior for that state > critical_threshold → RED alert.
    - If > 0.25 → AMBER.
    - Else → GREEN.

    Returns
    -------
    dict with: alert_color, alert_label, critical_state_idx, latest_critical_prob
    """
    labels = state_interp["state_labels"]
    ranking = state_interp["ranking"]

    # Most degraded state = last in ranking
    critical_state = ranking[-1]
    label_str = labels[critical_state]

    latest_prob = float(posteriors[-1, critical_state])

    if latest_prob >= critical_threshold:
        color = "🔴 RED"
        level = "CRITICAL"
    elif latest_prob >= 0.25:
        color = "🟡 AMBER"
        level = "WARNING"
    else:
        color = "🟢 GREEN"
        level = "NORMAL"

    return {
        "alert_color":          color,
        "alert_label":          level,
        "critical_state_idx":   critical_state,
        "critical_state_label": label_str,
        "latest_critical_prob": latest_prob,
        "threshold":            critical_threshold,
        "explanation": (
            f"Alert level is based on the posterior probability of state '{label_str}' "
            f"(the most degraded inferred state) at the latest time step.\n"
            f"Current probability: **{latest_prob:.1%}**. "
            f"Threshold for RED alert: **{critical_threshold:.0%}**.\n\n"
            "⚠️ This alert is derived from probabilistic model inference on sensor data. "
            "It is **not** a certified equipment failure prediction."
        ),
    }
