"""
visualizations.py
=================
Plotly-based chart factory for the Machine Health HMM dashboard.
All functions return Plotly Figure objects for use in Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = dict(
    bg="#0f1923",
    panel="#162032",
    grid="#1e2d3d",
    text="#d0dde8",
    accent="#4c8ef7",
    success="#2ecc71",
    warning="#f39c12",
    danger="#e74c3c",
)

LAYOUT_BASE = dict(
    paper_bgcolor=THEME["bg"],
    plot_bgcolor=THEME["panel"],
    font=dict(color=THEME["text"], family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    xaxis=dict(gridcolor=THEME["grid"], zeroline=False),
    yaxis=dict(gridcolor=THEME["grid"], zeroline=False),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ---------------------------------------------------------------------------
# Sensor time-series
# ---------------------------------------------------------------------------

def sensor_timeseries(
    seq: pd.DataFrame,
    sensor_cols: List[str],
    cycle_col: str = "cycle",
    state_sequence: Optional[np.ndarray] = None,
    state_colors: Optional[Dict[int, str]] = None,
    state_labels: Optional[Dict[int, str]] = None,
    title: str = "Sensor Readings Over Time",
) -> go.Figure:
    """Multi-panel sensor time-series with optional state shading."""
    n = len(sensor_cols)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=sensor_cols, vertical_spacing=0.06)

    cycles = seq[cycle_col].values

    for i, col in enumerate(sensor_cols, start=1):
        y = seq[col].values
        fig.add_trace(
            go.Scatter(
                x=cycles, y=y, mode="lines",
                name=col,
                line=dict(color=THEME["accent"], width=1.5),
                showlegend=(i == 1),
            ),
            row=i, col=1,
        )
        # Shading for state transitions
        if state_sequence is not None and state_colors and len(state_sequence) == len(cycles):
            _add_state_shading(fig, cycles, state_sequence, state_colors,
                               state_labels, row=i, showlegend=(i == 1))

    fig.update_layout(title=dict(text=title, x=0.5), height=220 * n)
    _apply_theme(fig)
    return fig


def _add_state_shading(fig, cycles, states, colors, labels, row, showlegend=True):
    """Add vertical background rectangles for each state run."""
    seen = set()
    current = states[0]
    start_idx = 0

    def add_rect(s, e, state):
        color = colors.get(state, "#888888")
        label = labels.get(state, f"State {state}") if labels else f"State {state}"
        name_str = label
        show = showlegend and (state not in seen)
        seen.add(state)
        fig.add_vrect(
            x0=cycles[s], x1=cycles[min(e, len(cycles) - 1)],
            fillcolor=color, opacity=0.15, line_width=0,
            row=row, col=1,
            annotation_text="",
        )

    for t in range(1, len(states)):
        if states[t] != current:
            add_rect(start_idx, t - 1, current)
            current   = states[t]
            start_idx = t
    add_rect(start_idx, len(states) - 1, current)


# ---------------------------------------------------------------------------
# State timeline
# ---------------------------------------------------------------------------

def state_timeline(
    cycles: np.ndarray,
    state_sequence: np.ndarray,
    state_colors: Dict[int, str],
    state_labels: Dict[int, str],
    title: str = "Inferred Hidden State Timeline",
) -> go.Figure:
    """Colored scatter plot showing state at each cycle."""
    colors = [state_colors.get(s, "#888") for s in state_sequence]
    labels_arr = [state_labels.get(s, f"State {s}") for s in state_sequence]

    fig = go.Figure()
    unique_states = sorted(set(state_sequence))
    for s in unique_states:
        mask = state_sequence == s
        fig.add_trace(go.Scatter(
            x=cycles[mask], y=np.zeros(mask.sum()),
            mode="markers",
            marker=dict(color=state_colors.get(s, "#888"), size=10, symbol="square"),
            name=state_labels.get(s, f"State {s}"),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        yaxis=dict(visible=False, range=[-1, 1]),
        xaxis=dict(title="Cycle"),
        height=120,
        showlegend=True,
        legend=dict(orientation="h", y=1.3),
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Posterior heatmap
# ---------------------------------------------------------------------------

def posterior_heatmap(
    posteriors: np.ndarray,
    state_labels: Dict[int, str],
    cycles: Optional[np.ndarray] = None,
    title: str = "Posterior State Probabilities",
) -> go.Figure:
    """Heatmap of P(state | observations) across time."""
    n_states = posteriors.shape[1]
    y_labels = [state_labels.get(i, f"State {i}") for i in range(n_states)]
    x_labels = cycles.tolist() if cycles is not None else list(range(len(posteriors)))

    fig = go.Figure(go.Heatmap(
        z=posteriors.T,
        x=x_labels,
        y=y_labels,
        colorscale="Viridis",
        colorbar=dict(title="Probability"),
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Cycle",
        yaxis_title="Hidden State",
        height=300,
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Transition matrix heatmap
# ---------------------------------------------------------------------------

def transition_matrix_heatmap(
    transmat: np.ndarray,
    state_labels: Dict[int, str],
    title: str = "Learned Transition Matrix",
) -> go.Figure:
    n = transmat.shape[0]
    labels = [state_labels.get(i, f"State {i}") for i in range(n)]

    text = [[f"{transmat[i, j]:.3f}" for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=transmat,
        x=labels,
        y=labels,
        colorscale="Blues",
        colorbar=dict(title="Probability"),
        text=text,
        texttemplate="%{text}",
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="To State",
        yaxis_title="From State",
        height=350,
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Emission means bar chart
# ---------------------------------------------------------------------------

def emission_means_chart(
    means: np.ndarray,
    feature_cols: List[str],
    state_labels: Dict[int, str],
    title: str = "Emission Means per Hidden State",
) -> go.Figure:
    n_states, n_feat = means.shape
    fig = go.Figure()
    colors_list = [THEME["success"], THEME["warning"], THEME["danger"],
                   THEME["accent"], "#9b59b6", "#1abc9c"]

    for s in range(n_states):
        fig.add_trace(go.Bar(
            name=state_labels.get(s, f"State {s}"),
            x=feature_cols,
            y=means[s],
            marker_color=colors_list[s % len(colors_list)],
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode="group",
        xaxis_title="Sensor Feature",
        yaxis_title="Scaled Mean Value",
        height=350,
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# State occupancy donut chart
# ---------------------------------------------------------------------------

def state_occupancy_pie(
    occupancy: Dict,
    state_labels: Dict[int, str],
    state_colors: Dict[int, str],
    title: str = "State Occupancy",
) -> go.Figure:
    labels  = [state_labels.get(s, f"State {s}") for s in occupancy]
    values  = [occupancy[s]["percent"] for s in occupancy]
    colors  = [state_colors.get(s, "#888") for s in occupancy]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        hole=0.45,
        textinfo="label+percent",
    ))
    fig.update_layout(title=dict(text=title, x=0.5), height=320)
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Sensor Correlation Matrix",
) -> go.Figure:
    labels = corr_matrix.columns.tolist()
    text   = [[f"{corr_matrix.iloc[i, j]:.2f}" for j in range(len(labels))]
              for i in range(len(labels))]

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=labels, y=labels,
        colorscale="RdBu",
        zmid=0,
        text=text,
        texttemplate="%{text}",
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(title=dict(text=title, x=0.5), height=380)
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# AIC / BIC comparison
# ---------------------------------------------------------------------------

def aic_bic_chart(
    compare_results: List[Dict],
    title: str = "Model Selection: AIC / BIC vs Number of States",
) -> go.Figure:
    n_states = [r["n_states"] for r in compare_results]
    aic      = [r.get("aic",  float("nan")) for r in compare_results]
    bic      = [r.get("bic",  float("nan")) for r in compare_results]
    ll       = [r.get("log_likelihood", float("nan")) for r in compare_results]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("AIC & BIC (lower = better)",
                                        "Training Log-Likelihood (higher = better)"))

    fig.add_trace(go.Scatter(x=n_states, y=aic, mode="lines+markers",
                             name="AIC", line=dict(color=THEME["warning"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=n_states, y=bic, mode="lines+markers",
                             name="BIC", line=dict(color=THEME["danger"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=n_states, y=ll, mode="lines+markers",
                             name="Log-Likelihood", line=dict(color=THEME["success"])), row=1, col=2)

    fig.update_layout(title=dict(text=title, x=0.5), height=350)
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_chart(
    cm: List[List[int]],
    labels: List[str],
    title: str = "Confusion Matrix",
) -> go.Figure:
    cm_arr = np.array(cm)
    text   = [[str(cm_arr[i, j]) for j in range(len(labels))]
              for i in range(len(labels))]

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=labels, y=labels,
        colorscale="Blues",
        text=text,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Forward table (educational)
# ---------------------------------------------------------------------------

def forward_table_chart(
    alpha: np.ndarray,
    state_names: List[str],
    obs_names: List[str],
    obs: List[int],
    scaled: bool = False,
    title: str = "Forward Probability Table",
) -> go.Figure:
    T, N = alpha.shape
    x_labels = [f"t={t} ({obs_names[obs[t]]})" for t in range(T)]
    y_labels  = state_names

    text = [[f"{alpha[t, s]:.4e}" for t in range(T)] for s in range(N)]

    fig = go.Figure(go.Heatmap(
        z=alpha.T,
        x=x_labels,
        y=y_labels,
        colorscale="Plasma",
        text=text,
        texttemplate="%{text}",
        colorbar=dict(title="α" + ("̂" if scaled else "")),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Time Step (Observation)",
        yaxis_title="Hidden State",
        height=300,
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Viterbi DP table
# ---------------------------------------------------------------------------

def viterbi_delta_chart(
    delta: np.ndarray,
    state_names: List[str],
    obs_names: List[str],
    obs: List[int],
    title: str = "Viterbi Log-Probability Table (δ)",
) -> go.Figure:
    T, N = delta.shape
    x_labels = [f"t={t} ({obs_names[obs[t]]})" for t in range(T)]
    y_labels  = state_names

    text = [[f"{delta[t, s]:.2f}" for t in range(T)] for s in range(N)]

    fig = go.Figure(go.Heatmap(
        z=delta.T,
        x=x_labels,
        y=y_labels,
        colorscale="Magma",
        text=text,
        texttemplate="%{text}",
        colorbar=dict(title="log δ"),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Time Step",
        yaxis_title="Hidden State",
        height=300,
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Log-likelihood convergence
# ---------------------------------------------------------------------------

def loglikelihood_convergence(
    scores: List[float],
    title: str = "Training Log-Likelihood per Initialization",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Init {i+1}" for i in range(len(scores))],
        y=scores,
        marker_color=[THEME["success"] if s == max(scores) else THEME["accent"]
                      for s in scores],
        text=[f"{s:.1f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Initialization",
        yaxis_title="Log-Likelihood",
        height=320,
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Scaling factors demo
# ---------------------------------------------------------------------------

def scaling_factors_chart(underflow_result: Dict, title: str = "Scaling Factors Demonstration") -> go.Figure:
    c = underflow_result["scaling_c"]
    T = len(c)
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Scaling Factors c_t (per time step)",
                                        "Cumulative Product (shows underflow risk)"))

    fig.add_trace(go.Bar(x=list(range(T)), y=c,
                         name="c_t", marker_color=THEME["accent"]), row=1, col=1)

    cum_prod = np.cumprod(c + 1e-300)
    fig.add_trace(go.Scatter(x=list(range(T)), y=cum_prod,
                              mode="lines+markers", name="∏c_t",
                              line=dict(color=THEME["warning"])), row=2, col=1)

    fig.update_layout(title=dict(text=title, x=0.5), height=400)
    _apply_theme(fig)
    return fig
