"""
data_loader.py
==============
Handles dataset loading, synthetic data generation, and validation
for the Predictive Machine Health Monitoring project.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import io
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_COLS = ["temperature", "vibration", "pressure", "rotation_speed"]
LABEL_COL   = "health_stage"   # optional ground-truth label (0=healthy,1=degrading,2=critical)
ID_COL      = "machine_id"
CYCLE_COL   = "cycle"

SYNTHETIC_RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_machines: int = 10,
    min_cycles: int = 80,
    max_cycles: int = 200,
    noise_level: float = 0.15,
    random_seed: int = SYNTHETIC_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a reproducible synthetic machine sensor dataset.

    Each machine progresses through three health stages:
      0 – Healthy     : stable baselines, low noise
      1 – Degrading   : gradual drift upward/downward
      2 – Critical    : larger fluctuations, clear degradation

    The stage lengths are randomly sampled per machine so the dataset
    contains meaningful temporal structure rather than i.i.d. noise.

    Returns
    -------
    pd.DataFrame with columns:
        machine_id, cycle, temperature, vibration, pressure,
        rotation_speed, health_stage
    """
    rng = np.random.default_rng(random_seed)
    records: List[Dict] = []

    for machine_id in range(1, n_machines + 1):
        n_cycles = int(rng.integers(min_cycles, max_cycles + 1))

        # Stage boundaries: 50 / 30 / 20 percent split (±noise)
        b1 = int(n_cycles * rng.uniform(0.35, 0.55))   # healthy → degrading
        b2 = int(n_cycles * rng.uniform(0.65, 0.82))   # degrading → critical

        for cycle in range(1, n_cycles + 1):
            t = cycle / n_cycles  # normalised progress [0, 1]

            if cycle <= b1:
                stage = 0
                # Healthy: stable, low-amplitude oscillation
                temp   = 350 + 5 * np.sin(cycle * 0.3) + rng.normal(0, noise_level * 10)
                vib    = 0.5 + 0.05 * np.sin(cycle * 0.5) + rng.normal(0, noise_level * 0.1)
                pres   = 50  + 1.0 * np.cos(cycle * 0.2) + rng.normal(0, noise_level * 1.0)
                rot    = 3000 + 20 * np.sin(cycle * 0.4) + rng.normal(0, noise_level * 30)

            elif cycle <= b2:
                stage = 1
                # Degrading: linear drift + increasing noise
                d = (cycle - b1) / max(1, b2 - b1)   # progress within stage
                temp   = 350 + 5 * np.sin(cycle * 0.3) + 40 * d   + rng.normal(0, noise_level * 15)
                vib    = 0.5 + 0.05 * np.sin(cycle * 0.5) + 0.8 * d + rng.normal(0, noise_level * 0.2)
                pres   = 50  + 1.0 * np.cos(cycle * 0.2) - 8 * d   + rng.normal(0, noise_level * 2.0)
                rot    = 3000 + 20 * np.sin(cycle * 0.4) - 150 * d  + rng.normal(0, noise_level * 50)

            else:
                stage = 2
                # Critical: high drift + high-amplitude irregular oscillation
                d = (cycle - b2) / max(1, n_cycles - b2)
                temp   = 390 + 40 * d + 8  * np.sin(cycle * 0.8) + rng.normal(0, noise_level * 25)
                vib    = 1.3 + 1.2 * d + 0.15 * np.sin(cycle * 1.2) + rng.normal(0, noise_level * 0.4)
                pres   = 42  - 10 * d - 3  * np.cos(cycle * 0.9)   + rng.normal(0, noise_level * 3.0)
                rot    = 2850 - 200 * d + 30 * np.sin(cycle * 0.6)  + rng.normal(0, noise_level * 70)

            records.append({
                ID_COL:   machine_id,
                CYCLE_COL: cycle,
                "temperature":    round(float(temp), 3),
                "vibration":      round(float(vib),  4),
                "pressure":       round(float(pres), 3),
                "rotation_speed": round(float(rot),  2),
                LABEL_COL:        int(stage),
            })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_builtin_dataset() -> pd.DataFrame:
    """
    Return the pre-generated synthetic dataset.
    Creates the CSV on first call if it doesn't exist.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sensor_data.csv")
    csv_path = os.path.normpath(csv_path)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = _cast_types(df)
        return df

    # Generate and save
    df = generate_synthetic_dataset()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def load_uploaded_csv(file_obj) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load and validate a user-uploaded CSV file.

    Returns
    -------
    (df, None)            on success
    (None, error_message) on failure
    """
    try:
        if hasattr(file_obj, "read"):
            content = file_obj.read()
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_csv(file_obj)
    except Exception as exc:
        return None, f"Could not parse CSV: {exc}"

    err = _validate_dataframe(df)
    if err:
        return None, err

    df = _cast_types(df)
    return df, None


# ---------------------------------------------------------------------------
# Validation and type-casting
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> Optional[str]:
    """Return an error string if df is not usable, else None."""
    if df.empty:
        return "The CSV file is empty."

    # Must have at least one numeric sensor column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return (
            "The CSV must contain at least two numeric sensor columns. "
            f"Found: {list(df.columns)}"
        )

    # machine_id column is strongly recommended
    if ID_COL not in df.columns:
        return (
            f"Column '{ID_COL}' not found. "
            "Please add a 'machine_id' column to distinguish different machines."
        )

    # cycle / time column
    if CYCLE_COL not in df.columns:
        return (
            f"Column '{CYCLE_COL}' not found. "
            "Please add a 'cycle' (or rename your time column to 'cycle')."
        )

    if len(df) < 20:
        return "Dataset has fewer than 20 rows, which is too small to train an HMM reliably."

    return None


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent dtypes."""
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(int)
    if CYCLE_COL in df.columns:
        df[CYCLE_COL] = df[CYCLE_COL].astype(int)
    for col in SENSOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def get_machine_ids(df: pd.DataFrame) -> List[int]:
    """Sorted list of unique machine IDs."""
    return sorted(df[ID_COL].unique().tolist())


def get_machine_sequence(df: pd.DataFrame, machine_id: int) -> pd.DataFrame:
    """Return rows for a single machine, sorted by cycle."""
    seq = df[df[ID_COL] == machine_id].sort_values(CYCLE_COL).reset_index(drop=True)
    return seq


def available_sensor_cols(df: pd.DataFrame) -> List[str]:
    """Numeric columns excluding ID, cycle, and label."""
    exclude = {ID_COL, CYCLE_COL, LABEL_COL}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Return a plain-dict summary of the dataset for UI display.
    """
    sensor_cols = available_sensor_cols(df)
    machine_ids = get_machine_ids(df)

    seq_lengths = []
    for mid in machine_ids:
        seq_lengths.append(len(get_machine_sequence(df, mid)))

    return {
        "n_machines":     len(machine_ids),
        "n_observations": len(df),
        "n_sensor_cols":  len(sensor_cols),
        "sensor_cols":    sensor_cols,
        "has_labels":     LABEL_COL in df.columns,
        "machine_ids":    machine_ids,
        "missing_values": int(df.isnull().sum().sum()),
        "seq_lengths_min": int(min(seq_lengths)) if seq_lengths else 0,
        "seq_lengths_max": int(max(seq_lengths)) if seq_lengths else 0,
        "seq_lengths_mean": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
    }
