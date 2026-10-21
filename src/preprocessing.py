"""
preprocessing.py
================
Feature scaling and sequence preparation for HMM training.
Rules:
  - Fit scaler on training sequences only.
  - Apply the same fitted scaler to validation/test sequences.
  - Never mix sequences across machines during training.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler

from .data_loader import (
    get_machine_ids,
    get_machine_sequence,
    available_sensor_cols,
    ID_COL,
    CYCLE_COL,
    LABEL_COL,
)


# ---------------------------------------------------------------------------
# Core preprocessing pipeline
# ---------------------------------------------------------------------------

class SensorPreprocessor:
    """
    Fits a StandardScaler on training machine sequences and transforms
    any sequence through the fitted scaler.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols: Optional[List[str]] = feature_cols
        self.scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, train_machine_ids: List[int]) -> "SensorPreprocessor":
        """
        Fit the scaler on all observations belonging to train_machine_ids.
        """
        if self.feature_cols is None:
            self.feature_cols = available_sensor_cols(df)

        rows = df[df[ID_COL].isin(train_machine_ids)][self.feature_cols].dropna()
        if rows.empty:
            raise ValueError("No training data found for the specified machine IDs.")

        self.scaler.fit(rows)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform_sequence(self, seq: pd.DataFrame) -> np.ndarray:
        """
        Transform a single machine's observation DataFrame → 2-D array
        of shape (n_cycles, n_features).
        Missing values are filled with column means (estimated from training).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform_sequence().")

        X = seq[self.feature_cols].copy()
        # Fill NaNs with training column means
        for i, col in enumerate(self.feature_cols):
            if X[col].isnull().any():
                X[col] = X[col].fillna(self.scaler.mean_[i])

        return self.scaler.transform(X).astype(np.float64)

    # ------------------------------------------------------------------
    def get_labels(self, seq: pd.DataFrame) -> Optional[np.ndarray]:
        """Return integer label array if labels exist, else None."""
        if LABEL_COL in seq.columns and not seq[LABEL_COL].isnull().all():
            return seq[LABEL_COL].fillna(-1).astype(int).values
        return None

    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        if self.feature_cols is None:
            return 0
        return len(self.feature_cols)

    # ------------------------------------------------------------------
    def feature_stats(self) -> pd.DataFrame:
        """Return per-feature mean and std learned during fitting."""
        return pd.DataFrame({
            "feature": self.feature_cols,
            "mean":    self.scaler.mean_,
            "std":     self.scaler.scale_,
        })


# ---------------------------------------------------------------------------
# Train / validation / test split  (machine-level, not observation-level)
# ---------------------------------------------------------------------------

def split_machine_ids(
    machine_ids: List[int],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    random_seed: int  = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split machines into train / validation / test sets.
    The test fraction is inferred as 1 - train_frac - val_frac.

    Returns
    -------
    (train_ids, val_ids, test_ids)
    """
    rng = np.random.default_rng(random_seed)
    ids = np.array(sorted(machine_ids))
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, int(np.floor(n * train_frac)))
    n_val   = max(1, int(np.floor(n * val_frac)))
    # ensure at least one test machine when n ≥ 3
    n_test  = max(1, n - n_train - n_val) if n >= 3 else 0
    if n_test == 0:
        n_val = max(0, n - n_train)

    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val :].tolist()
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Sequence preparation for hmmlearn
# ---------------------------------------------------------------------------

def prepare_sequences(
    df: pd.DataFrame,
    preprocessor: SensorPreprocessor,
    machine_ids: List[int],
) -> Tuple[np.ndarray, List[int]]:
    """
    Concatenate scaled sequences for hmmlearn.

    hmmlearn expects:
      X       : (total_observations, n_features)
      lengths : list of per-sequence lengths, same order as X

    Returns
    -------
    X, lengths
    """
    arrays  = []
    lengths = []
    for mid in machine_ids:
        seq = get_machine_sequence(df, mid)
        X_i = preprocessor.transform_sequence(seq)
        if len(X_i) > 0:
            arrays.append(X_i)
            lengths.append(len(X_i))

    if not arrays:
        raise ValueError("No valid sequences found for the given machine IDs.")

    X = np.vstack(arrays)
    return X, lengths


# ---------------------------------------------------------------------------
# Utility: raw sensor statistics
# ---------------------------------------------------------------------------

def compute_sensor_stats(df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    """Return a summary statistics DataFrame for the given sensor columns."""
    return df[sensor_cols].describe().T.reset_index().rename(columns={"index": "sensor"})


def compute_correlation(df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    """Pearson correlation matrix among sensor columns."""
    return df[sensor_cols].corr()
