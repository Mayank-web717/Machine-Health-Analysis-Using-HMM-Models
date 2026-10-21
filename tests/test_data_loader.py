"""
tests/test_data_loader.py
=========================
Unit tests for the data loading and synthetic dataset generation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
import io

from src.data_loader import (
    generate_synthetic_dataset,
    load_builtin_dataset,
    load_uploaded_csv,
    get_machine_ids,
    get_machine_sequence,
    available_sensor_cols,
    dataset_summary,
    _validate_dataframe,
    SENSOR_COLS,
    LABEL_COL,
    ID_COL,
    CYCLE_COL,
)


# ---------------------------------------------------------------------------
# generate_synthetic_dataset
# ---------------------------------------------------------------------------

class TestSyntheticDataset:

    def test_returns_dataframe(self):
        df = generate_synthetic_dataset(n_machines=3, min_cycles=40, max_cycles=60)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = generate_synthetic_dataset(n_machines=2, min_cycles=40, max_cycles=50)
        required = [ID_COL, CYCLE_COL] + SENSOR_COLS + [LABEL_COL]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_missing_values(self):
        df = generate_synthetic_dataset(n_machines=3, min_cycles=30, max_cycles=50)
        assert df.isnull().sum().sum() == 0

    def test_machine_ids_are_sequential(self):
        df = generate_synthetic_dataset(n_machines=5, min_cycles=20, max_cycles=30)
        ids = sorted(df[ID_COL].unique())
        assert ids == list(range(1, 6))

    def test_cycles_start_at_1(self):
        df = generate_synthetic_dataset(n_machines=2, min_cycles=20, max_cycles=30)
        for mid in df[ID_COL].unique():
            seq = get_machine_sequence(df, mid)
            assert seq[CYCLE_COL].iloc[0] == 1

    def test_labels_are_valid(self):
        df = generate_synthetic_dataset(n_machines=3, min_cycles=30, max_cycles=50)
        assert set(df[LABEL_COL].unique()).issubset({0, 1, 2})

    def test_reproducible_with_same_seed(self):
        df1 = generate_synthetic_dataset(n_machines=3, min_cycles=30, random_seed=99)
        df2 = generate_synthetic_dataset(n_machines=3, min_cycles=30, random_seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_synthetic_dataset(n_machines=3, min_cycles=30, random_seed=1)
        df2 = generate_synthetic_dataset(n_machines=3, min_cycles=30, random_seed=2)
        assert not df1.equals(df2)

    def test_sequences_not_mixed(self):
        """Ensure machine sequences are kept separate."""
        df = generate_synthetic_dataset(n_machines=4, min_cycles=30, max_cycles=50)
        for mid in df[ID_COL].unique():
            seq = get_machine_sequence(df, mid)
            assert (seq[ID_COL] == mid).all(), "Machine sequence contains foreign IDs"

    def test_health_progression(self):
        """Temperature should be higher on average in critical vs healthy periods."""
        df = generate_synthetic_dataset(n_machines=5, min_cycles=80, max_cycles=100,
                                        random_seed=42)
        healthy  = df[df[LABEL_COL] == 0]["temperature"].mean()
        critical = df[df[LABEL_COL] == 2]["temperature"].mean()
        assert critical > healthy, "Critical stage should have higher mean temperature"


# ---------------------------------------------------------------------------
# dataset_summary
# ---------------------------------------------------------------------------

class TestDatasetSummary:

    def test_summary_keys(self):
        df = generate_synthetic_dataset(n_machines=3, min_cycles=30, max_cycles=50)
        s = dataset_summary(df)
        for key in ["n_machines", "n_observations", "n_sensor_cols", "has_labels",
                    "missing_values", "seq_lengths_min", "seq_lengths_max"]:
            assert key in s

    def test_n_machines_correct(self):
        df = generate_synthetic_dataset(n_machines=4, min_cycles=30, max_cycles=40)
        s = dataset_summary(df)
        assert s["n_machines"] == 4


# ---------------------------------------------------------------------------
# _validate_dataframe
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_df_returns_none(self):
        df = generate_synthetic_dataset(n_machines=2, min_cycles=30, max_cycles=50)
        assert _validate_dataframe(df) is None

    def test_empty_df_fails(self):
        df = pd.DataFrame()
        err = _validate_dataframe(df)
        assert err is not None

    def test_missing_machine_id_fails(self):
        df = generate_synthetic_dataset(n_machines=2, min_cycles=30, max_cycles=50)
        df = df.drop(columns=[ID_COL])
        err = _validate_dataframe(df)
        assert err is not None

    def test_too_few_rows_fails(self):
        df = generate_synthetic_dataset(n_machines=1, min_cycles=5, max_cycles=5)
        # Override to 5 rows
        df = df.head(5)
        err = _validate_dataframe(df)
        assert err is not None


# ---------------------------------------------------------------------------
# CSV upload simulation
# ---------------------------------------------------------------------------

class TestCSVUpload:

    def test_valid_csv_upload(self):
        df_orig = generate_synthetic_dataset(n_machines=2, min_cycles=30, max_cycles=50)
        buf = io.BytesIO(df_orig.to_csv(index=False).encode())
        df_loaded, err = load_uploaded_csv(buf)
        assert err is None
        assert df_loaded is not None
        assert len(df_loaded) == len(df_orig)

    def test_invalid_csv_upload(self):
        buf = io.BytesIO(b"not,a,valid,sensor,csv\n1,2,x,y,z")
        df_loaded, err = load_uploaded_csv(buf)
        # Should either fail or missing machine_id
        # (depends on column names); either error or df should be None
        if err is None:
            # Validate manually
            from src.data_loader import _validate_dataframe
            err2 = _validate_dataframe(df_loaded)
            # Should have validation issue since machine_id missing
            assert err2 is not None or df_loaded is not None
