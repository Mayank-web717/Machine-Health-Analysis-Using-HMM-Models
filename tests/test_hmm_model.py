"""
tests/test_hmm_model.py
=======================
Unit tests for the Gaussian HMM wrapper (MachineHealthHMM).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import warnings

from src.data_loader import generate_synthetic_dataset, get_machine_ids
from src.preprocessing import (
    SensorPreprocessor, split_machine_ids, prepare_sequences
)
from src.hmm_model import MachineHealthHMM, compare_n_states
from src.state_interpretation import interpret_states, compute_alert_level


# ---------------------------------------------------------------------------
# Fixture: small dataset + preprocessor + trained model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_setup():
    df = generate_synthetic_dataset(n_machines=6, min_cycles=60, max_cycles=80, random_seed=0)
    machine_ids = get_machine_ids(df)
    train_ids, val_ids, test_ids = split_machine_ids(machine_ids, random_seed=0)

    prep = SensorPreprocessor(feature_cols=["temperature", "vibration", "pressure"])
    prep.fit(df, train_ids)

    X_train, lengths_train = prepare_sequences(df, prep, train_ids)

    hmm = MachineHealthHMM(n_states=3, n_iter=50, n_init=2, random_seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmm.fit(X_train, lengths_train)

    return df, prep, hmm, train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# MachineHealthHMM — basic properties
# ---------------------------------------------------------------------------

class TestMachineHealthHMM:

    def test_is_fitted(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        assert hmm.is_fitted

    def test_best_log_prob_finite(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        assert np.isfinite(hmm.best_log_prob)

    def test_startprob_sums_to_one(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        assert abs(hmm.startprob.sum() - 1.0) < 1e-6

    def test_transmat_rows_sum_to_one(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        for i in range(hmm.n_states):
            assert abs(hmm.transmat[i].sum() - 1.0) < 1e-6

    def test_means_shape(self, trained_setup):
        df, prep, hmm, *_ = trained_setup
        assert hmm.means.shape == (3, 3)  # n_states × n_features

    def test_covars_shape(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        # diag covariance: (n_states, n_features)
        assert hmm.covars.shape == (3, 3)

    def test_predict_states_returns_valid_sequence(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        from src.data_loader import get_machine_sequence
        seq = get_machine_sequence(df, train_ids[0])
        X = prep.transform_sequence(seq)
        states = hmm.predict_states(X)
        assert len(states) == len(X)
        assert all(0 <= s < 3 for s in states)

    def test_predict_proba_shape(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        from src.data_loader import get_machine_sequence
        seq = get_machine_sequence(df, train_ids[0])
        X = prep.transform_sequence(seq)
        post = hmm.predict_proba(X)
        assert post.shape == (len(X), 3)

    def test_posteriors_sum_to_one(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        from src.data_loader import get_machine_sequence
        seq = get_machine_sequence(df, train_ids[0])
        X = prep.transform_sequence(seq)
        post = hmm.predict_proba(X)
        np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-5)

    def test_posteriors_between_0_and_1(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        from src.data_loader import get_machine_sequence
        seq = get_machine_sequence(df, train_ids[0])
        X = prep.transform_sequence(seq)
        post = hmm.predict_proba(X)
        assert np.all(post >= -1e-9)
        assert np.all(post <= 1 + 1e-9)

    def test_unfitted_model_raises(self):
        hmm_new = MachineHealthHMM(n_states=2)
        with pytest.raises(RuntimeError):
            hmm_new.predict_states(np.zeros((10, 2)))


# ---------------------------------------------------------------------------
# Multiple initializations
# ---------------------------------------------------------------------------

class TestMultiInit:

    def test_n_init_log_probs_recorded(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        assert len(hmm.training_log_probs) == hmm.n_init

    def test_best_log_prob_equals_max(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        assert abs(hmm.best_log_prob - max(hmm.training_log_probs)) < 1e-3


# ---------------------------------------------------------------------------
# State interpretation
# ---------------------------------------------------------------------------

class TestStateInterpretation:

    def test_returns_all_keys(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        interp = interpret_states(hmm.n_states, hmm.means,
                                   ["temperature", "vibration", "pressure"])
        for key in ["state_labels", "state_colors", "state_scores",
                    "confidence", "explanation", "ranking"]:
            assert key in interp

    def test_all_states_have_labels(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        interp = interpret_states(hmm.n_states, hmm.means,
                                   ["temperature", "vibration", "pressure"])
        for i in range(hmm.n_states):
            assert i in interp["state_labels"]

    def test_ranking_is_permutation(self, trained_setup):
        _, _, hmm, *_ = trained_setup
        interp = interpret_states(hmm.n_states, hmm.means,
                                   ["temperature", "vibration", "pressure"])
        assert sorted(interp["ranking"]) == list(range(hmm.n_states))


# ---------------------------------------------------------------------------
# Alert level
# ---------------------------------------------------------------------------

class TestAlertLevel:

    def test_alert_keys(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        from src.data_loader import get_machine_sequence
        interp = interpret_states(hmm.n_states, hmm.means,
                                   ["temperature", "vibration", "pressure"])
        seq = get_machine_sequence(df, train_ids[0])
        X   = prep.transform_sequence(seq)
        post = hmm.predict_proba(X)
        alert = compute_alert_level(post, interp)
        for key in ["alert_color", "alert_label", "critical_state_idx",
                    "latest_critical_prob"]:
            assert key in alert

    def test_prob_in_range(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        from src.data_loader import get_machine_sequence
        interp = interpret_states(hmm.n_states, hmm.means,
                                   ["temperature", "vibration", "pressure"])
        seq  = get_machine_sequence(df, train_ids[0])
        X    = prep.transform_sequence(seq)
        post = hmm.predict_proba(X)
        alert = compute_alert_level(post, interp)
        assert 0.0 <= alert["latest_critical_prob"] <= 1.0


# ---------------------------------------------------------------------------
# compare_n_states
# ---------------------------------------------------------------------------

class TestCompareNStates:

    def test_returns_list(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        X, lengths = prepare_sequences(df, prep, train_ids)
        results = compare_n_states(X, lengths, [2, 3], n_features=3, n_iter=30, n_init=1)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_result_keys(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        X, lengths = prepare_sequences(df, prep, train_ids)
        results = compare_n_states(X, lengths, [2], n_features=3, n_iter=30, n_init=1)
        for key in ["n_states", "log_likelihood", "aic", "bic", "converged"]:
            assert key in results[0]

    def test_log_likelihood_finite(self, trained_setup):
        df, prep, hmm, train_ids, *_ = trained_setup
        X, lengths = prepare_sequences(df, prep, train_ids)
        results = compare_n_states(X, lengths, [2, 3], n_features=3, n_iter=30, n_init=1)
        for r in results:
            assert np.isfinite(r["log_likelihood"]), f"Non-finite LL for {r['n_states']} states"
