"""
tests/test_hmm_algorithms.py
============================
Unit tests for educational HMM algorithm implementations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.hmm_algorithms import (
    forward_algorithm,
    forward_algorithm_scaled,
    backward_algorithm_scaled,
    forward_backward,
    viterbi,
    marginal_decoding,
    baum_welch_one_step,
    underflow_demo,
    validate_hmm_params,
    EXAMPLE_HMMS,
)


# ---------------------------------------------------------------------------
# Fixtures: small 2-state HMM
# ---------------------------------------------------------------------------

@pytest.fixture
def hmm_2state():
    pi = np.array([0.6, 0.4])
    A  = np.array([[0.7, 0.3],
                   [0.4, 0.6]])
    B  = np.array([[0.5, 0.4, 0.1],
                   [0.1, 0.3, 0.6]])
    return pi, A, B

@pytest.fixture
def obs_seq():
    return [0, 1, 2, 0, 1]


# ---------------------------------------------------------------------------
# validate_hmm_params
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_params(self, hmm_2state):
        pi, A, B = hmm_2state
        assert validate_hmm_params(pi, A, B) is None

    def test_invalid_pi_sum(self, hmm_2state):
        pi, A, B = hmm_2state
        pi_bad = np.array([0.8, 0.8])
        err = validate_hmm_params(pi_bad, A, B)
        assert err is not None

    def test_invalid_A_row_sum(self, hmm_2state):
        pi, A, B = hmm_2state
        A_bad = A.copy()
        A_bad[0, 0] = 0.9  # row 0 now sums to > 1
        err = validate_hmm_params(pi, A_bad, B)
        assert err is not None

    def test_invalid_B_row_sum(self, hmm_2state):
        pi, A, B = hmm_2state
        B_bad = B.copy()
        B_bad[0] = np.array([0.9, 0.9, 0.1])
        err = validate_hmm_params(pi, A, B_bad)
        assert err is not None

    def test_shape_mismatch(self, hmm_2state):
        pi, A, B = hmm_2state
        A_bad = np.ones((3, 3)) / 3
        err = validate_hmm_params(pi, A_bad, B)
        assert err is not None


# ---------------------------------------------------------------------------
# Forward algorithm
# ---------------------------------------------------------------------------

class TestForwardAlgorithm:

    def test_shape(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        alpha, _ = forward_algorithm(pi, A, B, obs_seq)
        assert alpha.shape == (len(obs_seq), 2)

    def test_log_prob_is_finite(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        _, lp = forward_algorithm(pi, A, B, obs_seq)
        assert np.isfinite(lp), f"log_prob is not finite: {lp}"

    def test_alpha_nonnegative(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        alpha, _ = forward_algorithm(pi, A, B, obs_seq)
        assert np.all(alpha >= 0)

    def test_initial_step_matches_pi_b(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        alpha, _ = forward_algorithm(pi, A, B, obs_seq)
        expected = pi * B[:, obs_seq[0]]
        np.testing.assert_allclose(alpha[0], expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Scaled forward algorithm
# ---------------------------------------------------------------------------

class TestForwardAlgorithmScaled:

    def test_shape(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        alpha_hat, c, lp = forward_algorithm_scaled(pi, A, B, obs_seq)
        assert alpha_hat.shape == (len(obs_seq), 2)
        assert c.shape == (len(obs_seq),)

    def test_rows_sum_to_one(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        alpha_hat, _, _ = forward_algorithm_scaled(pi, A, B, obs_seq)
        row_sums = alpha_hat.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(obs_seq)), atol=1e-6)

    def test_log_prob_matches_unscaled(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        _, lp_unscaled = forward_algorithm(pi, A, B, obs_seq)
        _, _, lp_scaled = forward_algorithm_scaled(pi, A, B, obs_seq)
        assert abs(lp_unscaled - lp_scaled) < 1e-6, (
            f"Unscaled ({lp_unscaled:.6f}) and scaled ({lp_scaled:.6f}) log-probs differ"
        )


# ---------------------------------------------------------------------------
# Backward algorithm
# ---------------------------------------------------------------------------

class TestBackwardAlgorithm:

    def test_shape(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        _, c, _ = forward_algorithm_scaled(pi, A, B, obs_seq)
        beta = backward_algorithm_scaled(A, B, obs_seq, c)
        assert beta.shape == (len(obs_seq), 2)

    def test_final_step_nonzero(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        _, c, _ = forward_algorithm_scaled(pi, A, B, obs_seq)
        beta = backward_algorithm_scaled(A, B, obs_seq, c)
        assert np.all(beta[-1] > 0)


# ---------------------------------------------------------------------------
# Forward-Backward
# ---------------------------------------------------------------------------

class TestForwardBackward:

    def test_gamma_sums_to_one(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = forward_backward(pi, A, B, obs_seq)
        gamma = result["gamma"]
        row_sums = gamma.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(obs_seq)), atol=1e-6,
                                   err_msg="γ rows must sum to 1")

    def test_gamma_shape(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = forward_backward(pi, A, B, obs_seq)
        assert result["gamma"].shape == (len(obs_seq), 2)

    def test_xi_shape(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = forward_backward(pi, A, B, obs_seq)
        assert result["xi"].shape == (len(obs_seq) - 1, 2, 2)

    def test_log_prob_finite(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = forward_backward(pi, A, B, obs_seq)
        assert np.isfinite(result["log_prob"])

    def test_gamma_values_in_0_1(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = forward_backward(pi, A, B, obs_seq)
        gamma = result["gamma"]
        assert np.all(gamma >= -1e-9)
        assert np.all(gamma <= 1 + 1e-9)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------

class TestViterbi:

    def test_best_path_length(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = viterbi(pi, A, B, obs_seq)
        assert len(result["best_path"]) == len(obs_seq)

    def test_states_in_valid_range(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = viterbi(pi, A, B, obs_seq)
        assert all(0 <= s < 2 for s in result["best_path"])

    def test_log_prob_finite(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = viterbi(pi, A, B, obs_seq)
        assert np.isfinite(result["log_prob"])

    def test_delta_shape(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = viterbi(pi, A, B, obs_seq)
        assert result["delta"].shape == (len(obs_seq), 2)

    def test_viterbi_log_prob_le_forward(self, hmm_2state, obs_seq):
        """
        The Viterbi log-prob (best single path) ≤ forward log-prob
        (sum over all paths).
        """
        pi, A, B = hmm_2state
        vit_result = viterbi(pi, A, B, obs_seq)
        _, lp_fwd = forward_algorithm(pi, A, B, obs_seq)
        assert vit_result["log_prob"] <= lp_fwd + 1e-6, (
            f"Viterbi log-prob {vit_result['log_prob']:.4f} > forward log-prob {lp_fwd:.4f}"
        )


# ---------------------------------------------------------------------------
# Marginal decoding
# ---------------------------------------------------------------------------

class TestMarginalDecoding:

    def test_length(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = forward_backward(pi, A, B, obs_seq)
        marg = marginal_decoding(result["gamma"])
        assert len(marg) == len(obs_seq)

    def test_valid_states(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        result = forward_backward(pi, A, B, obs_seq)
        marg = marginal_decoding(result["gamma"])
        assert all(0 <= s < 2 for s in marg)


# ---------------------------------------------------------------------------
# Baum-Welch one step
# ---------------------------------------------------------------------------

class TestBaumWelchStep:

    def test_log_likelihood_nondecreasing(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        ll_before = forward_backward(pi, A, B, obs_seq)["log_prob"]
        pi_new, A_new, B_new = baum_welch_one_step(pi, A, B, obs_seq)
        ll_after  = forward_backward(pi_new, A_new, B_new, obs_seq)["log_prob"]
        assert ll_after >= ll_before - 1e-6, (
            f"Log-likelihood decreased: {ll_before:.6f} → {ll_after:.6f}"
        )

    def test_pi_sums_to_one(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        pi_new, _, _ = baum_welch_one_step(pi, A, B, obs_seq)
        assert abs(pi_new.sum() - 1.0) < 1e-6

    def test_A_rows_sum_to_one(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        _, A_new, _ = baum_welch_one_step(pi, A, B, obs_seq)
        for i in range(A_new.shape[0]):
            assert abs(A_new[i].sum() - 1.0) < 1e-6

    def test_B_rows_sum_to_one(self, hmm_2state, obs_seq):
        pi, A, B = hmm_2state
        _, _, B_new = baum_welch_one_step(pi, A, B, obs_seq)
        for i in range(B_new.shape[0]):
            assert abs(B_new[i].sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Underflow demo
# ---------------------------------------------------------------------------

class TestUnderflowDemo:

    def test_keys_present(self, hmm_2state):
        pi, A, B = hmm_2state
        obs_long = [0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        result = underflow_demo(pi, A, B, obs_long)
        for key in ["unscaled_alpha", "scaled_alpha", "scaling_c",
                    "log_prob_scaled", "log_prob_unscaled", "underflow_occurred"]:
            assert key in result

    def test_scaled_log_prob_finite(self, hmm_2state):
        pi, A, B = hmm_2state
        obs_long = list(range(3)) * 10
        result = underflow_demo(pi, A, B, obs_long)
        assert np.isfinite(result["log_prob_scaled"])


# ---------------------------------------------------------------------------
# Example HMMs
# ---------------------------------------------------------------------------

class TestExampleHMMs:

    def test_all_examples_valid(self):
        for name, ex in EXAMPLE_HMMS.items():
            pi = ex["pi"]
            A  = ex["A"]
            B  = ex["B"]
            err = validate_hmm_params(pi, A, B)
            assert err is None, f"Invalid HMM '{name}': {err}"

    def test_forward_backward_on_all_examples(self):
        for name, ex in EXAMPLE_HMMS.items():
            pi, A, B = ex["pi"], ex["A"], ex["B"]
            M = B.shape[1]
            obs = list(range(M)) * 3
            result = forward_backward(pi, A, B, obs)
            assert np.isfinite(result["log_prob"]), f"log_prob not finite for '{name}'"
            row_sums = result["gamma"].sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                       err_msg=f"γ row sums ≠ 1 for '{name}'")
