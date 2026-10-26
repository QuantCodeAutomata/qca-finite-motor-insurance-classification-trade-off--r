"""Tests for src/classification.py: accuracy and fairness metrics."""

import numpy as np
import pytest

from src.data import RHO, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics, validate_classification_metrics

# ---------------------------------------------------------------------------
# Fixtures / shared setup
# ---------------------------------------------------------------------------

_DIST = build_joint_distribution()
_CLF_NAMES = ("C8", "C4", "C3", "C2")


def _metrics(name):
    """Return compute_classification_metrics for the named classification."""
    return compute_classification_metrics(_DIST, get_classification_map(name))


# Pre-compute once for speed.
_ALL_METRICS = {name: _metrics(name) for name in _CLF_NAMES}

# Expected Table-2 values (from specification, verified against exp1_results.json)
_EXPECTED_L     = {"C8": 0.000000, "C4": 3.970919, "C3": 8.431019, "C2": 19.046236}
_EXPECTED_GAMMA = {"C8": 0.115805, "C4": 0.107304, "C3": 0.094281, "C2": 0.067228}
_EXPECTED_SD    = {"C8": 0.000000, "C4": 0.175373, "C3": 0.391553, "C2": 0.891995}


# ---------------------------------------------------------------------------
# L(C) — within-class variance
# ---------------------------------------------------------------------------

def test_L_C8_is_zero():
    """C⁸ has no within-class spread, so L must be exactly 0."""
    L = _ALL_METRICS["C8"]["L"]
    assert abs(L) < 1e-12, f"L(C8) = {L}, expected 0"


def test_L_matches_variance_formula():
    """L and L_check (variance decomposition) must agree for every classification.

    We allow atol=1e-10 to accommodate floating-point accumulation across the
    sum-of-squares computation (empirically the worst case is ~3e-12 for C3).
    """
    for name in _CLF_NAMES:
        m = _ALL_METRICS[name]
        assert abs(m["L"] - m["L_check"]) < 1e-10, (
            f"{name}: L={m['L']}, L_check={m['L_check']}, diff={abs(m['L']-m['L_check'])}"
        )


def test_expected_table2_L_values():
    """L(C) must match the Table-2 expected values within 1e-2."""
    for name in _CLF_NAMES:
        L = _ALL_METRICS[name]["L"]
        assert abs(L - _EXPECTED_L[name]) < 1e-2, (
            f"{name}: L={L:.6f}, expected {_EXPECTED_L[name]:.6f}"
        )


# ---------------------------------------------------------------------------
# Gamma(C) — group-profile dispersion
# ---------------------------------------------------------------------------

def test_Gamma_two_formulas_agree():
    """Gamma (barycenter formula) and Gamma_pairwise must agree for every classification."""
    for name in _CLF_NAMES:
        m = _ALL_METRICS[name]
        diff = abs(m["Gamma"] - m["Gamma_pairwise"])
        assert diff < 1e-12, (
            f"{name}: Gamma={m['Gamma']}, Gamma_pairwise={m['Gamma_pairwise']}, diff={diff}"
        )


def test_expected_table2_Gamma_values():
    """Gamma(C) must match the Table-2 expected values within 1e-2."""
    for name in _CLF_NAMES:
        G = _ALL_METRICS[name]["Gamma"]
        assert abs(G - _EXPECTED_GAMMA[name]) < 1e-2, (
            f"{name}: Gamma={G:.6f}, expected {_EXPECTED_GAMMA[name]:.6f}"
        )


# ---------------------------------------------------------------------------
# Q matrix
# ---------------------------------------------------------------------------

def test_Q_row_sums():
    """Each row of Q (one per D-group) must sum to 1 — total probability over classes."""
    for name in _CLF_NAMES:
        Q = _ALL_METRICS[name]["Q"]
        row_sums = Q.sum(axis=1)
        assert np.abs(row_sums - 1.0).max() < 1e-12, (
            f"{name}: Q row sums = {row_sums}"
        )


def test_Q_shape():
    """Q must have shape (3, n_classes) for the correct number of classes."""
    expected_n = {"C8": 8, "C4": 4, "C3": 3, "C2": 2}
    for name, nc in expected_n.items():
        Q = _ALL_METRICS[name]["Q"]
        assert Q.shape == (3, nc), f"{name}: Q.shape = {Q.shape}, expected (3, {nc})"


def test_Q_nonnegative():
    """All Q entries are probabilities and must be non-negative."""
    for name in _CLF_NAMES:
        assert _ALL_METRICS[name]["Q"].min() >= 0.0, f"{name}: Q has negative entry"


# ---------------------------------------------------------------------------
# q vector  (marginal class probabilities)
# ---------------------------------------------------------------------------

def test_q_equals_Qt_rho():
    """q = Q^T @ rho must hold exactly for all classifications."""
    for name in _CLF_NAMES:
        m = _ALL_METRICS[name]
        q_direct = m["Q"].T @ RHO
        assert np.abs(m["q"] - q_direct).max() < 1e-12, (
            f"{name}: q={m['q']}, Q^T@rho={q_direct}"
        )


def test_q_sums_to_one():
    """Marginal class probabilities must sum to 1."""
    for name in _CLF_NAMES:
        total = _ALL_METRICS[name]["q"].sum()
        assert abs(total - 1.0) < 1e-12, f"{name}: q.sum() = {total}"


# ---------------------------------------------------------------------------
# w_C — pooling bias
# ---------------------------------------------------------------------------

def test_rho_T_wC_is_zero():
    """rho^T @ w_C = 0: the pooling bias is budget-neutral under the default prior."""
    for name in _CLF_NAMES:
        w_C = _ALL_METRICS[name]["w_C"]
        inner = float(RHO @ w_C)
        assert abs(inner) < 1e-12, (
            f"{name}: rho^T @ w_C = {inner}, expected 0"
        )


def test_wC_C8_is_zero():
    """Under C⁸ (no pooling), the bias vector w_C must be identically zero."""
    w_C = _ALL_METRICS["C8"]["w_C"]
    assert np.abs(w_C).max() < 1e-12, f"w_C(C8) = {w_C}"


# ---------------------------------------------------------------------------
# SD(w_C)
# ---------------------------------------------------------------------------

def test_SD_formula():
    """SD(w_C) must be non-negative for every classification."""
    for name in _CLF_NAMES:
        SD = _ALL_METRICS[name]["SD_w"]
        assert SD >= 0.0, f"{name}: SD_w = {SD} < 0"


def test_SD_C8_is_zero():
    """C⁸ has no pooling bias, so SD(w_C) must be exactly 0."""
    SD = _ALL_METRICS["C8"]["SD_w"]
    assert abs(SD) < 1e-12, f"SD_w(C8) = {SD}"


def test_expected_table2_SD_values():
    """SD(w_C) must match the Table-2 expected values within 1e-2."""
    for name in _CLF_NAMES:
        SD = _ALL_METRICS[name]["SD_w"]
        assert abs(SD - _EXPECTED_SD[name]) < 1e-2, (
            f"{name}: SD_w={SD:.6f}, expected {_EXPECTED_SD[name]:.6f}"
        )


# ---------------------------------------------------------------------------
# Monotonicity across coarsening
# ---------------------------------------------------------------------------

def test_coarsening_monotonicity_L():
    """Coarser classifications must have weakly greater information loss L(C)."""
    L = {n: _ALL_METRICS[n]["L"] for n in _CLF_NAMES}
    assert L["C8"] <= L["C4"] + 1e-12, f"L(C8)={L['C8']} > L(C4)={L['C4']}"
    assert L["C4"] <= L["C3"] + 1e-12, f"L(C4)={L['C4']} > L(C3)={L['C3']}"
    assert L["C3"] <= L["C2"] + 1e-12, f"L(C3)={L['C3']} > L(C2)={L['C2']}"


def test_coarsening_monotonicity_Gamma():
    """Coarser classifications must have weakly smaller group-profile dispersion Gamma(C)."""
    G = {n: _ALL_METRICS[n]["Gamma"] for n in _CLF_NAMES}
    assert G["C8"] >= G["C4"] - 1e-12, f"Gamma(C8)={G['C8']} < Gamma(C4)={G['C4']}"
    assert G["C4"] >= G["C3"] - 1e-12, f"Gamma(C4)={G['C4']} < Gamma(C3)={G['C3']}"
    assert G["C3"] >= G["C2"] - 1e-12, f"Gamma(C3)={G['C3']} < Gamma(C2)={G['C2']}"


# ---------------------------------------------------------------------------
# validate_classification_metrics passes without error
# ---------------------------------------------------------------------------

def test_validate_passes_for_all_classifications():
    """validate_classification_metrics must not raise for any valid classification."""
    for name in _CLF_NAMES:
        validate_classification_metrics(_ALL_METRICS[name])  # raises on failure
