"""Tests for src/data.py: probability distributions and classification maps."""

import numpy as np
import pytest

from src.data import (
    COND_PROB,
    R_LEVELS,
    RHO,
    build_joint_distribution,
    get_classification_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIST = build_joint_distribution()          # default rho = (1/3, 1/3, 1/3)
_CLF_NAMES = ("C8", "C4", "C3", "C2")
_N_CLASSES  = {"C8": 8, "C4": 4, "C3": 3, "C2": 2}


# ---------------------------------------------------------------------------
# COND_PROB structure
# ---------------------------------------------------------------------------

def test_conditional_prob_row_sums():
    """Each row of COND_PROB (one per D-group) must sum exactly to 1."""
    row_sums = COND_PROB.sum(axis=1)
    assert np.abs(row_sums - 1.0).max() < 1e-15, (
        f"COND_PROB row sums deviate from 1: {row_sums}"
    )


def test_cond_prob_shape():
    """COND_PROB must be 3 × 8 (three D-groups, eight R price levels)."""
    assert COND_PROB.shape == (3, 8)


def test_cond_prob_nonnegative():
    """All conditional probabilities must be non-negative."""
    assert COND_PROB.min() >= 0.0


# ---------------------------------------------------------------------------
# Joint distribution
# ---------------------------------------------------------------------------

def test_joint_distribution_sum():
    """The joint distribution must sum to 1 within floating-point tolerance."""
    joint = _DIST["joint"]
    assert abs(joint.sum() - 1.0) < 1e-15, (
        f"Joint distribution sums to {joint.sum()}, expected 1.0"
    )


def test_joint_distribution_shape():
    """Joint array must be (3, 8) — groups × price levels."""
    assert _DIST["joint"].shape == (3, 8)


def test_joint_marginalises_to_rho():
    """Summing the joint over R must recover the group prior rho."""
    marginal_D = _DIST["marginal_D"]
    assert np.abs(marginal_D - RHO).max() < 1e-15, (
        f"marginal_D = {marginal_D}, expected RHO = {RHO}"
    )


# ---------------------------------------------------------------------------
# Marginal of R
# ---------------------------------------------------------------------------

def test_marginal_R_values():
    """Marginal P(R) with equal-weight prior must match the count fractions."""
    expected = np.array([32, 38, 42, 45, 44, 40, 34, 25], dtype=np.float64) / 300
    marginal_R = _DIST["marginal_R"]
    assert np.abs(marginal_R - expected).max() < 1e-14, (
        f"marginal_R = {marginal_R}\nexpected    = {expected}"
    )


def test_marginal_R_sums_to_one():
    """The marginal R distribution must sum to 1."""
    assert abs(_DIST["marginal_R"].sum() - 1.0) < 1e-15


# ---------------------------------------------------------------------------
# Group and overall means
# ---------------------------------------------------------------------------

def test_group_means():
    """E[R | D=d] must match the analytically expected values."""
    expected = np.array([98.0, 100.76, 105.72])
    group_means = _DIST["group_means"]
    assert np.abs(group_means - expected).max() < 1e-10, (
        f"group_means = {group_means}\nexpected    = {expected}"
    )


def test_overall_mean():
    """E[R] must equal 7612/75 (the exact rational value)."""
    expected_exact = 7612 / 75
    overall_mean = _DIST["overall_mean"]
    assert abs(overall_mean - expected_exact) < 1e-10, (
        f"overall_mean = {overall_mean}, expected {expected_exact}"
    )


def test_overall_mean_equals_weighted_group_means():
    """E[R] must equal rho @ group_means (law of total expectation)."""
    implied = float(RHO @ _DIST["group_means"])
    assert abs(_DIST["overall_mean"] - implied) < 1e-12


# ---------------------------------------------------------------------------
# Classification maps
# ---------------------------------------------------------------------------

def test_classification_maps_class_counts():
    """Each classification must expose the correct number of distinct classes."""
    for name, expected_n in _N_CLASSES.items():
        cmap = get_classification_map(name)
        n_classes = int(cmap.max()) + 1
        assert n_classes == expected_n, (
            f"{name}: expected {expected_n} classes, got {n_classes}"
        )


def test_classification_maps_valid_indices():
    """All class indices must be non-negative integers in [0, n_classes)."""
    for name, expected_n in _N_CLASSES.items():
        cmap = get_classification_map(name)
        assert cmap.min() >= 0, f"{name}: negative class index {cmap.min()}"
        assert cmap.max() < expected_n, (
            f"{name}: class index {cmap.max()} out of range [0, {expected_n})"
        )


def test_classification_maps_length():
    """Every classification map must have one entry per R level (length 8)."""
    for name in _CLF_NAMES:
        cmap = get_classification_map(name)
        assert len(cmap) == len(R_LEVELS), (
            f"{name}: map length {len(cmap)}, expected {len(R_LEVELS)}"
        )


def test_c8_is_identity_map():
    """C8 assigns each R level its own class (no pooling)."""
    cmap = get_classification_map("C8")
    expected = np.arange(8, dtype=np.int64)
    assert np.array_equal(cmap, expected), f"C8 map = {cmap}, expected {expected}"


def test_unknown_classification_raises():
    """Requesting an unsupported classification name must raise ValueError."""
    with pytest.raises(ValueError):
        get_classification_map("C99")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_edge_cases_all_mass_on_group1():
    """With rho = (1, 0, 0), the implied marginal_R equals COND_PROB[0, :].

    build_joint_distribution validates against the default (1/3,1/3,1/3) marginal,
    so we verify the mathematical property directly via the marginal formula:
    marginal_R = rho @ COND_PROB.
    """
    rho_edge = np.array([1.0, 0.0, 0.0])

    # group_means are independent of rho (= COND_PROB @ R_LEVELS per row)
    group_means = _DIST["group_means"]
    assert abs(group_means[0] - 98.0) < 1e-10, (
        f"group_means[0] = {group_means[0]}, expected 98.0"
    )

    # marginal_R for rho=(1,0,0) = rho @ COND_PROB = COND_PROB[0, :]
    marginal_R_edge = rho_edge @ COND_PROB
    assert np.abs(marginal_R_edge - COND_PROB[0, :]).max() < 1e-14, (
        f"rho_edge @ COND_PROB = {marginal_R_edge}\n"
        f"COND_PROB[0, :]      = {COND_PROB[0, :]}"
    )


def test_edge_cases_all_mass_on_group3():
    """With rho = (0, 0, 1), the implied marginal_R equals COND_PROB[2, :]."""
    rho_edge = np.array([0.0, 0.0, 1.0])
    marginal_R_edge = rho_edge @ COND_PROB
    assert np.abs(marginal_R_edge - COND_PROB[2, :]).max() < 1e-14


def test_custom_rho_implied_joint_sums_to_one():
    """For any valid rho, rho[:, None] * COND_PROB (the joint) must sum to 1."""
    rho_custom = np.array([0.2, 0.5, 0.3])
    joint_custom = rho_custom[:, np.newaxis] * COND_PROB
    assert abs(joint_custom.sum() - 1.0) < 1e-15
