"""Tests for src/fairness.py: AVaR computation, fairness minimisers, and coverage optimisation."""

import numpy as np
import pytest

from src.data import build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics
from src.fairness import (
    avar_minimizer,
    compute_AVaR,
    coverage_improvement,
    kernel_dim,
    sd_minimizer,
)

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

_DIST = build_joint_distribution()


# ---------------------------------------------------------------------------
# compute_AVaR — discrete average value at risk
# ---------------------------------------------------------------------------

def test_AVaR_known_case():
    """Known result: for uniform distribution over {1,2,3} at level 2/3, AVaR = 3.0.

    With values=(1,2,3), weights=(1/3,1/3,1/3), alpha=2/3:
      VaR_{2/3} = 2  (F(1)=1/3 < 2/3, F(2)=2/3 ≥ 2/3)
      E[max(X-2,0)] = 0·(1/3) + 0·(1/3) + 1·(1/3) = 1/3
      AVaR_{2/3} = (1/3) / (1 - 2/3) + 2 = 1 + 2 = 3.0
    """
    values  = np.array([1.0, 2.0, 3.0])
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    alpha   = 2 / 3
    result  = compute_AVaR(values, weights, alpha)
    assert abs(result - 3.0) < 1e-12, f"AVaR = {result}, expected 3.0"


def test_AVaR_nonnegative_for_nonnegative_inputs():
    """AVaR of a non-negative random variable must be non-negative."""
    rng = np.random.default_rng(42)
    values  = rng.uniform(0, 10, size=20)
    weights = np.ones(20) / 20
    for alpha in (0.5, 0.75, 0.9, 0.95):
        avar = compute_AVaR(values, weights, alpha)
        assert avar >= 0.0, f"AVaR at alpha={alpha} is negative: {avar}"


def test_AVaR_exceeds_mean():
    """AVaR_alpha >= E[X] for alpha >= 0 (coherent risk measure property)."""
    values  = np.array([88.0, 92.0, 96.0, 100.0, 104.0, 108.0, 112.0, 116.0])
    weights = np.array([32, 38, 42, 45, 44, 40, 34, 25], dtype=float) / 300
    mean    = float(weights @ values)
    for alpha in (0.5, 0.75, 0.9, 0.95):
        avar = compute_AVaR(values, weights, alpha)
        assert avar >= mean - 1e-12, (
            f"AVaR({alpha}) = {avar} < mean = {mean}"
        )


def test_AVaR_monotone_in_alpha():
    """AVaR must be non-decreasing in alpha (higher level → larger tail average)."""
    values  = np.array([88.0, 92.0, 96.0, 100.0, 104.0, 108.0, 112.0, 116.0])
    weights = np.array([32, 38, 42, 45, 44, 40, 34, 25], dtype=float) / 300
    alphas  = [0.5, 0.75, 0.9, 0.95]
    avars   = [compute_AVaR(values, weights, a) for a in alphas]
    for i in range(len(avars) - 1):
        assert avars[i] <= avars[i + 1] + 1e-12, (
            f"AVaR is not monotone: AVaR({alphas[i]})={avars[i]} > AVaR({alphas[i+1]})={avars[i+1]}"
        )


# ---------------------------------------------------------------------------
# kernel_dim — null-space dimension of Q
# ---------------------------------------------------------------------------

def test_C3_no_kernel_freedom():
    """dim(ker(Q_C3)) = 0: C³ has a square invertible Q, leaving no kernel freedom."""
    assert kernel_dim("C3") == 0, f"kernel_dim(C3) = {kernel_dim('C3')}, expected 0"


def test_C4_one_dimensional_kernel():
    """dim(ker(Q_C4)) = 1: C⁴ has one degree of freedom for coverage optimisation."""
    assert kernel_dim("C4") == 1, f"kernel_dim(C4) = {kernel_dim('C4')}, expected 1"


def test_C8_kernel_dimension():
    """dim(ker(Q_C8)) = 5: 8 classes with rank-3 Q leaves a 5D kernel."""
    assert kernel_dim("C8") == 5, f"kernel_dim(C8) = {kernel_dim('C8')}, expected 5"


def test_C2_kernel_dimension():
    """dim(ker(Q_C2)) = 0: Q_C2 is 3×2 with rank 2, no right null space."""
    assert kernel_dim("C2") == 0, f"kernel_dim(C2) = {kernel_dim('C2')}, expected 0"


# ---------------------------------------------------------------------------
# sd_minimizer and avar_minimizer for C² — fairness-optimal compensation
# ---------------------------------------------------------------------------

def test_SD_AVaR_C2_differ():
    """SD-minimiser and AVaR_{2/3}-minimiser must find different optimal parameters."""
    t_SD,  _, _ = sd_minimizer("C2")
    t_AVaR, _, _ = avar_minimizer("C2", alpha=2 / 3)
    assert abs(t_SD - t_AVaR) > 1e-6, (
        f"t_SD={t_SD}, t_AVaR={t_AVaR}: optimisers should differ by more than 1e-6"
    )


def test_C2_SD_minimizer_SD_value():
    """SD-minimiser for C² must achieve SD ≈ 0.149."""
    _, SD_val, _ = sd_minimizer("C2")
    assert abs(SD_val - 0.149) < 1e-2, (
        f"SD-minimiser: SD = {SD_val:.6f}, expected ≈ 0.149"
    )


def test_C2_SD_minimizer_f_RA():
    """SD-minimiser for C² must have f_RA ≈ 26.718."""
    _, _, f_RA_val = sd_minimizer("C2")
    assert abs(f_RA_val - 26.718) < 1e-2, (
        f"SD-minimiser: f_RA = {f_RA_val:.6f}, expected ≈ 26.718"
    )


def test_C2_AVaR_minimizer_AVaR_value():
    """AVaR_{2/3}-minimiser for C² must achieve AVaR ≈ 0.107."""
    _, AVaR_val, _ = avar_minimizer("C2", alpha=2 / 3)
    assert abs(AVaR_val - 0.107) < 1e-2, (
        f"AVaR-minimiser: AVaR = {AVaR_val:.6f}, expected ≈ 0.107"
    )


def test_C2_AVaR_minimizer_f_RA():
    """AVaR_{2/3}-minimiser for C² must have f_RA ≈ 27.279."""
    _, _, f_RA_val = avar_minimizer("C2", alpha=2 / 3)
    assert abs(f_RA_val - 27.279) < 1e-2, (
        f"AVaR-minimiser: f_RA = {f_RA_val:.6f}, expected ≈ 27.279"
    )


def test_SD_minimizer_returns_nonnegative_SD():
    """SD-minimiser must always return a non-negative SD value."""
    _, SD_val, _ = sd_minimizer("C2")
    assert SD_val >= 0.0, f"SD-minimiser returned negative SD: {SD_val}"


def test_AVaR_minimizer_returns_nonnegative_AVaR():
    """AVaR-minimiser must always return a non-negative AVaR value."""
    _, AVaR_val, _ = avar_minimizer("C2", alpha=2 / 3)
    assert AVaR_val >= 0.0, f"AVaR-minimiser returned negative AVaR: {AVaR_val}"


# ---------------------------------------------------------------------------
# coverage_improvement for C⁴ — kernel-based premium optimisation
# ---------------------------------------------------------------------------

def test_coverage_C4_baseline_AVaR():
    """Baseline AVaR_{0.95} under the class-mean premium p₀ must be ≈ 116.322."""
    baseline_avar, _, _ = coverage_improvement("C4", alpha=0.95)
    assert abs(baseline_avar - 116.322) < 1.0, (
        f"Baseline AVaR_{{0.95}} = {baseline_avar:.3f}, expected ≈ 116.322"
    )


def test_coverage_C4_improved_AVaR():
    """Coverage-optimal AVaR_{0.95} must be ≈ 115.955."""
    _, improved_avar, _ = coverage_improvement("C4", alpha=0.95)
    assert abs(improved_avar - 115.955) < 1.0, (
        f"Improved AVaR_{{0.95}} = {improved_avar:.3f}, expected ≈ 115.955"
    )


def test_coverage_C4_improvement_is_positive():
    """Optimising over the kernel must strictly improve (reduce) the tail AVaR."""
    baseline_avar, improved_avar, _ = coverage_improvement("C4", alpha=0.95)
    assert improved_avar < baseline_avar, (
        f"No improvement: baseline={baseline_avar:.6f}, improved={improved_avar:.6f}"
    )


def test_target_preserved_after_coverage_optimisation():
    """v_star must lie in ker(Q_C4), i.e. Q_C4 @ v_star ≈ 0 (target preserved)."""
    _, _, v_star = coverage_improvement("C4", alpha=0.95)

    m   = compute_classification_metrics(_DIST, get_classification_map("C4"))
    Q   = m["Q"]
    err = Q @ v_star
    assert np.abs(err).max() < 1e-10, (
        f"Q_C4 @ v_star = {err} (max |err|={np.abs(err).max():.2e}), expected 0"
    )
