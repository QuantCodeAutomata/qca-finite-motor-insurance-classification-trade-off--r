"""Tests for src/transfer.py: exact compensation and risk-adjustment cost."""

import numpy as np
import pytest

from src.data import RHO, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics
from src.transfer import (
    class_means,
    class_probs,
    compute_f_RA,
    is_attainable,
    solve_min_cost,
)

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

_DIST = build_joint_distribution()
_CLF_NAMES = ("C8", "C4", "C3", "C2")

# y = 0  — exact compensation target (no residual group deviation)
_Y_ZERO = {name: np.zeros(3) for name in _CLF_NAMES}


# ---------------------------------------------------------------------------
# C⁸ — exact compensation always feasible at zero cost
# ---------------------------------------------------------------------------

def test_C8_exact_compensation_feasible():
    """For C⁸ (individual pricing), exact compensation with y=0 costs nothing."""
    y = _Y_ZERO["C8"]
    assert is_attainable("C8", y), "C8 exact compensation should be attainable"

    cost, p_y = solve_min_cost("C8", y)
    assert cost is not None, "solve_min_cost returned None for attainable C8"
    assert abs(cost) < 1e-10, f"c_C8(0) = {cost}, expected 0"


def test_C8_f_RA_is_zero():
    """f_RA = L(C⁸) + c_C⁸(0) = 0 + 0 = 0."""
    y = _Y_ZERO["C8"]
    f = compute_f_RA("C8", y)
    assert f is not None, "compute_f_RA returned None for attainable C8"
    assert abs(f) < 1e-10, f"f_RA(C8, y=0) = {f}, expected 0"


# ---------------------------------------------------------------------------
# C⁴ — feasible with small nonzero cost
# ---------------------------------------------------------------------------

def test_C4_exact_compensation_attainable():
    """y=0 is attainable for C⁴ (rank(Q_C4) = 3 = dim of group space)."""
    assert is_attainable("C4", _Y_ZERO["C4"])


def test_C4_exact_compensation_cost():
    """c_{C⁴}(0) must be approximately 0.230."""
    cost, _ = solve_min_cost("C4", _Y_ZERO["C4"])
    assert cost is not None
    assert abs(cost - 0.230) < 1e-2, f"c_C4(0) = {cost:.6f}, expected ≈ 0.230"


def test_C4_exact_compensation_f_RA():
    """f_RA(C⁴, y=0) must be approximately 4.201."""
    f = compute_f_RA("C4", _Y_ZERO["C4"])
    assert f is not None
    assert abs(f - 4.201) < 1e-2, f"f_RA(C4) = {f:.6f}, expected ≈ 4.201"


# ---------------------------------------------------------------------------
# C³ — feasible with moderate cost
# ---------------------------------------------------------------------------

def test_C3_exact_compensation_attainable():
    """y=0 is attainable for C³ (Q_C3 is square and invertible)."""
    assert is_attainable("C3", _Y_ZERO["C3"])


def test_C3_exact_compensation_cost():
    """c_{C³}(0) must be approximately 1.465."""
    cost, _ = solve_min_cost("C3", _Y_ZERO["C3"])
    assert cost is not None
    assert abs(cost - 1.465) < 1e-2, f"c_C3(0) = {cost:.6f}, expected ≈ 1.465"


def test_C3_exact_compensation_f_RA():
    """f_RA(C³, y=0) must be approximately 9.896."""
    f = compute_f_RA("C3", _Y_ZERO["C3"])
    assert f is not None
    assert abs(f - 9.896) < 1e-2, f"f_RA(C3) = {f:.6f}, expected ≈ 9.896"


# ---------------------------------------------------------------------------
# C² — exact compensation is infeasible
# ---------------------------------------------------------------------------

def test_C2_infeasibility():
    """Exact compensation y=0 must NOT be attainable for C²."""
    assert not is_attainable("C2", _Y_ZERO["C2"]), (
        "C2 exact compensation should be infeasible (mu_star ∉ col(Q_C2))"
    )


def test_C2_solve_min_cost_returns_none():
    """solve_min_cost must signal infeasibility for C² with y=0."""
    cost, p_y = solve_min_cost("C2", _Y_ZERO["C2"])
    assert cost is None and p_y is None, (
        f"Expected (None, None) for infeasible C2; got cost={cost}"
    )


def test_C2_f_RA_returns_none():
    """compute_f_RA must return None for an infeasible target."""
    f = compute_f_RA("C2", _Y_ZERO["C2"])
    assert f is None, f"Expected None for infeasible C2; got {f}"


# ---------------------------------------------------------------------------
# f_RA = L(C) + c_C(y) formula check
# ---------------------------------------------------------------------------

def test_f_RA_formula_vs_direct():
    """For every attainable target, f_RA must equal L(C) + c_C(y)."""
    attainable_names = ("C8", "C4", "C3")
    for name in attainable_names:
        y = _Y_ZERO[name]
        cost, _ = solve_min_cost(name, y)
        f_via_formula = compute_f_RA(name, y)

        m = compute_classification_metrics(_DIST, get_classification_map(name))
        L = m["L"]
        f_expected = L + cost

        assert abs(f_via_formula - f_expected) < 1e-8, (
            f"{name}: f_RA={f_via_formula:.10f}, L+c_C(y)={f_expected:.10f}, "
            f"diff={abs(f_via_formula - f_expected):.2e}"
        )


# ---------------------------------------------------------------------------
# Balance constraint:  q^T @ p_y ≈ q^T @ rbar = E[R]
# ---------------------------------------------------------------------------

def test_balance_constraint():
    """The expected premium q^T p_y must equal the overall mean E[R] for all attainable solutions."""
    attainable_names = ("C8", "C4", "C3")
    for name in attainable_names:
        y = _Y_ZERO[name]
        _, p_y = solve_min_cost(name, y)
        assert p_y is not None, f"{name}: expected a feasible solution"

        q = class_probs(name)
        rbar = class_means(name)

        balance_p_y  = float(q @ p_y)
        balance_rbar = float(q @ rbar)

        assert abs(balance_p_y - balance_rbar) < 1e-10, (
            f"{name}: q^T p_y={balance_p_y:.12f}, q^T rbar={balance_rbar:.12f}, "
            f"diff={abs(balance_p_y - balance_rbar):.2e}"
        )


# ---------------------------------------------------------------------------
# Target achievement:  Q @ p_y - mu_star ≈ y
# ---------------------------------------------------------------------------

def test_target_achievement():
    """The optimal premium must satisfy the group-compensation constraint Q @ p_y = mu_star + y."""
    attainable_names = ("C8", "C4", "C3")
    for name in attainable_names:
        y = _Y_ZERO[name]
        _, p_y = solve_min_cost(name, y)
        assert p_y is not None

        m = compute_classification_metrics(_DIST, get_classification_map(name))
        Q = m["Q"]
        mu_star = m["mu_star"]

        residual = Q @ p_y - mu_star - y
        assert np.abs(residual).max() < 1e-10, (
            f"{name}: Q@p_y - mu_star - y = {residual} (max |err|={np.abs(residual).max():.2e})"
        )


# ---------------------------------------------------------------------------
# Monotonicity of f_RA across coarsening
# ---------------------------------------------------------------------------

def test_f_RA_monotone_with_coarsening():
    """f_RA(C) must be non-decreasing as the classification becomes coarser."""
    f_C8 = compute_f_RA("C8", _Y_ZERO["C8"])
    f_C4 = compute_f_RA("C4", _Y_ZERO["C4"])
    f_C3 = compute_f_RA("C3", _Y_ZERO["C3"])

    assert f_C8 <= f_C4 + 1e-10, f"f_RA(C8)={f_C8} > f_RA(C4)={f_C4}"
    assert f_C4 <= f_C3 + 1e-10, f"f_RA(C4)={f_C4} > f_RA(C3)={f_C3}"
