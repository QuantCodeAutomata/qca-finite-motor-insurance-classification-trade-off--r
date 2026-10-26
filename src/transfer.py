"""Balanced-transfer attainability and minimum-risk-inadequacy premiums.

Implements the linear-algebra conditions and pseudoinverse formulas from
Section 4 of the paper:

  Attainability of target y ∈ R^k:
    (i)  ρ*^T y = b   (mean condition)
    (ii) (y − w_C) ∈ im(Q)   (image condition)

  Minimum-risk-inadequacy premium (when attainable):
    p_y = r̄ + D_q^{-1} Q^T H_C^† (y − w_C)

  Extra cost:
    c_C(y) = (y − w_C)^T H_C^† (y − w_C)

  Total risk inadequacy:
    f_RA(p_y) = L(C) + c_C(y)
"""

from typing import Optional, Tuple

import numpy as np

from src.data import RHO, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics

# ---------------------------------------------------------------------------
# Module-level cached distributions and metrics
# ---------------------------------------------------------------------------

_DIST = build_joint_distribution()
_RHO_STAR = RHO.copy()


def _get_metrics(clf_name: str) -> dict:
    """Build and return classification metrics for a named classification."""
    cm = get_classification_map(clf_name)
    return compute_classification_metrics(_DIST, cm, rho_reg=_RHO_STAR, rho_star=_RHO_STAR)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def class_means(clf_name: str) -> np.ndarray:
    """Return class conditional mean premiums r̄_c for a named classification.

    Parameters
    ----------
    clf_name : one of 'C8', 'C4', 'C3', 'C2'

    Returns
    -------
    np.ndarray of shape (m,) where m is the number of classes.
    """
    return _get_metrics(clf_name)["rbar"]


def class_probs(clf_name: str) -> np.ndarray:
    """Return class probabilities q_c = P(C = c) for a named classification.

    Parameters
    ----------
    clf_name : one of 'C8', 'C4', 'C3', 'C2'

    Returns
    -------
    np.ndarray of shape (m,).
    """
    return _get_metrics(clf_name)["q"]


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def _projection_residual(Q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute the component of v orthogonal to im(Q).

    Returns (I - Q @ pinv(Q)) @ v.  If v ∈ im(Q) the norm of this residual is 0.
    """
    # Q is (D, m); pinv(Q) is (m, D)
    Q_pinv = np.linalg.pinv(Q)
    return v - Q @ (Q_pinv @ v)


# ---------------------------------------------------------------------------
# Attainability
# ---------------------------------------------------------------------------

def is_attainable(
    clf_name: str,
    y: np.ndarray,
    tol: float = 1e-9,
) -> bool:
    """Check whether target group-effect vector y is attainable as a balanced transfer.

    Conditions (main case, b = 0):
      (i)  |ρ*^T y|  < tol           — mean condition
      (ii) ||(I − Q Q^†)(y − w_C)||  < tol — image condition

    Parameters
    ----------
    clf_name : classification name
    y : (D,) target group total premium-effect vector
    tol : numerical tolerance

    Returns
    -------
    bool — True iff both conditions pass.
    """
    y = np.asarray(y, dtype=np.float64)
    m = _get_metrics(clf_name)
    rho_star = _RHO_STAR
    Q = m["Q"]
    w_C = m["w_C"]
    mu_star = m["mu_star"]
    rbar = m["rbar"]
    q = m["q"]

    # b = E_{P*}[A^C(C(θ)) − R] = q_star^T rbar − rho_star^T mu_star
    # In the main case q_star = q^T rho_star (q is already under P = P*).
    b = float(q @ rbar) - float(rho_star @ mu_star)

    # (i) mean condition
    mean_ok = abs(float(rho_star @ y) - b) < tol

    # (ii) image condition: (y − w_C) ∈ im(Q)
    diff = y - w_C
    residual = _projection_residual(Q, diff)
    image_ok = float(np.linalg.norm(residual)) < tol

    return mean_ok and image_ok


# ---------------------------------------------------------------------------
# Minimum-risk-inadequacy premium
# ---------------------------------------------------------------------------

def solve_min_cost(
    clf_name: str,
    y: np.ndarray,
    tol: float = 1e-9,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Compute minimum-risk-inadequacy class premium for an attainable target.

    If y is not attainable, returns (None, None).

    Parameters
    ----------
    clf_name : classification name
    y : (D,) target group-effect vector
    tol : attainability tolerance

    Returns
    -------
    (c_C, p_y) : extra cost and optimal premium vector, or (None, None).
    """
    if not is_attainable(clf_name, y, tol=tol):
        return None, None

    y = np.asarray(y, dtype=np.float64)
    m = _get_metrics(clf_name)
    Q = m["Q"]
    rbar = m["rbar"]
    q = m["q"]
    w_C = m["w_C"]

    D_q_inv = np.diag(1.0 / q)                       # (m, m)
    H_C = Q @ D_q_inv @ Q.T                          # (3, 3)
    H_C_pinv = np.linalg.pinv(H_C)                   # (3, 3)

    diff = y - w_C                                    # (3,)
    # p_y = r̄ + D_q^{-1} Q^T H_C^† (y − w_C)
    p_y = rbar + D_q_inv @ Q.T @ H_C_pinv @ diff    # (m,)
    c_C = float(diff @ H_C_pinv @ diff)

    return c_C, p_y


# ---------------------------------------------------------------------------
# Total risk inadequacy
# ---------------------------------------------------------------------------

def compute_f_RA(
    clf_name: str,
    y: np.ndarray,
    tol: float = 1e-9,
) -> Optional[float]:
    """Compute the minimum total risk inadequacy f_RA(p_y) = L(C) + c_C(y).

    Returns None if y is not attainable.

    Parameters
    ----------
    clf_name : classification name
    y : (D,) target group-effect vector

    Returns
    -------
    float or None
    """
    c_C, _ = solve_min_cost(clf_name, y, tol=tol)
    if c_C is None:
        return None
    L = _get_metrics(clf_name)["L"]
    return L + c_C


# ---------------------------------------------------------------------------
# Full transfer solver (for experiment scripts)
# ---------------------------------------------------------------------------

def solve_transfer(
    clf_name: str,
    y: np.ndarray,
    tol: float = 1e-9,
) -> dict:
    """Full transfer analysis for classification clf_name with target y.

    Parameters
    ----------
    clf_name : classification name
    y : (D,) target group-effect vector
    tol : attainability tolerance

    Returns
    -------
    dict with keys:
        'attainable'         : bool
        'c_C'                : extra cost (None if infeasible)
        'p_y'                : optimal premium vector (None if infeasible)
        'lambda_y'           : transfer = p_y − r̄ (None if infeasible)
        'f_RA'               : L + c_C (None if infeasible)
        'f_RA_direct'        : direct computation E[(p_{C} − R)^2] (None if infeasible)
        'Q_p_residual'       : ||Q p_y − μ* − y|| (None if infeasible)
        'balance_residual'   : |q^T p_y − q^T r̄| (None if infeasible)
        'L'                  : L(C) (always)
        'Gamma'              : Γ(C) (always)
        'w_C'                : pooling effect (always)
        'Q'                  : Q matrix (always)
        'rbar'               : class means (always)
        'q'                  : class probs (always)
    """
    from src.data import R_LEVELS
    m = _get_metrics(clf_name)
    cm = get_classification_map(clf_name)
    marginal_R = _DIST["marginal_R"]
    attainable = is_attainable(clf_name, y, tol=tol)

    result: dict = {
        "attainable": attainable,
        "c_C": None,
        "p_y": None,
        "lambda_y": None,
        "f_RA": None,
        "f_RA_direct": None,
        "Q_p_residual": None,
        "balance_residual": None,
        "L": m["L"],
        "Gamma": m["Gamma"],
        "w_C": m["w_C"],
        "Q": m["Q"],
        "rbar": m["rbar"],
        "q": m["q"],
    }

    if not attainable:
        return result

    c_C, p_y = solve_min_cost(clf_name, y, tol=tol)
    lambda_y = p_y - m["rbar"]
    f_RA = m["L"] + c_C

    # Direct check
    f_RA_direct = float(
        np.sum(marginal_R * (p_y[cm] - R_LEVELS) ** 2)
    )

    Q_p_res = float(np.linalg.norm(m["Q"] @ p_y - m["mu_star"] - y))
    bal_res = float(abs(m["q"] @ p_y - m["q"] @ m["rbar"]))

    result.update(
        c_C=c_C,
        p_y=p_y,
        lambda_y=lambda_y,
        f_RA=f_RA,
        f_RA_direct=f_RA_direct,
        Q_p_residual=Q_p_res,
        balance_residual=bal_res,
    )
    return result
