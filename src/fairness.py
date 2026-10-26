"""Fairness objectives for motor-insurance balanced transfers.

Provides exact discrete AVaR computation, SD and AVaR minimisation over
balanced class-premium transfers for C² and C⁴ classifications, and
coverage improvement via null-space degrees of freedom.
"""

from typing import Tuple

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar

from src.data import R_LEVELS, RHO, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics
from src.transfer import solve_min_cost

# Cached distribution (used by all helpers below)
_DIST = build_joint_distribution()
_RHO_STAR = RHO.copy()


# ---------------------------------------------------------------------------
# Discrete AVaR
# ---------------------------------------------------------------------------

def compute_AVaR(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Compute AVaR_alpha (Conditional Value-at-Risk) for a discrete distribution.

    Uses the exact formula:
      VaR_alpha  = smallest y such that P(Y ≤ y) ≥ alpha
      AVaR_alpha = VaR_alpha + (1/(1-alpha)) E[max(Y − VaR_alpha, 0)]

    Parameters
    ----------
    values  : (n,) array of outcome values
    weights : (n,) array of probabilities summing to 1
    alpha   : tail level in [0, 1)

    Returns
    -------
    float — AVaR_alpha
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if alpha < 0 or alpha >= 1:
        raise ValueError(f"alpha must be in [0, 1); got {alpha}")
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError(f"weights must sum to 1; sum = {weights.sum()}")

    # Sort ascending
    idx = np.argsort(values)
    v_sorted = values[idx]
    w_sorted = weights[idx]

    # VaR: smallest value such that cumulative weight >= alpha
    cumw = np.cumsum(w_sorted)
    var_idx = int(np.searchsorted(cumw, alpha, side="left"))
    var_idx = min(var_idx, len(v_sorted) - 1)
    VaR = v_sorted[var_idx]

    # E[max(Y - VaR, 0)]
    excess = np.maximum(v_sorted - VaR, 0.0)
    E_excess = float(weights[idx] @ excess)

    return VaR + E_excess / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Kernel dimension
# ---------------------------------------------------------------------------

def kernel_dim(clf_name: str, rcond: float = 1e-10) -> int:
    """Return the dimension of ker(Q) for a named classification.

    Parameters
    ----------
    clf_name : classification name ('C8', 'C4', 'C3', 'C2')
    rcond    : singular-value cutoff for numerical rank

    Returns
    -------
    int — dim(ker Q) = n_classes − rank(Q)
    """
    cm = get_classification_map(clf_name)
    m = compute_classification_metrics(_DIST, cm, rho_reg=_RHO_STAR, rho_star=_RHO_STAR)
    Q = m["Q"]  # (3, n_classes)
    rank_Q = np.linalg.matrix_rank(Q, tol=rcond)
    return int(Q.shape[1] - rank_Q)


# ---------------------------------------------------------------------------
# C² balanced-transfer parameterisation helpers
# ---------------------------------------------------------------------------

def _c2_transfer_direction() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (q, rbar, w_C, direction) for C² where direction = Q @ e.

    The balanced-transfer parameterisation for C² (2 classes):
      p(t) = rbar + t * e,   e = (1, −q[0]/q[1])
      y(t) = w_C + t * direction,   direction = Q @ e
    """
    cm = get_classification_map("C2")
    m = compute_classification_metrics(_DIST, cm, rho_reg=_RHO_STAR, rho_star=_RHO_STAR)
    q = m["q"]
    rbar = m["rbar"]
    Q = m["Q"]
    w_C = m["w_C"]

    # Balanced direction: e[0]=1, e[1]=-q[0]/q[1], so q^T e = q[0] - q[0] = 0
    e = np.array([1.0, -q[0] / q[1]])
    direction = Q @ e   # (3,)

    return q, rbar, w_C, direction


def _c2_y(t: float, w_C: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Group-effect vector y(t) = w_C + t * direction."""
    return w_C + t * direction


def _c2_SD(t: float, w_C: np.ndarray, direction: np.ndarray) -> float:
    """Weighted standard deviation of y(t) under equal weights ρ* = (1/3,1/3,1/3)."""
    y = _c2_y(t, w_C, direction)
    rho = _RHO_STAR
    y_mean = float(rho @ y)
    return float(np.sqrt(rho @ (y - y_mean) ** 2))


def _c2_AVaR(t: float, w_C: np.ndarray, direction: np.ndarray, alpha: float) -> float:
    """AVaR_alpha of y(t) under equal weights ρ* = (1/3,1/3,1/3)."""
    y = _c2_y(t, w_C, direction)
    return compute_AVaR(y, _RHO_STAR, alpha)


def _c2_f_RA(t: float, q: np.ndarray, L: float) -> float:
    """f_RA(t) = L(C²) + ||t·e||_q² = L(C²) + q[0]*t²/q[1].

    Derivation: p - rbar = t*(1, -q[0]/q[1])
    ||p - rbar||_q² = q[0]*t² + q[1]*(t*q[0]/q[1])² = q[0]*t²*(1 + q[0]/q[1])
                    = q[0]*t²*(q[0]+q[1])/q[1] = q[0]*t²/q[1]
    """
    return L + q[0] * t ** 2 / q[1]


# ---------------------------------------------------------------------------
# SD minimiser for C²
# ---------------------------------------------------------------------------

def sd_minimizer(clf_name: str = "C2") -> Tuple[float, float, float]:
    """Minimise weighted SD of group premium effects for classification clf_name.

    Currently implemented for the C² case.

    Parameters
    ----------
    clf_name : classification name (currently only 'C2' supported)

    Returns
    -------
    (t_star, SD_star, f_RA_star)
    """
    if clf_name != "C2":
        raise NotImplementedError(f"sd_minimizer only implemented for C2; got {clf_name}")

    q, rbar, w_C, direction = _c2_transfer_direction()
    cm = get_classification_map(clf_name)
    m_clf = compute_classification_metrics(_DIST, cm)
    L = m_clf["L"]

    # Analytic minimiser: t* = -direction^T w_C / ||direction||^2
    # (where SD² = (1/3) sum_d (w_d + t*direction_d)^2, mean = 0 since rho^T direction = 0)
    t_star = float(-(direction @ w_C) / (direction @ direction))
    SD_star = _c2_SD(t_star, w_C, direction)
    f_RA_star = _c2_f_RA(t_star, q, L)

    return t_star, SD_star, f_RA_star


# ---------------------------------------------------------------------------
# AVaR minimiser for C²
# ---------------------------------------------------------------------------

def avar_minimizer(clf_name: str = "C2", alpha: float = 2 / 3) -> Tuple[float, float, float]:
    """Minimise AVaR_alpha of group premium effects for classification clf_name.

    Uses a scalar search over the balanced-transfer parameter t.
    For three equal-weight groups at alpha = 2/3, AVaR_{2/3} = max(y).

    Parameters
    ----------
    clf_name : classification name (currently only 'C2' supported)
    alpha    : tail level

    Returns
    -------
    (t_star, AVaR_star, f_RA_star)
    """
    if clf_name != "C2":
        raise NotImplementedError(f"avar_minimizer only implemented for C2; got {clf_name}")

    q, rbar, w_C, direction = _c2_transfer_direction()
    cm = get_classification_map(clf_name)
    m_clf = compute_classification_metrics(_DIST, cm)
    L = m_clf["L"]

    # For 3 equal-weight groups with alpha = 2/3, AVaR_{2/3}(y) = max(y).
    # Minimise max(y(t)) = piecewise-linear convex function of t.
    # Minimiser is where the two largest components are equal.
    # Direction analysis: find t where y_i(t) = y_j(t) for each pair (i,j)
    # then evaluate AVaR at each and take the overall minimum.

    # Search over t with refined grid around analytic breakpoints
    def objective(t: float) -> float:
        return _c2_AVaR(t, w_C, direction, alpha)

    # Compute pairwise-crossing t values (where y_i = y_j)
    breakpoints = []
    for i in range(3):
        for j in range(i + 1, 3):
            ddir = direction[i] - direction[j]
            if abs(ddir) > 1e-15:
                t_cross = float(-(w_C[i] - w_C[j]) / ddir)
                breakpoints.append(t_cross)

    # Evaluate at all breakpoints and nearby
    candidates = []
    for tb in breakpoints:
        for dt in np.linspace(-0.5, 0.5, 51):
            candidates.append(tb + dt)
    # Also do a coarse grid
    candidates.extend(np.linspace(-10, 10, 2001).tolist())

    best_t = min(candidates, key=objective)

    # Refine with Brent's method in a neighbourhood
    result = minimize_scalar(objective, bounds=(best_t - 1, best_t + 1), method="bounded")
    t_star = float(result.x)

    AVaR_star = float(objective(t_star))
    f_RA_star = _c2_f_RA(t_star, q, L)

    return t_star, AVaR_star, f_RA_star


# ---------------------------------------------------------------------------
# Coverage improvement via ker(Q)
# ---------------------------------------------------------------------------

def coverage_improvement(
    clf_name: str,
    alpha: float = 0.95,
    delta: float = 1.0,
    tol: float = 1e-9,
) -> Tuple[float, float, np.ndarray]:
    """Improve coverage AVaR_alpha via null-space perturbation of the min-risk premium.

    Computes p_0 (the minimum-risk premium for exact compensation y=0), then
    optimises v in ker(Q) subject to ||v||_q² ≤ delta to minimise AVaR_alpha of
    Z = Y - p_{C(θ)}.  Claim model: Y = 2R with probability 1/2 and Y = 0 otherwise.

    Parameters
    ----------
    clf_name : classification name ('C4' or 'C3')
    alpha    : AVaR tail level (default 0.95)
    delta    : extra risk-inadequacy budget ||v||_q² ≤ delta (default 1.0)
    tol      : attainability tolerance

    Returns
    -------
    (baseline_avar, improved_avar, v_star)
    baseline_avar : AVaR_alpha under p_0
    improved_avar : AVaR_alpha under p_0 + v_star
    v_star        : (m,) optimal kernel perturbation
    """
    # Compute exact-compensation minimum-risk premium p_0
    y_zero = np.zeros(3)
    c_C, p_0 = solve_min_cost(clf_name, y_zero, tol=tol)
    if p_0 is None:
        raise ValueError(f"Exact compensation infeasible for {clf_name}")

    cm = get_classification_map(clf_name)
    m_clf = compute_classification_metrics(_DIST, cm)
    Q = m_clf["Q"]
    q = m_clf["q"]

    # Build claim scenarios: (scenario_prob, residual_loss = Y - p_{C(R)})
    # Y = 0 or Y = 2*R, each with prob 1/2 conditional on (R, D)
    # P(R=r_j, D=d) = joint[d, j]
    joint = _DIST["joint"]   # (3, 8)

    n_d, n_r = joint.shape

    # Baseline residual losses under p_0
    def _build_losses(p: np.ndarray):
        losses = []
        probs = []
        for d in range(n_d):
            for j in range(n_r):
                p_ij = joint[d, j] / 2.0  # half prob for each Y outcome
                c = int(cm[j])
                # Y = 0: loss = 0 - p[c] = -p[c]
                losses.append(-p[c])
                probs.append(p_ij)
                # Y = 2R: loss = 2*R_j - p[c]
                losses.append(2.0 * R_LEVELS[j] - p[c])
                probs.append(p_ij)
        return np.array(losses), np.array(probs)

    losses_0, probs = _build_losses(p_0)
    probs = probs / probs.sum()   # normalise (should already be 1)
    baseline_avar = compute_AVaR(losses_0, probs, alpha)

    # Compute null space of Q
    N = null_space(Q, rcond=1e-10)   # (m, dim_ker)
    dim_ker = N.shape[1]

    if dim_ker == 0:
        # No kernel freedom — return baseline unchanged
        return baseline_avar, baseline_avar, np.zeros_like(p_0)

    # Weighted null-space metric: N^T D_q N
    D_q = np.diag(q)
    NtDqN = N.T @ D_q @ N   # (dim_ker, dim_ker)

    if dim_ker == 1:
        # Scalar search: v = x * N[:,0]
        n0 = N[:, 0]
        n0_qnorm_sq = float(n0 @ D_q @ n0)
        x_max = np.sqrt(delta / n0_qnorm_sq) if n0_qnorm_sq > 0 else 0.0

        def obj(x: float) -> float:
            p_trial = p_0 + x * n0
            losses_trial, _ = _build_losses(p_trial)
            return compute_AVaR(losses_trial, probs, alpha)

        grid = np.linspace(-x_max, x_max, 2001)
        best_x = min(grid, key=obj)

        # Refine
        res = minimize_scalar(
            obj,
            bounds=(max(best_x - 0.05 * x_max, -x_max),
                    min(best_x + 0.05 * x_max, x_max)),
            method="bounded",
        )
        x_star = float(res.x)
        v_star = x_star * n0

    else:
        # General case: grid search over the ball in dim_ker dimensions
        # Simplified: minimise along the direction that reduces AVaR fastest
        # Use scipy minimize with quadratic constraint
        from scipy.optimize import minimize as sp_minimize

        def obj_vec(x: np.ndarray) -> float:
            p_trial = p_0 + N @ x
            losses_trial, _ = _build_losses(p_trial)
            return compute_AVaR(losses_trial, probs, alpha)

        def constraint_budget(x: np.ndarray) -> float:
            return float(delta - x @ NtDqN @ x)

        from scipy.optimize import NonlinearConstraint
        con = NonlinearConstraint(lambda x: x @ NtDqN @ x, 0, delta)

        x0 = np.zeros(dim_ker)
        result = sp_minimize(
            obj_vec,
            x0,
            constraints=[con],
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        x_star_vec = result.x
        # Project to feasible set if constraint slightly violated
        qnorm_sq = float(x_star_vec @ NtDqN @ x_star_vec)
        if qnorm_sq > delta:
            x_star_vec *= np.sqrt(delta / qnorm_sq)
        v_star = N @ x_star_vec

    p_star = p_0 + v_star
    losses_star, _ = _build_losses(p_star)
    improved_avar = compute_AVaR(losses_star, probs, alpha)

    return baseline_avar, improved_avar, v_star
