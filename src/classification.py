"""Motor-insurance classification experiment: fairness and accuracy metrics."""

import numpy as np
from typing import Dict, Optional

from src.data import R_LEVELS, D_LEVELS


def compute_classification_metrics(
    dist: Dict[str, np.ndarray],
    class_map: np.ndarray,
    rho_reg: Optional[np.ndarray] = None,
    rho_star: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute accuracy and group-fairness metrics for a given classification.

    Parameters
    ----------
    dist:
        Output of ``build_joint_distribution()``.
    class_map:
        (8,) int array mapping each price-level index j to a class index c (0-based).
    rho_reg:
        (3,) regulatory group probabilities used in computing ``s`` and ``eta``.
        Defaults to ``dist['marginal_D']``, which equals RHO = (1/3, 1/3, 1/3) in
        the main experiment.
    rho_star:
        (3,) target group distribution used in ``eta_bar``, ``w_C``, and ``SD_w``.
        Defaults to ``dist['marginal_D']``.

    Returns
    -------
    dict with keys:
        'n_classes'       : int  — number of classes m
        'q'               : (m,) P(C = c) under the population measure P
        's'               : (m,) regulatory P^reg(C = c); equals q in the main case
        'rbar'            : (m,) E[R | C = c]
        'eta'             : (m, 3) group profiles; eta[c, d] = P^reg(D = d | C = c)
        'eta_bar'         : (3,)  = rho_star
        'Q'               : (3, m) Q[d, c] = P(C = c | D = d)
        'L'               : scalar L(C) = E[(R - rbar(C))^2]  (within-class variance)
        'L_check'         : scalar E[R^2] - sum_c q[c]*rbar[c]^2  (should equal L)
        'Gamma'           : scalar 2 * sum_c s[c] * ||eta[c] - eta_bar||^2
        'Gamma_pairwise'  : scalar sum_{c,c'} s[c]*s[c']*||eta[c]-eta[c']||^2
                            (equals Gamma when rho_reg == rho_star)
        'w_C'             : (3,) pooling bias; w_C[d] = (Q @ rbar)[d] - mu_star[d]
        'SD_w'            : scalar std-dev of w_C weighted by rho_star
        'mu_star'         : (3,) group conditional means E[R | D = d]
    """
    if rho_reg is None:
        rho_reg = dist["marginal_D"].copy()
    if rho_star is None:
        rho_star = dist["marginal_D"].copy()

    rho_reg = np.asarray(rho_reg, dtype=np.float64)
    rho_star = np.asarray(rho_star, dtype=np.float64)

    marginal_R: np.ndarray = dist["marginal_R"]          # (8,)
    cond_prob: np.ndarray = dist["cond_prob_R_given_D"]  # (3, 8)
    mu_star: np.ndarray = dist["group_means"]            # (3,)

    n_r = len(R_LEVELS)
    n_d = len(D_LEVELS)
    n_classes = int(class_map.max()) + 1

    # ------------------------------------------------------------------
    # q[c] = P(C = c) = sum_j marginal_R[j] * 1[class_map[j] == c]
    # ------------------------------------------------------------------
    q = np.zeros(n_classes, dtype=np.float64)
    np.add.at(q, class_map, marginal_R)

    # ------------------------------------------------------------------
    # Q[d, c] = P(C = c | D = d) = sum_j cond_prob[d, j] * 1[class_map[j] == c]
    # ------------------------------------------------------------------
    Q = np.zeros((n_d, n_classes), dtype=np.float64)
    for j in range(n_r):
        Q[:, class_map[j]] += cond_prob[:, j]

    # ------------------------------------------------------------------
    # s[c] = P^reg(C = c) = sum_d rho_reg[d] * Q[d, c]
    # ------------------------------------------------------------------
    s = rho_reg @ Q  # (n_classes,)

    # ------------------------------------------------------------------
    # rbar[c] = E[R | C = c]
    # ------------------------------------------------------------------
    rbar_num = np.zeros(n_classes, dtype=np.float64)
    np.add.at(rbar_num, class_map, R_LEVELS * marginal_R)
    rbar = rbar_num / q  # (n_classes,)

    # ------------------------------------------------------------------
    # eta[c, d] = P^reg(D = d | C = c) = rho_reg[d] * Q[d, c] / s[c]
    # Shape:  (rho_reg * Q).T / s  -> (n_classes, n_d)
    # ------------------------------------------------------------------
    eta = (rho_reg[:, np.newaxis] * Q).T / s[:, np.newaxis]  # (n_classes, n_d)

    eta_bar = rho_star.copy()

    # ------------------------------------------------------------------
    # L(C) = sum_j marginal_R[j] * (R_LEVELS[j] - rbar[class_map[j]])^2
    # L_check = E[R^2] - sum_c q[c]*rbar[c]^2   (algebraic equivalent)
    # ------------------------------------------------------------------
    L = float(np.sum(marginal_R * (R_LEVELS - rbar[class_map]) ** 2))
    E_R2 = float(np.sum(marginal_R * R_LEVELS ** 2))
    L_check = E_R2 - float(np.sum(q * rbar ** 2))

    # ------------------------------------------------------------------
    # Gamma = 2 * sum_c s[c] * ||eta[c] - eta_bar||^2
    # ------------------------------------------------------------------
    diff_eta = eta - eta_bar                                   # (n_classes, n_d)
    sq_norms = np.sum(diff_eta ** 2, axis=1)                   # (n_classes,)
    Gamma = float(2.0 * np.dot(s, sq_norms))

    # ------------------------------------------------------------------
    # Gamma_pairwise = sum_{c, c'} s[c]*s[c'] * ||eta[c] - eta[c']||^2
    # Equivalent to Gamma when rho_reg == rho_star (barycenter identity)
    # ------------------------------------------------------------------
    # Use the identity:  sum_{c,c'} s[c]*s[c']*||eta[c]-eta[c']||^2
    #   = 2*(sum_c s[c]*||eta[c]||^2 - ||sum_c s[c]*eta[c]||^2)
    eta_sq = np.sum(eta ** 2, axis=1)                          # (n_classes,)
    bary = s @ eta                                             # (n_d,) = sum_c s[c]*eta[c]
    Gamma_pairwise = float(2.0 * (np.dot(s, eta_sq) - np.dot(bary, bary)))

    # ------------------------------------------------------------------
    # w_C[d] = (Q @ rbar)[d] - mu_star[d]
    # ------------------------------------------------------------------
    w_C = Q @ rbar - mu_star                                   # (n_d,)

    # ------------------------------------------------------------------
    # SD_w = sqrt(sum_d rho_star[d] * (w_C[d] - rho_star @ w_C)^2)
    # ------------------------------------------------------------------
    w_mean = float(rho_star @ w_C)
    SD_w = float(np.sqrt(np.dot(rho_star, (w_C - w_mean) ** 2)))

    return {
        "n_classes": n_classes,
        "q": q,
        "s": s,
        "rbar": rbar,
        "eta": eta,
        "eta_bar": eta_bar,
        "Q": Q,
        "L": L,
        "L_check": L_check,
        "Gamma": Gamma,
        "Gamma_pairwise": Gamma_pairwise,
        "w_C": w_C,
        "SD_w": SD_w,
        "mu_star": mu_star,
    }


def validate_classification_metrics(metrics: Dict[str, object]) -> None:
    """Assert algebraic identities in computed classification metrics.

    Parameters
    ----------
    metrics:
        Output of ``compute_classification_metrics()``.

    Raises
    ------
    AssertionError if any identity fails at tolerance 1e-12.
    """
    tol = 1e-12

    q: np.ndarray = metrics["q"]
    s: np.ndarray = metrics["s"]
    eta: np.ndarray = metrics["eta"]
    Q: np.ndarray = metrics["Q"]
    L: float = metrics["L"]
    L_check: float = metrics["L_check"]
    Gamma: float = metrics["Gamma"]
    Gamma_pairwise: float = metrics["Gamma_pairwise"]

    assert np.isclose(q.sum(), 1.0, atol=tol), (
        f"q must sum to 1; got {q.sum()}"
    )
    assert np.isclose(s.sum(), 1.0, atol=tol), (
        f"s must sum to 1; got {s.sum()}"
    )
    assert np.allclose(eta.sum(axis=1), 1.0, atol=tol), (
        f"eta rows must sum to 1; row sums: {eta.sum(axis=1)}"
    )
    assert np.allclose(Q.sum(axis=1), 1.0, atol=tol), (
        f"Q rows must sum to 1; row sums: {Q.sum(axis=1)}"
    )
    assert np.isclose(L, L_check, atol=tol), (
        f"L ({L}) != L_check ({L_check}); diff = {abs(L - L_check)}"
    )
    assert np.isclose(Gamma, Gamma_pairwise, atol=tol), (
        f"Gamma ({Gamma}) != Gamma_pairwise ({Gamma_pairwise}); "
        f"diff = {abs(Gamma - Gamma_pairwise)}"
    )
