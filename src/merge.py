"""Merge-effect formulas and verification for classification merging.

Implements the exact analytical identities:
  ΔL = L(C^{a,b}) - L(C) = (q_a * q_b / (q_a + q_b)) * (r̄_a - r̄_b)^2
  ΔΓ = Γ(C) - Γ(C^{a,b}) = (2 * s_a * s_b / (s_a + s_b)) * ||η_a - η_b||^2
"""

import numpy as np
from typing import Dict, Tuple

from src.data import R_LEVELS, build_joint_distribution
from src.classification import compute_classification_metrics, validate_classification_metrics


def compute_merge_delta_L(q_a: float, q_b: float, rbar_a: float, rbar_b: float) -> float:
    """Compute the increase in risk inadequacy when merging two classes.

    Formula: ΔL = (q_a * q_b / (q_a + q_b)) * (r̄_a - r̄_b)^2

    Parameters
    ----------
    q_a, q_b : class probabilities before merge
    rbar_a, rbar_b : class conditional means before merge

    Returns
    -------
    float — predicted L(C^{a,b}) - L(C)
    """
    return (q_a * q_b / (q_a + q_b)) * (rbar_a - rbar_b) ** 2


def compute_merge_delta_Gamma(
    s_a: float, s_b: float, eta_a: np.ndarray, eta_b: np.ndarray
) -> float:
    """Compute the decrease in inferential disparity when merging two classes.

    Formula: ΔΓ = (2 * s_a * s_b / (s_a + s_b)) * ||η_a - η_b||^2

    Parameters
    ----------
    s_a, s_b : regulatory class probabilities before merge
    eta_a, eta_b : (k,) group profiles for the two classes

    Returns
    -------
    float — predicted Γ(C) - Γ(C^{a,b})
    """
    return (2 * s_a * s_b / (s_a + s_b)) * np.sum((eta_a - eta_b) ** 2)


def merge_classification(class_map: np.ndarray, class_a: int, class_b: int) -> np.ndarray:
    """Build a new class_map by merging class_a and class_b.

    The merged class takes the smaller index; remaining classes are re-indexed
    contiguously from 0.

    Parameters
    ----------
    class_map : (8,) int array of class assignments
    class_a, class_b : indices to merge (0-based)

    Returns
    -------
    np.ndarray of shape (8,) with merged class assignments, 0-based contiguous.
    """
    target = min(class_a, class_b)
    other = max(class_a, class_b)
    new_map = class_map.copy()
    # Redirect 'other' to 'target'
    new_map[new_map == other] = target
    # Re-index to fill gap
    old_classes = np.unique(new_map)
    remap = {old: new for new, old in enumerate(old_classes)}
    for j in range(len(new_map)):
        new_map[j] = remap[new_map[j]]
    return new_map


def verify_merge_formulas(
    dist: Dict,
    class_map: np.ndarray,
    class_a: int,
    class_b: int,
    tol: float = 1e-12,
) -> Dict:
    """Verify ΔL and ΔΓ formulas numerically for a single class merge.

    Parameters
    ----------
    dist : output of build_joint_distribution()
    class_map : current classification map
    class_a, class_b : classes to merge
    tol : tolerance for pass/fail check

    Returns
    -------
    dict with:
        'delta_L_direct' : L(merged) - L(pre)
        'delta_L_formula' : formula-predicted ΔL
        'delta_Gamma_direct' : Γ(pre) - Γ(merged)
        'delta_Gamma_formula' : formula-predicted ΔΓ
        'L_ok', 'Gamma_ok' : bool pass/fail
    """
    metrics_pre = compute_classification_metrics(dist, class_map)
    validate_classification_metrics(metrics_pre)

    merged_map = merge_classification(class_map, class_a, class_b)
    metrics_post = compute_classification_metrics(dist, merged_map)
    validate_classification_metrics(metrics_post)

    q_pre = metrics_pre["q"]
    s_pre = metrics_pre["s"]
    rbar_pre = metrics_pre["rbar"]
    eta_pre = metrics_pre["eta"]

    q_a, q_b = q_pre[class_a], q_pre[class_b]
    s_a, s_b = s_pre[class_a], s_pre[class_b]
    rbar_a, rbar_b = rbar_pre[class_a], rbar_pre[class_b]
    eta_a, eta_b = eta_pre[class_a], eta_pre[class_b]

    delta_L_direct = metrics_post["L"] - metrics_pre["L"]
    delta_L_formula = compute_merge_delta_L(q_a, q_b, rbar_a, rbar_b)

    delta_Gamma_direct = metrics_pre["Gamma"] - metrics_post["Gamma"]
    delta_Gamma_formula = compute_merge_delta_Gamma(s_a, s_b, eta_a, eta_b)

    L_ok = bool(abs(delta_L_direct - delta_L_formula) < tol)
    Gamma_ok = bool(abs(delta_Gamma_direct - delta_Gamma_formula) < tol)

    return {
        "class_a": class_a,
        "class_b": class_b,
        "delta_L_direct": delta_L_direct,
        "delta_L_formula": delta_L_formula,
        "delta_Gamma_direct": delta_Gamma_direct,
        "delta_Gamma_formula": delta_Gamma_formula,
        "L_ok": L_ok,
        "Gamma_ok": Gamma_ok,
        "L_residual": abs(delta_L_direct - delta_L_formula),
        "Gamma_residual": abs(delta_Gamma_direct - delta_Gamma_formula),
    }
