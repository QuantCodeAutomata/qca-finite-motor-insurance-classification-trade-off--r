"""Experiment 6: Changed reference-law experiment.

Holds portfolio law P and classification C⁴ fixed, but changes the
reference/balancing law P* to ρ* = (0.5, 0.3, 0.2).

Reproduces: q* ≈ (0.280, 0.298, 0.259, 0.163), b ≈ 0.062.
Shows exact compensation infeasible, group neutrality feasible.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import COND_PROB, R_LEVELS, RHO, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics
from src.transfer import _projection_residual

EXPECTED = {
    "q_star": [0.280, 0.298, 0.259, 0.163],
    "b": 0.062,
}

# Changed reference-law group probabilities
RHO_STAR_NEW = np.array([0.5, 0.3, 0.2])


def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(x) for x in obj]
    return obj


def build_changed_ref_law(rho_star: np.ndarray) -> dict:
    """Build changed reference law P*(R, D) = ρ*_d * P(R | D).

    Returns joint, marginal_R, marginal_D (all under P*) and shared Q.
    """
    cond_prob = COND_PROB   # (3, 8): P(R=r_j | D=d)

    # Joint under P*
    joint_star = rho_star[:, np.newaxis] * cond_prob   # (3, 8)
    marginal_R_star = joint_star.sum(axis=0)            # (8,)

    return {
        "joint_star": joint_star,
        "marginal_R_star": marginal_R_star,
        "cond_prob": cond_prob,
        "rho_star": rho_star,
    }


def compute_q_star(cm: np.ndarray, joint_star: np.ndarray) -> np.ndarray:
    """Compute class probabilities q_c* = P*(C = c)."""
    n_classes = int(cm.max()) + 1
    q_star = np.zeros(n_classes)
    for j, c in enumerate(cm):
        q_star[c] += joint_star[:, j].sum()
    return q_star


def compute_mu_star(cond_prob: np.ndarray, r_levels: np.ndarray) -> np.ndarray:
    """Per-group means μ*_d = E_{P*}[R | D = d] = E_P[R | D = d] (unchanged)."""
    return cond_prob @ r_levels  # (3,)


def compute_Q_matrix(cm: np.ndarray, cond_prob: np.ndarray) -> np.ndarray:
    """Q_{d,c} = P*(C = c | D = d) = sum_{j: c_j = c} P(R = r_j | D = d).

    Depends only on P(R | D) which is unchanged by the reference-law swap.
    """
    n_classes = int(cm.max()) + 1
    n_d = cond_prob.shape[0]
    Q = np.zeros((n_d, n_classes))
    for j, c in enumerate(cm):
        Q[:, c] += cond_prob[:, j]
    return Q


def compute_b(q_star: np.ndarray, rbar: np.ndarray,
               marginal_R_star: np.ndarray, r_levels: np.ndarray) -> float:
    """Scalar b = E_{P*}[A^C(C(θ)) − R] = (q*)^T r̄ − E_{P*}[R].

    r̄ comes from the portfolio law P (aggregation rule A^C),
    but the expectation is taken under P* (changed reference law).
    Hence the class weights used are q_c* = P*(C=c), not q_c = P(C=c).
    """
    return float(q_star @ rbar) - float(marginal_R_star @ r_levels)


def test_attainability_changed(Q: np.ndarray, rbar: np.ndarray, q: np.ndarray,
                                rho_star: np.ndarray, mu_star: np.ndarray,
                                y: np.ndarray, b: float,
                                tol: float = 1e-9) -> tuple:
    """Test attainability under changed reference law.

    Conditions:
      (i)  ρ*^T y = b
      (ii) (y − w_C) ∈ im(Q)
    where w_C = Q r̄ − μ*.
    """
    w_C = Q @ rbar - mu_star
    mean_ok = abs(float(rho_star @ y) - b) < tol
    diff = y - w_C
    residual = _projection_residual(Q, diff)
    image_ok = float(np.linalg.norm(residual)) < tol
    return mean_ok, image_ok, w_C


def solve_group_neutral(Q: np.ndarray, rbar: np.ndarray, q: np.ndarray,
                         rho_star: np.ndarray, mu_star: np.ndarray, b: float) -> dict:
    """Group neutrality target y = b * 1_D and solve for minimum-risk premium."""
    n_d = Q.shape[0]
    y_neutral = b * np.ones(n_d)

    mean_ok, image_ok, w_C = test_attainability_changed(
        Q, rbar, q, rho_star, mu_star, y_neutral, b
    )

    result = {"target_y": y_neutral, "mean_ok": mean_ok, "image_ok": image_ok, "feasible": mean_ok and image_ok}

    if result["feasible"]:
        D_q_inv = np.diag(1.0 / q)
        H_C = Q @ D_q_inv @ Q.T
        H_C_pinv = np.linalg.pinv(H_C)
        diff = y_neutral - w_C
        p_y = rbar + D_q_inv @ Q.T @ H_C_pinv @ diff
        # Verify residuals
        result["p_y"] = p_y
        result["Q_p_residual"] = float(np.linalg.norm(Q @ p_y - mu_star - y_neutral))
        result["balance_residual"] = float(abs(q @ p_y - q @ rbar))

    return result


def main() -> None:
    dist = build_joint_distribution()
    cm = get_classification_map("C4")

    # Portfolio law P metrics (unchanged)
    m_P = compute_classification_metrics(dist, cm, rho_reg=RHO, rho_star=RHO)
    rbar = m_P["rbar"]   # from P
    q_P = m_P["q"]      # class probs under P

    # Changed reference law
    rho_star_new = RHO_STAR_NEW
    ref = build_changed_ref_law(rho_star_new)

    # Q is unchanged (depends only on P(R|D))
    Q = compute_Q_matrix(cm, ref["cond_prob"])
    mu_star = compute_mu_star(ref["cond_prob"], R_LEVELS)
    q_star = compute_q_star(cm, ref["joint_star"])

    # Scalar b = E_{P*}[r̄_{C(R)} - R] = (q*)^T r̄ − E_{P*}[R]
    b = compute_b(q_star, rbar, ref["marginal_R_star"], R_LEVELS)

    print("\n" + "=" * 70)
    print("Exp 6: Changed Reference Law (ρ* = (0.5, 0.3, 0.2))")
    print("=" * 70)

    print(f"\n  q* = {np.array2string(q_star, precision=4)}")
    print(f"  Expected ≈ {EXPECTED['q_star']}")
    print(f"\n  b = {b:.6f}  (expected ≈ {EXPECTED['b']:.3f})")

    # Row sums of Q
    assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-12), "Q rows must sum to 1"
    # q* = Q^T ρ*
    q_from_formula = Q.T @ rho_star_new
    assert np.allclose(q_star, q_from_formula, atol=1e-12), (
        f"q* mismatch: computed {q_star} vs formula {q_from_formula}"
    )

    print("\n  Q row sums = 1 ✓")
    print("  q* = Q^T ρ* ✓")

    # Exact compensation y = 0
    print("\n--- Exact Compensation (y = 0) ---")
    mean_ok_0, image_ok_0, w_C = test_attainability_changed(Q, rbar, q_P, rho_star_new, mu_star, np.zeros(3), b)
    print(f"  Mean condition  (ρ*^T 0 = b = {b:.4f}?): {mean_ok_0} — {'PASS' if mean_ok_0 else 'FAIL (expected)'}")
    print(f"  Image condition: {image_ok_0}")
    print(f"  Exact compensation feasible: {mean_ok_0 and image_ok_0}  (expected: False)")
    assert not (mean_ok_0 and image_ok_0), "Exact compensation should be infeasible"
    print("  Infeasibility confirmed ✓")

    # Group neutrality y = b * 1
    print("\n--- Group Neutrality (y = b·1) ---")
    gn = solve_group_neutral(Q, rbar, q_P, rho_star_new, mu_star, b)
    print(f"  Target y = {np.array2string(gn['target_y'], precision=4)}")
    print(f"  Mean condition: {gn['mean_ok']}")
    print(f"  Image condition: {gn['image_ok']}")
    print(f"  Feasible: {gn['feasible']}  (expected: True)")
    if gn["feasible"]:
        print(f"  p_y = {np.array2string(gn['p_y'], precision=5)}")
        print(f"  ‖Q p_y − μ* − y‖ = {gn['Q_p_residual']:.2e}")
        print(f"  Balance residual  = {gn['balance_residual']:.2e}")
    assert gn["feasible"], "Group neutrality should be feasible"
    print("  Group neutrality feasibility ✓")

    # Sensitivity comparison
    print("\n--- Sensitivity Summary: Main vs. Changed Reference Law (C⁴) ---")
    m_main = compute_classification_metrics(dist, cm, rho_reg=RHO, rho_star=RHO)
    q_main = m_main["q"]
    # Under main case P* = P so q* = q_P and marginal_R* = marginal_R under P
    marginal_R_main = (COND_PROB * RHO[:, np.newaxis]).sum(axis=0)
    b_main = compute_b(q_P, rbar, marginal_R_main, R_LEVELS)
    print(f"  Main case:    q* = {np.array2string(q_main, precision=4)}, b = {b_main:.4f}")
    print(f"  Changed case: q* = {np.array2string(q_star, precision=4)}, b = {b:.4f}")
    print(f"  Exact comp feasible (main):    True")
    print(f"  Exact comp feasible (changed): False")

    # Verifications
    assert abs(q_star[0] - EXPECTED["q_star"][0]) < 0.01, f"q*[0] = {q_star[0]:.4f}"
    assert abs(b - EXPECTED["b"]) < 0.01, f"b = {b:.4f}"
    print("\n  Numerical checks PASSED ✓")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    combined = {
        "exp": "exp_6",
        "rho_star_new": rho_star_new.tolist(),
        "q_star": q_star.tolist(),
        "b": float(b),
        "Q": Q.tolist(),
        "rbar": rbar.tolist(),
        "mu_star": mu_star.tolist(),
        "exact_compensation": {"mean_ok": mean_ok_0, "image_ok": image_ok_0, "feasible": False},
        "group_neutrality": _to_serialisable(gn),
    }
    path = os.path.join(out_dir, "exp6_results.json")
    with open(path, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
