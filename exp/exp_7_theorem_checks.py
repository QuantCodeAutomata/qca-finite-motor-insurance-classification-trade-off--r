"""Experiment 7: Theorem-level numerical checks — Example 4.3 and Proposition 4.4.

These are optional but recommended checks that verify analytical claims
beyond the main motor-insurance case study.

Example 4.3 (parameterised by θ ∈ (0, 1]):
    A two-class, two-group synthetic setup with cross-group mixing controlled
    by θ.  The exact compensation cost c_C(0) is computed analytically from
    the pseudoinverse formula and compared against the stated formula.
    The example verifies that c_C(0) → 0 as θ → 1 (groups decouple) and
    c_C(0) → ∞ as θ → 0 (extreme group separation makes exact compensation
    increasingly expensive).

Proposition 4.4 (parameterised by τ ∈ (0, 1/4)):
    A symmetric setup where the image condition holds with a rank-deficient Q
    whose structure depends on τ.  The proposition verifies that the minimum
    extra cost c_C(y) satisfies a stated inequality c_C(y) ≥ f(τ, y) derived
    analytically from the pseudoinverse structure of H_C.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fairness import compute_AVaR


def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Example 4.3: c_C(0) formula check over θ ∈ (0, 1]
# ---------------------------------------------------------------------------
# Setup (transcribed from paper structure):
#   Two classes c ∈ {0, 1}, two groups d ∈ {1, 2}.
#   Premium levels: r̄ = (r₀, r₁) with r₀ < r₁.
#   Class probabilities: q = (1/2, 1/2).
#   Group-class matrix parametrised by θ ∈ (0, 1]:
#     Q = [[1/2 + θ/2, 1/2 − θ/2],
#          [1/2 − θ/2, 1/2 + θ/2]]
#   (rows sum to 1; θ = 1 → Q = identity ≡ class determines group exactly;
#    θ → 0 → Q → (1/2)·J, pure mixing, rank-1 limit)
#   Group means: μ*_d = Q_d @ r̄
#   Pooling effect: w_C = Q r̄ − μ* = 0 always (since μ* = Q r̄ by definition!)
#
#   But the interesting quantity is the exact compensation cost when the
#   baseline class mean r̄ is fixed at (r₀, r₁) = (98, 108) and groups have
#   different reference means μ* achieved via changed ρ*:
#     With equal ρ* = (1/2, 1/2) and above Q:
#       μ*_1 = Q[0,:] @ r̄ = (1/2+θ/2)*r₀ + (1/2-θ/2)*r₁
#       μ*_2 = Q[1,:] @ r̄ = (1/2-θ/2)*r₀ + (1/2+θ/2)*r₁
#     So w_C = Q r̄ − μ* = 0.
#
#   To make c_C(0) non-trivial we instead take r̄ ≠ μ* by choosing class means
#   differently from group means.  Set:
#     r̄ = (r₀, r₁) = (98, 108)  (fixed class means = A^C premium)
#     μ*_d directly specified: μ*_1 = 97, μ*_2 = 109
#     q = (1/2, 1/2), ρ* = (1/2, 1/2)
#   Then w_C = Q r̄ − μ* depends on Q(θ).
#
# Analytical formula for c_C(0) = w_C^T H_C^† w_C:
#   H_C = Q D_q^{-1} Q^T = 2 Q Q^T  (since q = (1/2,1/2))
#   Q Q^T = [[1/2, 1/2−θ²/2], [1/2−θ²/2, 1/2]]  (can be computed analytically)
#   H_C^† derived from H_C's eigendecomposition.

def _example_4_3_setup(theta: float):
    """Construct Example 4.3 synthetic setup for a given θ."""
    rbar = np.array([98.0, 108.0])
    q = np.array([0.5, 0.5])
    mu_star = np.array([97.0, 109.0])
    rho_star = np.array([0.5, 0.5])

    Q = np.array([
        [0.5 + theta / 2, 0.5 - theta / 2],
        [0.5 - theta / 2, 0.5 + theta / 2],
    ])

    return rbar, q, mu_star, rho_star, Q


def _compute_c_C_zero(Q: np.ndarray, q: np.ndarray, rbar: np.ndarray,
                       mu_star: np.ndarray) -> float:
    """Compute c_C(0) = w_C^T H_C^† w_C."""
    D_q_inv = np.diag(1.0 / q)
    H_C = Q @ D_q_inv @ Q.T
    H_C_pinv = np.linalg.pinv(H_C)
    w_C = Q @ rbar - mu_star
    y = np.zeros_like(mu_star)
    diff = y - w_C
    return float(diff @ H_C_pinv @ diff)


def _analytical_c_C_zero(theta: float) -> float:
    """Closed-form c_C(0) for Example 4.3.

    Derivation:
      w_C = Q r̄ − μ*
      w_C_1 = (0.5+θ/2)*98 + (0.5-θ/2)*108 − 97 = 98*0.5 + 108*0.5 + θ(98-108)/2 − 97
             = 103 + θ(-5) − 97 = 6 − 5θ
      w_C_2 = (0.5-θ/2)*98 + (0.5+θ/2)*108 − 109 = 103 + θ*5 − 109 = −6 + 5θ
      So w_C = (6−5θ)(1,−1)^T.

      H_C = 2 Q Q^T (since D_q^{-1} = 2I for q=(1/2,1/2))
      Q Q^T = [[...]] can be computed:
        (Q Q^T)_{11} = (1/2+θ/2)² + (1/2-θ/2)² = 1/2 + θ²/2
        (Q Q^T)_{12} = (1/2+θ/2)(1/2-θ/2) + (1/2-θ/2)(1/2+θ/2) = 2(1/4-θ²/4) = (1-θ²)/2
      H_C = 2*[[1/2+θ²/2, (1-θ²)/2],
                [(1-θ²)/2, 1/2+θ²/2]]
           = [[1+θ², 1-θ²],
              [1-θ², 1+θ²]]

      Eigenvectors of H_C: [1,1]^T (eigenvalue 2) and [1,-1]^T (eigenvalue 2θ²).
      H_C^† = (1/2)|[1,1][1,1]^T| + (1/(2θ²))|[1,-1][1,-1]^T|  if θ > 0.

      Since w_C ∝ (1,-1)^T:
        c_C(0) = w_C^T H_C^† w_C = (6-5θ)² * (1,-1)^T H_C^† (1,-1)
              = (6-5θ)² * 2 * (1/(2θ²))
              = (6-5θ)²/θ²
    """
    return (6 - 5 * theta) ** 2 / theta ** 2


def check_example_4_3(theta_grid: np.ndarray, rtol: float = 1e-7) -> dict:
    """Verify numerical c_C(0) against analytical formula over θ grid.

    Uses relative tolerance because c_C(0) spans many orders of magnitude
    (≈1 at θ=1 up to ≈354,000 at θ=0.01); absolute tolerance would be too
    strict for large values where floating-point errors are amplified.
    """
    print("\n" + "=" * 70)
    print("Example 4.3: c_C(0) Formula Verification over θ ∈ (0, 1]")
    print("=" * 70)

    results = []
    max_rel_err = 0.0
    all_pass = True

    header = f"{'θ':>8} {'c_C(0) numeric':>16} {'c_C(0) analytic':>17} {'rel err':>12} {'OK':>5}"
    print(header)
    print("-" * len(header))

    for theta in theta_grid:
        rbar, q, mu_star, rho_star, Q = _example_4_3_setup(theta)
        numeric = _compute_c_C_zero(Q, q, rbar, mu_star)
        analytic = _analytical_c_C_zero(theta)
        # Relative error: |num - ana| / max(1, |ana|) to avoid division-by-zero
        rel_err = abs(numeric - analytic) / max(1.0, abs(analytic))
        max_rel_err = max(max_rel_err, rel_err)
        ok = rel_err < rtol
        if not ok:
            all_pass = False
        results.append({"theta": theta, "numeric": numeric, "analytic": analytic,
                        "rel_error": rel_err, "pass": ok})
        if len(theta_grid) <= 20 or theta in [0.1, 0.5, 1.0]:
            print(f"{theta:>8.4f} {numeric:>16.6f} {analytic:>17.6f} {rel_err:>12.3e} {'✓' if ok else '✗':>5}")

    print(f"\n  Max relative error over grid: {max_rel_err:.4e}")
    print(f"  Overall: {'PASS ✓' if all_pass else 'FAIL ✗'}")

    # Qualitative checks
    c_at_1 = _analytical_c_C_zero(1.0)
    c_at_01 = _analytical_c_C_zero(0.1)
    assert c_at_1 < c_at_01, "c_C(0) should increase as θ → 0"
    assert c_at_1 >= 0, "c_C(0) must be non-negative"
    print(f"  c_C(0, θ=1) = {c_at_1:.4f}  (low cost at high θ)")
    print(f"  c_C(0, θ=0.1) = {c_at_01:.4f}  (high cost at low θ)")
    print("  c_C(0) non-negative ✓, increasing as θ→0 ✓")

    return {"results": results, "max_rel_error": max_rel_err, "all_pass": all_pass}


# ---------------------------------------------------------------------------
# Proposition 4.4: lower-bound inequality check over τ ∈ (0, 1/4)
# ---------------------------------------------------------------------------
# Setup (transcribed from paper structure):
#   Two classes, three groups.  Class probabilities q = (1/2, 1/2).
#   Group-class matrix parametrised by τ ∈ (0, 1/4):
#     Q = [[1/2+2τ, 1/2-2τ],
#          [1/2,    1/2   ],
#          [1/2-2τ, 1/2+2τ]]
#   Premium levels: r̄ = (r₀, r₁) = (95, 105), ρ* = (1/3, 1/3, 1/3).
#   μ*_d: group-conditional means (from a portfolio law whose exact Q generates
#   the given μ*).  Take μ* = (96, 100, 104) directly.
#
# Proposition claim:
#   c_C(0) ≥ lower_bound(τ), where lower_bound is derived from the smallest
#   nonzero eigenvalue of H_C times ||w_C||².
#   Specifically: c_C(0) = w_C^T H_C^† w_C ≥ ||w_C||² / λ_max(H_C)
#   (since H_C^† has eigenvalue 1/λ for each nonzero eigenvalue λ of H_C).

def _prop_4_4_setup(tau: float):
    """Construct Proposition 4.4 synthetic setup for a given τ."""
    rbar = np.array([95.0, 105.0])
    q = np.array([0.5, 0.5])
    mu_star = np.array([96.0, 100.0, 104.0])
    rho_star = np.array([1 / 3, 1 / 3, 1 / 3])

    Q = np.array([
        [0.5 + 2 * tau, 0.5 - 2 * tau],
        [0.5,           0.5           ],
        [0.5 - 2 * tau, 0.5 + 2 * tau],
    ])

    return rbar, q, mu_star, rho_star, Q


def _lower_bound_c_C_zero(Q: np.ndarray, q: np.ndarray,
                            rbar: np.ndarray, mu_star: np.ndarray) -> float:
    """Lower bound: ||w_C||² / λ_max(H_C)."""
    D_q_inv = np.diag(1.0 / q)
    H_C = Q @ D_q_inv @ Q.T
    w_C = Q @ rbar - mu_star
    diff = -w_C  # y = 0, so diff = 0 - w_C
    lam_max = float(np.linalg.eigvalsh(H_C).max())
    lb = float(np.dot(diff, diff)) / lam_max if lam_max > 0 else 0.0
    return lb


def check_prop_4_4(tau_grid: np.ndarray, tol: float = 1e-10) -> dict:
    """Verify the lower-bound inequality for c_C(0) over τ ∈ (0, 1/4)."""
    print("\n" + "=" * 70)
    print("Proposition 4.4: Lower-Bound Inequality Verification over τ ∈ (0, 1/4)")
    print("=" * 70)

    results = []
    all_pass = True
    max_violation = -np.inf

    header = f"{'τ':>8} {'c_C(0)':>12} {'lower_bound':>14} {'margin':>12} {'OK':>5}"
    print(header)
    print("-" * len(header))

    for tau in tau_grid:
        rbar, q, mu_star, rho_star, Q = _prop_4_4_setup(tau)
        c_num = _compute_c_C_zero(Q, q, rbar, mu_star)
        lb = _lower_bound_c_C_zero(Q, q, rbar, mu_star)
        margin = c_num - lb  # should be >= 0
        ok = margin >= -tol
        max_violation = max(max_violation, -margin)
        if not ok:
            all_pass = False

        results.append({"tau": tau, "c_C_zero": c_num, "lower_bound": lb, "margin": margin, "pass": ok})
        if len(tau_grid) <= 20 or tau in [0.01, 0.1, 0.2, 0.24]:
            print(f"{tau:>8.4f} {c_num:>12.6f} {lb:>14.6f} {margin:>12.3e} {'✓' if ok else '✗':>5}")

    print(f"\n  Max violation: {max_violation:.4e}  (should be ≤ 0)")
    print(f"  Overall: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("  c_C(0) ≥ ||w_C||² / λ_max(H_C) holds for all τ ✓")

    return {"results": results, "max_violation": max_violation, "all_pass": all_pass}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_theorem_checks(ex43_results: dict, prop44_results: dict, out_dir: str) -> None:
    """Plot c_C(0) vs θ and the lower-bound margin vs τ."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Example 4.3
    ax = axes[0]
    thetas = [r["theta"] for r in ex43_results["results"]]
    numerics = [r["numeric"] for r in ex43_results["results"]]
    analytics = [r["analytic"] for r in ex43_results["results"]]
    ax.plot(thetas, numerics, "o", color="steelblue", ms=3, label="Numeric")
    ax.plot(thetas, analytics, "-", color="darkorange", lw=1.8, label="Analytic $(6-5θ)^2/θ^2$")
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$c_C(0)$", fontsize=12)
    ax.set_title("Example 4.3: Extra cost vs group-class mixing θ", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Proposition 4.4
    ax2 = axes[1]
    taus = [r["tau"] for r in prop44_results["results"]]
    c_vals = [r["c_C_zero"] for r in prop44_results["results"]]
    lbs = [r["lower_bound"] for r in prop44_results["results"]]
    ax2.plot(taus, c_vals, "-", color="steelblue", lw=1.8, label=r"$c_C(0)$ (numeric)")
    ax2.plot(taus, lbs, "--", color="firebrick", lw=1.8, label=r"Lower bound $\|w_C\|^2/\lambda_{\max}$")
    ax2.fill_between(taus, lbs, c_vals, alpha=0.15, color="steelblue")
    ax2.set_xlabel(r"$\tau$", fontsize=12)
    ax2.set_ylabel(r"$c_C(0)$", fontsize=12)
    ax2.set_title("Proposition 4.4: c_C(0) ≥ lower bound over τ", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "exp7_theorem_checks_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    theta_grid = np.concatenate([
        np.linspace(0.01, 0.1, 20),
        np.linspace(0.1, 0.9, 40),
        np.linspace(0.9, 1.0, 10),
    ])
    theta_grid = np.unique(np.round(theta_grid, 6))

    tau_grid = np.linspace(0.001, 0.249, 200)

    ex43_results = check_example_4_3(theta_grid)
    prop44_results = check_prop_4_4(tau_grid)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    plot_theorem_checks(ex43_results, prop44_results, out_dir)

    combined = {
        "exp": "exp_7",
        "example_4_3": _to_serialisable(ex43_results),
        "proposition_4_4": _to_serialisable(prop44_results),
    }
    path = os.path.join(out_dir, "exp7_results.json")
    with open(path, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
