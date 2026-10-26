"""Experiment 5: Coverage improvement at fixed protected-group target using ker(Q).

For C⁴ with exact-compensation target y = 0:
  - Compute baseline premium p₀ and baseline AVaR_{0.95}(Z)
  - Find v ∈ ker(Q) with ‖v‖_q² ≤ δ = 1 minimising AVaR_{0.95}
  - Verify that the improved p* still satisfies the fairness target

For C³: show that ker(Q) = {0}, so no improvement is possible.

Expected values: baseline AVaR ≈ 116.322, improved AVaR ≈ 115.955, total f_RA ≈ 5.201.
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

from src.data import R_LEVELS, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics
from src.fairness import compute_AVaR, coverage_improvement, kernel_dim
from src.transfer import solve_min_cost


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


EXPECTED = {
    "C4_p0": [89.47382, 97.92100, 106.15698, 114.38168],
    "C4_baseline_avar": 116.322,
    "C4_improved_avar": 115.955,
    "C4_total_f_RA": 5.201,
}


def run_c4_coverage(dist: dict) -> dict:
    """C⁴: compute p₀, baseline coverage, and improved coverage via ker(Q)."""
    print("\n" + "=" * 70)
    print("C⁴ Coverage Improvement (ker(Q) optimisation, δ = 1)")
    print("=" * 70)

    y_zero = np.zeros(3)
    delta = 1.0
    alpha = 0.95

    # Minimum-risk premium for exact compensation
    c0, p_0 = solve_min_cost("C4", y_zero)
    assert p_0 is not None, "Exact compensation should be feasible for C4"

    print(f"\n  p₀ = {np.array2string(p_0, precision=5)}")
    print(f"  Expected ≈ {EXPECTED['C4_p0']}")
    print(f"  f_RA(p₀) = {c0 + compute_classification_metrics(dist, get_classification_map('C4'))['L']:.4f}  (expected ≈ 4.201)")

    # Coverage improvement
    baseline_avar, improved_avar, v_star = coverage_improvement("C4", alpha=alpha, delta=delta)

    m = compute_classification_metrics(dist, get_classification_map("C4"))
    q = m["q"]
    Q = m["Q"]
    L = m["L"]
    f_RA_improved = L + c0 + float(v_star @ np.diag(q) @ v_star)

    print(f"\n  dim(ker Q_C4) = {kernel_dim('C4')}")
    print(f"  Baseline AVaR_{{0.95}} = {baseline_avar:.4f}  (expected ≈ {EXPECTED['C4_baseline_avar']:.3f})")
    print(f"  Improved AVaR_{{0.95}} = {improved_avar:.4f}  (expected ≈ {EXPECTED['C4_improved_avar']:.3f})")
    print(f"  Improvement          = {baseline_avar - improved_avar:.4f}")
    print(f"  ‖v*‖_q²              = {float(v_star @ np.diag(q) @ v_star):.4f}  (budget δ = {delta})")
    print(f"  f_RA(p* = p₀ + v*)   = {f_RA_improved:.4f}  (expected ≈ {EXPECTED['C4_total_f_RA']:.3f})")

    # Verify target preserved: Q @ v* ≈ 0
    Qv = Q @ v_star
    print(f"  Q @ v*               = {np.array2string(Qv, precision=6)}  (should be ≈ 0)")
    assert np.abs(Qv).max() < 1e-8, f"Target not preserved: Q@v* = {Qv}"
    print("  Target preservation ✓")

    return {
        "p0": p_0,
        "v_star": v_star,
        "p_star": p_0 + v_star,
        "baseline_avar": baseline_avar,
        "improved_avar": improved_avar,
        "improvement": baseline_avar - improved_avar,
        "f_RA_base": L + c0,
        "f_RA_improved": f_RA_improved,
        "Q_v_residual": float(np.abs(Qv).max()),
    }


def run_c3_kernel_check() -> dict:
    """C³: verify ker(Q) = {0} → no fairness-preserving coverage improvement."""
    print("\n" + "=" * 70)
    print("C³ Kernel Check (no coverage improvement possible)")
    print("=" * 70)

    dim_ker = kernel_dim("C3")
    print(f"\n  dim(ker Q_C3) = {dim_ker}  (expected 0)")
    assert dim_ker == 0, f"Expected ker(Q_C3) = {{0}}, got dim = {dim_ker}"
    print("  ker(Q_C3) = {0} ✓  — no null-space freedom for fairness-preserving improvement")

    # Run coverage_improvement to confirm it returns no change
    baseline, improved, v = coverage_improvement("C3", alpha=0.95, delta=1.0)
    assert np.allclose(baseline, improved, atol=1e-10), (
        f"Expected no improvement for C3: baseline={baseline:.4f}, improved={improved:.4f}"
    )
    print(f"  coverage_improvement confirms baseline = improved = {baseline:.4f} ✓")

    return {"kernel_dim": dim_ker, "baseline_avar": baseline, "improved_avar": improved}


def plot_coverage_curve(dist: dict, out_dir: str) -> None:
    """Plot AVaR vs extra budget δ for C⁴."""
    print("\nComputing coverage curve for C⁴ (several δ values)...")
    y_zero = np.zeros(3)
    c0, p_0 = solve_min_cost("C4", y_zero)
    alpha = 0.95

    deltas = np.linspace(0, 3, 31)
    baseline_avar, _, _ = coverage_improvement("C4", alpha=alpha, delta=0.0)
    avars = []

    for d in deltas:
        _, improved, _ = coverage_improvement("C4", alpha=alpha, delta=float(d))
        avars.append(improved)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(deltas, avars, "o-", color="steelblue", lw=1.5, markersize=5)
    ax.axhline(baseline_avar, color="gray", ls="--", lw=1.2, label=f"Baseline = {baseline_avar:.3f}")
    ax.axvline(1.0, color="darkorange", ls="--", lw=1.2, label=r"$\delta = 1$ (budget)")
    ax.set_xlabel(r"Risk-inadequacy budget $\delta$", fontsize=12)
    ax.set_ylabel(r"AVaR$_{0.95}$(Z)", fontsize=12)
    ax.set_title(r"Coverage Improvement vs. Budget ($C^4$, exact compensation)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = os.path.join(out_dir, "exp5_coverage_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Coverage curve plot saved to {path}")


def main() -> None:
    dist = build_joint_distribution()

    c4_results = run_c4_coverage(dist)
    c3_results = run_c3_kernel_check()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    plot_coverage_curve(dist, out_dir)

    combined = {
        "exp": "exp_5",
        "C4": _to_serialisable(c4_results),
        "C3": _to_serialisable(c3_results),
    }
    path = os.path.join(out_dir, "exp5_results.json")
    with open(path, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
