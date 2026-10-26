"""Experiment 4: SD vs AVaR fairness objectives under C².

Compares two balanced-transfer fairness criteria for C²:
  1. Minimise SD of group premium effects
  2. Minimise AVaR_{2/3} of group premium effects

Reproduces Table 4 and shows that different objectives select different
transfers with different risk-adequacy costs.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import RHO, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics
from src.fairness import (
    _c2_AVaR,
    _c2_SD,
    _c2_f_RA,
    _c2_transfer_direction,
    _c2_y,
    avar_minimizer,
    compute_AVaR,
    sd_minimizer,
)

EXPECTED_SD = {
    "y": (0.141, -0.205, 0.065),
    "SD": 0.149,
    "AVaR_2_3": 0.141,
    "f_RA": 26.718,
}

EXPECTED_AVAR = {
    "y": (0.107, -0.215, 0.107),
    "SD": 0.152,
    "AVaR_2_3": 0.107,
    "f_RA": 27.279,
}


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


def run_experiment() -> dict:
    """Solve both fairness optimisation problems under C² and collect results."""
    q, rbar, w_C, direction = _c2_transfer_direction()
    cm = get_classification_map("C2")
    dist = build_joint_distribution()
    m_clf = compute_classification_metrics(dist, cm)
    L = m_clf["L"]

    # SD minimiser
    t_SD, SD_val, f_RA_SD = sd_minimizer("C2")
    y_SD = _c2_y(t_SD, w_C, direction)
    AVaR_SD = compute_AVaR(y_SD, RHO, 2 / 3)
    p_SD = rbar + t_SD * np.array([1.0, -q[0] / q[1]])
    lam_SD = p_SD - rbar

    # AVaR minimiser
    t_AVAR, AVaR_val, f_RA_AVAR = avar_minimizer("C2", alpha=2 / 3)
    y_AVAR = _c2_y(t_AVAR, w_C, direction)
    SD_AVAR = _c2_SD(t_AVAR, w_C, direction)
    p_AVAR = rbar + t_AVAR * np.array([1.0, -q[0] / q[1]])
    lam_AVAR = p_AVAR - rbar

    # Validate balance constraints
    q_balance_SD = float(q @ p_SD - q @ rbar)
    q_balance_AVAR = float(q @ p_AVAR - q @ rbar)

    # Validate f_RA via direct formula
    marginal_R = dist["marginal_R"]
    class_map = get_classification_map("C2")
    from src.data import R_LEVELS
    f_RA_direct_SD = float(np.sum(marginal_R * (p_SD[class_map] - R_LEVELS) ** 2))
    f_RA_direct_AVAR = float(np.sum(marginal_R * (p_AVAR[class_map] - R_LEVELS) ** 2))

    return {
        "SD_minimizer": {
            "t_star": t_SD, "y": y_SD, "p": p_SD, "lambda": lam_SD,
            "SD": SD_val, "AVaR_2_3": AVaR_SD, "f_RA": f_RA_SD,
            "f_RA_direct": f_RA_direct_SD, "balance_residual": q_balance_SD,
        },
        "AVaR_minimizer": {
            "t_star": t_AVAR, "y": y_AVAR, "p": p_AVAR, "lambda": lam_AVAR,
            "SD": SD_AVAR, "AVaR_2_3": AVaR_val, "f_RA": f_RA_AVAR,
            "f_RA_direct": f_RA_direct_AVAR, "balance_residual": q_balance_AVAR,
        },
        "L_C2": L,
        "w_C": w_C,
        "q": q,
        "rbar": rbar,
    }


def print_table_4(results: dict) -> None:
    """Print Table 4 comparison."""
    print("\n" + "=" * 70)
    print("Table 4: SD vs AVaR₂/₃ Fairness Objectives under C²")
    print("=" * 70)

    header = f"{'Objective':<22} {'M_Δ (y)':<30} {'SD':>8} {'AVaR₂/₃':>10} {'f_RA':>8}"
    print(header)
    print("-" * 80)

    def fmt_y(y):
        return f"({y[0]:.3f}, {y[1]:.3f}, {y[2]:.3f})"

    res_SD = results["SD_minimizer"]
    res_AV = results["AVaR_minimizer"]

    exp_y_SD = fmt_y(EXPECTED_SD["y"])
    exp_y_AV = fmt_y(EXPECTED_AVAR["y"])

    print(f"{'Min SD':<22} {fmt_y(res_SD['y']):<30} {res_SD['SD']:>8.3f} "
          f"{res_SD['AVaR_2_3']:>10.3f} {res_SD['f_RA']:>8.3f}")
    print(f"{'  Expected':<22} {exp_y_SD:<30} {EXPECTED_SD['SD']:>8.3f} "
          f"{EXPECTED_SD['AVaR_2_3']:>10.3f} {EXPECTED_SD['f_RA']:>8.3f}")
    print()
    print(f"{'Min AVaR₂/₃':<22} {fmt_y(res_AV['y']):<30} {res_AV['SD']:>8.3f} "
          f"{res_AV['AVaR_2_3']:>10.3f} {res_AV['f_RA']:>8.3f}")
    print(f"{'  Expected':<22} {exp_y_AV:<30} {EXPECTED_AVAR['SD']:>8.3f} "
          f"{EXPECTED_AVAR['AVaR_2_3']:>10.3f} {EXPECTED_AVAR['f_RA']:>8.3f}")
    print("-" * 80)


def verify_results(results: dict) -> None:
    """Check computed values against expected targets."""
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    tol = 0.015
    res_SD = results["SD_minimizer"]
    res_AV = results["AVaR_minimizer"]

    assert abs(res_SD["SD"] - EXPECTED_SD["SD"]) < tol, (
        f"SD-min SD: {res_SD['SD']:.4f} vs expected {EXPECTED_SD['SD']:.3f}"
    )
    assert abs(res_SD["f_RA"] - EXPECTED_SD["f_RA"]) < tol, (
        f"SD-min f_RA: {res_SD['f_RA']:.4f} vs expected {EXPECTED_SD['f_RA']:.3f}"
    )
    assert abs(res_AV["AVaR_2_3"] - EXPECTED_AVAR["AVaR_2_3"]) < tol, (
        f"AVaR-min AVaR: {res_AV['AVaR_2_3']:.4f} vs expected {EXPECTED_AVAR['AVaR_2_3']:.3f}"
    )
    assert abs(res_AV["f_RA"] - EXPECTED_AVAR["f_RA"]) < tol, (
        f"AVaR-min f_RA: {res_AV['f_RA']:.4f} vs expected {EXPECTED_AVAR['f_RA']:.3f}"
    )

    # The two minimisers must be different
    assert abs(res_SD["t_star"] - res_AV["t_star"]) > 1e-6, (
        "SD and AVaR minimisers should have different parameters"
    )

    # Balance constraints
    assert abs(res_SD["balance_residual"]) < 1e-10, (
        f"SD-min balance violation: {res_SD['balance_residual']:.2e}"
    )
    assert abs(res_AV["balance_residual"]) < 1e-10, (
        f"AVaR-min balance violation: {res_AV['balance_residual']:.2e}"
    )

    # f_RA formula matches direct
    assert abs(res_SD["f_RA"] - res_SD["f_RA_direct"]) < 1e-8, (
        f"SD-min f_RA formula vs direct: {abs(res_SD['f_RA'] - res_SD['f_RA_direct']):.2e}"
    )

    print("  SD-min  SD   ✓, f_RA ✓, balance ✓")
    print("  AVaR-min AVaR ✓, f_RA ✓, balance ✓")
    print("  t_SD ≠ t_AVaR ✓")
    print("\n  All checks PASSED ✓")


def plot_objectives(results: dict, out_dir: str) -> None:
    """Plot SD and AVaR_{2/3} as functions of the transfer parameter t."""
    q, rbar, w_C, direction = _c2_transfer_direction()
    cm = get_classification_map("C2")
    dist = build_joint_distribution()
    m_clf = compute_classification_metrics(dist, cm)
    L = m_clf["L"]

    t_range = np.linspace(-5, 2, 3001)
    SD_vals = [_c2_SD(t, w_C, direction) for t in t_range]
    AVaR_vals = [_c2_AVaR(t, w_C, direction, 2 / 3) for t in t_range]
    fRA_vals = [_c2_f_RA(t, q, L) for t in t_range]

    t_SD = results["SD_minimizer"]["t_star"]
    t_AV = results["AVaR_minimizer"]["t_star"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(t_range, SD_vals, "steelblue", lw=1.5, label="SD(y(t))")
    ax.plot(t_range, AVaR_vals, "darkorange", lw=1.5, label=r"AVaR$_{2/3}$(y(t))")
    ax.axvline(t_SD, color="steelblue", ls="--", lw=1, label=f"t*(SD) = {t_SD:.3f}")
    ax.axvline(t_AV, color="darkorange", ls="--", lw=1, label=f"t*(AVaR) = {t_AV:.3f}")
    ax.set_xlabel("Transfer parameter t", fontsize=12)
    ax.set_ylabel("Fairness metric", fontsize=12)
    ax.set_title("Fairness Metrics vs Transfer Parameter (C²)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[1]
    ax2.plot(t_range, fRA_vals, "forestgreen", lw=1.5, label=r"$f_{RA}(t)$")
    ax2.axvline(t_SD, color="steelblue", ls="--", lw=1)
    ax2.axvline(t_AV, color="darkorange", ls="--", lw=1)
    ax2.set_xlabel("Transfer parameter t", fontsize=12)
    ax2.set_ylabel(r"$f_{RA}$ (total risk inadequacy)", fontsize=12)
    ax2.set_title(r"Risk Inadequacy $f_{RA}(t)$ under C²", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "exp4_comparison_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


def main() -> None:
    results = run_experiment()
    print_table_4(results)
    verify_results(results)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    plot_objectives(results, out_dir)

    path = os.path.join(out_dir, "exp4_results.json")
    with open(path, "w") as fh:
        import json
        json.dump(_to_serialisable(results), fh, indent=2)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
