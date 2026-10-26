"""Experiment 2: Merge formulas and dynamic-programming verification.

Part A: Verify the exact analytical merge-effect formulas ΔL and ΔΓ for
        each adjacent merge among C⁸→C⁴→C³→C².

Part B: Implement the dynamic-programming recursion for optimal contiguous
        partitions under the weighted objective L(C) + λ·Γ(C), and verify
        against brute-force enumeration for K=8 levels.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import COND_PROB, RHO, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics
from src.merge import verify_merge_formulas, merge_classification
from src.dp import precompute_block_costs, dp_optimal_partition, verify_dp_vs_brute_force

LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def _to_serialisable(obj):
    """Convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Part A: Merge formula verification
# ---------------------------------------------------------------------------

def part_a_merge_verification(dist: dict) -> dict:
    """Verify merge-effect formulas for each step in the coarsening sequence."""
    print("\n" + "=" * 70)
    print("Part A: Merge Formula Verification")
    print("=" * 70)

    merge_steps = [
        # (start_classification, class_a, class_b, description)
        ("C8", 0, 1, "C⁸: merge classes 0,1 → adjacent lowest pair"),
        ("C8", 2, 3, "C⁸: merge classes 2,3"),
        ("C8", 4, 5, "C⁸: merge classes 4,5"),
        ("C8", 6, 7, "C⁸: merge classes 6,7"),
        ("C4", 0, 1, "C⁴: merge classes 0,1 → C³-like first step"),
        ("C4", 1, 2, "C⁴: merge classes 1,2"),
        ("C4", 2, 3, "C⁴: merge classes 2,3"),
        ("C3", 0, 1, "C³: merge classes 0,1"),
        ("C3", 1, 2, "C³: merge classes 1,2"),
    ]

    all_results = []
    all_pass = True

    header = (
        f"{'Merge':<40} {'ΔL_direct':>12} {'ΔL_formula':>12} "
        f"{'ΔΓ_direct':>12} {'ΔΓ_formula':>12} {'OK':>5}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for clf_name, ca, cb, desc in merge_steps:
        cm = get_classification_map(clf_name)
        res = verify_merge_formulas(dist, cm, ca, cb, tol=1e-10)
        ok = res["L_ok"] and res["Gamma_ok"]
        if not ok:
            all_pass = False

        print(
            f"{desc:<40} {res['delta_L_direct']:>12.6f} {res['delta_L_formula']:>12.6f} "
            f"{res['delta_Gamma_direct']:>12.6f} {res['delta_Gamma_formula']:>12.6f} "
            f"{'✓' if ok else '✗':>5}"
        )
        all_results.append({"description": desc, **_to_serialisable(res)})

    print(sep)
    status = "ALL PASS ✓" if all_pass else "SOME FAIL ✗"
    print(f"Merge formula verification: {status}")
    return {"merge_steps": all_results, "all_pass": all_pass}


# ---------------------------------------------------------------------------
# Part B: Dynamic-programming verification
# ---------------------------------------------------------------------------

def part_b_dp_verification(dist: dict) -> dict:
    """Run DP and verify against brute force for each λ in the grid."""
    print("\n" + "=" * 70)
    print("Part B: Dynamic-Programming Verification vs. Brute Force")
    print("=" * 70)

    marginal_R = dist["marginal_R"]
    cond_prob = dist["cond_prob_R_given_D"]

    all_lambda_results = {}
    all_dp_match = True

    for lam in LAMBDA_GRID:
        block_costs, block_info = precompute_block_costs(
            marginal_R, cond_prob, RHO, lam
        )
        verify_res = verify_dp_vs_brute_force(block_costs, K=8, tol=1e-10)
        all_lambda_results[lam] = verify_res

        lam_match = all(
            v.get("match", True)
            for k, v in verify_res.items()
            if k != "all_match" and isinstance(v, dict)
        )
        if not lam_match:
            all_dp_match = False

        dp_res = dp_optimal_partition(block_costs, K=8, max_classes=8)
        V = dp_res["V"]
        parts = dp_res["partitions"]

        print(f"\n  λ = {lam:.2f}")
        print(f"  {'n_classes':>9} {'Obj(DP)':>12} {'Obj(BF)':>12} {'Match':>7}  Optimal partition")
        print(f"  {'-'*9} {'-'*12} {'-'*12} {'-'*7}  {'-'*30}")

        for n in range(1, 9):
            entry = verify_res.get(n, {})
            if not entry.get("feasible", False):
                continue
            dp_obj = entry["dp_objective"]
            bf_obj = entry["bf_objective"]
            match = entry["match"]
            partition = parts.get(n, [])
            part_str = " | ".join(f"[{i+1},{j+1}]" for i, j in partition)
            print(
                f"  {n:>9}  {dp_obj:>11.6f} {bf_obj:>12.6f} {'✓' if match else '✗':>7}  {part_str}"
            )

    global_status = "ALL PASS ✓" if all_dp_match else "SOME FAIL ✗"
    print(f"\nDP vs. brute-force overall: {global_status}")
    return {"lambda_results": _to_serialisable(all_lambda_results), "all_pass": all_dp_match}


# ---------------------------------------------------------------------------
# Pareto frontier plot
# ---------------------------------------------------------------------------

def plot_frontier(dist: dict, out_dir: str) -> None:
    """Plot the L–Γ Pareto frontier from DP results across λ values."""
    from src.data import R_LEVELS
    marginal_R = dist["marginal_R"]
    cond_prob = dist["cond_prob_R_given_D"]

    L_frontier = []
    G_frontier = []

    for lam in np.linspace(0, 20, 200):
        block_costs, block_info = precompute_block_costs(
            marginal_R, cond_prob, RHO, lam
        )
        dp_res = dp_optimal_partition(block_costs, K=8, max_classes=8)
        parts = dp_res["partitions"]

        for n_cls, partition in parts.items():
            # Build class map from partition
            cm = np.zeros(8, dtype=np.int64)
            for c_idx, (i, j) in enumerate(partition):
                for k in range(i, j + 1):
                    cm[k] = c_idx

            from src.classification import compute_classification_metrics
            m = compute_classification_metrics(dist, cm)
            L_frontier.append(m["L"])
            G_frontier.append(m["Gamma"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(G_frontier, L_frontier, alpha=0.3, s=15, color="lightblue", label="DP solutions")

    # Overlay named classifications
    named = {}
    for clf in ["C8", "C4", "C3", "C2"]:
        cm = get_classification_map(clf)
        m = compute_classification_metrics(dist, cm)
        named[clf] = (m["Gamma"], m["L"])
        ax.scatter(m["Gamma"], m["L"], s=100, zorder=5)
        ax.annotate(
            f"$C^{{{clf[1:]}}}$",
            xy=(m["Gamma"], m["L"]),
            xytext=(m["Gamma"] + 0.001, m["L"] + 0.3),
            fontsize=11,
        )

    ax.set_xlabel(r"$\Gamma(C)$  [inferential disparity]", fontsize=12)
    ax.set_ylabel(r"$L(C)$  [risk inadequacy]", fontsize=12)
    ax.set_title("Pareto Frontier: Contiguous Partitions of 8 Price Levels", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "exp2_frontier_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nFrontier plot saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dist = build_joint_distribution()

    results_a = part_a_merge_verification(dist)
    results_b = part_b_dp_verification(dist)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    combined = {"exp": "exp_2", "merge_verification": results_a, "dp_verification": results_b}
    path = os.path.join(out_dir, "exp2_results.json")
    with open(path, "w") as fh:
        json.dump(_to_serialisable(combined), fh, indent=2)
    print(f"\nResults saved to {path}")

    plot_frontier(dist, out_dir)


if __name__ == "__main__":
    main()
