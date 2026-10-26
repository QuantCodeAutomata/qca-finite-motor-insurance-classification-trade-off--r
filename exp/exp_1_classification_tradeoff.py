"""Experiment 1: Motor-insurance classification accuracy vs. group-fairness trade-off.

Evaluates four classifications (C8, C4, C3, C2) and reports Table 2 metrics:
  - max within-class spread of R
  - L(C)  : within-class variance (accuracy loss)
  - Gamma(C): group-profile dispersion (fairness cost)
  - w_C   : per-group pooling bias vector
  - SD(w_C): standard deviation of pooling bias

Results are saved to results/exp1_results.json and a trade-off plot to
results/exp1_tradeoff_plot.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure the project root is on the path when the script is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import R_LEVELS, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics, validate_classification_metrics

CLASSIFICATIONS = ["C8", "C4", "C3", "C2"]

# Expected rounded values from the specification
EXPECTED = {
    "C8": dict(L=0.000, Gamma=0.116, w=(0.000,  0.000,  0.000), SD=0.000),
    "C4": dict(L=3.971, Gamma=0.107, w=(0.196,  0.034, -0.230), SD=0.175),
    "C3": dict(L=8.431, Gamma=0.094, w=(0.473,  0.012, -0.485), SD=0.392),
    "C2": dict(L=19.046, Gamma=0.067, w=(1.065,  0.053, -1.118), SD=0.892),
}


def max_within_class_spread(class_map: np.ndarray) -> float:
    """Return max over classes of (max R in class − min R in class).

    Parameters
    ----------
    class_map:
        (8,) int array produced by ``get_classification_map``.

    Returns
    -------
    float: largest intra-class R range across all classes.
    """
    n_classes = int(class_map.max()) + 1
    spreads = []
    for c in range(n_classes):
        r_in_class = R_LEVELS[class_map == c]
        spreads.append(float(r_in_class.max() - r_in_class.min()))
    return max(spreads)


def _to_serialisable(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    return obj


def run_experiment() -> dict:
    """Run the full classification trade-off experiment.

    Returns
    -------
    dict mapping each classification name to its computed metrics plus
    'max_spread' and 'name' entries.
    """
    dist = build_joint_distribution()

    results = {}
    for name in CLASSIFICATIONS:
        class_map = get_classification_map(name)
        metrics = compute_classification_metrics(dist, class_map)
        validate_classification_metrics(metrics)

        metrics["name"] = name
        metrics["max_spread"] = max_within_class_spread(class_map)
        results[name] = metrics

    return results


def print_table(results: dict) -> None:
    """Print Table 2 to stdout."""
    header = (
        f"{'Classification':<16} {'MaxSpread':>10} {'L(C)':>10} "
        f"{'Gamma(C)':>10} {'w_C':>28} {'SD(w_C)':>10}"
    )
    sep = "-" * len(header)

    print("\nTable 2: Classification Accuracy–Fairness Trade-off")
    print(sep)
    print(header)
    print(sep)

    for name in CLASSIFICATIONS:
        m = results[name]
        w = m["w_C"]
        w_str = f"({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})"
        print(
            f"{name:<16} {m['max_spread']:>10.1f} {m['L']:>10.3f} "
            f"{m['Gamma']:>10.3f} {w_str:>28} {m['SD_w']:>10.3f}"
        )

    print(sep)


def print_comparison(results: dict) -> None:
    """Print expected vs. computed comparison for each classification."""
    print("\nExpected vs. Computed Comparison")
    print("=" * 70)
    for name in CLASSIFICATIONS:
        m = results[name]
        exp = EXPECTED[name]
        print(f"\n  {name}:")
        print(f"    L(C)    : expected {exp['L']:.3f}  | computed {m['L']:.3f}")
        print(f"    Gamma(C): expected {exp['Gamma']:.3f}  | computed {m['Gamma']:.3f}")
        w = m["w_C"]
        ew = exp["w"]
        print(
            f"    w_C     : expected ({ew[0]:+.3f},{ew[1]:+.3f},{ew[2]:+.3f}) "
            f"| computed ({w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f})"
        )
        print(f"    SD(w_C) : expected {exp['SD']:.3f}  | computed {m['SD_w']:.3f}")


def verify_key_results(results: dict) -> None:
    """Check epsilon_0 and L(C^8) = 0."""
    gamma_c8 = results["C8"]["Gamma"]
    L_c8 = results["C8"]["L"]

    print("\nKey Verifications")
    print("-" * 40)
    print(f"  epsilon_0 = Gamma(C^8) = {gamma_c8:.6f}  (expected ≈ 0.116)")
    print(f"  L(C^8)                 = {L_c8:.6f}  (expected = 0.000)")

    assert abs(gamma_c8 - 0.116) < 0.001, (
        f"epsilon_0 = Gamma(C^8) = {gamma_c8:.6f}, expected ≈ 0.116"
    )
    assert abs(L_c8) < 1e-12, f"L(C^8) = {L_c8}, expected exactly 0"
    print("  Both checks passed.")


def save_results(results: dict, out_dir: str) -> None:
    """Serialise results to JSON."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "exp1_results.json")
    with open(path, "w") as fh:
        json.dump(_to_serialisable(results), fh, indent=2)
    print(f"\nResults saved to {path}")


def save_plot(results: dict, out_dir: str) -> None:
    """Save L vs Gamma trade-off scatter plot."""
    os.makedirs(out_dir, exist_ok=True)

    Ls = [results[n]["L"] for n in CLASSIFICATIONS]
    Gammas = [results[n]["Gamma"] for n in CLASSIFICATIONS]
    labels = CLASSIFICATIONS

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(Gammas, Ls, "o-", color="steelblue", linewidth=1.5, markersize=7,
            zorder=3)

    for name, x, y in zip(labels, Gammas, Ls):
        offset = (0.001, 0.4) if name != "C2" else (-0.004, 0.4)
        ax.annotate(
            f"$C^{{{name[1:]}}}$",
            xy=(x, y),
            xytext=(x + offset[0], y + offset[1]),
            fontsize=11,
        )

    ax.set_xlabel(r"$\Gamma(C)$  [group-profile dispersion]", fontsize=12)
    ax.set_ylabel(r"$L(C)$  [within-class variance]", fontsize=12)
    ax.set_title("Accuracy–Fairness Trade-off (Motor Insurance)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = os.path.join(out_dir, "exp1_tradeoff_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


def main() -> None:
    """Entry point for the experiment."""
    results = run_experiment()

    print_table(results)
    verify_key_results(results)
    print_comparison(results)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_results(results, out_dir)
    save_plot(results, out_dir)


if __name__ == "__main__":
    main()
