"""Experiment 3: Balanced-transfer attainability and minimum risk-inadequacy cost.

For each classification C ∈ {C⁸, C⁴, C³, C²}, tests attainability of the
exact-compensation target y = 0 and, when feasible, computes the minimum-risk
class-premium vector p₀ and extra cost c_C(0). Reproduces Table 3.

For C², tests infeasibility and solves the minimum-dispersion problem.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import R_LEVELS, RHO, build_joint_distribution, get_classification_map
from src.classification import compute_classification_metrics
from src.transfer import (
    class_means,
    class_probs,
    compute_f_RA,
    is_attainable,
    solve_min_cost,
    solve_transfer,
)

CLASSIFICATIONS = ["C8", "C4", "C3", "C2"]

EXPECTED = {
    "C8": {"feasible": True,  "c_C": 0.000,  "f_RA": 0.000},
    "C4": {"feasible": True,  "c_C": 0.230,  "f_RA": 4.201},
    "C3": {"feasible": True,  "c_C": 1.465,  "f_RA": 9.896},
    "C2": {"feasible": False, "min_SD": 0.149, "min_f_RA": 26.718},
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


def run_exact_compensation(dist: dict) -> dict:
    """Test exact compensation (y = 0) for all four classifications."""
    y_zero = np.zeros(3)
    results = {}

    print("\n" + "=" * 70)
    print("Exact Compensation (y = 0) Attainability and Cost")
    print("=" * 70)

    for name in CLASSIFICATIONS:
        res = solve_transfer(name, y_zero)
        results[name] = res
        att = res["attainable"]
        print(f"\n  {name}:")
        print(f"    Attainable: {att}")
        if att:
            print(f"    c_C(0)     = {res['c_C']:.6f}  (expected ≈ {EXPECTED[name]['c_C']:.3f})")
            print(f"    f_RA(p₀)   = {res['f_RA']:.6f}  (expected ≈ {EXPECTED[name]['f_RA']:.3f})")
            print(f"    p₀         = {np.array2string(res['p_y'], precision=5)}")
            print(f"    λ₀         = {np.array2string(res['lambda_y'], precision=5)}")
            print(f"    ‖Q p₀ − μ* − y‖ = {res['Q_p_residual']:.2e}")
            print(f"    |q^T p₀ − q^T r̄| = {res['balance_residual']:.2e}")
        else:
            print(f"    Exact compensation infeasible (as expected for {name})")

    return results


def run_c2_minimum_dispersion() -> dict:
    """For C², solve the minimum-dispersion problem over balanced transfers."""
    print("\n" + "=" * 70)
    print("C² Minimum-Dispersion Problem (SD minimisation over balanced transfers)")
    print("=" * 70)

    from src.fairness import sd_minimizer, avar_minimizer, compute_AVaR

    t_SD, SD_val, f_RA_SD = sd_minimizer("C2")

    # Compute the group-effect vector at t_SD
    from src.fairness import _c2_transfer_direction, _c2_y, _c2_AVaR
    q, rbar, w_C, direction = _c2_transfer_direction()
    y_SD = _c2_y(t_SD, w_C, direction)
    AVaR_SD = compute_AVaR(y_SD, RHO, 2 / 3)

    print(f"\n  SD-minimising balanced transfer for C²:")
    print(f"    t*             = {t_SD:.6f}")
    print(f"    y(t*)          = {np.array2string(y_SD, precision=4)}")
    print(f"    SD(y)          = {SD_val:.6f}  (expected ≈ 0.149)")
    print(f"    AVaR_{{2/3}}(y)  = {AVaR_SD:.6f}")
    print(f"    f_RA           = {f_RA_SD:.6f}  (expected ≈ 26.718)")

    result = {
        "t_star": t_SD,
        "y_star": y_SD,
        "SD": SD_val,
        "AVaR_2_3": AVaR_SD,
        "f_RA": f_RA_SD,
    }
    return result


def print_table_3(exact_results: dict, c2_disp: dict) -> None:
    """Print Table 3 summary."""
    print("\n" + "=" * 70)
    print("Table 3: Transfer Attainability and Risk-Adequacy Cost")
    print("=" * 70)

    header = (
        f"{'Classification':<16} {'Feasible':>9} {'c_C(0)':>10} "
        f"{'f_RA':>10} {'min_SD':>10} {'notes'}"
    )
    print(header)
    print("-" * 70)

    for name in CLASSIFICATIONS:
        if name == "C2":
            sd_val = c2_disp["SD"]
            f_ra_val = c2_disp["f_RA"]
            print(
                f"{name:<16} {'NO':>9} {'N/A':>10} {f_ra_val:>10.3f} "
                f"{sd_val:>10.3f}  (min-disp optimum)"
            )
        else:
            res = exact_results[name]
            c_C = res["c_C"] if res["c_C"] is not None else float("nan")
            f_ra = res["f_RA"] if res["f_RA"] is not None else float("nan")
            print(
                f"{name:<16} {'YES':>9} {c_C:>10.3f} {f_ra:>10.3f} "
                f"{'—':>10}  exact compensation"
            )

    print("-" * 70)


def verify_against_expected(exact_results: dict, c2_disp: dict) -> None:
    """Check computed values against expected rounded targets."""
    print("\n" + "=" * 70)
    print("Verification Against Expected Values")
    print("=" * 70)
    tol = 0.01

    # C8, C4, C3
    for name in ["C8", "C4", "C3"]:
        exp = EXPECTED[name]
        res = exact_results[name]
        assert res["attainable"], f"{name}: should be feasible"
        assert abs(res["c_C"] - exp["c_C"]) < tol, (
            f"{name}: c_C = {res['c_C']:.6f}, expected ≈ {exp['c_C']:.3f}"
        )
        assert abs(res["f_RA"] - exp["f_RA"]) < tol, (
            f"{name}: f_RA = {res['f_RA']:.6f}, expected ≈ {exp['f_RA']:.3f}"
        )
        print(f"  {name}: c_C = {res['c_C']:.4f} ✓, f_RA = {res['f_RA']:.4f} ✓")

    # C2 infeasible
    assert not exact_results["C2"]["attainable"], "C2 should be infeasible"
    exp_c2 = EXPECTED["C2"]
    assert abs(c2_disp["SD"] - exp_c2["min_SD"]) < tol, (
        f"C2 min_SD = {c2_disp['SD']:.6f}, expected ≈ {exp_c2['min_SD']:.3f}"
    )
    assert abs(c2_disp["f_RA"] - exp_c2["min_f_RA"]) < tol, (
        f"C2 min_f_RA = {c2_disp['f_RA']:.6f}, expected ≈ {exp_c2['min_f_RA']:.3f}"
    )
    print(f"  C2: infeasible ✓, min_SD = {c2_disp['SD']:.4f} ✓, min_f_RA = {c2_disp['f_RA']:.4f} ✓")
    print("\n  All checks PASSED ✓")


def main() -> None:
    dist = build_joint_distribution()

    exact_results = run_exact_compensation(dist)
    c2_disp = run_c2_minimum_dispersion()

    print_table_3(exact_results, c2_disp)
    verify_against_expected(exact_results, c2_disp)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    combined = {
        "exp": "exp_3",
        "exact_compensation": _to_serialisable(exact_results),
        "c2_min_dispersion": _to_serialisable(c2_disp),
    }
    path = os.path.join(out_dir, "exp3_results.json")
    with open(path, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
