# Experiment Results: Finite Motor-Insurance Classification Trade-Off

## Repository: qca-finite-motor-insurance-classification-trade-off--r

All experiments use a fully synthetic finite motor-insurance distribution from the paper's Table 1.  
**Test suite: 71/71 tests PASS** (`pytest tests/ -v`, 0.97 s).

---

## Exp 1 — Classification Trade-Off: Table 2 Replication

**Objective:** Reproduce Table 2 values for classifications C⁸, C⁴, C³, C².

### Summary Table (Table 2)

| Classification | Max spread | L(C)   | Γ(C)  | w_C = (D1, D2, D3)           | SD(w_C) |
|----------------|-----------|--------|-------|------------------------------|---------|
| C⁸ (= Cᵖ)     | 4         | 0.000  | 0.116 | (0.000, 0.000, 0.000)        | 0.000   |
| C⁴             | 8         | 3.971  | 0.107 | (+0.196, +0.034, −0.230)     | 0.175   |
| C³             | 12        | 8.431  | 0.094 | (+0.473, +0.012, −0.485)     | 0.392   |
| C²             | 28        | 19.046 | 0.067 | (+1.065, +0.053, −1.118)     | 0.892   |

**Key findings:**
- C⁸ = Cᵖ attains **perfect risk adequacy** L(C⁸) = 0 and zero pooling effects.
- Threshold ε₀ = Γ(Cᵖ) ≈ **0.116** — the inferential disparity at the finest classification.
- As classifications coarsen (C⁸ → C² ), Γ(C) **decreases** (0.116 → 0.067) while L(C) **increases** (0 → 19.046) and group pooling effects worsen.
- All values match expected rounded targets to 3 decimal places. ✓

**Output files:** `results/exp1_results.json`, `results/exp1_tradeoff_plot.png`

---

## Exp 2 — Merge Formulas and Dynamic Programming

**Objective:** Verify analytical merge identities ΔL, ΔΓ and DP optimality vs. exhaustive enumeration.

### Merge Identity Verification

For every adjacent pair of contiguous classes tested:
- **ΔL formula:** `L(C^{a,b}) − L(C) = (qₐ qᵦ / (qₐ+qᵦ)) (r̄ₐ − r̄ᵦ)²` — verified to machine precision (<1e-14).
- **ΔΓ formula:** `Γ(C) − Γ(C^{a,b}) = (2 sₐ sᵦ / (sₐ+sᵦ)) ‖ηₐ−ηᵦ‖²` — verified to machine precision (<1e-14).

### Dynamic Programming vs. Brute Force

For λ ∈ {0, 0.1, 0.25, 0.5, 1, 2, 5, 10} and class counts ℓ = 1, …, 8:
- DP objective **matches brute-force exactly** for every (λ, ℓ) combination (tolerance 1e-12). ✓
- All ✓ markers in the comparison table — **ALL PASS**.

### Sample DP Frontier (λ=1, 4 classes)

Optimal partition: [1,2] | [3,4] | [5,6] | [7,8] → L+Γ = 5.044

**Output files:** `results/exp2_results.json`, `results/exp2_frontier_plot.png`

---

## Exp 3 — Balanced-Transfer Attainability: Table 3 Replication

**Objective:** Reproduce Table 3 — attainability, extra cost c_C(0), minimum f_RA.

### Summary Table (Table 3)

| Classification | Exact comp. feasible? | c_C(0)  | min f_RA | Min residual SD | Notes                       |
|----------------|----------------------|---------|----------|-----------------|-----------------------------|
| C⁸             | YES                  | 0.000   | 0.000    | —               | Perfect, zero cost          |
| C⁴             | YES                  | 0.230   | 4.201    | —               | Small corrective transfer   |
| C³             | YES                  | 1.465   | 9.896    | —               | Moderate corrective transfer|
| C²             | NO                   | N/A     | 26.718   | 0.149           | Infeasible; min-dispersion  |

**Key findings:**
- Attainability is governed by **image conditions on Q**, not by Γ(C) alone.
- C⁴: verified p₀ ≈ (89.474, 97.921, 106.157, 114.382), Q p₀ − μ* = 0. ✓
- C²: infeasibility confirmed — w_C ∉ im(Q) under balanced constraint.
- All values match expected targets. ✓

**Output files:** `results/exp3_results.json`

---

## Exp 4 — SD vs. AVaR₂/₃ Fairness Objectives Under C²: Table 4 Replication

**Objective:** Compare SD-minimising and AVaR₂/₃-minimising balanced transfers under C².

### Summary Table (Table 4)

| Objective     | M_Δ = (y₁, y₂, y₃)       | SD(y)  | AVaR₂/₃(y) | f_RA   |
|---------------|--------------------------|--------|------------|--------|
| Min SD        | (0.141, −0.205, 0.065)   | 0.149  | 0.141      | 26.718 |
| Min AVaR₂/₃   | (0.107, −0.215, 0.107)   | 0.152  | 0.107      | 27.279 |

**Key findings:**
- SD-minimiser and AVaR-minimiser produce **different transfers** and **different risk-adequacy costs**. ✓
- AVaR₂/₃-minimiser achieves lower tail adverse effect (0.107 < 0.141) but incurs higher f_RA (27.279 > 26.718).
- Confirms: fairness criteria are **not interchangeable**.
- All values match expected rounded targets to 3 decimal places. ✓

**Output files:** `results/exp4_results.json`, `results/exp4_comparison_plot.png`

---

## Exp 5 — Coverage Improvement via Kernel Freedom

**Objective:** Demonstrate coverage improvement under C⁴ using ker(Q); confirm C³ offers none.

### C⁴ Results

| Quantity                      | Value        | Expected     |
|-------------------------------|--------------|--------------|
| Baseline premium p₀           | (89.474, 97.921, 106.157, 114.382) | ✓ |
| Baseline f_RA                 | 4.2014       | ≈ 4.201      |
| Baseline AVaR₀.₉₅(Z)         | 116.3224     | ≈ 116.322    |
| dim(ker Q_{C⁴})               | 1            | 1            |
| Improved AVaR₀.₉₅ (δ=1)     | 115.9554     | ≈ 115.955    |
| Total f_RA after improvement  | 5.2014       | ≈ 5.201      |
| ‖v*‖²_q (budget used)        | 1.0000       | = δ = 1      |
| Q v* ≈ 0 (target preserved)  | max|·| ≈ 1e-16 | ✓          |

### C³ Results

| Quantity              | Value |
|-----------------------|-------|
| dim(ker Q_{C³})       | 0     |
| Coverage improvement  | 0 (impossible by construction) |

**Key findings:**
- C⁴ has a **1-dimensional null space** in Q, enabling coverage improvement while preserving the fairness target.
- C³ has **ker(Q) = {0}** — no nontrivial fairness-preserving premium adjustment exists.
- Full risk-inadequacy budget δ = 1 is consumed by the optimal v*. ✓

**Output files:** `results/exp5_results.json`, `results/exp5_coverage_plot.png`

---

## Exp 6 — Changed Reference-Law Sensitivity

**Objective:** Under C⁴, change reference-law group weights to ρ* = (0.5, 0.3, 0.2) and test feasibility.

### Results

| Quantity          | Main case (ρ*=1/3 each) | Changed case (ρ*=(0.5,0.3,0.2)) |
|-------------------|-------------------------|---------------------------------|
| q*                | (0.233, 0.290, 0.280, 0.197) | **(0.280, 0.298, 0.259, 0.163)** |
| b = E_{P*}[A^C − R] | ≈ 0.000              | **≈ 0.062**                     |
| Exact comp. (y=0) feasible? | YES           | **NO** (mean condition fails: 0 ≠ b) |
| Group neutrality (y=b·1) feasible? | N/A    | **YES**                         |

**Key findings:**
- q* ≈ (0.280, 0.298, 0.259, 0.163) matches expected. ✓
- b ≈ 0.062 matches expected. ✓
- Exact compensation is immediately **infeasible** because (ρ*)ᵀ·0 = 0 ≠ b ≈ 0.062.
- Group neutrality y = b·1 is **feasible** — confirmed by image and balance checks.
- Conclusion: changing the balancing population **alters feasibility** without changing the portfolio distribution or classification.

**Output files:** `results/exp6_results.json`

---

## Exp 7 — Theorem-Level Numerical Checks

**Objective:** Validate Example 4.3 and Proposition 4.4 over parameter grids.

### Example 4.3 — c_C(0) Formula Check (θ ∈ (0, 1])

Analytical formula: `c_C(0) = (6 − 5θ)² / θ²`

| θ     | Numeric       | Analytic      | Rel. error |
|-------|---------------|---------------|------------|
| 0.01  | ≈ 354,025     | ≈ 354,025     | < 1e-12    |
| 0.1   | 3025.000000   | 3025.000000   | 4.5e-16    |
| 1.0   | 1.000000      | 1.000000      | 0.0        |

- Max relative error over 70-point grid: **4.9 × 10⁻¹³** (machine precision). ✓
- Qualitative: c_C(0) non-negative ✓, increasing as θ → 0 ✓.
- **Overall: PASS ✓**

### Proposition 4.4 — Lower-Bound Inequality (τ ∈ (0, 1/4))

Claim: `c_C(0) ≥ ‖w_C‖² / λ_max(H_C)`

- Verified over 200-point grid in (0.001, 0.249).
- Max violation: **−5.6 × 10⁻⁵** (negative = inequality holds). ✓
- **Overall: PASS ✓**

**Output files:** `results/exp7_results.json`, `results/exp7_theorem_checks_plot.png`

---

## Overall Test Results

```
71 passed in 0.97s
```

| Test file                  | Tests | Result  |
|----------------------------|-------|---------|
| test_data.py               | 18    | ✓ PASS  |
| test_classification.py     | 18    | ✓ PASS  |
| test_transfer.py           | 13    | ✓ PASS  |
| test_fairness_coverage.py  | 22    | ✓ PASS  |
| **Total**                  | **71**| **PASS**|

---

## Repository Structure

```
.
├── data/               # Synthetic data generation (no raw market data)
├── src/
│   ├── data.py         # Joint distribution construction & validation
│   ├── classification.py  # L(C), Γ(C), Q, w_C, SD computations
│   ├── merge.py        # ΔL, ΔΓ merge-formula verification
│   ├── dp.py           # Dynamic-programming optimal segmentation
│   ├── transfer.py     # Balanced-transfer attainability & pseudoinverse
│   └── fairness.py     # AVaR, SD optimisation, coverage improvement
├── exp/
│   ├── exp_1_classification_tradeoff.py
│   ├── exp_2_merge_dp.py
│   ├── exp_3_balanced_transfer.py
│   ├── exp_4_fairness_comparison.py
│   ├── exp_5_coverage_improvement.py
│   ├── exp_6_changed_reference_law.py
│   └── exp_7_theorem_checks.py
├── tests/
│   ├── test_data.py
│   ├── test_classification.py
│   ├── test_transfer.py
│   └── test_fairness_coverage.py
└── results/
    ├── RESULTS.md          ← this file
    ├── exp1_results.json / exp1_tradeoff_plot.png
    ├── exp2_results.json / exp2_frontier_plot.png
    ├── exp3_results.json
    ├── exp4_results.json / exp4_comparison_plot.png
    ├── exp5_results.json / exp5_coverage_plot.png
    ├── exp6_results.json
    └── exp7_results.json / exp7_theorem_checks_plot.png
```
