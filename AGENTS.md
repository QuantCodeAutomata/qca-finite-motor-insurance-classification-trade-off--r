# Repository: Motor-Insurance Classification Trade-off Experiment

## Project Overview

Pure synthetic experiment — no external data needed.  All probability mass is derived
from the 3×8 conditional-probability table `COND_PROB` in `src/data.py`.

## Project Layout

```
src/           Python package (has __init__.py)
  data.py          R_LEVELS, COND_PROB, RHO, build_joint_distribution(),
                   get_classification_map()
  classification.py  compute_classification_metrics(), validate_classification_metrics()
  transfer.py      solve_min_cost(), is_attainable(), compute_f_RA(), class_means(),
                   class_probs()
  fairness.py      compute_AVaR(), sd_minimizer(), avar_minimizer(),
                   coverage_improvement(), kernel_dim()

exp/           Experiment scripts (run directly; project root added to sys.path)
  exp_1_classification_tradeoff.py

results/       Generated output (JSON + PNG); not version-controlled
  exp1_results.json
  exp1_tradeoff_plot.png

data/          Reserved for raw data files (empty; fully synthetic)
tests/         pytest test suite
  test_data.py, test_classification.py, test_transfer.py, test_fairness_coverage.py
```

## Running

```bash
python exp/exp_1_classification_tradeoff.py   # produces results/
pytest tests/                                  # run all tests
pytest tests/test_data.py -v                   # single module
```

## Key Design Decisions

### src/data.py
- `R_LEVELS` (8,), `D_LEVELS` (3,), `COND_PROB` (3×8), `RHO` (3,) are float64 constants.
- `build_joint_distribution(rho)` runs 5 internal checks; raises `ValueError` on failure.
- `get_classification_map(name)` → (8,) int64 array; supports 'C8', 'C4', 'C3', 'C2'.

### src/classification.py
- `compute_classification_metrics(dist, class_map, rho_reg=None, rho_star=None)` returns:
  - `Q` (3×m): Q[d,c] = Σ_{j: C(j)=c} COND_PROB[d,j] = P(C=c | D=d)
  - `q` (m,): marginal class probabilities = Q^T @ rho
  - `rbar` (m,): class means E[R | C=c]
  - `mu_star` (3,): group means E[R | D=d]  [= group_means(); independent of rho]
  - `w_C` (3,): pooling bias = Q @ rbar − mu_star
  - `L`, `L_check`, `Gamma`, `Gamma_pairwise`, `SD_w`, `eta`, `s`
- `validate_classification_metrics(metrics)` asserts six identities at tol 1e-12.

### src/transfer.py
- **Transfer problem**: given classification C, find class-level premium p ∈ ℝ^m that
  satisfies Q @ p = mu_star + y (group-level compensation target) at minimum q-weighted
  cost c_C(y) = ‖p − r̄‖²_q  (no 1/2 factor).
- Feasibility: y is attainable iff mu_star + y ∈ col(Q).
  - C⁸, C⁴, C³: col(Q) = ℝ³ (Q has rank 3) → always attainable.
  - C²: col(Q) is a 2D subspace → mu_star ∉ col(Q_C2), so y=0 is infeasible.
- Solution (when feasible): p_y = r̄ + W⁻¹ Q^T (Q W⁻¹ Q^T)⁻¹ (mu_star + y − Q r̄)
  where W = diag(q).  The pseudoinverse H_C^† implements this via the Moore-Penrose
  pseudoinverse of H_C = Q.
- `compute_f_RA(clf_name, y)` = L(C) + c_C(y).
- Budget-neutrality: q @ p_y = q @ r̄ = E[R] follows automatically from Q @ p_y = mu_star.

### src/fairness.py
- `compute_AVaR(values, weights, alpha)`: discrete AVaR at level α.
  - VaR_α = smallest x with CDF ≥ α.
  - AVaR_α = (1/(1−α)) · E[max(X − VaR_α, 0)] + VaR_α.
  - Known check: values=(1,2,3), weights=(1/3,1/3,1/3), α=2/3 → AVaR = 3.0.
- `kernel_dim(clf_name)`: dim(ker(Q)) = n_classes − rank(Q).
  - C⁸: 5, C⁴: 1, C³: 0, C²: 0.
- `sd_minimizer` / `avar_minimizer`: optimize over the 1D kernel freedom of C⁴ (or the
  achievable set of C²) to minimise SD(w_C) or AVaR_α(w_C).
- `coverage_improvement(clf_name, alpha)`: finds v_star ∈ ker(Q_C4) minimising
  AVaR_0.95 of coverage.  Q @ v_star = 0 by construction (target preserved).

## Metric Formulas
- **L(C)** = Σ_j p(R_j)·(R_j − r̄_{C(j)})²   (within-class variance)
- **Γ(C)** = 2·Σ_c s(c)·‖η_c − η̄‖²          (group-profile dispersion)
- **w_C[d]** = (Q·r̄)[d] − μ*[d]               (pooling bias for group d)
- **SD(w_C)** = √(Σ_d ρ*_d·(w_C[d]−ρ*·w_C)²)
- **c_C(y)** = ‖p_y − r̄‖²_q                   (transfer cost, q-weighted)
- **f_RA** = L(C) + c_C(y)                      (total risk-adjustment cost)

## Verified Numerical Values

### Table 2 Metrics (6 s.f.)

| Classification | L(C) | Γ(C) | w_C | SD(w_C) |
|---|---|---|---|---|
| C⁸ | 0.000000 | 0.115805 | (0, 0, 0) | 0.000000 |
| C⁴ | 3.970919 | 0.107304 | (0.196, 0.034, −0.230) | 0.175373 |
| C³ | 8.431019 | 0.094281 | (0.473, 0.012, −0.485) | 0.391553 |
| C² | 19.046236 | 0.067228 | (1.065, 0.053, −1.118) | 0.891995 |

### Transfer Costs c_C(0) and f_RA

| Classification | c_C(0) | f_RA = L + c_C(0) | Attainable? |
|---|---|---|---|
| C⁸ | 0.000000 | 0.000000 | Yes |
| C⁴ | 0.230470 | 4.201389 | Yes |
| C³ | 1.464547 | 9.895566 | Yes |
| C² | — | — | No (infeasible) |

### Kernel Dimensions

| Classification | Q shape | rank(Q) | dim(ker(Q)) |
|---|---|---|---|
| C⁸ | (3, 8) | 3 | 5 |
| C⁴ | (3, 4) | 3 | 1 |
| C³ | (3, 3) | 3 | 0 |
| C² | (3, 2) | 2 | 0 |

### Fairness Minimisers for C²

| Method | Metric | f_RA |
|---|---|---|
| SD-minimiser | SD ≈ 0.149 | ≈ 26.718 |
| AVaR_{2/3}-minimiser | AVaR ≈ 0.107 | ≈ 27.279 |

### Coverage Improvement for C⁴ (AVaR_{0.95})

| Scenario | AVaR_{0.95} |
|---|---|
| Baseline p₀ | ≈ 116.322 |
| Optimised p₀ + v_star | ≈ 115.955 |

## Dependencies
- Python ≥ 3.11, numpy, matplotlib
- Optional: pandas, scipy, seaborn
- No external data; everything is self-contained.
