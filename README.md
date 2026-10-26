# Motor-Insurance Classification Trade-Off Analysis

A pure synthetic experiment examining the accuracy–fairness trade-off that arises when
motor insurers are required to use progressively coarser risk classifications.

---

## Mathematical Framework

The experiment operates on a **synthetic finite population** with two observable quantities:

- **R ∈ {88, 92, 96, 100, 104, 108, 112, 116}** — individual repair-cost level (8 price
  levels, in hundreds of currency units).
- **D ∈ {1, 2, 3}** — driver risk group (low, medium, high risk).

The joint distribution P(R, D) is defined through:

- A fixed prior **ρ = (1/3, 1/3, 1/3)** over groups.
- A conditional probability table **P(R | D)** (Table 1 in the source paper), encoded as
  the 3 × 8 matrix `COND_PROB` in `src/data.py`.

Key derived quantities:

| Quantity | Value |
|---|---|
| Marginal P(R) | (32, 38, 42, 45, 44, 40, 34, 25) / 300 |
| Group means E[R\|D=d] | (98.00, 100.76, 105.72) |
| Overall mean E[R] | 7612 / 75 ≈ 101.493 |

---

## Experiments

Seven experiments (scripts in `exp/`) cover distinct aspects of the trade-off:

| Script | Description |
|---|---|
| `exp_1_classification_tradeoff.py` | Accuracy–fairness metrics for all four classifications |
| `exp_2_*.py` | *(planned)* Q-matrix analysis and group profiles |
| `exp_3_*.py` | *(planned)* Transfer problem and exact compensation |
| `exp_4_*.py` | *(planned)* Infeasibility analysis for C² |
| `exp_5_*.py` | *(planned)* SD-minimising and AVaR-minimising transfers |
| `exp_6_*.py` | *(planned)* Coverage optimisation via kernel freedom |
| `exp_7_*.py` | *(planned)* End-to-end comparison of all strategies |

### Four Classifications

| Name | Classes | Grouping |
|---|---|---|
| **C⁸** | 8 | Singleton {88}, {92}, …, {116} |
| **C⁴** | 4 | Pairs {88,92}, {96,100}, {104,108}, {112,116} |
| **C³** | 3 | Triples {88,92,96}, {100,104}, {108,112,116} |
| **C²** | 2 | Halves {88,…,100}, {104,…,116} |

### Key Expected Values (Table 2)

| Classification | L(C) | Γ(C) | SD(w_C) |
|---|---|---|---|
| C⁸ | 0.000 | 0.116 | 0.000 |
| C⁴ | 3.971 | 0.107 | 0.175 |
| C³ | 8.431 | 0.094 | 0.392 |
| C² | 19.046 | 0.067 | 0.892 |

---

## Repository Structure

```
motor-insurance-tradeoff/
├── data/               Reserved for raw data files (none needed; fully synthetic)
│   └── __init__.py
├── src/                Python source package
│   ├── __init__.py
│   ├── data.py         R_LEVELS, COND_PROB, RHO, build_joint_distribution(),
│   │                   get_classification_map()
│   ├── classification.py  compute_classification_metrics(), validate_classification_metrics()
│   ├── transfer.py     solve_min_cost(), is_attainable(), compute_f_RA()
│   └── fairness.py     compute_AVaR(), sd_minimizer(), avar_minimizer(),
│                       coverage_improvement(), kernel_dim()
├── exp/                Experiment entry-point scripts
│   └── exp_1_classification_tradeoff.py
├── tests/              pytest test suite
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_classification.py
│   ├── test_transfer.py
│   └── test_fairness_coverage.py
├── results/            Generated output (JSON + PNG); not version-controlled
│   ├── exp1_results.json
│   └── exp1_tradeoff_plot.png
├── .gitignore
├── README.md
└── AGENTS.md
```

---

## How to Run

```bash
# From the project root

# Run experiment 1
python exp/exp_1_classification_tradeoff.py

# Run the full test suite
pytest tests/

# Run a specific test module
pytest tests/test_data.py -v

# Run a single test
pytest tests/test_classification.py::test_L_C8_is_zero -v
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations, linear algebra, pseudoinverse |
| `pandas` | Tabular result formatting (optional) |
| `scipy` | Additional optimisation helpers (optional) |
| `matplotlib` | Trade-off plots |
| `seaborn` | Enhanced statistical plots (optional) |

Install all at once:

```bash
pip install numpy pandas scipy matplotlib seaborn
```

No external data files are required; the entire experiment is self-contained within the
source code.
