# TSRDE: Two-Stage Robust Differential Evolution for Multimodal Optimization

Reference implementation for the paper:

> **Two-Stage Robust Differential Evolution for Multimodal Optimization** (under review at *IEEE Transactions on Cybernetics*)
>
> The extended version of the paper (12-page main text + appendix with full derivations, per-problem tables, and additional figures) is [`TSRDE_extended_version.pdf`](TSRDE_extended_version.pdf).

## Overview

Standard multimodal optimization algorithms locate all peaks without distinguishing robust (wide-basin) optima from fragile ones. This paper addresses that gap with three coordinated contributions:

1. **RobBench** — a 20-problem benchmark suite with five groups stress-testing different aspects of robust peak identification (dimensionality scaling, robustness ratio, peak count, anisotropy, spatial traps). All peaks are equal in height, so ground-truth robustness is analytically exact.

2. **RobF1** — an F1-style metric that simultaneously penalizes redundant recommendations and missed robust peaks, yielding a single scalar score for robust solution discovery.

3. **TSRDE** — Two-Stage Robust Differential Evolution that decouples global multimodal exploration from robust refinement:
   - **Phase 1**: k-NN rival crowding DE with random restarts — efficient multi-peak coverage with zero robustness evaluation cost.
   - **Phase 2**: Coarse robustness screening, competition-elimination loop, and deep refinement — selects the `n_sol` most robust and spatially diverse solutions.

A key theoretical contribution is the dimension-adaptive sampling count **M\*(D)**, derived from a quadratic peak model, which guarantees statistically reliable robustness estimation at all dimensions.

TSRDE exposes **a single user parameter** (`n_sol` — the desired number of robust solutions) and is empirically robust to ±2 variations around the true count.

---

## Requirements

```bash
pip install -r requirements.txt
```

Tested on Python 3.10+.

---

## Quick Start

```python
from src.tsrde import TSRDE
from src.problems.CMMOP.ProblemMM import ProblemMM

# Load a CEC multimodal problem (problem ID 1, dimension 2)
problem = ProblemMM(pid=1, pin=1, dim=2)

# Run TSRDE — Track B: find the 3 most robust peaks
algo = TSRDE(seed=42, n_sol=3, search_robust=True)
solutions = algo._run(problem)

print(f"Found {len(solutions)} robust solutions:")
for i, x in enumerate(solutions):
    print(f"  x{i+1} = {x}")
```

For Track A (find all distinct peaks, no robustness filtering):

```python
algo = TSRDE(seed=42, search_robust=False)
solutions = algo._run(problem)
```

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── quick_example.py         # Runnable quick-start script
├── TSRDE_extended_version.pdf  # Extended version of the paper (main text + appendix)
├── results/                    # Per-problem result tables and offline audits (Markdown)
│   ├── robbench_per_problem.md
│   ├── cec2026_per_problem.md
│   ├── ablation.md
│   ├── nsol_sensitivity.md
│   └── audits.md
└── src/
    ├── tsrde.py             # Main TSRDE algorithm
    ├── evri_utils.py        # Robustness measure E[f(x+δ)] + M*(D) theory
    ├── base_algorithm.py    # Abstract base class for all algorithms
    ├── archive.py           # Capacity-bounded archive with radius deduplication
    ├── baselines/           # Comparison algorithms
    │   ├── adea_mmo.py      # ADEA-MMO
    │   ├── drea.py          # DREA (TEVC 2024)
    │   ├── foa.py           # FOA
    │   ├── gecco2020.py     # GECCO2020 baseline
    │   ├── lips.py          # LIPS
    │   ├── ncde.py          # NCDE
    │   └── robust_baselines.py  # Post-hoc robust wrappers (PBRO, RPSO-IC, SRI)
    └── problems/
        ├── CMMOP/           # CEC2013/2026 multimodal benchmark
        └── RobBench/        # Our RobBench suite (20 problems, 5 groups)
```

---

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_sol` | 3 | Target number of robust peaks to return (Track B only) |
| `search_robust` | True | `True` = Track B (robust), `False` = Track A (all peaks) |
| `seed` | None | Random seed for reproducibility |

**Note**: `n_sol` is the only parameter users need to tune. It corresponds to the expected number of robust peaks in the problem. Experiments show TSRDE is robust to `n_sol ± 2` deviations.

For RobBench problems, `n_sol` is automatically set to the number of ground-truth robust peaks.

---

## Robustness Measure

TSRDE uses the expected-fitness robustness definition:

```
Robustness(x) = E_{δ ~ U(-σ, σ)^D}[f(x + δ)]
```

where `σ = 0.1 × Rs` (Rs = archive deduplication radius).

Lower expected fitness = more robust (wider basin, less sensitive to perturbation).

The dimension-adaptive sample count formula:

```
M*(D) = max(M_stat(D), M_cov(D))

M_stat(D) = ceil(z² · (2 − 6/(7D)) / ε²)   [asymptotic RSD bound]
M_cov(D)  = 2D                               [directional coverage]
```

Derived from a quadratic peak model, ensuring reliable variance estimation for all D.

---

## RobBench Suite

| Group | Stress Factor | Problems |
|-------|---------------|----------|
| A | Dimensionality (D = 2, 3, 5, 10) | rb_a1–rb_a4 |
| B | Robustness ratio (25%–75% robust peaks) | rb_b1–rb_b4 |
| C | Peak count (2–8 peaks) | rb_c1–rb_c4 |
| D | Anisotropy (axis-aligned vs. rotated narrow dims) | rb_d1–rb_d4 |
| E | Spatial trap (robust peak in low-density region) | rb_e1–rb_e4 |

In total: 20 problems, 100 peaks, 41 robust peaks. All peaks are equal in objective value; robustness is determined solely by basin width. Ground-truth robust peaks and their exact robustness values are analytically known.

---

## RobF1 Metric

RobF1 extends the standard CEC peak ratio to penalize both missed robust peaks and redundant recommendations:

```
Precision = (correct robust peaks found) / (solutions returned)
Recall    = (correct robust peaks found) / (total ground-truth robust peaks)
RobF1     = 2 · Precision · Recall / (Precision + Recall)
```

A solution is credited to a robust peak if it falls within the niche radius and its fitness error is within tolerance.

---

## Results

Per-problem results and post-hoc audits accompanying the paper, all recomputed from the frozen experimental data:

| File | Contents |
|------|----------|
| [`results/robbench_per_problem.md`](results/robbench_per_problem.md) | RobF1 (mean ± std, 10 runs) on all 20 RobBench problems, nine algorithms |
| [`results/cec2026_per_problem.md`](results/cec2026_per_problem.md) | CEC2026 competition score on all 16 problems at D = 2/5/10/20 |
| [`results/ablation.md`](results/ablation.md) | Group-level RobF1 for full TSRDE and three ablated variants |
| [`results/nsol_sensitivity.md`](results/nsol_sensitivity.md) | RobF1 vs. `n_sol` sweep (1,400 runs) and the Δn tolerance curve |
| [`results/audits.md`](results/audits.md) | Near-optimality audit, robustness-ordering consistency, sample-count sensitivity |

Headline numbers: overall RobF1 **0.952** on RobBench (best of nine algorithms, 19 of 20 problems; Wilcoxon signed-rank with Bonferroni correction, p < 1e-5) and overall score **0.550** on CEC2026 (next-best baseline 0.302).

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{tsrde2026,
  title   = {Two-Stage Robust Differential Evolution for Multimodal Optimization},
  author  = {Yao, Xingyi and Li, Wenhua and Wang, Rui and Zhang, Tao and Lu, Xin},
  year    = {2026},
  note    = {Manuscript under review (IEEE Transactions on Cybernetics)}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
