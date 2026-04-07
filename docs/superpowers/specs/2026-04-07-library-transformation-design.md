# Design Spec: Structural Economics Library Transformation

**Date:** 2026-04-07
**Author:** Pranjal Rawat + Claude
**Status:** Draft

---

## Context

The repo `structural-economics` contains ~609 files across MATLAB, Python, Julia, and R implementing ~40-50 distinct computational/structural economics models. The code is functional but unpolished: no standardized output, inconsistent naming, no shared utilities, no project infrastructure.

**Goal:** Transform this into a polished, pedagogical library where every model is:
- Executable via a single `python run.py`
- Self-documented via an auto-generated `README.md` with equations, solutions, visualizations, and economic takeaways
- Written in JAX/Python (migrating from MATLAB/Julia/R where possible)
- Browsable directly on GitHub as the primary UX

**Constraints:**
- Delete nothing — all original files preserved in `_legacy/`
- Keep Dynare `.mod` files as-is (irreplaceable for DSGE)
- All ~40-50 models get the full treatment

---

## Architecture

### Repository Structure

```
structural-economics/
├── README.md                          # Visual catalog hub with thumbnails
├── CLAUDE.md                          # Root-level Claude instructions
├── .gitignore                         # Comprehensive gitignore
├── pyproject.toml                     # Project config + dependencies
├── requirements.txt                   # Pinned dependencies
│
├── lib/                               # Shared JAX utility library
│   ├── __init__.py
│   ├── grids.py                       # Asset/state grid construction
│   ├── discretize.py                  # Tauchen, Rouwenhorst methods
│   ├── interpolate.py                 # Linear/cubic interpolation
│   ├── vfi.py                         # Generic VFI loop (JAX)
│   ├── plotting.py                    # Matplotlib style + save helpers
│   └── output.py                      # ModelReport class (README generator)
│
├── dynamic-programming/
│   ├── CLAUDE.md                      # Folder-specific Claude instructions
│   ├── README.md                      # Section index
│   ├── cake-eating/
│   │   ├── run.py                     # Single self-contained script
│   │   ├── README.md                  # Auto-generated output
│   │   └── figures/
│   │       ├── value-function.png
│   │       ├── policy-function.png
│   │       └── thumb.png              # 200x150 thumbnail for catalog
│   ├── optimal-growth/
│   ├── consumption-savings/
│   ├── job-search-mccall/
│   ├── rbc/
│   ├── aiyagari/
│   ├── diamond-mortensen-pissarides/
│   ├── solow-growth/
│   ├── asset-pricing/
│   ├── heterogeneous-agents/
│   └── deep-learning-vfi/             # Neural net value function approx
│
├── dynare/                            # Kept largely as-is
│   ├── CLAUDE.md
│   ├── README.md
│   ├── rbc/
│   │   ├── model.mod                  # Original Dynare source
│   │   ├── run.py                     # Python wrapper (parses IRFs, generates README)
│   │   ├── README.md
│   │   └── figures/
│   ├── new-keynesian/
│   ├── asset-pricing-news/
│   └── ar-processes/
│
├── industrial-organization/
│   ├── CLAUDE.md
│   ├── README.md
│   ├── bertrand-logit-demand/
│   ├── blp-random-coefficients/
│   ├── merger-simulation/
│   ├── effective-hhi/
│   ├── collusion-detection/
│   ├── nash-in-nash/
│   ├── dynamic-entry-exit/
│   └── static-games/
│
├── optimal-control/
│   ├── CLAUDE.md
│   ├── README.md
│   ├── finite-difference/
│   ├── phase-diagrams/
│   ├── continuous-cake-eating/
│   └── ramsey-growth/
│
├── choice/
│   ├── CLAUDE.md
│   ├── README.md
│   ├── revealed-preference-afriat/
│   ├── garp-warshall/
│   ├── preference-recoverability/
│   ├── bayesian-learning/
│   └── logit-discrete-choice/
│
├── continuous-time/                   # Renamed from ContinousTime (typo fix)
│   ├── CLAUDE.md
│   ├── README.md
│   ├── huggett-incomplete-markets/
│   └── hjb-growth/
│
├── heterogeneous-agents/              # Promoted from misc/docs/HA_codes
│   ├── CLAUDE.md
│   ├── README.md
│   ├── vfi-deterministic/
│   ├── vfi-iid-income/
│   ├── endogenous-grid-points/
│   ├── egp-aiyagari/
│   └── envelope-equation-iteration/
│
├── global-dsge/                       # Renamed from gdgse
│   ├── CLAUDE.md
│   ├── README.md
│   ├── rbc-nonlinear/
│   ├── rbc-capital-tax/
│   └── rbc-irreversible-investment/
│
├── time-series/                       # Promoted from misc/time-series
│   ├── CLAUDE.md
│   ├── README.md
│   ├── fred-macro-data/
│   └── stock-watson/
│
└── _legacy/                           # Complete mirror of original repo
    ├── dynamic-programming/
    ├── dynare/
    ├── choice/
    ├── ...
    └── README.md                      # Original README
```

### Changes from Current Structure
1. **Renamed:** `ContinousTime` → `continuous-time` (typo + kebab-case)
2. **Renamed:** `gdgse` → `global-dsge` (descriptive)
3. **Promoted:** `misc/docs/HA_codes` → `heterogeneous-agents/` (top-level topic)
4. **Promoted:** `misc/time-series` → `time-series/` (top-level topic)
5. **Dissolved:** `misc/` — contents distributed to proper topics or `_legacy/`
6. **All folder names:** kebab-case, descriptive, no abbreviations

---

## Shared Library (`lib/`)

### `lib/output.py` — ModelReport Class

The core abstraction. Every `run.py` uses it to generate a standardized README.md.

```python
class ModelReport:
    """Generates a standardized GitHub-flavored Markdown report."""

    def __init__(self, title: str, description: str = ""):
        self.title = title
        self.description = description
        self.sections: list[tuple[str, str]] = []

    def add_overview(self, text: str) -> None: ...
    def add_equations(self, latex: str, description: str = "") -> None: ...
    def add_model_setup(self, text: str) -> None: ...
    def add_solution_method(self, text: str) -> None: ...
    def add_figure(self, path: str, caption: str, fig) -> None:
        """Save matplotlib figure to path and add to report."""
    def add_table(self, path: str, caption: str, df) -> None:
        """Save DataFrame as CSV and render as markdown table."""
    def add_results(self, text: str) -> None: ...
    def add_takeaway(self, text: str) -> None: ...
    def add_references(self, refs: list[str]) -> None: ...
    def write(self, path: str = "README.md") -> None:
        """Write the full markdown report to disk."""
    def generate_thumbnail(self, source_fig: str, path: str = "figures/thumb.png") -> None:
        """Create 200x150 thumbnail from the first figure."""
```

### Standard README.md Section Order

Every auto-generated README follows this structure:

```markdown
# Model Title

> One-line description of the economic question this model answers.

## Overview
Brief context: what problem, why it matters, key reference paper.

## Equations
$$V(a,z) = \max_{c,a'} \left\{ u(c) + \beta \mathbb{E}[V(a',z') | z] \right\}$$
Subject to: budget constraint, borrowing limit, etc.

## Model Setup
Parameters, calibration, grid construction details.

## Solution Method
Algorithm description: VFI, EGP, contraction mapping, etc.

## Results

### Figures
![Value Function](figures/value-function.png)
*Caption describing what the figure shows.*

### Tables
| Parameter | Value | Description |
|-----------|-------|-------------|
| β         | 0.96  | Discount factor |

## Economic Takeaway
What do we learn? Policy implications, comparative statics, key insight.

## Reproduce
```bash
cd dynamic-programming/cake-eating
python run.py
```

## References
- Author (Year). "Title." *Journal*.
```

### `lib/grids.py`

```python
def uniform_grid(a_min, a_max, n): ...
def exponential_grid(a_min, a_max, n, density=3.0): ...
def chebyshev_nodes(a_min, a_max, n): ...
```

### `lib/discretize.py`

```python
def tauchen(rho, sigma, n, m=3): ...
def rouwenhorst(rho, sigma, n): ...
```

### `lib/interpolate.py`

```python
def linear_interp(x_grid, y_values, x_new): ...  # JAX-compatible
```

### `lib/vfi.py`

```python
def solve_vfi(bellman_operator, v_init, tol=1e-6, max_iter=1000): ...
```

### `lib/plotting.py`

```python
STYLE_CONFIG = { ... }  # Consistent academic matplotlib style

def setup_style(): ...
def save_figure(fig, path, dpi=150): ...
def save_thumbnail(fig, path, size=(200, 150)): ...
```

---

## Per-Model `run.py` Pattern

Each `run.py` is a single self-contained file (~100-300 lines) that:

1. Imports from `lib/` for shared utilities
2. Defines model-specific economics (parameters, Bellman equation, etc.)
3. Solves the model using JAX
4. Generates all output via `ModelReport`

```python
#!/usr/bin/env python3
"""Cake-Eating Problem: Optimal Consumption Under Finite Resources.

Solves the infinite-horizon cake-eating problem using value function iteration
with JAX. Demonstrates the simplest dynamic programming problem: how to
optimally consume a non-renewable resource over time.

Reference: Stokey, Lucas, and Prescott (1989), Ch. 4.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add lib to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.grids import exponential_grid
from lib.vfi import solve_vfi
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # === Parameters ===
    beta = 0.96      # Discount factor
    sigma = 2.0      # CRRA coefficient
    n_grid = 500     # Grid points
    cake_max = 10.0  # Initial cake size

    # === Grid ===
    w_grid = exponential_grid(1e-6, cake_max, n_grid)

    # === Utility ===
    def u(c):
        return jnp.where(sigma == 1, jnp.log(c), c**(1 - sigma) / (1 - sigma))

    # === Bellman Operator ===
    @jax.jit
    def bellman(v):
        # ... model-specific logic
        return v_new, policy

    # === Solve ===
    v_star, policy = solve_vfi(bellman, v_init)

    # === Generate Report ===
    setup_style()
    report = ModelReport(
        "Cake-Eating Problem",
        "Optimal consumption of a non-renewable resource over infinite horizon."
    )
    report.add_overview("""...""")
    report.add_equations(r"""
$$V(W) = \max_{0 \le c \le W} \left\{ u(c) + \beta V(W - c) \right\}$$
where $W$ is remaining cake, $c$ is consumption, $\beta$ is the discount factor.
""")
    # ... figures, tables, takeaway
    report.write("README.md")


if __name__ == "__main__":
    main()
```

---

## Root README.md — Visual Catalog

```markdown
# Structural Economics: Computational Examples

A library of executable models in computational and structural economics.
Every model is self-contained, runs with `python run.py`, and produces a
documented report with equations, solutions, visualizations, and economic
takeaways.

**Built with:** JAX, NumPy, Matplotlib | **License:** MIT

## Quick Start
```bash
pip install -r requirements.txt
cd dynamic-programming/cake-eating
python run.py
```

---

## Dynamic Programming

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="dynamic-programming/cake-eating/figures/thumb.png" width="120"> | [**Cake Eating**](dynamic-programming/cake-eating/) | Value Function Iteration | Optimal depletion of finite resource |
| <img src="dynamic-programming/optimal-growth/figures/thumb.png" width="120"> | [**Optimal Growth**](dynamic-programming/optimal-growth/) | VFI | Capital accumulation dynamics |
| ... | ... | ... | ... |

## Industrial Organization

| | Model | Method | Key Insight |
|:---:|---|---|---|
| <img src="industrial-organization/bertrand-logit-demand/figures/thumb.png" width="120"> | [**Bertrand-Nash Logit**](industrial-organization/bertrand-logit-demand/) | Fixed Point | Oligopoly pricing with differentiation |
| ... | ... | ... | ... |

## Choice Models
...

## Optimal Control
...

## Continuous Time
...

## Heterogeneous Agents
...

## Global DSGE Solutions
...

## Time Series
...

## Dynare (DSGE)
...
```

---

## CLAUDE.md Files

### Root `CLAUDE.md`

```markdown
# Structural Economics Library

## Project Overview
Pedagogical library of computational/structural economics models.
Each model lives in a self-contained folder with `run.py` → `README.md`.

## Code Style
- Python 3.11+, JAX for numerical computation
- Type hints on all public functions
- Google-style docstrings
- kebab-case folder names, snake_case file/function names
- Each run.py is self-contained: economics + computation + report generation

## Shared Library
- `lib/` contains shared JAX utilities (grids, discretize, interpolate, vfi, output, plotting)
- Models import from lib/ — do not duplicate utility code

## Output Standard
Every model produces:
- `README.md` (auto-generated via ModelReport)
- `figures/*.png` (matplotlib, 150 DPI, academic style)
- `tables/*.csv` (optional, if model has tabular results)
- `figures/thumb.png` (200x150 thumbnail for root catalog)

## Section Order in README.md
Overview → Equations → Model Setup → Solution Method → Results → Economic Takeaway → Reproduce → References

## Do Not
- Delete any files — originals are in `_legacy/`
- Use NumPy where JAX works (prefer jnp over np)
- Add interactive/Plotly visualizations — matplotlib only
- Create notebooks — this repo is .py + .md only (new code)
```

### Per-folder `CLAUDE.md` (example: `dynamic-programming/CLAUDE.md`)

```markdown
# Dynamic Programming Models

## Models in this section
Each subfolder is a self-contained DP model using value function iteration
or related methods (policy iteration, EGP).

## Common patterns
- State spaces defined on asset/wealth grids (lib/grids.py)
- Income processes discretized via Tauchen/Rouwenhorst (lib/discretize.py)
- VFI loop via lib/vfi.py with JAX JIT compilation
- Bellman operators are model-specific, defined in each run.py

## Key economics
- Discount factor β, CRRA utility with σ parameter
- Budget constraints, borrowing limits
- Stationary distributions via forward iteration
```

---

## .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
.eggs/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.venv/
env/
venv/

# MATLAB
*.asv
*.mex*

# Dynare generated (keep .mod source, ignore generated)
dynare/**/+model/
dynare/**/*.log
dynare/**/*_dynamic.m
dynare/**/*_static.m
dynare/**/*_results.mat

# Large files
*.pdf
!_legacy/**/*.pdf
```

---

## Complete Model Inventory

### Dynamic Programming (12 models)

| # | Model | Source Files | JAX Migration |
|---|-------|-------------|---------------|
| 1 | Cake Eating | cake1-4.m, cake5-6.ipynb | Full rewrite |
| 2 | Optimal Growth | growth1-3.m, growth4.ipynb | Full rewrite |
| 3 | Consumption-Savings | saving1-2.m | Full rewrite |
| 4 | Job Search (McCall) | search1-2.ipynb | Port from NumPy to JAX |
| 5 | RBC | rbc1-4.ipynb | Port to JAX |
| 6 | Aiyagari | Aiyagari-Model.ipynb, 1-3_*.ipynb | Port to JAX |
| 7 | DMP | dmp1.ipynb | Port to JAX |
| 8 | Solow Growth | solow1-2.ipynb | Port to JAX |
| 9 | Asset Pricing | assetPricing1.ipynb | Port to JAX |
| 10 | Heterogeneous Agents (intro) | hetagent1.ipynb | Port to JAX |
| 11 | Deep Learning VFI | deeplearning1-7.ipynb | Keep PyTorch/TF, add JAX variant |
| 12 | ODE Methods | ode1-2.ipynb, tools1-2.ipynb | Port to JAX |

### Industrial Organization (8 models)

| # | Model | Source Files | JAX Migration |
|---|-------|-------------|---------------|
| 1 | Bertrand-Nash Logit | 1_diff_prod_logit/ | Port .ipynb + .R to JAX |
| 2 | BLP Random Coefficients | 2_rand_coeff_logit/ | Port to JAX |
| 3 | Merger Simulation | 3_diff_prod_rand_logit/ | Port to JAX |
| 4 | Effective HHI | 4_effective_HHI/ | Port to JAX |
| 5 | Collusion Detection | 5_collusion/ | Port to JAX |
| 6 | Nash-in-Nash | 6_nash_in_nash/ | Port .R to JAX |
| 7 | Dynamic Entry/Exit | 7_dynamic_discrete_choice/ | Port .R to JAX |
| 8 | Static Games | 8_static_games/ | Port to JAX |

### Choice (5 models)

| # | Model | Source Files | JAX Migration |
|---|-------|-------------|---------------|
| 1 | Afriat Inequalities | RP/01_Afriat*.ipynb | Port to JAX |
| 2 | GARP (Warshall) | RP/02_Warshall*.ipynb | Port to JAX |
| 3 | Preference Recoverability | RP/03_Recoverability.ipynb | Port to JAX |
| 4 | Bayesian Learning | bayes/ (Python + MATLAB) | Port MATLAB to JAX |
| 5 | Logit Discrete Choice | logit/ | Port to JAX |

### Optimal Control (4 models)

| # | Model | Source Files | JAX Migration |
|---|-------|-------------|---------------|
| 1 | Finite Difference | 01-FiniteDifference.ipynb | Port to JAX |
| 2 | Phase Diagrams | 02-PhaseDiagrams.ipynb | Port to JAX |
| 3 | Continuous Cake Eating | 03-CakeEating.ipynb | Port to JAX |
| 4 | Ramsey Growth | 04-OptimalGrowth.ipynb | Port to JAX |

### Continuous Time (2 models)

| # | Model | Source Files | JAX Migration |
|---|-------|-------------|---------------|
| 1 | Huggett Incomplete Markets | ContinousTime/Huggett/*.m | Full rewrite from MATLAB |
| 2 | HJB Growth/Savings | ContinousTime/Simple/*.m | Full rewrite from MATLAB |

### Heterogeneous Agents (5 models)

| # | Model | Source Files | JAX Migration |
|---|-------|-------------|---------------|
| 1 | VFI Deterministic | misc/docs/HA_codes/vfi_deterministic.* | Port Python version to JAX |
| 2 | VFI with IID Income | misc/docs/HA_codes/vfi_iid.* | Port to JAX |
| 3 | Endogenous Grid Points | misc/docs/HA_codes/egp_IID.* | Port to JAX |
| 4 | EGP Aiyagari | misc/docs/HA_codes/egp_IID_aiyagari.* | Port to JAX |
| 5 | Envelope Equation Iteration | misc/docs/HA_codes/eei_IID.* | Port to JAX |

### Global DSGE (3 models)

| # | Model | Source Files | JAX Migration |
|---|-------|-------------|---------------|
| 1 | RBC Nonlinear | gdgse/1_rbc/ | Rewrite from MATLAB to JAX |
| 2 | RBC Capital Tax | gdgse/2_rbc_capital_tax/ | Rewrite from MATLAB to JAX |
| 3 | RBC Irreversible Investment | gdgse/3_rbcIrr/ | Rewrite from MATLAB to JAX |

### Dynare DSGE (6 model groups — keep as Dynare)

| # | Model | Source Files | Migration |
|---|-------|-------------|-----------|
| 1 | RBC | dynare/rbc/, dynare/RBCNews/ | Keep .mod, add Python wrapper |
| 2 | New Keynesian | dynare/nkdsge/, dynare/nkdsgeNews/ | Keep .mod, add Python wrapper |
| 3 | NKPC | dynare/nkpcNews/ | Keep .mod, add Python wrapper |
| 4 | Asset Pricing News | dynare/assetNews/ | Keep .mod, add Python wrapper |
| 5 | Growth | dynare/growth/ | Keep .mod, add Python wrapper |
| 6 | AR Processes | dynare/ar1/, ar2/, etc. | Keep .mod, add Python wrapper |

### Time Series (2 models)

| # | Model | Source Files | JAX Migration |
|---|-------|-------------|---------------|
| 1 | FRED Macro Data | misc/time-series/fred-md.R | Port to Python/pandas |
| 2 | Stock-Watson | misc/time-series/stock-watson.* | Port to Python/JAX |

**Total: ~47 models**

---

## Dependencies

### `requirements.txt`

```
jax>=0.4.20
jaxlib>=0.4.20
numpy>=1.24
scipy>=1.11
matplotlib>=3.8
pandas>=2.0
```

### `pyproject.toml`

```toml
[project]
name = "structural-economics"
version = "1.0.0"
description = "Computational examples in structural economics"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "numpy>=1.24",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
dynare = ["dynare-python"]  # if available
ml = ["torch", "scikit-learn", "xgboost"]
```

---

## Verification Plan

### Per-model verification
For each migrated model:
1. `python run.py` completes without error
2. `README.md` is generated with all sections populated
3. All `figures/*.png` files are created and non-empty
4. Numerical results match original implementation (within tolerance)
5. Thumbnail `figures/thumb.png` exists

### Integration verification
1. Root `README.md` renders correctly on GitHub (all thumbnail images load, all links work)
2. `pip install -r requirements.txt` installs cleanly
3. Every `run.py` in the repo can be executed from a fresh virtualenv
4. `_legacy/` contains exact copy of original repo state

### Spot-check economics
For key models (Aiyagari, BLP, RBC), verify:
- Policy functions are monotone where expected
- Equilibrium prices/quantities match known analytical solutions
- Convergence behavior is reasonable (iterations, tolerance)

---

## Implementation Order

**Phase 0:** Infrastructure
- Create `_legacy/` backup
- Set up `lib/`, `.gitignore`, `CLAUDE.md`, `pyproject.toml`, `requirements.txt`
- Implement `ModelReport` class and plotting utilities

**Phase 1:** Dynamic Programming (12 models)
- Start with cake-eating (simplest, proves the pattern)
- Then optimal-growth, consumption-savings (similar VFI structure)
- Then RBC, Aiyagari (more complex)
- Deep learning VFI last (different dependencies)

**Phase 2:** Heterogeneous Agents (5 models)
- Already have Python versions — straightforward JAX port

**Phase 3:** Industrial Organization (8 models)
- BLP is the most complex — do Bertrand-logit first

**Phase 4:** Choice Models (5 models)
- Revealed preference first (cleanest), then Bayesian learning

**Phase 5:** Optimal Control (4 models)
- Straightforward notebook → run.py conversions

**Phase 6:** Continuous Time (2 models)
- MATLAB → JAX rewrite (Huggett is non-trivial)

**Phase 7:** Global DSGE (3 models)
- MATLAB → JAX rewrite of sparse grid methods

**Phase 8:** Time Series (2 models)
- R → Python/pandas port

**Phase 9:** Dynare (6 model groups)
- Add Python wrapper scripts around existing .mod files

**Phase 10:** Polish
- Root README with thumbnails
- Cross-link verification
- Final spot-check of all 47 models
