# Dynamic Programming Models

## Models in this section
Each subfolder is a self-contained DP model using value function iteration or related methods (policy iteration, EGP, neural network approximation).

## Catalog role
- Dynamic programming is the first substantive section in the root catalog.
- Keep the root README order pedagogical: shock grids and one-state Bellman
  problems first, then savings/search/asset-pricing, then business-cycle and
  general-equilibrium applications.
- It is fine for this section to include macro models when the tutorial is
  mainly teaching Bellman-equation reasoning. More specialized heterogeneous
  agent method comparisons belong under `heterogeneous-agents/`.
- Use titles that name both the economic problem and what the reader learns,
  not only the solver.

## Common patterns
- State spaces defined on asset/wealth grids (`lib/grids.py`)
- Income processes discretized via Tauchen/Rouwenhorst (`lib/discretize.py`)
- VFI loop via `lib/vfi.py` with JAX JIT compilation
- Bellman operators are model-specific, defined in each `run.py`
- CRRA utility: `u(c) = c^(1-sigma)/(1-sigma)` with log special case

## Key economics
- Discount factor beta, CRRA risk aversion sigma
- Budget constraints, borrowing limits
- Stationary distributions via forward iteration on policy functions
