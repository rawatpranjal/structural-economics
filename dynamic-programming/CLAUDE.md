# Dynamic Programming Models

## Models in this section
Each subfolder is a self-contained DP model using value function iteration or related methods (policy iteration, EGP, neural network approximation).

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
