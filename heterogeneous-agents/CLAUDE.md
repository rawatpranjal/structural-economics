# Heterogeneous Agent Models

## Models in this section
Canonical heterogeneous agent models comparing solution methods:
VFI, endogenous grid points (EGP), and envelope equation iteration.

## Catalog role
- Treat this as part of the broader Macroeconomics block, not as a standalone
  methods bucket.
- Order tutorials from the household problem to general equilibrium, then to
  alternative speedups or marginal-value methods.
- Catalog titles should make clear whether the tutorial is about income risk,
  Aiyagari equilibrium, EGP, or envelope-equation iteration.

## Common patterns
- Income risk discretized via Rouwenhorst (`lib/discretize.py`)
- Asset grids with exponential spacing (`lib/grids.py`)
- Forward simulation for wealth distributions and MPC computation
- Partial equilibrium (fixed r) and general equilibrium (Aiyagari)

## Key economics
- Precautionary savings motive under income uncertainty
- Wealth inequality and Gini coefficients
- Marginal propensity to consume (MPC) heterogeneity
- General equilibrium: capital market clearing pins down interest rate
