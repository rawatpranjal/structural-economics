# Heterogeneous Agent Models

## Models in this section
Canonical heterogeneous agent models comparing solution methods:
VFI, endogenous grid points (EGP), and envelope equation iteration.

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
