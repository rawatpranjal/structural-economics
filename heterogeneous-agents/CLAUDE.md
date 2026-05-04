# Heterogeneous Agent Models

## Models in this section
Fast solvers for incomplete-market economies:
endogenous grid points (EGP), envelope equation iteration (EEI), and
continuous-time HJB/KFE equilibrium.

## Catalog role
- Treat this as part of the broader Macroeconomics block, not as a generic
  methods bucket.
- Keep this section focused on method variants for incomplete-market models.
  The canonical VFI consumption-savings and Aiyagari tutorials live in
  `dynamic-programming/`.
- Order tutorials from Euler-equation inversion, to marginal-value iteration,
  to continuous-time HJB/KFE equilibrium.
- Catalog titles should make clear whether the tutorial is about EGP, EEI, or
  the Huggett equilibrium object.

## Common patterns
- Income risk discretized via Rouwenhorst (`lib/discretize.py`) when the model
  is discrete time
- Asset grids with exponential spacing (`lib/grids.py`)
- Euler-equation inversion or marginal-value iteration for household policies
- HJB and KFE blocks for continuous-time stationary equilibrium
- Market-clearing checks when an interest rate or aggregate object is solved

## Key economics
- Precautionary savings motive under income uncertainty
- Wealth inequality and Gini coefficients
- Marginal propensity to consume (MPC) heterogeneity
- Incomplete-market equilibrium: asset or bond market clearing pins down prices
