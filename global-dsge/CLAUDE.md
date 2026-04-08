# Global DSGE Solutions

## Models in this section
Nonlinear, global solutions for DSGE models using two approaches:
1. **VFI-based**: Value function iteration on grids (rbc-nonlinear, rbc-capital-tax, rbc-irreversible-investment)
2. **STPFI-based**: Simultaneous Transition and Policy Function Iterations from Cao, Luo, Nie (2023)

## STPFI Algorithm
The key innovation: solve for policy P and transition T functions simultaneously using
first-order conditions (not Bellman equations) and consistency equations for endogenous state transitions.

At each collocation point (z, s), solve: F(s, x, z, {s'(z'), P^n(z', s'(z'))}) = 0
- Unknowns: policy variables x AND future endogenous states {s'(z')}
- Uses `scipy.optimize.root` for the nonlinear system
- Complementary slackness via min(mu, constraint) formulation
- Convergence: sup-norm of policy function changes

Shared solver: `lib/stpfi.py` — `solve_stpfi(step, policy_init, trans_init)`

## STPFI Models
- **heaton-lucas**: Two agents, equity+bonds, wealth share dynamics (1D state)
- **barro-rare-disasters**: Heterogeneous risk-aversion, Epstein-Zin, IID disasters (1D state)
- **bianchi-sudden-stops**: Open economy, pecuniary externality, borrowing constraint (1D state)
- **mendoza-sudden-stops**: Collateral constraints, state transformation (2D state)
- **guvenen-asset-pricing**: Heterogeneous EIS, production economy (2D state)

## Common patterns
- Value function iteration or STPFI on grids (tensor-product)
- Policy function interpolation (1D linear or 2D cubic via RegularGridInterpolator)
- Simulation of nonlinear dynamics via `lib/simulate.py`
- Each model includes algorithm pseudocode in the Solution Method section

## Key economics
- Incomplete markets with heterogeneous agents
- Occasionally binding constraints (borrowing limits, short-sale, collateral)
- Endogenous state variables with implicit laws of motion (wealth shares, bonds)
- Why global solutions matter: captures kinks at constraint boundaries, risk premia, precautionary behavior
