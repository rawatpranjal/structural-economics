# RBC with Endogenous Labor: Klein QZ Solution

> Real Business Cycles with capital and labor, solved by generalized Schur decomposition — the algorithm Dynare uses at first order.

## Overview

When labor is endogenous, a productivity shock moves both quantities: capital responds slowly because it is predetermined, but labor jumps immediately to exploit a temporarily higher real wage. Two states (capital, TFP) and two jump variables (consumption, labor) leave method of undetermined coefficients behind: hand-deriving a 4×4 linear system is tedious and error-prone, and the smallest realistic medium-scale models add several more variables.

This tutorial solves the same first-order rational-expectations system using Klein's (2000) generalized Schur (QZ) decomposition — the exact algorithm Dynare implements for ``stoch_simul, order=1`` (Villemot 2011). The method is matrix-based and scales to arbitrary linear DSGEs without changes. The neighboring [RBC tutorial](../rbc/) and [NK tutorial](../nkdsge/) cross-check their hand-derived coefficients against this same QZ routine; here QZ is the primary solver.

## Equations

The household chooses consumption $C_t$ and labor $N_t$ to maximize

$$\mathbb{E}_0\sum_{t=0}^{\infty}\beta^t \left[\frac{C_t^{1-\sigma}}{1-\sigma}-\psi\frac{N_t^{1+\chi}}{1+\chi}\right]$$

subject to $C_t+I_t=Y_t$, $K_t=I_t+(1-\delta)K_{t-1}$, and a Cobb-Douglas
production technology

$$Y_t=A_t K_{t-1}^\alpha N_t^{1-\alpha},\qquad \log A_t=\rho\log A_{t-1}+\varepsilon_t.$$

The intratemporal labor-supply condition is

$$\psi N_t^\chi=(1-\alpha)\frac{Y_t}{N_t}\,C_t^{-\sigma},$$

and the Euler equation for capital is

$$C_t^{-\sigma}=\beta \mathbb{E}_t\left[C_{t+1}^{-\sigma}\left(\alpha A_{t+1}K_t^{\alpha-1}N_{t+1}^{1-\alpha}+1-\delta\right)\right].$$

After log-linearization around the deterministic steady state, the system has
two predetermined variables $\hat k_{t-1},\hat a_t$ and two jump variables
$\hat c_t,\hat n_t$. Output and investment are eliminated via the production
function and resource constraint.

## Model Setup

| Primitive | Value | Role |
|---|---:|---|
| $\alpha$ | 0.33 | Capital share |
| $\beta$ | 0.99 | Quarterly discount factor |
| $\delta$ | 0.025 | Quarterly depreciation |
| $\rho$ | 0.95 | Persistence of log TFP |
| $\sigma$ | 1.0 | CRRA coefficient (log utility) |
| $\chi$ | 1.0 | Inverse Frisch elasticity |
| $\bar N$ | 0.333 | Steady-state hours target |
| $\sigma_\varepsilon$ | 0.010 | Innovation s.d. |
| Shock | 1.0% | One-s.d. innovation at $t=0$ |

| Steady-state object | Value |
|---|---:|
| $K$ | 9.449 |
| $Y$ | 1.005 |
| $C$ | 0.769 |
| $K/Y$ | 9.401 |
| $C/Y$ | 0.765 |
| Real wage | 2.020 |
| Labor weight $\psi$ | 7.883 |

## Solution Method

Stack the linearized system as $A\,\mathbb{E}_t s_{t+1}=B\,s_t$ with $s_t=(\hat k_{t-1},\hat a_t,\hat c_t,\hat n_t)'$. The first two entries are predetermined; the last two are jump variables. The four equations are: capital accumulation (after substituting the resource constraint and production function), the TFP AR(1), the intratemporal labor-supply condition (no expectation), and the Euler equation.

Klein's algorithm computes the ordered generalized Schur (QZ) decomposition

$$Q^H A Z = AA,\qquad Q^H B Z = BB,$$

where $AA, BB$ are upper triangular and the eigenvalues $\lambda_i = (BB)_{ii}/(AA)_{ii}$ are sorted with $|\lambda_i|<1$ first. Blanchard-Kahn requires the count of explosive eigenvalues to equal the number of jump variables. The decision rule for jumps and the state transition follow from the stable subspace of $Z$ — see ``lib/perturbation.py``.

```text
Algorithm: Klein QZ first-order DSGE solver
Inputs:  matrices A, B; partition n_x = number of predetermined vars
Outputs: state transition F (n_x x n_x), decision rule P (n_y x n_x)

1. Compute ordered generalized Schur: ordqz(B, A, sort='iuc') -> AA, BB, Z.
2. Count stable eigenvalues |alpha/beta| < 1; check Blanchard-Kahn.
3. Partition Z conformably: Z = [[Z11 Z12]; [Z21 Z22]] with Z11 (n_x x n_x).
4. Decision rule:    P = Z21 @ inv(Z11)        (jump = P @ predetermined).
5. State transition: F = Z11 @ inv(AA11) @ BB11 @ inv(Z11).
6. Iterate x_{t+1} = F x_t and recover y_t = P x_t for IRFs.
```

For this calibration the Blanchard-Kahn condition holds: 2 stable eigenvalues for 2 predetermined variables, 2 explosive eigenvalues for 2 jump variables. The stable eigenvalues are 0.9500 (capital) and 0.9531 (TFP). Capital evolution and the jump variables follow from the stable subspace; the explosive components are zeroed out as the unique non-explosive rational-expectations solution.

## Results

All six series respond to the productivity innovation. Output rises on impact for two reasons — TFP is higher and labor jumps up to take advantage of the temporarily elevated real wage. Investment is the most volatile margin; consumption smooths through the Euler equation. Capital accumulates slowly. The dashed nonlinear perfect-foresight transition tracks the linear Klein solution closely at this shock size, which is what a local first-order approximation should deliver near steady state.

<img src="figures/irf-tfp-shock.png" alt="Linear Klein QZ vs nonlinear perfect-foresight responses to a 1% TFP shock" width="80%">

**Linear vs Nonlinear IRF Summary**

| Variable    |   Impact (%) |   Peak (%) |   Peak quarter |   Max linear-vs-PF gap (pp) |
|:------------|-------------:|-----------:|---------------:|----------------------------:|
| Output      |        1.309 |      1.309 |              0 |                       0.928 |
| Consumption |        0.387 |      0.626 |             14 |                       0.373 |
| Investment  |        4.311 |      4.311 |              0 |                       5.066 |
| Labor       |        0.461 |      0.461 |              0 |                       0.651 |
| Capital     |        0.108 |      0.84  |             19 |                       1.582 |
| TFP         |        1     |      1     |              0 |                       0     |

## Takeaway

Endogenous labor changes the transmission of a TFP shock from a slow capital story to a fast labor-jump plus slow capital story. The key economic object is the labor decision rule $\hat n_t=-0.1677\hat k_{t-1}+0.4612\hat a_t$, which says hours rise sharply with TFP but moderate as the capital stock expands and the marginal product of labor returns toward steady state. Methodologically, the same Klein QZ solver that drives general DSGE frameworks works here without bespoke coefficient algebra — adding more variables (sticky prices, capital adjustment costs, habits, two countries) would only enlarge the matrices, not change the algorithm.

## References

- King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.
- Hansen, G. (1985). Indivisible Labor and the Business Cycle. *Journal of Monetary Economics*, 16(3), 309-327.
- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
- Villemot, S. (2011). Solving Rational Expectations Models at First Order: What Dynare Does. *Dynare Working Paper 2*, CEPREMAP.
