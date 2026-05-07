# RBC Labor Supply and TFP Shocks by Klein QZ

> Endogenous hours shape business-cycle responses, and generalized Schur decomposition solves the local rational-expectations system.

## Overview

A positive productivity shock makes firms want more inputs today. Capital is largely inherited from yesterday, but hours can move at once, so the household must decide how much of the higher wage to take as labor income, consumption, and investment. In this small RBC economy, a one percent TFP innovation is enough to show the main tension: output jumps immediately, capital builds gradually, and hours absorb part of the short-run adjustment.

The computation starts after log-linearizing the equilibrium around steady state. The local system has lagged capital and current TFP as predetermined variables, with consumption and labor as jump variables. Hand-solving those coefficients is possible but unappealing once labor enters. Klein's (2000) generalized Schur decomposition finds the stable subspace of the linear rational-expectations system and converts it into a state transition plus decision rules. Those rules let us trace impulse responses and compare the local solution with a nonlinear perfect-foresight path near steady state.

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

After log-linearization around the deterministic steady state, the local object
is a decision rule for consumption and labor as functions of lagged capital and
current TFP. Output and investment are then recovered from production and the
resource constraint.

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

Stack the linearized equilibrium as $A\,\mathbb{E}_t s_{t+1}=B\,s_t$ with $s_t=(\hat k_{t-1},\hat a_t,\hat c_t,\hat n_t)'$. The first two entries are states, and the last two are jump variables. The rows collect capital accumulation, the TFP law of motion, intratemporal labor supply, and the Euler equation for capital. The solution is a pair of matrices

$$x_{t+1}=F x_t,\qquad y_t=P x_t,$$

where $x_t=(\hat k_{t-1},\hat a_t)'$ and $y_t=(\hat c_t,\hat n_t)'$. The economic content of $P$ is immediate: it tells us how consumption and hours respond when productivity rises or when the inherited capital stock changes.

Klein's algorithm computes an ordered generalized Schur decomposition. The ordering separates the stable roots from the explosive roots, then uses the stable subspace to pick the unique non-explosive path for the jump variables.

```text
Algorithm: Klein QZ for the linearized RBC system
Inputs:  matrices A and B; number of state variables n_x = 2
Outputs: state transition F, decision rule P, and impulse responses

1. Compute the ordered QZ decomposition of (B, A), with stable roots first.
2. Check Blanchard-Kahn: the number of stable roots must equal n_x.
3. Partition the Schur vectors into state and jump blocks.
4. Recover P from the jump block relative to the state block.
5. Recover F from the stable triangular blocks.
6. Start from x_0 = (0, sigma_e) and iterate x_{t+1} = F x_t, y_t = P x_t.
```

For this calibration the Blanchard-Kahn condition holds: 2 stable eigenvalues for 2 state variables, and 2 explosive eigenvalues for 2 jump variables. The stable eigenvalues are 0.9500 and 0.9531. Once these roots are ordered, the rest of the calculation is linear algebra.

## Results

The productivity innovation raises output on impact because TFP is higher and hours rise with the temporarily higher marginal product of labor. Investment moves more than consumption because capital is the state that carries the shock forward. Consumption responds smoothly through the Euler equation, while capital accumulates over many quarters. The nonlinear perfect-foresight path is included as a local benchmark. It preserves the same ranking of margins, with larger percentage-point gaps for investment and capital because investment is the residual margin that moves the stock.

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

Endogenous labor makes the TFP shock partly a labor-hours response and partly a capital-accumulation response. The labor decision rule $\hat n_t=-0.1677\hat k_{t-1}+0.4612\hat a_t$ says that hours rise strongly with productivity but fall as inherited capital becomes abundant. Klein QZ matters because it delivers that economic object from the stable subspace of the log-linear system. Adding more DSGE variables would enlarge the matrices, but the solution logic would stay the same.

## References

- King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.
- Hansen, G. (1985). Indivisible Labor and the Business Cycle. *Journal of Monetary Economics*, 16(3), 309-327.
- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
- Villemot, S. (2011). Solving Rational Expectations Models at First Order: What Dynare Does. *Dynare Working Paper 2*, CEPREMAP.
