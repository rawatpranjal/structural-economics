# RBC Labor Supply and TFP Shocks

> Endogenous hours shape a productivity shock, and Klein QZ solves the local rational-expectations system.

## Overview

A positive TFP shock raises the marginal product of inputs. Capital is mostly inherited, so hours carry much of the impact response.

The object is the local impulse response of output, consumption, investment, capital, and labor. The household chooses consumption and hours while firms use capital and labor to produce output.

Log-linearization gives a rational-expectations system with two states and two jump variables. Klein QZ selects the stable path and returns decision rules for consumption and labor.

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

Log-linearization turns the model into decision rules for consumption and labor.
The states are lagged capital and current TFP. Output and investment follow from
production and the resource constraint.

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

Stack the linearized equilibrium as

$$A\,\mathbb{E}_t s_{t+1}=B\,s_t.$$

Use $s_t=(\hat k_{t-1},\hat a_t,\hat c_t,\hat n_t)'$. The first two entries are states. The last two entries are jump variables.

The rows are capital accumulation, TFP, labor supply, and the Euler equation.

The solution is a pair of matrices

$$x_{t+1}=F x_t,\qquad y_t=P x_t,$$

where $x_t=(\hat k_{t-1},\hat a_t)'$ and $y_t=(\hat c_t,\hat n_t)'$. The matrix $P$ maps states into consumption and labor.

Klein QZ orders the generalized eigenvalues. The stable block gives the non-explosive path for the two jump variables.

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

The Blanchard-Kahn count matches: 2 stable roots for 2 states. The stable roots are 0.9500 and 0.9531. This selects one local equilibrium path.

## Results

The productivity innovation raises output on impact because TFP and hours rise together. Investment moves more than consumption because capital carries the shock forward. Consumption adjusts smoothly through the Euler equation.

The dashed nonlinear path is a local check on the linear solution. It keeps the same ranking of margins. The largest gaps appear for investment and capital, which move through the resource constraint.

<img src="figures/irf-tfp-shock.png" alt="Linear Klein QZ vs nonlinear perfect-foresight responses to a 1% TFP shock" width="80%">

**IRF Summary**

| Variable    |   Impact (%) |   Peak (%) |   Peak quarter |   Max linear-vs-PF gap (pp) |
|:------------|-------------:|-----------:|---------------:|----------------------------:|
| Output      |        1.309 |      1.309 |              0 |                       0.928 |
| Consumption |        0.387 |      0.626 |             14 |                       0.373 |
| Investment  |        4.311 |      4.311 |              0 |                       5.066 |
| Labor       |        0.461 |      0.461 |              0 |                       0.651 |
| Capital     |        0.108 |      0.84  |             19 |                       1.582 |
| TFP         |        1     |      1     |              0 |                       0     |

## Takeaway

The TFP shock splits into an hours response and a capital response. The labor rule $\hat n_t=-0.1677\hat k_{t-1}+0.4612\hat a_t$ rises with productivity and falls with inherited capital.

Klein QZ delivers that rule from the stable subspace.

## References

- King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.
- Hansen, G. (1985). Indivisible Labor and the Business Cycle. *Journal of Monetary Economics*, 16(3), 309-327.
- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
- Villemot, S. (2011). Solving Rational Expectations Models at First Order: What Dynare Does. *Dynare Working Paper 2*, CEPREMAP.
