# RBC TFP Shocks and Capital Propagation

## Overview

When TFP rises, the existing capital stock produces more output. Capital was chosen last period. The household can carry the shock forward only by investing some extra output.

The object is the impulse response of output, consumption, investment, and capital. A persistent TFP shock raises the marginal product of capital. The Euler equation governs how much consumption the household postpones.

We log-linearize the fixed-labor RBC model around steady state. Coefficient matching gives a stable capital decision rule. A Klein QZ solve checks the same coefficients. The figure compares that local rule with the exact nonlinear transition for the same shock path.

## Equations

This representative-agent RBC model follows a one-time technology innovation.
Let $A_t$ denote total factor productivity, $K_{t-1}$ predetermined capital,
$C_t$ consumption, $I_t$ investment, and $Y_t$ output.
Production and goods-market clearing are

$$
Y_t = A_t K_{t-1}^\alpha,
\qquad
Y_t = C_t + I_t,
$$

$$
K_t = I_t + (1-\delta)K_{t-1},
$$

Investment is the only choice that moves capital after the shock.
The household chooses consumption and investment subject to this law.
The Euler equation is

$$
C_t^{-\sigma} =
\beta \mathbb{E}_t\left[
C_{t+1}^{-\sigma}
\left(\alpha A_{t+1}K_t^{\alpha-1}+1-\delta\right)
\right].
$$

The TFP process is

$$
\log A_t = \rho \log A_{t-1} + \varepsilon_t,
\qquad
\varepsilon_t \sim N(0,\sigma_\varepsilon^2).
$$

At the deterministic steady state with $A=1$,

$$
\alpha K^{\alpha-1} = \frac{1}{\beta} - 1 + \delta,
\qquad
Y=K^\alpha,\qquad I=\delta K,\qquad C=Y-I.
$$

The calibration implies $K/Y=9.40$ and $C/Y=0.76$.

## Model Setup

| Primitive | Value | Role |
|---|---:|---|
| $\alpha$ | 0.33 | Capital share in production |
| $\beta$ | 0.99 | Quarterly discount factor |
| $\delta$ | 0.025 | Quarterly depreciation |
| $\rho$ | 0.95 | Persistence of log TFP |
| $\sigma$ | 1.0 | CRRA coefficient; here log utility |
| $\sigma_\varepsilon$ | 0.010 | Innovation standard deviation in log TFP |
| Shock | 1.0% | One-standard-deviation innovation at date 0 |
| IRF horizon | 40 quarters | Periods shown in the figure |

| Steady-state object | Value |
|---|---:|
| $K$ | 28.348 |
| $Y$ | 3.015 |
| $C$ | 2.307 |
| $I$ | 0.709 |
| $K/Y$ | 9.401 |
| $C/Y$ | 0.765 |

## Solution Method

The computation solves for the stable law of motion for capital. Write $\hat k_t=\log(K_t/K)$ and $\hat a_t=\log A_t$. The log-linear decision rule is:

$$
\hat k_t = 0.9621\hat k_{t-1} + 0.0801\hat a_t.
$$

This rule maps inherited capital and current productivity into next capital. Production and the resource constraint then recover output, investment, and consumption.

Because the capital root is below one, investment moves first and capital builds gradually.

```text
Algorithm: first-order RBC impulse response
Inputs: alpha, beta, delta, rho, sigma, shock size eps_0, horizon T
Outputs: paths for yhat_t, chat_t, ihat_t, khat_t

1. Compute the deterministic steady state K, Y, C, I.
2. Linearize the resource constraint and Euler equation in log deviations.
3. Guess khat_t = p khat_{t-1} + q ahat_t.
4. Substitute the guess into the linearized equations and match the
   coefficients on khat_{t-1} and ahat_t.
5. Iterate the decision rule along ahat_t = rho^t eps_0.
6. Recover output, investment, and consumption from the model equations.
7. Compare with the nonlinear transition for the same shock path.
```

The coefficient-matching residual is 2.9e-15. Klein QZ agrees to 1.5e-15, so the first-order system is solved correctly. The nonlinear transition uses the same shock path without later innovations. Its small gap is a local accuracy check.

## Results

Output rises immediately because the same capital is more productive. Investment jumps more than output because the household wants more capital while productivity remains high. Consumption rises by less on impact and keeps drifting upward for several quarters as the Euler equation smooths marginal utility. The dashed nonlinear transition sits almost on top of the first-order solution at this shock size.

<img src="figures/irf-tfp-shock.png" alt="Impulse responses of output, consumption, investment, and capital to a 1 percent TFP shock" width="80%">

The table separates impact effects from delayed peaks. Capital and consumption peak well after the shock because the state moves slowly. Investment peaks immediately because it changes the state.

**IRF Summary Statistics**

| Variable    |   Impact (%) |   Peak (%) |   Peak quarter |   Half-life after peak |   Max nonlinear gap (pp) |
|:------------|-------------:|-----------:|---------------:|-----------------------:|-------------------------:|
| Output      |        1     |      1     |              0 |                     26 |                    0     |
| Consumption |        0.323 |      0.528 |             16 |                     38 |                    0.001 |
| Investment  |        3.204 |      3.204 |              0 |                     11 |                    0.033 |
| Capital     |        0.08  |      0.687 |             21 |                     39 |                    0.001 |
| TFP         |        1     |      1     |              0 |                     14 |                    0     |

## Takeaway

In this RBC model, a productivity shock raises output on impact. Investment responds strongly because the marginal product of capital is temporarily high. Consumption moves more smoothly, and capital accumulates only gradually. First-order perturbation isolates that propagation mechanism near steady state.

## References

- Kydland, F. and Prescott, E. (1982). Time to Build and Aggregate Fluctuations. *Econometrica*, 50(6), 1345-1370.
- King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.
- Uhlig, H. (1999). A Toolkit for Analysing Nonlinear Dynamic Stochastic Models Easily. In *Computational Methods for the Study of Dynamic Economies*.
- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
