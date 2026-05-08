# Ramsey Saving by Saddle-Path Shooting

> A continuous-time Ramsey planner inherits capital and uses the consumption jump to reach the saddle path.

## Overview

In Ramsey growth, an economy inherits its capital stock. A capital-poor economy saves to build productive capacity. A capital-rich economy can consume more while capital falls toward its long-run level.

The object is the date-zero consumption choice $c_0$. History fixes $k_0$, but consumption can jump. The right jump places the economy on the saddle path to the Ramsey steady state.

Shooting treats $c_0$ as the unknown. Each guess defines a full path through the Euler equation and resource law. A root search chooses the guess whose terminal capital is near $k^{\ast}$.

## Equations

The planner chooses a feasible path $\{c(t)\}_{t\geq 0}$:

$$
\max_{\{c(t)\}} \int_0^\infty e^{-\rho t}
\frac{c(t)^{1-\sigma}}{1-\sigma}\,dt
\quad\text{s.t.}\quad
\dot{k}(t)=f(k(t))-\delta k(t)-c(t),
\qquad f(k)=Ak^\alpha .
$$

Here $\rho$ is the continuous-time discount rate. The parameter $\delta$ is
depreciation. The parameter $\sigma$ is the CRRA coefficient and inverse EIS.
The parameter $A$ is total factor productivity.

The Euler equation is the Keynes-Ramsey rule

$$
\frac{\dot{c}(t)}{c(t)}=
\frac{f'(k(t))-\delta-\rho}{\sigma}.
$$

Together with the resource law, this gives the two-dimensional system solved by
the code:

$$
\dot{k}=Ak^\alpha-\delta k-c,
\qquad
\dot{c}=\frac{\alpha A k^{\alpha-1}-\delta-\rho}{\sigma}c .
$$

The steady state satisfies

$$
f'(k^{\ast})=\rho+\delta,
\qquad
k^{\ast}=\left(\frac{\alpha A}{\rho+\delta}\right)^{1/(1-\alpha)},
\qquad
c^{\ast}=f(k^{\ast})-\delta k^{\ast}.
$$

The saddle path starts from the inherited $k_0$. It also satisfies the
infinite-horizon boundary condition

$$
\lim_{t\to\infty} e^{-\rho t}u'(c(t))k(t)=0
$$

along with the two differential equations above. The finite shooting
calculation chooses $c_0$ so the path is near $(k^{\ast},c^{\ast})$ at date $T$.

## Model Setup

The calibration is deterministic and close to textbook growth examples. Initial states range from scarce capital to excess capital. The terminal date approximates the transversality condition; it is not an economic horizon.

| Object | Value | Role |
|---|---:|---|
| $\alpha$ | 0.33 | Capital share in $Ak^\alpha$ |
| $\delta$ | 0.05 | Depreciation rate |
| $\rho$ | 0.03 | Discount rate |
| $\sigma$ | 2.0 | CRRA coefficient and inverse EIS |
| $A$ | 1.0 | Total factor productivity |
| $T$ | 150 | Terminal date for shooting |
| Initial capital | $0.25k^{\ast}$ to $2.00k^{\ast}$ | Predetermined state values |
| $k^{\ast}$ | 8.2898 | Ramsey steady-state capital |
| $c^{\ast}$ | 1.5952 | Ramsey steady-state consumption |

## Solution Method

Shooting solves the Ramsey boundary value problem with repeated initial value problems. For fixed $k_0$, define the terminal gap $G(c_0;k_0)=k(T;c_0)-k^{\ast}$. A trial $c_0$ gives a sign: positive $G$ means early consumption was too low, and negative $G$ means it was too high.

The algorithm brackets $c_0$ with one low guess and one high guess. Brent's method then searches for the jump that makes the terminal gap zero. The final integration gives the selected capital and consumption paths.

```text
Algorithm: finite-horizon shooting for Ramsey growth
Inputs: primitives (alpha, delta, rho, sigma, A), initial capital k0, terminal date T
1. Compute (k*, c*) from f'(k*) = rho + delta and c* = f(k*) - delta k*.
2. Pick a low c0 guess and a high c0 guess with opposite signs of k(T; c0) - k*.
3. For a trial c0, integrate
       dot{k} = f(k) - delta k - c,
       dot{c}/c = [f'(k) - delta - rho] / sigma
   from t = 0 to T, stopping early if feasibility fails.
4. Use bisection/Brent updates on c0 until abs(k(T; c0) - k*) is small.
5. Reintegrate the ODE with the selected c0 to obtain k(t) and c(t).
Output: the saddle-path initial consumption c0(k0) and transition path.
```

## Results

The phase diagram shows how shooting selects the path. The dashed curve is net output, where $\dot{k}=0$. The vertical line is $k^{\ast}$, where $\dot{c}=0$. Each colored path starts from a different $k_0$ and uses the chosen $c_0$. Below $k^{\ast}$, consumption starts low enough to build capital. Above $k^{\ast}$, consumption starts high enough to run capital down.

<img src="figures/phase-diagram.png" alt="Ramsey phase diagram with selected saddle paths from different initial capital stocks" width="80%">

The time paths show the saving rule along the selected path. A capital-poor economy keeps consumption below output and lets capital rise. A capital-rich economy consumes more than net output and lets capital fall. Consumption moves with the Euler equation as the marginal product changes.

<img src="figures/time-paths.png" alt="Ramsey transition paths for capital and consumption after shooting selects c0" width="80%">

The log-scale plot shows convergence after the saddle path is selected. Far from the steady state, the paths bend as the marginal product changes. Near $k^{\ast}$, $|k(t)-k^{\ast}|$ falls close to the stable-eigenvalue rate.

<img src="figures/convergence-speed.png" alt="Log convergence of capital to the Ramsey steady state with the stable eigenvalue benchmark" width="80%">

The table records the jump chosen by the root search. The consumption ratio is below one when the planner builds capital. It is above one when the planner runs capital down. The last column reports the terminal capital gap.

**Shooting Diagnostics**

|   $k_0/k^{\ast}$ |   $c_0$ from shooting |   $c_0/[f(k_0)-\delta k_0]$ |   $k(50)/k^{\ast}$ |   $c(50)/c^{\ast}$ |   Terminal capital gap |
|--------------:|----------------------:|----------------------------:|----------------:|----------------:|-----------------------:|
|          0.25 |              0.867114 |                       0.742 |          0.9548 |          0.979  |               2.75e-07 |
|          0.5  |              1.16845  |                       0.84  |          0.9714 |          0.9868 |               3.7e-07  |
|          0.75 |              1.39947  |                       0.923 |          0.9862 |          0.9936 |               3.17e-06 |
|          1.5  |              1.92645  |                       1.15  |          1.0259 |          1.0118 |               7.69e-08 |
|          2    |              2.20938  |                       1.302 |          1.0503 |          1.0228 |               4.36e-10 |

## Takeaway

History fixes $k_0$, but optimality selects $c_0$. A wrong jump sends the economy toward capital exhaustion or overaccumulation. Shooting finds the jump that keeps the path feasible and near the Ramsey steady state.

The selected path gives the Ramsey saving logic. Build capital when it is scarce. Run capital down when it is abundant. Converge toward the modified golden-rule point $f'(k^{\ast})=\rho+\delta$.

## References

- Ramsey, F. (1928). "A Mathematical Theory of Saving." *Economic Journal*, 38(152).
- Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.
- Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 8.
- Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 2.
