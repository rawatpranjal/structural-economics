# Ramsey Saving by Saddle-Path Shooting

> A continuous-time Ramsey planner inherits capital and uses the consumption jump to reach the saddle path.

## Overview

The Ramsey-Cass-Koopmans planner starts with an inherited capital stock and decides how much output to consume today versus invest for future production. A capital-poor economy should save enough to build productive capacity. A capital-rich economy can consume more while capital runs down. In both cases, history fixes $k_0$, but optimality must select the date-zero consumption jump $c_0$.

The difficulty comes from the boundary condition at infinity. Once we pick $c_0$, the Euler equation and resource law determine the whole path. Almost every initial consumption choice fails: a high guess exhausts capital, while a low guess leaves the planner overaccumulating relative to the transversality condition. Shooting turns that long-run economic restriction into a scalar root search over $c_0$.

The page computes that selection for economies that begin at one quarter, one half, three quarters, one and a half, and twice the Ramsey steady-state capital stock. It uses the same planner as the neighboring [Ramsey phase-diagram](../phase-diagrams/) example and the [HJB growth](../hjb-growth/) tutorial, but represents the stable arm as the root of a terminal capital gap.

## Equations

The planner chooses a feasible path $\{c(t)\}_{t\geq 0}$:

$$
\max_{\{c(t)\}} \int_0^\infty e^{-\rho t}
\frac{c(t)^{1-\sigma}}{1-\sigma}\,dt
\quad\text{s.t.}\quad
\dot{k}(t)=f(k(t))-\delta k(t)-c(t),
\qquad f(k)=Ak^\alpha .
$$

Here $\rho$ is the continuous-time discount rate, $\delta$ is depreciation,
$\sigma$ is the CRRA coefficient and inverse intertemporal elasticity of
substitution, and $A$ is total factor productivity.

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

The saddle path is the feasible path that starts from the inherited $k_0$ and
satisfies the infinite-horizon boundary condition

$$
\lim_{t\to\infty} e^{-\rho t}u'(c(t))k(t)=0
$$

along with the two differential equations above. The finite shooting
calculation approximates this condition by choosing $c_0$ so that the path is
near $(k^{\ast},c^{\ast})$ at a long terminal date.

## Model Setup

The calibration is deterministic and stays close to textbook growth examples. The low initial states mimic economies that begin with too little capital, and the high initial states mimic economies that have accumulated past the long-run Ramsey level. The terminal date is a numerical device for approximating the infinite-horizon transversality condition; it is not an economic horizon.

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

Shooting solves the Ramsey boundary value problem through repeated initial value problems. For a fixed $k_0$, define the terminal gap $G(c_0;k_0)=k(T;c_0)-k^{\ast}$. The algorithm guesses $c_0$, integrates the Ramsey system forward, and reads the sign of $G$ as an economic mistake: too much terminal capital means the planner consumed too little early on, while too little terminal capital means the planner consumed too much.

For this calibration, the relevant bracket has a single crossing. A low $c_0$ leaves too much capital at $T$; a high $c_0$ exhausts capital before the terminal date or leaves too little capital. The bracket must be wide enough for initial states above $k^{\ast}$, where the optimal path begins with consumption above net output so that capital decumulates. Brent's bracketing method then searches for the jump variable that makes the terminal gap zero.

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

The local speed check comes from the Jacobian of $(\dot{k},\dot{c})$ at the steady state. Its eigenvalues are $\lambda_s=-0.0584$ and $\lambda_u=0.0884$, so the local half-life is $\ln(2)/|\lambda_s|=11.9$ time units. On the computed path from $0.25k^{\ast}$, the fitted late-transition rate is $\hat{\lambda}=-0.0583$.

## Results

The phase diagram shows the economic selection behind the shooting calculation. The dashed curve is net output, where $\dot{k}=0$; the vertical line is $k^{\ast}$, where $\dot{c}=0$. Each colored path starts from a different $k_0$ and uses the $c_0$ found by shooting. Below $k^{\ast}$, consumption starts low enough for investment to build capital. Above $k^{\ast}$, consumption starts above net output, so capital is run down.

<img src="figures/phase-diagram.png" alt="Ramsey phase diagram with selected saddle paths from different initial capital stocks" width="80%">

The time paths make the saving logic easier to read. A capital-poor economy keeps consumption below output and lets capital rise; a capital-rich economy consumes more than current net output and moves down. Consumption is not fixed at a constant saving rate. It moves with the Euler equation as the marginal product of capital changes along the transition.

<img src="figures/time-paths.png" alt="Ramsey transition paths for capital and consumption after shooting selects c0" width="80%">

The log-scale convergence plot separates nonlinear transition dynamics from the local stable-root approximation. Far from the steady state, the paths bend because the marginal product changes quickly. Once the economy is close to $k^{\ast}$, the decline in $|k(t)-k^{\ast}|$ is approximately exponential at the stable eigenvalue.

<img src="figures/convergence-speed.png" alt="Log convergence of capital to the Ramsey steady state with the stable eigenvalue benchmark" width="80%">

The table records the jump variable selected by the root search. The consumption ratio is below one when the planner is building capital and above one when the planner is running capital down. The last column is the finite-horizon shooting residual, left visible so the boundary-condition approximation is auditable.

**Shooting Diagnostics**

|   $k_0/k^{\ast}$ |   $c_0$ from shooting |   $c_0/[f(k_0)-\delta k_0]$ |   $k(50)/k^{\ast}$ |   $c(50)/c^{\ast}$ |   Terminal capital gap |
|--------------:|----------------------:|----------------------------:|----------------:|----------------:|-----------------------:|
|          0.25 |              0.867114 |                       0.742 |          0.9548 |          0.979  |               2.75e-07 |
|          0.5  |              1.16845  |                       0.84  |          0.9714 |          0.9868 |               3.7e-07  |
|          0.75 |              1.39947  |                       0.923 |          0.9862 |          0.9936 |               3.17e-06 |
|          1.5  |              1.92645  |                       1.15  |          1.0259 |          1.0118 |               7.69e-08 |
|          2    |              2.20938  |                       1.302 |          1.0503 |          1.0228 |               4.36e-10 |

## Takeaway

The computation enforces the economic restriction that the planner cannot choose an arbitrary date-zero consumption jump. History fixes $k_0$, optimality selects $c_0$, and the root search finds the value that keeps the path feasible while satisfying the transversality condition.

Saddle-path systems are easy to state but delicate to compute. A small error in $c_0$ sends the economy toward capital exhaustion or overaccumulation. Once the correct path is selected, the model delivers the standard Ramsey logic: invest when capital is scarce, decumulate when capital is abundant, and converge toward the modified golden-rule point $f'(k^{\ast})=\rho+\delta$.

## References

- Ramsey, F. (1928). "A Mathematical Theory of Saving." *Economic Journal*, 38(152).
- Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.
- Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 8.
- Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 2.
