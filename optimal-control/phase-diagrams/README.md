# Ramsey Consumption Choice and Saddle Paths

> Trace the stable arm that selects initial consumption in continuous-time growth.

## Overview

A Ramsey planner inherits an initial capital stock and must choose consumption today before the rest of the path is known. Starting below the steady state makes the tradeoff concrete: consuming too much leaves too little investment, while consuming too little builds capital that the planner would rather have consumed earlier.

The phase diagram turns that intertemporal choice into geometry. Capital is the state, consumption is the control, and each point $(k,c)$ has an arrow showing how the economy moves next. The nullclines show where capital or consumption is temporarily flat, and their intersection gives the Ramsey steady state. The optimum needs one more object: the stable arm, the curve of initial $(k,c)$ pairs that converges to the saddle-point steady state and satisfies the present-value boundary condition.

The computation traces that curve. We linearize the differential equation system at the steady state, take the stable eigenvector as a local guide, and integrate the nonlinear Ramsey ODE backward to draw the stable arm away from the steady state. The same selection problem appears in shooting and HJB methods, but the phase plane lets the reader see the economics before choosing a numerical solver.

## Equations

The planner solves

$$
\max_{\{c(t)\}_{t \geq 0}}
\int_0^\infty e^{-\rho t}\frac{c(t)^{1-\sigma}}{1-\sigma}\,dt
\quad\text{s.t.}\quad
\dot{k}(t)=Ak(t)^\alpha-\delta k(t)-c(t).
$$

The Euler equation and the resource law form the two-dimensional system

$$
\dot{k}=f(k)-\delta k-c,
\qquad
\frac{\dot{c}}{c}=\frac{f'(k)-\delta-\rho}{\sigma},
\qquad
f(k)=Ak^\alpha .
$$

The capital nullcline is

$$
\dot{k}=0
\quad\Longleftrightarrow\quad
c=f(k)-\delta k .
$$

The consumption nullcline is

$$
\dot{c}=0
\quad\Longleftrightarrow\quad
f'(k)=\rho+\delta
\quad\Longleftrightarrow\quad
k=k^{\ast}
=\left(\frac{\alpha A}{\rho+\delta}\right)^{1/(1-\alpha)} .
$$

Steady-state consumption is $c^{\ast}=f(k^{\ast})-\delta k^{\ast}$. The golden-rule
capital stock satisfies $f'(k_{GR})=\delta$, so with $\rho>0$ the Ramsey steady
state lies to the left of the golden rule. The boundary condition selecting the
planner's path is the transversality condition

$$
\lim_{t\to\infty} e^{-\rho t}u'(c(t))k(t)=0 .
$$

## Model Setup

The calibration is deterministic, so every arrow in the diagram reflects the planner's consumption-investment tradeoff. Output is Cobb-Douglas, preferences are CRRA, and the discount rate places the Ramsey steady state below the golden-rule capital stock.

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | 0.30 | Capital share |
| $\delta$ | 0.05 | Depreciation rate |
| $\rho$ | 0.04 | Continuous-time discount rate |
| $\sigma$ | 2.0 | CRRA coefficient and inverse EIS |
| $A$ | 1.0 | Total factor productivity |
| $k^{\ast}$ | 5.5843 | Ramsey steady-state capital |
| $c^{\ast}$ | 1.3961 | Ramsey steady-state consumption |
| $k_{GR}$ | 12.9314 | Golden-rule capital |
| $c_{GR}$ | 1.5087 | Golden-rule sustainable consumption |

## Solution Method

The steady state anchors the calculation because the saddle structure is local. The Jacobian of $(\dot{k},\dot{c})$ at $(k^{\ast},c^{\ast})$ is

$$
J=
\begin{bmatrix}
f'(k^{\ast})-\delta & -1 \\
c^{\ast}f''(k^{\ast})/\sigma & 0
\end{bmatrix}.
$$

The eigenvalues are $\lambda_s=-0.0710$ and $\lambda_u=0.1110$. One is negative and one is positive, so nearby paths split into stable and unstable directions. The stable eigenvector has local slope $dc/dk=0.1110$. That line gives the slope of the planner's path at the steady state, but it is only a first-order approximation. To draw the nonlinear stable arm, the code starts near the steady state along that eigenvector and integrates the full Ramsey system backward.

```text
Algorithm: trace the Ramsey stable arm
Inputs: primitives (alpha, delta, rho, sigma, A), bounds for plotted k and c
1. Compute (k*, c*) from f'(k*) = rho + delta and c* = f(k*) - delta k*.
2. Form the Jacobian J of F(k,c) = (dot{k}, dot{c}) at (k*, c*).
3. Let lambda_s < 0 and v_s = (1, m_s) be the stable eigenpair.
4. Start just below and just above (k*, c*) along v_s.
5. Integrate d(k,c)/d tau = -F(k,c) away from the steady state.
6. Stop when a branch leaves the plotted economic region.
7. Sort the branches by k and read c(k) as the selected initial consumption rule.
Output: nullclines, local linear arm, nonlinear stable arm, and sample forward paths.
```

Backward integration is a drawing device. Forward in economic time, points on the traced arm converge to the steady state; points above or below it miss the present-value boundary condition.

## Results

The blue curve and red line give sign information. Below net output, capital accumulates; left of $k^{\ast}$, consumption grows because the marginal product is high. The black curve does more than show local motion. For each capital stock on the plotted branch, it gives the initial consumption level that converges to the steady state and satisfies the transversality condition. The dashed line shows that linearization works near the steady state but not globally; over $k \in [0.5k^{\ast},1.5k^{\ast}]$ its largest consumption gap from the nonlinear reference is 0.050.

<img src="figures/phase-diagram.png" alt="Ramsey phase plane with nullclines, local linear arm, and nonlinear stable arm" width="80%">

Starting below steady-state capital, the selected path keeps consumption low enough for investment to be high. As capital approaches $k^{\ast}$, the marginal product falls, investment slows, and consumption rises toward $c^{\ast}$. The local stable eigenvalue implies a half-life of about $\ln(2)/|\lambda_s|=9.8$ time units near the steady state; the early part of the transition is nonlinear.

<img src="figures/time-paths.png" alt="Capital and consumption converge to the Ramsey steady state along the stable arm" width="80%">

Holding initial capital fixed makes the saddle-path logic explicit. A higher initial consumption choice starts above the stable arm and runs capital down. A lower choice starts below it and accumulates too much capital relative to the present-value boundary condition. The arrows help explain motion, but they do not select the path. The planner needs the one initial consumption level that puts the economy on the stable arm.

<img src="figures/path-selection.png" alt="Forward trajectories from the same initial capital but different initial consumption" width="80%">

The table keeps the main numbers auditable. The Ramsey steady state has $r^{\ast}=\rho$, while the golden-rule point has more capital because it ignores impatience. The eigenvalue pair verifies the saddle classification. The last row records how far the nonlinear stable arm moves away from the local linear approximation over the central part of the graph.

**Steady-State and Stable-Arm Diagnostics**

| Quantity                    |   Value | Description                                    |
|:----------------------------|--------:|:-----------------------------------------------|
| $k^{\ast}$                     |  5.5843 | Ramsey steady-state capital                    |
| $c^{\ast}$                     |  1.3961 | Ramsey steady-state consumption                |
| $y^{\ast}$                     |  1.6753 | Steady-state output                            |
| $r^{\ast}=f'(k^{\ast})-\delta$    |  0.04   | Net return equals rho in steady state          |
| $k_{GR}$                    | 12.9314 | Capital that maximizes sustainable consumption |
| $c_{GR}$                    |  1.5087 | Maximum sustainable consumption                |
| $\lambda_s$                 | -0.071  | Stable eigenvalue                              |
| $\lambda_u$                 |  0.111  | Unstable eigenvalue                            |
| $dc/dk$ on local stable arm |  0.111  | Consumption slope of the linearized stable arm |
| Max nonlinear-linear gap    |  0.05   | Largest c gap on k in [0.5 k*, 1.5 k*]         |

## Takeaway

A Ramsey phase diagram is an economic selection device. Nullclines say which way capital and consumption move, but they do not choose the initial consumption level. The transversality condition selects the stable arm. Linearization gives the local slope and convergence speed; backward integration of the nonlinear ODE shows how the selected path bends away from the steady state. Shooting algorithms and HJB methods solve the same selection problem in different ways, while the phase plane keeps the consumption-investment tradeoff visible.

## References

- Ramsey, F. (1928). "A Mathematical Theory of Saving." *Economic Journal*, 38(152).
- Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.
- Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 8.
