# Phase Diagrams: Ramsey Optimal Growth

> Continuous-time dynamics of consumption and capital with saddle-path stability.

## Overview

Phase diagrams are the primary tool for analyzing continuous-time dynamic economic models. The Ramsey-Cass-Koopmans model has a two-dimensional state space $(k, c)$ with a unique steady state that is a *saddle point* — only one path (the stable manifold) converges to it.

This module visualizes the phase plane: nullclines where $\dot{k} = 0$ and $\dot{c} = 0$, the vector field showing direction of motion, and the saddle path that the economy must follow for an interior optimum.

## Equations

**Capital accumulation:**
$$\dot{k} = f(k) - \delta k - c$$

**Euler equation (consumption):**
$$\dot{c} = \frac{1}{\sigma} \left( f'(k) - \delta - \rho \right) c$$

**Nullclines:**
- $\dot{k} = 0$: $c = f(k) - \delta k$ (hump-shaped in $k$)
- $\dot{c} = 0$: $f'(k) = \delta + \rho$, i.e., $k = k^*$ (vertical line)

**Steady state:** $k^* = \left(\frac{\alpha A}{\rho + \delta}\right)^{1/(1-\alpha)}$, $c^* = f(k^*) - \delta k^*$

**Transversality condition** selects the saddle path as the unique optimal trajectory.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | 0.3 | Capital share |
| $\delta$ | 0.05 | Depreciation rate |
| $\rho$ | 0.04 | Discount rate |
| $\sigma$ | 2.0 | CRRA coefficient |
| $k^*$ | 5.5843 | Steady-state capital |
| $c^*$ | 1.3961 | Steady-state consumption |

## Solution Method

**Linearization:** The Jacobian at the steady state has eigenvalues $\lambda_1 = -0.0710$ (stable) and $\lambda_2 = 0.1110$ (unstable). This confirms the steady state is a **saddle point**.

**Saddle path:** The stable manifold is traced using the eigenvector associated with $\lambda_1$. Near the steady state, the saddle path slope is approximately 0.1110.

**Integration:** Time paths computed via `scipy.integrate.solve_ivp` (RK45).

## Results

![Phase diagram with nullclines, vector field, and saddle path](figures/phase-diagram.png)
*Phase diagram with nullclines, vector field, and saddle path*

![Capital and consumption converge to steady state along the saddle path](figures/time-paths.png)
*Capital and consumption converge to steady state along the saddle path*

![Only trajectories starting on the saddle path converge to steady state](figures/four-regions.png)
*Only trajectories starting on the saddle path converge to steady state*

**Steady-State Values and Eigenvalues**

| Quantity    |   Value | Description                     |
|:------------|--------:|:--------------------------------|
| $k^*$       |  5.5843 | Steady-state capital            |
| $c^*$       |  1.3961 | Steady-state consumption        |
| $y^*$       |  1.6753 | Steady-state output             |
| $r^*$       |  0.04   | Net interest rate (= rho at ss) |
| $\lambda_1$ | -0.071  | Stable eigenvalue               |
| $\lambda_2$ |  0.111  | Unstable eigenvalue             |

## Economic Takeaway

Phase diagrams reveal the qualitative dynamics of the Ramsey model:

**Key insights:**
- The steady state is a **saddle point**: most trajectories diverge. Only the saddle path (stable manifold) converges — the transversality condition selects it.
- Above the saddle path, agents *over-consume*, depleting capital. Below it, they *over-save*, accumulating capital without bound.
- The $\dot{k}=0$ nullcline is the golden rule line — maximum sustainable consumption. The Ramsey steady state lies *below* the golden rule because agents are impatient ($\rho > 0$).
- The **speed of convergence** depends on $|\lambda_1|$: a half-life of $\ln(2)/|\lambda_1| \approx 9.8$ periods for capital to close half the gap to steady state.

## Reproduce

```bash
python run.py
```

## References

- Ramsey, F. (1928). "A Mathematical Theory of Saving." *Economic Journal*, 38(152).
- Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.
- Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 8.
