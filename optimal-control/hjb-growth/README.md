# Ramsey Capital Accumulation by HJB Upwinding

> A Ramsey planner allocates output between consumption and investment. An implicit upwind HJB computes the shadow value of capital.

## Overview

A Ramsey planner inherits aggregate capital $k$. Output can be consumed today or invested for future production. Scarce capital raises investment value. Abundant capital makes current consumption cheaper.

The object is the consumption policy $c(k)$ and the capital drift $\dot{k}$. Together they describe how the economy returns to its steady state.

The HJB gives the value of starting from each capital stock. Its derivative is the shadow value that pins down consumption. A finite-difference scheme is needed because the nonlinear HJB has no closed-form policy on the grid. Upwinding chooses the derivative side using the policy-implied drift.

## Equations

The planner solves

$$
\max_{\{c(t)\}_{t \geq 0}}
\int_0^\infty e^{-\rho t} u(c(t))\,dt
\quad\text{s.t.}\quad
\dot{k}(t)=f(k(t))-\delta k(t)-c(t),
$$

Here $f(k)=Ak^\alpha$ and $u(c)=c^{1-\sigma}/(1-\sigma)$ for
$\sigma \neq 1$. The parameter $\rho$ is the continuous-time discount rate.

The HJB equation is

$$
\rho V(k)=\max_{c>0}
\left[u(c)+V'(k)\left(f(k)-\delta k-c\right)\right].
$$

The first-order condition is

$$
u'(c^{\ast}(k))=V'(k)
\quad\Longrightarrow\quad
c^{\ast}(k)=\left(V'(k)\right)^{-1/\sigma}.
$$

Substituting this policy into the drift

$$
s(k)=\dot{k}=f(k)-\delta k-c^{\ast}(k)
$$

leaves a nonlinear equation in $V$. On the grid $k_1,\ldots,k_N$ with spacing
$\Delta k$, the upwind derivative uses the direction implied by the drift:

$$
D_i V =
\begin{cases}
(V_{i+1}-V_i)/\Delta k, & s_i>0,\\
(V_i-V_{i-1})/\Delta k, & s_i<0,\\
\left(f(k_i)-\delta k_i\right)^{-\sigma}, & s_i=0.
\end{cases}
$$

The steady state satisfies $s(k_{ss})=0$ and the Euler condition
$f'(k_{ss})=\rho+\delta$, so

$$
k_{ss}=\left(\frac{\alpha A}{\rho+\delta}\right)^{1/(1-\alpha)}.
$$

## Model Setup

The calibration uses one aggregate capital state, Cobb-Douglas production, CRRA utility, and no shocks. The grid spans low and high capital around the Ramsey steady state.

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\rho$   | 0.05 | Discount rate |
| $\sigma$ | 2.0 | CRRA coefficient |
| $\alpha$ | 0.36 | Capital share |
| $\delta$ | 0.05 | Depreciation rate |
| $A$       | 1.0 | TFP |
| Baseline HJB grid | 500 points | $k \in [0.1, 14.80]$ |
| $k_{ss}$ | 7.3998 | Steady-state capital |
| $c_{ss}$ | 1.6855 | Steady-state consumption |
| $y_{ss}$ | 2.0555 | Steady-state output |

## Solution Method

Start from a guessed value function on the capital grid. The update compares forward and backward slopes, maps each slope into consumption, and chooses the slope consistent with capital drift. That choice aligns the derivative with the economic law of motion.

```text
Inputs: grid {k_i}, primitives (rho, sigma, alpha, delta, A), tolerance eps
Initialize V^0_i = u(f(k_i)) / rho
For n = 0, 1, ... until ||V^{n+1} - V^n||_infinity < eps:
    1. Form forward and backward slopes D^+ V^n_i and D^- V^n_i.
    2. Use the FOC to compute candidate consumption:
       c^+_i = (D^+ V^n_i)^(-1/sigma), c^-_i = (D^- V^n_i)^(-1/sigma).
    3. Compute candidate drifts s^+_i = f(k_i) - delta k_i - c^+_i
       and s^-_i = f(k_i) - delta k_i - c^-_i.
    4. Choose the upwind derivative D_i V^n using the sign of the drift;
       at s_i = 0 use the steady-state marginal utility.
    5. Set c^n_i = (D_i V^n)^(-1/sigma) and build the tridiagonal
       generator G^n from the positive and negative drift parts.
    6. Solve the implicit linear system
       [(1/Delta + rho) I - G^n] V^{n+1}
       = u(c^n) + V^n / Delta.
Output: value V, consumption policy c(k), and drift s(k)=dot{k}
```

The implicit step solves one sparse tridiagonal linear system. The large pseudo-time step $\Delta=1000$ is numerical, not a model period. It stabilizes the fixed-point update while keeping the continuous-time HJB as the target.

The HJB converged in **16 iterations** with final change 5.34e-07.

## Results

The value function is increasing and concave. Extra capital raises future consumption, but diminishing marginal product lowers the marginal gain.

<img src="figures/value-function.png" alt="Value function from the upwind HJB" width="80%">

The consumption rule comes from marginal value. Below the steady state, consumption stays below net output, so capital rises. Above it, consumption exceeds net output, so capital falls.

<img src="figures/consumption-policy.png" alt="Consumption policy and net output" width="80%">

The drift $s(k)=\dot{k}$ drives transitions and selects the upwind derivative. Positive drift points to capital accumulation. Negative drift points to decumulation. The zero crossing is the Ramsey steady state.

<img src="figures/savings-policy.png" alt="Capital drift with accumulation below steady state and decumulation above it" width="80%">

The policy-implied law of motion sends each initial capital stock toward $k_{ss}$. Low-capital economies invest because marginal product is high. High-capital economies consume more than net output and move down.

<img src="figures/transition-dynamics.png" alt="Transition dynamics k(t) from different initial conditions converging to steady state" width="80%">

The closed-form steady state checks the finite-difference solution. The grid locates zero drift within one step.

**Steady-State Values and HJB Diagnostics**

| Variable                              | Analytical   |   Baseline HJB |
|:--------------------------------------|:-------------|---------------:|
| $k_{ss}$ (capital)                    | 7.3998       |       7.4057   |
| $c_{ss}$ (consumption)                | 1.6855       |       1.6858   |
| $y_{ss}$ (output)                     | 2.0555       |       2.0561   |
| $i_{ss} = \delta k_{ss}$ (investment) | 0.3700       |       0.3703   |
| $s = i/y$ (saving rate)               | 0.1800       |       0.1801   |
| $f'(k_{ss})$ (MPK)                    | 0.1000       |       0.0999   |
| HJB iterations                        | --           |      16        |
| HJB residual                          | --           |       5.34e-07 |

## Takeaway

The computed policy follows the Ramsey Euler logic. Investment is high when capital has high marginal product. Consumption rises once capital is abundant. The path converges to $f'(k)=\rho+\delta$.

The HJB turns this logic into a value derivative. Upwinding uses the direction of capital movement to choose the derivative. After that choice, the update is a sparse linear solve.

## References

- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies*, 89(1), 45-86.
- Moll, B. (2022). "Lecture notes on continuous-time methods in macroeconomics." https://benjaminmoll.com/lectures/
- Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition.
