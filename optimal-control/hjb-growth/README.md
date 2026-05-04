# HJB Growth and Capital Accumulation

> A Ramsey planner chooses consumption and capital accumulation in continuous time; the HJB is solved by implicit upwind finite differences.

## Overview

This is the continuous-time version of the planner's growth problem. The state is aggregate capital $k$, the control is consumption $c$, and the economic question is how much output should be consumed today rather than invested for future production.

The Hamilton-Jacobi-Bellman equation gives the value of capital in units of lifetime utility. Once the marginal value $V'(k)$ is known, the consumption choice follows from the first-order condition. The numerical problem is therefore not a search over consumption; it is a problem of computing the right derivative of the value function along the capital drift.

The tutorial uses the implicit upwind scheme from continuous-time macro. The same economic dynamics also appear in the neighboring [Ramsey phase-diagram](../phase-diagrams/) and [Ramsey shooting](../ramsey-growth/) tutorials; here the focus is the HJB representation and the finite-difference policy calculation.

## Equations

The planner solves

$$
\max_{\{c(t)\}_{t \geq 0}}
\int_0^\infty e^{-\rho t} u(c(t))\,dt
\quad\text{s.t.}\quad
\dot{k}(t)=f(k(t))-\delta k(t)-c(t),
$$

where $f(k)=Ak^\alpha$, $u(c)=c^{1-\sigma}/(1-\sigma)$ for
$\sigma \neq 1$, and $\rho$ is the continuous-time discount rate.

The HJB equation is

$$
\rho V(k)=\max_{c>0}
\left[u(c)+V'(k)\left(f(k)-\delta k-c\right)\right].
$$

The first-order condition is

$$
u'(c^{*}(k))=V'(k)
\quad\Longrightarrow\quad
c^{*}(k)=\left(V'(k)\right)^{-1/\sigma}.
$$

Substituting this policy into the drift

$$
s(k)=\dot{k}=f(k)-\delta k-c^{*}(k)
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

The calibration is intentionally small: one capital state, Cobb-Douglas technology, CRRA utility, and no shocks. The baseline HJB grid is used for the reported policy functions. A finer HJB grid is solved only as a same-model reference for the figures; it is not a different economic model.

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\rho$   | 0.05 | Discount rate |
| $\sigma$ | 2.0 | CRRA coefficient |
| $\alpha$ | 0.36 | Capital share |
| $\delta$ | 0.05 | Depreciation rate |
| $A$       | 1.0 | TFP |
| Baseline HJB grid | 500 points | $k \in [0.1, 14.80]$ |
| Fine-grid reference | 1600 points | Same capital interval |
| Discrete-time check | 200 points | Same capital interval |
| $k_{ss}$ | 7.3998 | Steady-state capital |
| $c_{ss}$ | 1.6855 | Steady-state consumption |
| $y_{ss}$ | 2.0555 | Steady-state output |

## Solution Method

The HJB is solved by implicit iteration in pseudo-time. Given a value guess, the algorithm computes two candidate marginal values, turns each into a consumption rule, and then chooses the derivative from the side that is upwind relative to the implied capital drift. The derivative and the policy are therefore chosen together, which is the main numerical discipline in the continuous-time formulation.

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

The linear solve is sparse and tridiagonal. The large pseudo-time step $\Delta=1000$ is a numerical device, not an economic period length; it stabilizes the fixed-point update while leaving the continuous-time HJB as the target equation.

The baseline continuous-time HJB converged in **16 iterations** (change = 5.34e-07). The fine-grid HJB reference converged in **14 iterations** (change = 6.94e-07). A coarse discrete-time VFI check, included only for orientation, converged in **243 iterations** (error = 9.87e-07).

## Results

The value function is increasing and concave because extra capital raises future consumption possibilities but at a diminishing marginal product. The baseline HJB solution lies almost on top of the fine-grid HJB reference, while the discrete-time VFI line is best read as a separate Bellman-equation check rather than the target continuous-time object.

<img src="figures/value-function.png" alt="Value function with baseline HJB, fine-grid HJB reference, and discrete-time VFI check" width="80%">

The consumption rule comes directly from marginal value: $c(k)=(V'(k))^{-1/\sigma}$. Below the steady state, consumption stays below net output so the planner accumulates capital. Above it, consumption exceeds net output and capital is run down. The fine-grid reference confirms that the baseline grid is already resolving the policy shape.

<img src="figures/consumption-policy.png" alt="Consumption policy with fine-grid HJB reference and net output" width="80%">

The drift $s(k)=\dot{k}$ is the object that determines both economic transitions and the upwind derivative. Positive drift means the economy moves toward higher capital; negative drift means it moves back down. The zero crossing is the Ramsey steady state, and the fine-grid line shows that the baseline grid locates it accurately.

<img src="figures/savings-policy.png" alt="Capital drift with accumulation below steady state and decumulation above it" width="80%">

Integrating the policy-implied law of motion gives the familiar convergence picture. Low-capital economies invest because marginal product is high; high-capital economies consume more than net output and move down. The single-state planner has a unique stable path back to $k_{ss}$.

<img src="figures/transition-dynamics.png" alt="Transition dynamics k(t) from different initial conditions converging to steady state" width="80%">

The steady state has a closed-form target, so it is a useful check on the finite-difference solution. The baseline grid locates the zero drift within one grid step, and the finer grid tightens that comparison without changing the economic calculation.

**Steady-State Values and HJB Diagnostics**

| Variable                              | Analytical   |   Baseline HJB |   Fine-grid HJB |
|:--------------------------------------|:-------------|---------------:|----------------:|
| $k_{ss}$ (capital)                    | 7.3998       |       7.4057   |        7.3993   |
| $c_{ss}$ (consumption)                | 1.6855       |       1.6858   |        1.6855   |
| $y_{ss}$ (output)                     | 2.0555       |       2.0561   |        2.0555   |
| $i_{ss} = \delta k_{ss}$ (investment) | 0.3700       |       0.3703   |        0.37     |
| $s = i/y$ (saving rate)               | 0.1800       |       0.1801   |        0.18     |
| $f'(k_{ss})$ (MPK)                    | 0.1000       |       0.0999   |        0.1      |
| HJB iterations                        | --           |      16        |       14        |
| HJB residual                          | --           |       5.34e-07 |        6.94e-07 |

## Takeaway

The economic content is the Ramsey accumulation logic: invest when the marginal product of capital is high, consume more when capital is abundant, and converge to the point where $f'(k)=\rho+\delta$. The computational content is that the HJB turns this logic into a derivative problem. Once $V'(k)$ is approximated from the correct side, consumption follows from the FOC and the remaining step is a sparse linear solve.

This is why the upwind choice matters. It is not a cosmetic numerical detail; it encodes the direction of capital movement. That same idea becomes central in continuous-time heterogeneous-agent models, where the HJB policy and the forward equation for the distribution have to use compatible drift directions.

## References

- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies*, 89(1), 45-86.
- Moll, B. (2022). "Lecture notes on continuous-time methods in macroeconomics." https://benjaminmoll.com/lectures/
- Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition.
