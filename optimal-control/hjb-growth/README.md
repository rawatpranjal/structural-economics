# Ramsey Capital Accumulation by HJB Upwinding

> A Ramsey planner allocates output between consumption and investment in continuous time; an implicit upwind HJB computes the shadow value of capital.

## Overview

Consider a planner who inherits aggregate capital $k$. Output can be consumed now or reinvested, and the marginal product of capital changes along the path. When capital is scarce, investment is valuable. When capital is abundant, the planner can afford more current consumption. The tutorial computes this Ramsey transition in continuous time.

The Hamilton-Jacobi-Bellman equation records the lifetime value of starting with each capital stock. Its derivative, $V'(k)$, is the shadow value of one more unit of capital, and the first-order condition maps that shadow value directly into consumption. Computation is needed because the nonlinear HJB does not give a closed-form policy on the capital grid.

Implicit upwind finite differences approximate $V'(k)$ from the side consistent with the policy-implied capital drift. That is why the method matters for the economics: the derivative choice and the direction of capital movement have to agree. The same Ramsey dynamics also appear in the neighboring [Ramsey phase-diagram](../phase-diagrams/) and [Ramsey shooting](../ramsey-growth/) tutorials; here the HJB representation makes the planner's shadow value explicit.

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

The model keeps the economics deliberately clean: one aggregate capital state, Cobb-Douglas production, CRRA utility, and no shocks. The baseline HJB grid produces the reported policies. A finer HJB grid is solved only as a same-model reference for the figures; it is not a different economic environment.

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

Start from a guessed value function on the capital grid. The update compares forward and backward slopes, converts each slope into a consumption rule, and then uses the slope whose direction matches the implied motion of capital. That upwind choice keeps the finite-difference derivative aligned with the economic law of motion.

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

The implicit step then solves one sparse tridiagonal linear system. The large pseudo-time step $\Delta=1000$ is a numerical device, not an economic period length; it stabilizes the fixed-point update while leaving the continuous-time HJB as the target equation.

The baseline continuous-time HJB converged in **16 iterations** (change = 5.34e-07). The fine-grid HJB reference converged in **14 iterations** (change = 6.94e-07). A coarse discrete-time VFI check, used only for orientation, converged in **243 iterations** (error = 9.87e-07).

## Results

The value function is increasing and concave because extra capital raises future consumption possibilities but at a diminishing marginal product. The baseline HJB solution lies almost on top of the fine-grid HJB reference, while the discrete-time VFI line is best read as a separate Bellman-equation check rather than the target continuous-time object.

<img src="figures/value-function.png" alt="Value function with baseline HJB, fine-grid HJB reference, and discrete-time VFI check" width="80%">

The consumption rule comes directly from marginal value: $c(k)=(V'(k))^{-1/\sigma}$. Below the steady state, consumption stays below net output so the planner accumulates capital. Above it, consumption exceeds net output and capital is run down. The fine-grid reference confirms that the baseline grid is already resolving the policy shape.

<img src="figures/consumption-policy.png" alt="Consumption policy with fine-grid HJB reference and net output" width="80%">

The drift $s(k)=\dot{k}$ determines both economic transitions and the upwind derivative. Positive drift means the economy moves toward higher capital; negative drift means it moves back down. The zero crossing is the Ramsey steady state, and the fine-grid line shows that the baseline grid locates it accurately.

<img src="figures/savings-policy.png" alt="Capital drift with accumulation below steady state and decumulation above it" width="80%">

Integrating the policy-implied law of motion produces the standard convergence picture. Low-capital economies invest because marginal product is high; high-capital economies consume more than net output and move down. The single-state planner has a unique stable path back to $k_{ss}$.

<img src="figures/transition-dynamics.png" alt="Transition dynamics k(t) from different initial conditions converging to steady state" width="80%">

The steady state has a closed-form target, which provides a check on the finite-difference solution. The baseline grid locates the zero drift within one grid step, and the finer grid tightens that comparison without changing the economic calculation.

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

The Ramsey logic is visible in the computed policy: invest when the marginal product of capital is high, consume more when capital is abundant, and converge to the point where $f'(k)=\rho+\delta$. The HJB makes that logic operational through the marginal value $V'(k)$. Once the derivative is approximated from the correct side, consumption follows from the FOC and the remaining update is a sparse linear solve.

Upwinding is an economic consistency check as well as a numerical choice. It uses the direction of capital movement to pick the derivative. The same idea becomes central in continuous-time heterogeneous-agent models, where the HJB policy and the forward equation for the distribution must use compatible drift directions.

## References

- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies*, 89(1), 45-86.
- Moll, B. (2022). "Lecture notes on continuous-time methods in macroeconomics." https://benjaminmoll.com/lectures/
- Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition.
