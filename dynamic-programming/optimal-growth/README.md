# Optimal Growth by Value Function Iteration

> Productive capital, the Ramsey transition, and a closed-form audit for VFI.

## Overview

This is the discrete-time Ramsey-Cass-Koopmans planner: one good, one factor (capital), and a representative agent who chooses consumption to maximize discounted utility. Compared with [cake eating](../cake-eating/), the only new ingredient is that the state is *productive*. Saving a unit of output as capital delivers $\alpha A k^{\alpha-1}$ extra units of output tomorrow rather than the gross return of one that disciplines the cake problem. That single change introduces diminishing returns, an interior steady state, and the trade-off between the impatience rate $1/\beta - 1$ and the marginal product of capital.

With log utility, Cobb-Douglas production, and full depreciation, the planner's problem has a closed form: the optimal saving rate is the constant $\alpha\beta$, the value function is affine in $\log k$, and the transition to the Ramsey steady state $k_{ss}=(\alpha\beta A)^{1/(1-\alpha)}$ is monotone. This is the only one-sector growth calibration where every numerical object has an exact analytical twin, which is what makes it the natural audit for a generic Bellman solver before risk, partial depreciation, labor, or equilibrium prices break the closed form. The same recursion reappears, with different state spaces, in the [RBC tutorial](../rbc/) once productivity shocks are added and in [Aiyagari](../aiyagari/) once the planner is replaced by a continuum of constrained households facing market-determined factor prices.

## Equations

Capital $k_t$ produces output $y_t = A k_t^{\alpha}$ with $A>0$ and
$\alpha\in(0,1)$. Capital fully depreciates each period, so the resource
constraint is

$$c_t + k_{t+1} \;=\; A k_t^{\alpha},
\qquad c_t > 0,\; k_{t+1} \ge 0.$$

The planner maximizes discounted log utility,

$$\sum_{t=0}^{\infty} \beta^{t} \log c_t,
\qquad \beta \in (0,1),$$

with state $k$ summarizing the entire future. The Bellman equation is

$$V(k) \;=\; \max_{0 < k' < A k^{\alpha}}
\{\, \log(A k^{\alpha}-k') + \beta\, V(k') \,\}.$$

Let $g(k)$ denote the optimal $k'$ and $c^{\ast}(k) = A k^{\alpha} - g(k)$ the
implied consumption. Differentiating inside the max and applying the envelope
theorem $V'(k) = u'(c^{\ast}(k))\, f'(k)$ delivers the **Euler equation**

$$u'(c_t) \;=\; \beta\, f'(k_{t+1})\, u'(c_{t+1}),
\qquad f'(k) = \alpha A k^{\alpha-1}.$$

The shadow value of capital today equals discounted shadow value tomorrow
*scaled by the gross return on capital*. The cake-eating Euler equation is the
$f'(k)\equiv 1$ special case.

For log utility and Cobb-Douglas production, conjecture $g(k) = s A k^{\alpha}$
with constant saving rate $s$. Substituting into the Euler equation gives
$s = \alpha\beta$, so

$$g(k) \;=\; \alpha\beta\, A\, k^{\alpha},
\qquad
c^{\ast}(k) \;=\; (1-\alpha\beta)\, A\, k^{\alpha}.$$

The value function is affine in $\log k$,

$$V(k) \;=\; E + B\, \log k,
\qquad
B \;=\; \frac{\alpha}{1-\alpha\beta},$$

with intercept

$$E \;=\; \frac{1}{1-\beta}\!\left[\,\log(A(1-\alpha\beta))
+ \frac{\beta\alpha}{1-\alpha\beta}\,\log(A\,\alpha\beta)\,\right].$$

The steady state solves $k = g(k)$, equivalently $\beta f'(k_{ss}) = 1$:

$$k_{ss} \;=\; (\alpha\beta A)^{1/(1-\alpha)},
\qquad c_{ss} \;=\; A k_{ss}^{\alpha} - k_{ss}.$$

The closed form depends on all three assumptions jointly. Drop log utility,
introduce partial depreciation, or replace the production function and the
Ramsey transition still exists, but $g$ and $V$ have to be solved numerically.
That generic case is exactly what VFI is for; the calibration here is the
sharpest available test of whether the solver gets it right.

## Model Setup

| Symbol | Value | Role |
|--------|-------|------|
| $\alpha$ | 0.3 | Capital share in $A k^{\alpha}$ |
| $A$ | 18.5 | Total factor productivity |
| $\beta$ | 0.9 | Discount factor; pins down impatience and the saving rate |
| $k_{ss}$ | 9.9519 | Closed-form steady-state capital $(\alpha\beta A)^{1/(1-\alpha)}$ |
| $c_{ss}$ | 26.9071 | Steady-state consumption $A k_{ss}^{\alpha} - k_{ss}$ |
| $k$ domain | $[0.01,\, 24.88]$ | Capital range; upper bound is $2.5\,k_{ss}$ |
| $N_k$ | 500 | Uniform state grid for $k$ |
| $N_{k'}$ | 500 | Inner choice grid for $k'$ at each Bellman update |
| Tolerance $\varepsilon$ | 1e-06 | Sup-norm convergence threshold |
| $T_{sim}$ | 50 | Simulation horizon |
| $k_0$ | $0.1\, k_{ss}\approx0.9952$ | Initial capital for the transition path |

## Solution Method

Define the Bellman operator on bounded continuous functions of capital,

$$(TV)(k) \;=\; \max_{0 < k' < A k^{\alpha}}\{\, \log(A k^{\alpha} - k') + \beta\, V(k') \,\}.$$

Blackwell's monotonicity and discounting conditions hold, so $T$ is a contraction with modulus $\beta$. Successive iterates satisfy $\|V_n - V\|_{\infty} \le \beta^{n}\|V_0 - V\|_{\infty}$, which fixes the convergence rate and the stopping rule. With $\beta=0.9$ the bound predicts roughly $\log(\varepsilon)/\log(\beta)$ iterations to reach tolerance $\varepsilon$.

Numerically, $V$ is tabulated on a uniform state grid for $k$. At each state, the maximizer is searched on a finer grid of candidate next-period capital values, and the continuation $V(k')$ is recovered by linear interpolation against the current iterate. Two implementation choices matter economically: (i) the upper end of the state grid sits well above $k_{ss}$ so that the policy converges to $g(k)<k$ before hitting the boundary, and (ii) the inner choice grid is at least as fine as the state grid, because policy errors propagate directly into the simulated transition. The closed-form policy is *not* used inside the loop; it is computed afterwards solely as a benchmark.

```text
Algorithm — Optimal-Growth VFI with continuous k', interpolated continuation
Input : capital grid {k_i}_{i=1..N_k}, choice grid size N_kp,
        primitives (A, alpha, beta), utility u(c) = log c, tolerance epsilon
Output: value V*(k_i), capital policy g(k_i)
  initialise V_0(k_i) = u(A k_i^alpha)             # eat-everything guess
  for n = 0, 1, 2, ... :
      for each state k_i :
          y_i    <- A * k_i^alpha
          kp_max <- min(y_i, k_max)
          kp     <- N_kp points uniform on [k_min, kp_max)
          c      <- y_i - kp                         # period consumption
          V_cont <- interp(V_n, kp)                  # off-grid continuation
          obj    <- log(c) + beta * V_cont
          V_{n+1}(k_i) <- max(obj)
          g(k_i)       <- argmax(obj)
      err <- max_i | V_{n+1}(k_i) - V_n(k_i) |
      stop when err < epsilon
```

With the calibration above the iteration converges in **143 steps** to a sup-norm residual of **9.32e-07**, consistent with the geometric bound. The Euler equation could also be solved in one pass by endogenous-grid points or a shooting method, but VFI is what generalizes to the stochastic and constrained problems later in the catalog.

## Results

The value function is increasing and concave because more capital relaxes the resource constraint while marginal product diminishes. With log utility and Cobb-Douglas production it is exactly affine in $\log k$, and the numerical curve sits on top of the closed-form curve over the whole economically relevant range. Outside the bottom decile of the grid the largest sup-norm gap is **1.91e-05**, which is essentially interpolation noise; the wider deviation near $k=0$ is the usual artifact of the log singularity on a uniform grid.

<img src="figures/value-function.png" alt="Numerical value function plotted against the closed-form $E + B\log k$" width="80%">

The economic content of the model lives in this picture. The policy crosses the $45^{\circ}$ line exactly at $k_{ss}$: below the steady state the planner accumulates ($k' > k$), above it the planner runs capital down ($k' < k$). The slope at the crossing is less than one, which is what makes the steady state stable and the transition monotone. Under log utility the saving rate is the constant $\alpha\beta = 0.27$ regardless of the level of capital; off log utility, $g$ would still cross the $45^{\circ}$ line at $k_{ss}$ but its curvature would shift with the intertemporal elasticity. The largest pointwise policy gap outside the bottom decile is **2.87e-02**, with a corresponding consumption-policy gap of **2.87e-02**.

<img src="figures/policy-function.png" alt="Capital policy $g(k)$ versus the closed-form rule $\alpha\beta A k^{\alpha}$" width="80%">

Iterating $k_{t+1}=g(k_t)$ from $k_0 = 0.1\,k_{ss}$ traces the Ramsey transition. Capital rises quickly at first because marginal product is high when capital is scarce, then convergence slows as $f'(k)$ falls toward $1/\beta$. Consumption inherits the same hump-free monotonicity here because the saving rate is constant; with non-log utility the consumption path could overshoot or undershoot $c_{ss}$ even when capital does not. The numerical and closed-form trajectories are visually indistinguishable, with sup-norm capital-path error **2.39e-02**.

<img src="figures/simulation.png" alt="Capital and consumption transitions starting from $k_0=1.00$" width="80%">

The audit table reports both objects at eight representative capital states. Value-function residuals are uniformly tight; policy residuals are larger but smooth in $k$ and never reverse sign in a way that would suggest a spurious local optimum. The relevant diagnostic for downstream simulations is the policy column, since policies are what get forward-iterated.

**Numerical vs closed-form solution at selected capital states**

|      k |   V numerical |   V closed form |   V error |   k' numerical |   k' closed form |   k' error |
|-------:|--------------:|----------------:|----------:|---------------:|-----------------:|-----------:|
|  2.502 |       32.3565 |         32.3565 | -1.53e-05 |         6.5967 |           6.5769 |    0.0198  |
|  5.692 |       32.6943 |         32.6943 | -1.53e-05 |         8.4329 |           8.416  |    0.0168  |
|  8.881 |       32.8771 |         32.8771 | -1.53e-05 |         9.629  |           9.6179 |    0.0111  |
| 12.071 |       33.0032 |         33.0032 | -1.53e-05 |        10.5261 |          10.5453 |   -0.0192  |
| 15.261 |       33.0996 |         33.0996 | -1.53e-05 |        11.3235 |          11.3138 |    0.00972 |
| 18.451 |       33.1776 |         33.1776 | -1.53e-05 |        11.9715 |          11.9767 |   -0.00528 |
| 21.64  |       33.2431 |         33.2431 | -1.14e-05 |        12.5695 |          12.5636 |    0.00592 |
| 24.88  |       33.3004 |         33.3005 | -1.53e-05 |        13.1178 |          13.1006 |    0.0172  |

## Takeaway

Optimal growth is the cake-eating Bellman equation with one extra ingredient: the resource constraint runs through a production function, so saving today delivers $f'(k_{t+1})$ extra units of consumption tomorrow. The Euler equation absorbs that change cleanly, and under log utility, Cobb-Douglas production, and full depreciation it collapses to a constant saving rate $\alpha\beta = 0.27$ and a closed-form transition toward $k_{ss} = 9.95$. VFI recovers that policy to interpolation accuracy, which is the right calibration to take into the stochastic, partially depreciated, or constrained settings later in the catalog, where Euler residuals and equilibrium consistency replace the closed form as the only available checks.

## References

- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 2 & 4.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.
- Ramsey, F. P. (1928). A Mathematical Theory of Saving. *Economic Journal*, 38(152), 543-559.
- Cass, D. (1965). Optimum Growth in an Aggregative Model of Capital Accumulation. *Review of Economic Studies*, 32(3), 233-240.
- Koopmans, T. C. (1965). On the Concept of Optimal Economic Growth. In *The Econometric Approach to Development Planning*. North-Holland.
