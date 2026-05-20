# Optimal Growth by Value Function Iteration

## Overview

A planner allocates output between consumption today and capital tomorrow. Capital produces future output, so saving has a return that falls with $k$. The economy settles where impatience balances the marginal product of capital.

The object is the policy rule $g(k)$ for next-period capital. Given $g(k)$, consumption is $c^{\ast}(k)=A k^{\alpha}-g(k)$, where $A>0$ is total factor productivity and $\alpha\in(0,1)$ is the capital share.

The log Cobb-Douglas case has the closed-form saving rate $\alpha\beta$, where $\beta\in(0,1)$ is the discount factor. Value function iteration solves the Bellman equation on a grid. Here the closed form audits the computed value and policy point by point.

## Equations

Capital $k_t$ produces output $y_t = A k_t^{\alpha}$ with $A>0$ and
$\alpha\in(0,1)$. Capital fully depreciates each period, so the resource
constraint is

$$c_t + k_{t+1} = A k_t^{\alpha},
\qquad c_t > 0, k_{t+1} \ge 0.$$

The planner maximizes discounted log utility,

$$\sum_{t=0}^{\infty} \beta^{t} \log c_t,
\qquad \beta \in (0,1),$$

with state $k$ summarizing the entire future. The Bellman equation is

$$V(k) = \max_{0 < k' < A k^{\alpha}}
\{\, \log(A k^{\alpha}-k') + \beta\, V(k') \,\}.$$

Let $g(k)$ denote the optimal $k'$ and $c^{\ast}(k) = A k^{\alpha} - g(k)$ the
implied consumption. The first-order and envelope conditions deliver the
Euler equation

$$u'(c_t) = \beta\, f'(k_{t+1})\, u'(c_{t+1}),
\qquad f'(k) = \alpha A k^{\alpha-1}.$$

For log utility and Cobb-Douglas production, conjecture $g(k) = s A k^{\alpha}$
with constant saving rate $s$. Substituting into the Euler equation gives
$s = \alpha\beta$, so

$$g(k) = \alpha\beta\, A\, k^{\alpha},
\qquad
c^{\ast}(k) = (1-\alpha\beta)\, A\, k^{\alpha}.$$

The value function is affine in $\log k$,

$$V(k) = E + B\, \log k,
\qquad
B = \frac{\alpha}{1-\alpha\beta},$$

with intercept

$$E = \frac{1}{1-\beta}\left[\,\log(A(1-\alpha\beta))
+ \frac{\beta\alpha}{1-\alpha\beta}\,\log(A\,\alpha\beta)\,\right].$$

The steady state solves $k = g(k)$, equivalently $\beta f'(k_{ss}) = 1$:

$$k_{ss} = (\alpha\beta A)^{1/(1-\alpha)},
\qquad c_{ss} = A k_{ss}^{\alpha} - k_{ss}.$$

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

$$(TV)(k) = \max_{0 < k' < A k^{\alpha}}\{\, \log(A k^{\alpha} - k') + \beta\, V(k') \,\}.$$

VFI starts from an initial value on the capital grid. At each $k_i$, the code searches over feasible $k'$ values. The feasible range is $k' \in [k_{min},\, A k_i^{\alpha})$ where $k_{min}=0.01$ is the lower bound on next-period capital. It chooses the $k'$ with the highest current utility plus interpolated continuation value. The loop stops when the sup-norm change in $V$ is below $\varepsilon$.

```text
Algorithm: Optimal-growth VFI with continuous k'
Input : capital grid {k_i}_{i=1..N_k}, choice grid size N_{k'},
        k_min (lower bound on k'; = 0.01), primitives (A, alpha, beta),
        utility u(c) = log c, tolerance epsilon
Output: value V*(k_i), capital policy g(k_i)
  initialise V_0(k_i) = u(A k_i^alpha)             # eat-everything guess
  for n = 0, 1, 2, ... :
      for each state k_i :
          y_i    <- A * k_i^alpha
          kp_max <- min(0.9999 * y_i, k_max)        # 0.9999 keeps c > 0 at the top node
          kp     <- N_{k'} points uniform on [k_min, kp_max]
          c      <- y_i - kp                         # period consumption
          V_cont <- interp(V_n, kp)                  # off-grid continuation
          obj    <- log(c) + beta * V_cont
          V_{n+1}(k_i) <- max(obj)
          g(k_i)       <- argmax(obj)
      err <- max_i | V_{n+1}(k_i) - V_n(k_i) |
      stop when err < epsilon
```

The iteration converges in **143 steps** with sup-norm residual **9.32e-07**. The closed-form rule is computed only after VFI finishes.

## Results

The value function rises and bends because capital has diminishing returns. The numerical curve matches $E+B\log k$ except near the lowest grid points. Outside the bottom decile, the largest value gap is **1.91e-05**.

<img src="figures/value-function.png" alt="Numerical value function plotted against the closed-form $E + B\log k$" width="80%">

The policy crosses the $45^{\circ}$ line at $k_{ss}$. Below $k_{ss}$, the planner accumulates capital. Above $k_{ss}$, the planner runs capital down. The log case saves the constant share $\alpha\beta = 0.27$ of output. The largest policy gap outside the bottom decile is **2.87e-02**.

<img src="figures/policy-function.png" alt="Capital policy $g(k)$ versus the closed-form rule $\alpha\beta A k^{\alpha}$" width="80%">

Starting from $0.1\,k_{ss}$, capital rises toward the steady state. It rises fastest when capital is scarce. Consumption also rises because the saving share is constant. The maximum capital-path error is **2.39e-02**.

<img src="figures/simulation.png" alt="Capital and consumption transitions starting from $k_0=0.9952$" width="80%">

The table checks eight representative capital states. Value errors are tiny at each selected state. Policy errors are larger because $k'$ is chosen on a finite grid.

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

The one-capital growth problem makes saving productive. In the log Cobb-Douglas case, the exact policy saves $\alpha\beta = 0.27$ of output. VFI recovers that rule to grid accuracy. The example shows how to audit a Bellman solver when an exact benchmark exists.

## References

- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 2 & 4.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.
- Ramsey, F. P. (1928). A Mathematical Theory of Saving. *Economic Journal*, 38(152), 543-559.
- Cass, D. (1965). Optimum Growth in an Aggregative Model of Capital Accumulation. *Review of Economic Studies*, 32(3), 233-240.
- Koopmans, T. C. (1965). On the Concept of Optimal Economic Growth. In *The Econometric Approach to Development Planning*. North-Holland.
