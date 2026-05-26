# Optimal Growth by Value Function Iteration

## Overview

Ramsey (1928) posed the infinite-horizon saving problem but left it disconnected from the neoclassical production framework. Solow (1956) modeled capital accumulation but fixed the saving rate exogenously. Cass (1965) and Koopmans (1965) merged the two: they embedded Ramsey's intertemporal consumer problem inside a neoclassical economy, making the saving rate endogenous and the steady state Pareto efficient.

The object of interest here is the *policy rule* mapping today's capital stock to next-period capital. Consumption falls out as the residual after that saving choice, given Cobb-Douglas production.

The log Cobb-Douglas case has a closed-form saving rate equal to the product of the capital share and the discount factor. Value function iteration solves the Bellman equation on a grid. The closed form then audits the computed value and policy point by point.

## Read before

- [Cake-eating problem](../cake-eating/README.md)
- [Consumption-savings under income risk](../consumption-savings/README.md)
- [Aiyagari saving and capital-market clearing](../aiyagari/README.md)

## Equations

Capital $`k_t`$ produces output $`y_t = A k_t^{\alpha}`$ with $`A>0`$ and
$`\alpha\in(0,1)`$. Capital fully depreciates each period, so the resource
constraint is

```math
c_t + k_{t+1} = A k_t^{\alpha},
\qquad c_t > 0, k_{t+1} \ge 0.
```

The planner maximizes discounted log utility,

```math
\sum_{t=0}^{\infty} \beta^{t} \log c_t,
\qquad \beta \in (0,1),
```

with state $`k`$ summarizing the entire future. The *Bellman equation* is

```math
V(k) = \max_{0 < k' < A k^{\alpha}}
\lbrace \log(A k^{\alpha}-k') + \beta V(k') \rbrace.
```

Let $`g(k)`$ denote the optimal $`k'`$ and $`c^{\ast}(k) = A k^{\alpha} - g(k)`$ the
implied consumption. The first-order and envelope conditions deliver the
Euler equation

```math
u'(c_t) = \beta f'(k_{t+1})  u'(c_{t+1}),
\qquad f'(k) = \alpha A k^{\alpha-1}.
```

For log utility and Cobb-Douglas production, conjecture $`g(k) = s A k^{\alpha}`$
with constant saving rate $`s`$. Substituting into the Euler equation gives
$`s = \alpha\beta`$, so

```math
g(k) = \alpha\beta A k^{\alpha},
\qquad
c^{\ast}(k) = (1-\alpha\beta)  A k^{\alpha}.
```

The value function is affine in $`\log k`$,

```math
V(k) = E + B \log k,
\qquad
B = \frac{\alpha}{1-\alpha\beta},
```

with intercept

```math
E = \frac{1}{1-\beta}\left[\log(A(1-\alpha\beta)) +
\frac{\beta\alpha}{1-\alpha\beta} \log(A\alpha\beta) \right].
```

The steady state solves $`k = g(k)`$, equivalently $`\beta f'(k_{ss}) = 1`$:

```math
k_{ss} = (\alpha\beta A)^{1/(1-\alpha)},
\qquad c_{ss} = A k_{ss}^{\alpha} - k_{ss}.
```

## Worked Numerical Example

Starting from the low-capital initial condition $`k_0 = 0.9952`$ (one-tenth of $`k_{ss}`$), trace one step of the *closed-form policy* to see how much capital the planner saves and how much is consumed.

Output at $`k_0`$ is

```math
y_0 = A k_0^{\alpha} = 18.5 \times 0.9952^{0.3} = 18.4733.
```

The closed-form saving rate is $`\alpha\beta = (0.3)(0.9) = 0.27`$, so next-period capital is

```math
k_1 = \alpha\beta \, A k_0^{\alpha} = 0.27 \times 18.4733 = \boxed{4.9878}.
```

Period-0 consumption follows as the residual,

```math
c_0 = (1 - \alpha\beta) A k_0^{\alpha} = 0.73 \times 18.4733 = 13.4855.
```

The steady state verifies the same formula: substituting $`k_{ss} = (\alpha\beta A)^{1/(1-\alpha)} = 4.995^{1/0.7} = 9.9519`$ gives $`k_1 = \alpha\beta \cdot A k_{ss}^{\alpha} = k_{ss}`$, confirming it is a fixed point.

Starting well below $`k_{ss}`$, the planner saves 27 percent of output regardless of the capital level. The log Cobb-Douglas case delivers a constant saving rate. The single step already moves capital from 0.9952 toward 4.99 because output is high relative to capital when $`k`$ is low and the Cobb-Douglas exponent $`\alpha = 0.3`$ moderates diminishing returns.

## Model Setup

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Capital share $`\alpha`$ | 0.3 | Total factor productivity $`A`$ | 18.5 |
| Discount factor $`\beta`$ | 0.9 | Steady-state capital $`k_{ss}`$ | 9.9519 |
| Steady-state consumption $`c_{ss}`$ | 26.9071 | Capital domain $`[k_{min}, k_{max}]`$ | $`[0.01, 24.88]`$ |
| State grid $`N_k`$ | 500 pts uniform | Choice grid $`N_{k'}`$ | 500 pts uniform |
| VFI tolerance $`\varepsilon`$ (sup-norm) | 1e-06 | Simulation horizon $`T_{sim}`$ | 50 periods |
| Initial capital $`k_0`$ | $`0.1 k_{ss} \approx 0.9952`$ | | |

## Solution Method

The *Bellman operator* maps bounded continuous functions of capital to themselves. At each grid point $`k_i`$, the code searches over feasible $`k'`$ values and picks the one that maximizes current utility plus the interpolated continuation value. The loop stops when the sup-norm change in $`V`$ is below tolerance.

```
              primitives (beta, alpha, A), k grid
                              |
                              v
              +---------- VFI loop ----------+
              |                              |
              |   V_k  -->  [ Bellman op ]  -->  V_{k+1}
              |                              |
              +------ err >= tol: repeat ----+
                              |
                          err < tol
                              v
                       V*(k), g(k) = k'*(k)
```

```python
# VFI: iterate until sup-norm change in V is below tolerance.
def solve_vfi(k_grid, n_kprime, k_min, k_max, A, alpha, beta, tol):
    v = np.log(A * k_grid ** alpha)       # eat-everything initial guess

    while True:
        v_new = np.zeros_like(v)
        g = np.zeros_like(v)
        for i, k in enumerate(k_grid):
            y = A * k ** alpha                           # output at state k
            kp = np.linspace(k_min, min(0.9999 * y, k_max), n_kprime)
            c = y - kp                                   # c = A k^alpha - k'
            obj = np.log(c) + beta * np.interp(kp, k_grid, v)   # u(c) + beta * V(k')
            best = np.argmax(obj)
            v_new[i] = obj[best]
            g[i] = kp[best]                              # g(k) = argmax_{k'} obj

        if np.max(np.abs(v_new - v)) < tol:
            return v_new, g
        v = v_new
```

The iteration converges in 143 steps with sup-norm residual 9.32e-07. The closed-form rule is computed only after VFI finishes and serves purely as an audit.

## Results

The value function rises and bends because capital has diminishing returns. The numerical curve matches $`E+B\log k`$ except near the lowest grid points. The policy crosses the $`45^{\circ}`$ line at $`k_{ss}`$. Below $`k_{ss}`$, the planner accumulates capital. Above $`k_{ss}`$, the planner runs capital down.

![Value function and capital policy versus closed-form benchmarks](figures/vf-policy.png)

Starting from $`0.1k_{ss}`$, capital rises toward the steady state fastest when capital is scarce. Consumption also rises because the saving share is constant. The VFI sup-norm residual falls geometrically on a log scale, confirming the Bellman operator is a *contraction*.

![Capital transition path and VFI sup-norm convergence](figures/simulation-convergence.png)

The table checks eight representative capital states. Value errors are tiny at each selected state. Policy errors are larger because $`k'`$ is chosen on a finite grid.

### Numerical vs closed-form solution at selected capital states

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

*Endogenizing* the saving rate was the Cass-Koopmans contribution. Ramsey had the optimization. Solow had the production structure. Cass and Koopmans connected them and showed the steady state is efficient, not a planning accident. The quantitative surprise was that the optimal saving rate in the log Cobb-Douglas case is simply the product of the capital share and the discount factor, a formula any undergraduate can verify but that Solow's exogenous rate could not deliver. The framework became the backbone of modern representative-agent macroeconomics, from real business cycle theory through DSGE.

## See also

- [Consumption-savings under income risk](../consumption-savings/README.md)
- [Aiyagari saving and capital-market clearing](../aiyagari/README.md)
- [Real business cycle model](../rbc/README.md)

## References

- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 2 & 4.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.
- Ramsey, F. P. (1928). A Mathematical Theory of Saving. *Economic Journal*, 38(152), 543-559.
- Cass, D. (1965). Optimum Growth in an Aggregative Model of Capital Accumulation. *Review of Economic Studies*, 32(3), 233-240.
- Koopmans, T. C. (1965). On the Concept of Optimal Economic Growth. In *The Econometric Approach to Development Planning*. North-Holland.
