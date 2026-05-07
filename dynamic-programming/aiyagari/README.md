# Aiyagari Saving and Capital-Market Clearing

> An incomplete-markets economy where household precautionary saving determines aggregate capital and the equilibrium real rate.

## Overview

The Aiyagari (1994) economy is an incomplete-markets closure of the single-household saving problem from the [buffer-stock tutorial](../consumption-savings/). Each household faces a persistent idiosyncratic productivity shock $z_t$, can save in a single risk-free asset, and cannot borrow. A representative firm rents that asset and labor at competitive prices. The point of the tutorial is the equilibrium price: the interest rate $r$ adjusts until the cross-section of self-insuring households supplies exactly as much capital as the firm wants to hire.

Two forces collide at $r$. Households want a precautionary buffer because income risk is uninsurable and the borrowing limit binds in bad states; that motive grows with the persistence of $z$ and with the curvature of $u$. Firms want capital up to where the marginal product equals $r+\delta$; that schedule slopes down. The equilibrium $r^{\ast}$ sits below the rate of time preference $1/\beta - 1 = 0.0417$ — Aiyagari's headline result. If markets were complete, the household problem would collapse to the deterministic Euler equation and pin $r$ at exactly that value; incompleteness pushes $r$ down and pushes mean assets up.

What the tutorial computes is a fixed point of fixed points. The household block is value-function iteration on a discretized $(a,z)$ state space; the outer loop is bisection on $r$, with the wage falling out of the firm's first-order condition at every candidate price. This is intentionally the slow algorithm. The [endogenous-grid-points tutorial](../../heterogeneous-agents/endogenous-grid-points/) and the [envelope-iteration tutorial](../../heterogeneous-agents/envelope-equation-iteration/) show faster Euler-based household solvers; the [Huggett tutorial](../../heterogeneous-agents/huggett-incomplete-markets/) runs the same equilibrium logic with a pure-exchange bond instead of physical capital. Keeping the global VFI step here lets the algorithmic structure of the equilibrium fixed point stay visible.

## Equations

**Household.** Let $a_t\in[\underline a,\bar a]$ be beginning-of-period assets and
$z_t$ idiosyncratic labor efficiency. With prices $(r,w)$ and a no-borrowing
constraint $\underline a=0$, the household chooses next-period assets
$a_{t+1}=a'$ and consumes

$$c_t = (1+r)a_t + w z_t - a_{t+1},\qquad c_t>0.$$

Preferences are time-separable CRRA,

$$U_0 = \mathbb{E}_0\sum_{t=0}^{\infty}\beta^t u(c_t),\qquad
u(c)=\frac{c^{1-\sigma}}{1-\sigma},$$

with $\beta\in(0,1)$ and $\sigma>0$. Log productivity is a Gaussian AR(1),

$$\log z_{t+1} = \rho\log z_t + \varepsilon_{t+1},\qquad
\varepsilon_{t+1}\sim\mathcal{N}(0,\sigma_\varepsilon^2),$$

approximated by an $N$-state Rouwenhorst chain with grid $\{z_j\}$ and
transition matrix $P_{jk}=\Pr(z_{t+1}=z_k\mid z_t=z_j)$. The chain is exact
for the AR(1) variance and persistence, and is normalized so $\mathbb{E}[z]=1$.
The Bellman equation is

$$
V(a,z_j) = \max_{a'\in[\underline a,(1+r)a+wz_j)}
[u((1+r)a+wz_j-a') + \beta\sum_k P_{jk} V(a',z_k)],
$$

with optimal asset policy $g_a(a,z_j)$ and consumption policy
$c^{\ast}(a,z_j) = (1+r)a + w z_j - g_a(a,z_j)$. The borrowing constraint
binds whenever $g_a(a,z_j)=\underline a$.

**Stationary cross-section.** Given $g_a$ and $P$, the long-run distribution
$\mu$ over $(a,z)$ satisfies the operator equation

$$
\mu(a',z_k) = \sum_j P_{jk}\sum_{i:\,g_a(a_i,z_j)=a'} \mu(a_i,z_j),
$$

and aggregate household assets are
$K^s(r) = \sum_{i,j} a_i\,\mu(a_i,z_j)$.

**Firm.** Cobb-Douglas technology $Y = K^{\alpha} L^{1-\alpha}$ with capital
share $\alpha$ and depreciation $\delta$ delivers competitive factor prices

$$r(K) = \alpha(\tfrac{K}{L})^{\alpha-1}-\delta,\qquad
w(K) = (1-\alpha)(\tfrac{K}{L})^{\alpha}.$$

With aggregate efficient labor normalized to $L=1$, capital demand at $r$ is

$$K^d(r) = (\tfrac{r+\delta}{\alpha})^{1/(\alpha-1)}.$$

**Stationary equilibrium.** A price $r^{\ast}$, wage
$w^{\ast}=w(K^d(r^{\ast}))$, household policy $g_a$, and stationary
distribution $\mu$ such that the household problem is solved at
$(r^{\ast},w^{\ast})$, $\mu$ is invariant under $(g_a,P)$, and the capital
market clears:

$$K^s(r^{\ast}) = \sum_{i,j} a_i\,\mu(a_i,z_j) = K^d(r^{\ast}).$$

A standard result is $r^{\ast}<1/\beta-1$: any $r\geq 1/\beta-1$ would induce
unbounded precautionary saving and excess capital supply.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Discount factor $\beta$ | 0.96 | Annual time preference |
| Impatience benchmark $1/\beta-1$ | 0.0417 | Complete-markets ceiling on $r^{\ast}$ |
| CRRA $\sigma$ | 2.0 | Curvature; controls precautionary motive |
| Capital share $\alpha$ | 0.36 | Cobb-Douglas exponent on $K$ |
| Depreciation $\delta$ | 0.08 | Pinning $K^d(r)$ |
| Income persistence $\rho$ | 0.9 | AR(1) coefficient on $\log z$ |
| Innovation s.d. $\sigma_\varepsilon$ | 0.2 | AR(1) shock scale |
| Income states $N$ | 7 | Rouwenhorst nodes for $\{z_j\}$ |
| Asset bracket | $[0,50]$ | $\underline a$ at no-borrowing limit |
| Asset grid (coarse) | 200 pts | Exponential, denser at $\underline a$ |
| Asset grid (fine, audit) | 600 pts | Discretization benchmark at $r^{\ast}$ |
| Capital-market tolerance | 5e-04 | Stop when $\lvert K^s-K^d\rvert/K^d$ falls below |
| Bracket-width tolerance | 1e-06 | Backup stop on $r_H-r_L$ |
| VFI tolerance | 1e-07 | Sup-norm on $V$ |

## Solution Method

Two nested fixed points. The inner one is the household Bellman; the outer one is the price that clears the capital market.

**Inner block (Bellman contraction).** The operator

$$ (TV)(a,z_j) = \max_{a'} [u((1+r)a+wz_j-a') + \beta \textstyle\sum_k P_{jk}V(a',z_k)] $$

is a $\beta$-contraction in the sup norm on bounded continuous $V$, so iterates converge geometrically at rate $\beta=0.96$. The nontrivial wrinkle in the discrete implementation is the occasionally-binding borrowing constraint: when $g_a(a,z_j)=\underline a$, the Euler equation holds with strict inequality, and Euler-based solvers have to handle that case explicitly. Grid-search VFI handles it trivially because the constraint is built into the feasible set.

**Outer block (capital-market clearing).** Capital demand $K^d(r)$ is strictly decreasing and analytic, so a price that equates supply and demand exists if $K^s(r)$ is monotone enough. On a discrete grid, $K^s(r)$ is a piecewise-constant step function with small upward jumps where the policy switches grid points. Bisection therefore converges in $r$ but can stop on the bracket width rather than on the relative gap; here the gap closes to **4.94e-04** and the outer search exits because the capital-market gap met the relative tolerance.

```text
Algorithm 1: household block at (r, w)
Inputs    asset grid {a_i} (i=1..N_a), income chain ({z_j}, P)
          preferences (β, σ), prices (r, w), tolerance ε_V
Output    value V_{ij}, asset policy g(a_i, z_j),
          consumption policy c*(a_i, z_j)

Initialise V_{ij}^{(0)} = u((1+r)a_i + w z_j) / (1 − β)
repeat n = 0, 1, 2, …
    EV_{mk} = Σ_l P_{kl} V_{ml}^{(n)}                # continuation in z
    for each (i, j):
        c_{ijm} = (1+r) a_i + w z_j − a_m
        m*(i, j) = argmax_m  u(c_{ijm}) + β · EV_{mj}    over c_{ijm} > 0
        V_{ij}^{(n+1)} = the matching maximum value
        g(a_i, z_j) = a_{m*(i, j)}
    err = max_{i,j} |V_{ij}^{(n+1)} − V_{ij}^{(n)}|
until err < ε_V
set c*(a_i, z_j) = (1+r) a_i + w z_j − g(a_i, z_j)
```

```text
Algorithm 2: stationary general equilibrium
Inputs    primitives, bracket [r_L, r_H], tolerances ε_K, ε_r
Output    r*, w*, K*, stationary distribution μ

repeat:
    r ← (r_L + r_H) / 2
    K^d(r)  = ((r + δ)/α)^{1/(α−1)}
    w(r)    = (1 − α) · K^d(r)^α
    Algorithm 1 at (r, w)  →  policy g
    iterate μ_{n+1}(a',z') = Σ_{a,z: g(a,z)=a'} μ_n(a,z) · P(z, z')
        until ||μ_{n+1} − μ_n||_∞ < ε_μ
    K^s(r) = Σ_{i,j} a_i μ(a_i, z_j)
    if K^s(r) > K^d(r):  r_H ← r       # excess saving lowers the rate
    else:                r_L ← r
until |K^s(r) − K^d(r)| / K^d(r) < ε_K  or  r_H − r_L < ε_r
```

**Discretization audit.** The household problem is resolved at the coarse-grid equilibrium $r^{\ast}=0.0260$ on a finer asset grid with $N_a=600$. The asset policy moves by at most **0.500** units anywhere on the grid, and by **3.21%** in relative terms away from the constraint. Aggregate quantities are more sensitive: at the same price the fine-grid stationary cross-section accumulates $K^s=6.992$, **3.27%** above the coarse-grid $K^s=6.763$, because tail mass multiplies large $a_i$ in the integral. The qualitative results — Gini $G=0.526$ vs. $G=0.519$, constraint mass 2.5% vs. 2.4%, equilibrium $r^{\ast}$ well below $1/\beta-1$ — are robust to the resolution. A finer-grid bisection would shift $r^{\ast}$ down by a few basis points without changing the picture.

At convergence the outer search took **12** iterations, and the final household VFI took **188** iterations to drive the sup-norm residual below the 1e-07 tolerance.

## Results

The downward-sloping firm schedule is the analytic $K^d(r)=((r+\delta)/\alpha)^{1/(\alpha-1)}$. The upward-sloping household schedule is $K^s(r)$, computed at sixteen evenly-spaced rates by re-solving the inner Bellman and forwarding the cross-section to its stationary measure at each rate. As $r$ approaches $1/\beta-1$ the household curve bends sharply right: in the limit the precautionary buffer is unbounded, so the equilibrium has to sit strictly below that vertical asymptote. At the calibration $r^{\ast}=0.0260$, roughly 38% below the complete-markets benchmark.

<img src="figures/capital-market.png" alt="Capital demand and household supply schedules with the stationary equilibrium" width="80%">

Solid lines are the coarse-grid solution at five income states; the dotted overlays in the right panel are the fine-grid policy ($N_a=600$) interpolated onto the coarse grid. The two are visually indistinguishable across the bulk of the state space, with the largest residual concentrated near the constraint where the policy is steeper. Vertical spread between income states is concave: at low assets a one-step income drop has a large value cost because the constraint limits consumption-smoothing; at high assets the buffer absorbs the shock and the income tier matters mostly through the budget. In the right panel, every income line eventually crosses below the 45-degree line; that crossing is the stochastic buffer-stock target Carroll (1997) describes, and it is what keeps assets from drifting to infinity at $r<1/\beta-1$.

<img src="figures/savings-policy.png" alt="Equilibrium value function and asset policy with fine-grid benchmark overlay" width="80%">

The stationary distribution is the joint output of the asset policy and the income chain — it is what the equilibrium interest rate $r^{\ast}=0.0260$ delivers, not an independent assumption. Mean wealth ($\bar a=6.76$) sits well above the median ($\tilde a=4.47$), with $2.5\%$ of households exactly at the borrowing limit and a long right tail driven by households who strung together good income draws. The Gini of $G=0.526$ is what a finite-state Rouwenhorst income process and a no-borrowing constraint produce on their own — without ex-ante heterogeneity in patience, returns, or bequests. Adding any of those ingredients (as in Krusell-Smith or Castañeda-Díaz-Giménez-Ríos-Rull-style calibrations) is what generates the much higher empirical Gini.

<img src="figures/wealth-distribution.png" alt="Stationary wealth distribution and Lorenz curve at the equilibrium prices" width="80%">

The capital-market gap is the residual at the bisection's stop, not a model object — refining the asset grid and tightening the bracket tolerance pushes it toward zero. The fine-grid row reports $K^s$ on a $600$-point asset grid evaluated at the same $r^{\ast}$; the few-percent shift is informative about discretization noise in tail aggregates rather than evidence of a different equilibrium.

**Stationary equilibrium diagnostics**

| Variable                     | Value                     |
|:-----------------------------|:--------------------------|
| Interest rate $r^{\ast}$     | 0.025959                  |
| Wage $w^{\ast}$              | 1.2734                    |
| Aggregate capital $K^{\ast}$ | 6.7599                    |
| Output $Y^{\ast}$            | 1.9897                    |
| Capital-output ratio $K/Y$   | 3.3975                    |
| Mean wealth $\bar a$         | 6.7633                    |
| Median wealth $\tilde a$     | 4.4728                    |
| P90 wealth                   | 16.3145                   |
| Gini                         | 0.5261                    |
| Mass at constraint           | 0.0245                    |
| Relative market-clearing gap | +4.939e-04                |
| Fine-grid $K^s$ check        | 6.9919 (rel gap 3.27e-02) |

## Takeaway

Aiyagari (1994) is the canonical worked example of how missing insurance markets reshape an aggregate price. With a single asset, a borrowing limit, and uninsurable persistent income risk, households want to hold a buffer; in equilibrium that buffer is the economy's capital stock, and the rate that supports it is $r^{\ast}=0.0260$, strictly below $1/\beta-1=0.0417$. The wealth distribution is right-skewed without any ex-ante heterogeneity in preferences, with Gini $G=0.526$ generated purely by the interaction of an AR(1) income process and self-insurance through a single asset. Faster solvers — endogenous grid points, envelope iteration, sequence-space Jacobian methods — change the cost of computing this fixed point but not its content. Closing the model on the firm side is what turns a household saving rule into an aggregate price implication.

## References

- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.
- Huggett, M. (1993). The Risk-Free Rate in Heterogeneous-Agent Incomplete-Insurance Economies. *Journal of Economic Dynamics and Control*, 17(5-6), 953-969.
- Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.
- Kaplan, G., Moll, B., and Violante, G. L. (2018). Monetary Policy According to HANK. *American Economic Review*, 108(3), 697-743.
