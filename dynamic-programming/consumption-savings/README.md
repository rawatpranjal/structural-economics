# Income Risk and Buffer-Stock Saving

> A partial-equilibrium savings problem with persistent idiosyncratic income.

## Overview

This tutorial is the first dynamic-programming example here where uncertainty is central to the household problem. In [cake eating](../cake-eating/) and [optimal growth](../optimal-growth/), the state moves deterministically once the agent chooses how much to carry forward. Here the household also faces persistent income risk, so assets become self-insurance.

The exercise is deliberately partial equilibrium: the risk-free return $r$ is fixed. That keeps attention on the individual policy rules. In the [Aiyagari tutorial](../aiyagari/), these same household decisions are aggregated and $r$ is pinned down by capital-market clearing. The Rouwenhorst income chain used below is the same object studied in [shock discretization](../shock-discretization/): it enters the Bellman equation through expected continuation values, not merely through simulated histories.

## Equations

Let $a_t$ be beginning-of-period assets, $z_t$ labor income, and
$R=1+r$ the gross risk-free return. The household chooses next-period assets
$a_{t+1}=a'$ and consumes the residual

$$c_t = R a_t + z_t - a_{t+1}.$$

Assets are bounded below by the no-borrowing constraint

$$a_{t+1}\geq \underline a = 0,$$

and the numerical problem also uses an upper grid bound $\bar a$. Preferences are
CRRA,

$$u(c)=\frac{c^{1-\sigma}}{1-\sigma}, \qquad \sigma>0,\quad \sigma\neq 1.$$

Log income follows

$$\log z_{t+1}=\rho \log z_t+\varepsilon_{t+1},\qquad
\varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2),$$

and is approximated by income states $z_1,\ldots,z_J$ with transition matrix
$P$, where $P_{jk}=\Pr(z_{t+1}=z_k\mid z_t=z_j)$. The Bellman equation is

$$
V(a,z_j)=
\max_{\underline a\leq a'\leq \bar a,\ a'\leq R a+z_j}
\left[
u(Ra+z_j-a')+
\beta\sum_{k=1}^J P_{jk}V(a',z_k)
\right].
$$

The asset policy is $g_a(a,z)=a'$ and the consumption policy is
$c^{\ast}(a,z)=Ra+z-g_a(a,z)$. At an interior choice the Euler condition is

$$u'(c_t)=\beta R\,\mathbb E_t[u'(c_{t+1})],$$

with the usual inequality when the borrowing constraint binds.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$ | 0.95 | Discount factor |
| $r$ | 0.03 | Exogenous risk-free interest rate |
| $R$ | 1.03 | Gross return on assets |
| $\beta R$ | 0.9785 | Impatience margin; below one here |
| $\sigma$ | 2.0 | CRRA risk aversion |
| $\rho$ | 0.9 | Persistence of log income |
| $\sigma_\varepsilon$ | 0.1 | Innovation standard deviation |
| $\underline{a}$ | 0.0 | No-borrowing lower bound |
| $a \in$ | [0.0, 20.0] | Asset grid support |
| Asset state grid | 300 points | Exponential spacing near $\underline{a}$ |
| Next-asset choice grid | 900 points | Candidate $a'$ values in each Bellman update |
| Refined diagnostic grid | 600 states, 1500 choices | Held-out check for the median income state |
| Income states | 5 | Rouwenhorst approximation to log income |
| Simulation panel | 3000 agents, 400 periods | Used only to illustrate the induced asset distribution |

## Solution Method

The state is the pair $(a,z)$. For each income state, the transition matrix turns a guessed value function into an expected continuation-value schedule. The value function lives on the asset state grid, while the inner maximization searches over a denser grid of feasible next-period assets and interpolates continuation values between state points.

```text
Algorithm: grid VFI for the income fluctuation problem
Input: asset state grid A, next-asset grid G, income grid Z, transition matrix P, beta, R, utility u, tolerance epsilon
Output: value function V(a,z), asset policy g_a(a,z), consumption policy c*(a,z)
Initialize V_0(a_i,z_j) = u(R*a_i + z_j) / (1 - beta)
repeat for n = 0, 1, 2, ...:
    for each income state z_j:
        continuation on A: C(a_i) = sum_k P_jk * V_n(a_i, z_k)
        interpolate C from A to each next-asset choice g in G
        for each asset state a_i:
            feasible choices are g in G with g <= R*a_i + z_j
            choose g that maximizes u(R*a_i + z_j - g) + beta * C(g)
            record V_{n+1}(a_i,z_j) and g_a(a_i,z_j)
    error = max_{i,j} |V_{n+1}(a_i,z_j) - V_n(a_i,z_j)|
until error < epsilon
set c*(a_i,z_j) = R*a_i + z_j - g_a(a_i,z_j)
```

The main grid converged in **260 iterations** with sup-norm error **9.91e-07**. Because this model has no closed form, the report also solves the same Bellman equation on a refined state and choice grid and uses the median-income policy as a held-out approximation check.

## Results

Higher current income raises lifetime utility, but the income-state gap is largest near the borrowing constraint because low-asset households cannot borrow much against future mean reversion. Farther out on the asset grid, self-insurance makes the current income state less decisive.

<img src="figures/value-functions.png" alt="Value functions by income state" width="80%">

The consumption rules are increasing and concave in assets. For the median income state, the average marginal propensity to consume is about **0.52** near the constraint and **0.04** near the top of the plotted grid. That decline is the buffer-stock mechanism: extra assets are most valuable when liquidity is scarce. The dashed median-income curve comes from the refined grid; its maximum gap from the main-grid policy is **2.55e-02**.

<img src="figures/consumption-policy.png" alt="Consumption policy functions with a refined-grid median-income check" width="80%">

Net saving separates the insurance motive from the level of consumption. A high income realization pushes the household toward asset accumulation, especially when assets are low. A low income realization does the opposite: the household draws down buffers, but the no-borrowing constraint prevents dissaving below zero. The horizontal line marks zero net saving, not an equilibrium condition.

<img src="figures/savings-policy.png" alt="Net saving by asset and income state" width="80%">

Simulated histories translate the policy into asset dynamics. Agents are ex ante identical, but persistent income realizations push them to different parts of the asset grid. In the 3,000-agent panel, median assets after 400 periods are **0.20**, the 90th percentile is **1.85**, and **20.5%** of agents sit essentially at the borrowing constraint.

<img src="figures/simulated-paths.png" alt="Simulated asset paths and the induced asset distribution" width="80%">

The table gives pointwise policy values rather than a separate result. At zero assets and low income, the household cannot borrow, so consumption is pinned down by current income. At the same asset level and high income, the household saves part of the temporary cash-on-hand increase.

**Policy functions at selected asset states**

|   Assets a |   c*(a,z_low) |   c*(a,z_mid) |   c*(a,z_high) |   g_a(a,z_low) |   g_a(a,z_mid) |   g_a(a,z_high) |
|-----------:|--------------:|--------------:|---------------:|---------------:|---------------:|----------------:|
|       0    |        0.632  |        0.9931 |         1.3087 |         0      |         0.0069 |          0.2736 |
|       0.06 |        0.6761 |        1.0059 |         1.3131 |         0.0131 |         0.0512 |          0.3263 |
|       0.46 |        0.7917 |        1.0635 |         1.3416 |         0.3135 |         0.4098 |          0.7139 |
|       1.57 |        0.9424 |        1.1646 |         1.4101 |         1.3058 |         1.4516 |          1.7883 |
|       3.68 |        1.1199 |        1.3039 |         1.5202 |         3.2983 |         3.4823 |          3.8482 |
|       7.23 |        1.3589 |        1.4985 |         1.7116 |         6.7203 |         6.9487 |          7.3179 |
|      12.55 |        1.6203 |        1.798  |         1.9447 |        11.9409 |        12.1312 |         12.5668 |
|      20    |        1.9572 |        2.1292 |         2.3154 |        19.2748 |        19.4708 |         19.8668 |

## Takeaway

Uninsurable persistent income risk changes the shape of saving. Assets are valuable because they relax tomorrow's constraint after bad income draws. The policy therefore has high MPCs near zero assets, positive saving after favorable income shocks, and dissaving after unfavorable shocks. This partial-equilibrium object is the household block used in Aiyagari-style equilibrium models, where the same precautionary motive feeds into aggregate capital demand and the equilibrium interest rate.

## References

- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.
- Bewley, T. (1986). Stationary Monetary Equilibrium with a Continuum of Independently Fluctuating Consumers. In W. Hildenbrand and A. Mas-Colell (eds.), *Contributions to Mathematical Economics in Honor of Gerard Debreu*. North-Holland.
- Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.
- Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.
