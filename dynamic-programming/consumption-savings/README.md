# Income Risk and Buffer-Stock Saving

> A partial-equilibrium savings problem with persistent idiosyncratic income.

## Overview

A single household holds one risk-free asset, faces persistent labor income, and cannot borrow. The decision is recursive in $(a,z)$: how much of current cash-on-hand to carry forward as $a'$, given that next period's $z'$ is random and that $a'$ has to stay nonnegative.

Two extensions of the deterministic Bellman problem are doing the work. Continuation values now involve an expectation over $z'$, so the income discretization shows up directly in the recursion rather than only in simulations. And the no-borrowing constraint is a complementary-slackness condition on the Euler equation: it binds with positive probability for low-income, low-asset states, which is exactly where consumption is most responsive to a marginal dollar of wealth. [Cake eating](../cake-eating/) and [optimal growth](../optimal-growth/) keep neither.

The interest rate $r$ is fixed so the focus stays on individual behaviour. Closing the loop is what [Aiyagari](../aiyagari/) does for capital and what [Huggett](../../heterogeneous-agents/huggett-incomplete-markets/) does for a zero-net-supply bond, using exactly this household block. The Rouwenhorst chain entering $\mathbb{E}_t V(a',z')$ is the same object built in [shock discretization](../shock-discretization/). Faster solvers for the same primitives live next door: [endogenous grid points](../../heterogeneous-agents/endogenous-grid-points/) inverts the Euler equation, and [envelope-equation iteration](../../heterogeneous-agents/envelope-equation-iteration/) iterates marginal values directly. Doing brute VFI here first is useful because the operator is the one that reappears throughout the catalog.

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
$c^{\ast}(a,z)=Ra+z-g_a(a,z)$. At an interior choice the Euler equation reads

$$u'(c_t)=\beta R\,\mathbb E_t[u'(c_{t+1})],$$

and the no-borrowing constraint replaces it by $u'(c_t)\geq \beta R\,\mathbb
E_t[u'(c_{t+1})]$ whenever it binds, with $a_{t+1}=0$.

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

The state is the pair $(a,z)$, with $a$ continuous and $z$ on a five-point Rouwenhorst grid for $\log z$. The Bellman operator is

$$(TV)(a,z_j)=\max_{0\leq a'\leq Ra+z_j}\left[u(Ra+z_j-a')+\beta\sum_{k=1}^J P_{jk}V(a',z_k)\right],$$

a $\beta$-contraction on bounded continuous functions of $(a,z)$. The expectation enters only through the sum $\sum_k P_{jk} V(\cdot,z_k)$, which is recomputed on the asset state grid at every sweep and then interpolated onto the (denser) choice grid. The state grid is exponentially spaced near $\underline{a}$, where the policy is steepest; the choice grid is finer (900 vs 300 points) because policy noise dominates value-function noise once it is forward-iterated through simulations.

Two pieces of structure are worth flagging because they shape every figure below.

First, $\beta R = 0.9785 < 1$. Without uncertainty the household would prefer current consumption and decumulate to the constraint. With persistent risk and prudence ($u'''>0$), a precautionary motive pushes in the opposite direction. The two forces balance at a finite *target* wealth around the modal income state, and the policy inherits the buffer-stock shape that Carroll (1997) characterizes analytically in a neighbourhood of that target.

Second, the no-borrowing constraint binds for low-income, low-asset states. At those points the FOC is replaced by the kink $a'=0$ and $c=Ra+z$, so consumption tracks current cash-on-hand one-for-one. This is what produces high MPCs near the constraint and the visible kink in the asset policy.

```text
Algorithm  Income-fluctuation VFI
Inputs   asset state grid A = {a_i}, asset choice grid G = {g_l},
           income grid Z = {z_j}, transition P with P_{jk} = Pr(z' = z_k | z = z_j),
           primitives (beta, R, sigma), utility u, tolerance epsilon
Outputs  V*(a_i, z_j), asset policy g_a(a_i, z_j),
           consumption policy c*(a_i, z_j) = R a_i + z_j - g_a(a_i, z_j)

Initialise V_0(a_i, z_j) <- u(R a_i + z_j) / (1 - beta)        # eat-cash-on-hand guess
for n = 0, 1, 2, ...:
    for each income state z_j:
        EV(a_i) <- sum_k P_{jk} * V_n(a_i, z_k)                # expected continuation on A
        EV_hat(g_l) <- interp(EV from A to G)                  # off-state continuation on G
        for each asset state a_i:
            feasible(g_l) := { g_l <= R a_i + z_j }            # respects no-borrowing
            obj(g_l) <- u(R a_i + z_j - g_l) + beta * EV_hat(g_l)
            g_a(a_i, z_j) <- argmax_{feasible} obj
            V_{n+1}(a_i, z_j) <- max obj
    err <- max_{i,j} | V_{n+1}(a_i, z_j) - V_n(a_i, z_j) |
    stop when err < epsilon
```

The main grid converges in **260 iterations** to sup-norm residual **9.91e-07**. There is no closed form to audit against here, so the same Bellman equation is also resolved on a refined grid (600 state points, 1500 choice points). The median-income consumption policy from that refined solve is overlaid on the figure below and acts as ground truth.

## Results

$V(a,z_j)$ is increasing in both arguments and concave in $a$. The vertical spread across income states is largest near $\underline{a}=0$, because at low assets the household has no cushion against mean reversion: a bad $z_j$ today translates almost one-for-one into low cash-on-hand and a binding constraint tomorrow. Deeper into the asset grid the spread compresses, exactly as the precautionary motive predicts: assets substitute for the missing insurance market.

<img src="figures/value-functions.png" alt="Value functions by income state" width="80%">

Consumption is increasing and concave in $a$ at every income state, with the steepest slope right at the constraint. For the median income state the average MPC is about **0.52** in the bottom-decile of the asset grid and falls to **0.04** near the top. The drop is the buffer-stock mechanism in one number: an extra dollar of wealth is almost entirely consumed when assets are scarce and liquidity matters, but mostly saved once the household has built up enough buffer to absorb bad income draws on its own. The dashed line is the median-income policy from the 600-state refined solve; the maximum vertical distance to the main-grid policy is **2.55e-02**, which is interpolation noise rather than economic disagreement.

<img src="figures/consumption-policy.png" alt="Consumption policy with refined-grid benchmark on the median income state" width="80%">

Plotting $g_a(a,z_j)-a$ separates the insurance motive from the level of consumption. After a high $z_j$ the household saves out of the windfall, and most of that saving happens at low $a$ where the constraint is closest to binding next period. After a low $z_j$ it dissaves to smooth consumption, but the no-borrowing constraint truncates dissaving from below: the bottom income curve runs along the floor for small $a$ before drift pulls it back. The level at which the median-income net-saving curve crosses zero is the buffer-stock target for the modal $z$, the analogue of $k_{ss}$ in [optimal growth](../optimal-growth/) but stochastic and income-state-dependent.

<img src="figures/savings-policy.png" alt="Net saving by asset and income state" width="80%">

Forward-iterating the policy under the Rouwenhorst chain turns the static rule into asset dynamics. Five sample agents start with identical median income but quickly disperse: each path is a long sequence of buffer accumulation interrupted by runs of bad luck that grind assets back to the constraint. Persistence is what makes those runs visible; with i.i.d. income the histories would look much smoother.

The right panel collapses 3,000 such agents into a cross-section after 400 periods. Median wealth is **0.20**, the 90th percentile is **1.85**, and **20.5%** of agents sit essentially at $\underline{a}$. The pile-up at the constraint and the long right tail are exactly the features that show up in Aiyagari's stationary distribution once $r$ is endogenized; here they are pure consequences of policy plus persistence.

<img src="figures/simulated-paths.png" alt="Simulated asset paths and the induced cross-sectional asset distribution" width="80%">

Reading the rows confirms what the figures show. At $a=0$ and the lowest income state, $g_a=0$ exactly: the constraint binds and consumption is the entire cash-on-hand $z_{\rm low}$. At the same $a$ but the highest income, the household saves part of the windfall instead of consuming all of it, anticipating mean reversion. Higher up the asset grid, the income spread in $g_a$ stays roughly constant in level but shrinks as a share of $a$, which is the discrete-state version of the buffer-stock target shrinking the marginal value of an extra dollar.

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

Persistent income risk plus a no-borrowing constraint make the asset policy nonlinear, state-contingent, and steepest right at the constraint. With $\beta R = 0.9785<1$, the deterministic household would run assets to zero; prudence is what stops it, and the balance shows up as a stochastic buffer-stock target that depends on $z$. The same household block is what gets aggregated in [Aiyagari](../aiyagari/) and [Huggett](../../heterogeneous-agents/huggett-incomplete-markets/) once $r$ is forced to clear capital or bond markets, and what gets solved more efficiently in [endogenous grid points](../../heterogeneous-agents/endogenous-grid-points/) and [envelope-equation iteration](../../heterogeneous-agents/envelope-equation-iteration/).

## References

- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *Quarterly Journal of Economics*, 109(3), 659-684.
- Bewley, T. (1986). Stationary Monetary Equilibrium with a Continuum of Independently Fluctuating Consumers. In W. Hildenbrand and A. Mas-Colell (eds.), *Contributions to Mathematical Economics in Honor of Gerard Debreu*. North-Holland.
- Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.
- Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 18.
