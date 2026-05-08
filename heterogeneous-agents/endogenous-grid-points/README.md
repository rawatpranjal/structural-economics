# Buffer-Stock Saving by Endogenous Grid Points

> Euler-equation inversion for a partial-equilibrium income-risk household problem.

## Overview

An impatient household faces IID labor-income risk and cannot borrow below zero. Bad income draws make the household hold assets as a buffer.

The object is the consumption and saving policy over assets and current income. The same policy also implies a stationary wealth distribution and MPCs.

Grid search over next assets is slow inside household blocks. EGP avoids that search by inverting the Euler equation on a next-asset grid.

## Equations

The household enters the period with assets $a$ and income $y_j$. Income is IID
on $\{y_1,\dots,y_{n_y}\}$ with probabilities $\pi_j$. With gross return
$R=1+r$, the household chooses next-period assets $a'=g(a,y_j)$. Consumption is
the residual, and the borrowing limit is $\underline a$:

$$
V(a,y_j) = \max_{a'\geq \underline a}
  [\,u(R a + y_j - a') + \beta\,\sum_{\ell=1}^{n_y}\pi_\ell\, V(a',y_\ell)\,],
\qquad c(a,y_j) = R a + y_j - g(a,y_j).
$$

Because income is IID, the continuation $\mathbb{E}V(a',y')$ depends only on
$a'$. Preferences are CRRA, so marginal utility has an analytic inverse:

$$
u'(c) = c^{-\gamma}, \qquad (u')^{-1}(\mu) = \mu^{-1/\gamma}.
$$

At an interior optimum the Euler equation equates today's marginal utility
with the discounted marginal benefit of saving,

$$
\underbrace{u'(c(a,y_j))}_{\text{cost of saving today}}
= \beta R\,
\underbrace{\sum_{\ell=1}^{n_y}\pi_\ell\,u'\!(c(g(a,y_j),y_\ell))}_{\text{expected marginal utility tomorrow}}.
$$

When the borrowing limit binds, $g(a,y_j)=\underline a$. The Euler condition
holds as an inequality:

$$
u'(c(a,y_j)) \geq \beta R \sum_\ell \pi_\ell u'(c(\underline a,y_\ell)).
$$

This constraint margin creates high MPCs at low wealth. A small transfer relaxes
the constraint before it mainly raises saving.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| CRRA $\gamma$ | 2.0 | Curvature; sets the strength of precautionary motive and shapes MPCs |
| Discount factor $\beta$ | 0.95 | Annual time preference |
| Net rate $r$ | 0.03 | Exogenous risk-free return |
| Patience-return product $\beta R$ | 0.9785 | $<1$ rules out the unbounded-saving target of Carroll (1997) |
| Income mean $\mu_y$ | 1.0 | Normalization |
| Income s.d. $\sigma_y$ | 0.2 | Width of the IID labor-income shock |
| Income states $n_y$ | 5 | Width-fitted equal-spaced normal grid |
| Borrowing limit $\underline a$ | 0.0 | Hard zero; binds with positive mass |
| Upper grid bound $\bar a$ | 20.0 | Set wide enough to contain the simulated tail |
| EGP asset grid | 120 pts | Exponential, denser at $\underline a$ |
| Audit grid | 900 pts | Fine-grid reference for the discretization check |
| Convergence tolerance | 1e-06 | Sup-norm on consumption iterates |
| Simulation | 50,000 households, 550 periods | Forward-iterated cross section under $g_a$ |

## Solution Method

EGP places the grid on candidate next assets. For each $a_i'$, the current
policy guess gives expected marginal utility tomorrow. Euler inversion turns
that expectation into current consumption $c_i$. The budget identity then gives
the current asset that would choose $a_i'$:

$$
c_i = (u')^{-1}\!(\beta R \sum_{\ell} \pi_\ell\, u'(c_n(a_i', y_\ell))),
\qquad
a^{\text{endo}}_{ij} = \frac{c_i + a_i' - y_j}{R}.
$$

Each income state produces a monotone endogenous grid. Linear interpolation maps
it back to the exogenous current-asset grid. Points below the first endogenous
asset use the borrowing limit.

```text
Algorithm: EGP for IID-income buffer-stock saving
Inputs    grid {a_i'} (also the exogenous current-asset grid),
          income chain ({y_j}, {pi_j}), primitives (beta, R, gamma),
          borrowing limit a_min, tolerance eps
Output    consumption policy c(a, y), saving policy g(a, y)

Initialise c_0(a_i, y_j) = (R-1) a_i + y_j        # consume current resources
repeat n = 0, 1, 2, ...
    # 1. Euler inversion at each candidate next asset a_i'
    M_i  = sum_l pi_l * u'(c_n(a_i', y_l))         # expected MU tomorrow
    c_i  = (u')^{-1}(beta R M_i)                  # consumption today

    for each income state y_j:
        # 2. Endogenous current asset
        a^endo_{i,j} = (c_i + a_i' - y_j) / R

        # 3. Invert by interpolation onto the exogenous grid A = {a_i}
        g_{n+1}(a_i, y_j) = interp(a_i; a^endo_{:,j}, a'_:)

        # 4. Constrained branch
        for each a_i <= a^endo_{1,j}:
            g_{n+1}(a_i, y_j) = a_min

        c_{n+1}(a_i, y_j) = R a_i + y_j - g_{n+1}(a_i, y_j)

    err = max_{i,j} |c_{n+1}(a_i, y_j) - c_n(a_i, y_j)|
until err < eps
```

**Convergence and accuracy.** The 120-point grid converged in
**103 EGP iterations**. The final consumption
sup-norm residual is 9.77e-07. A 900-point
grid gives a reference policy on the same calibration. On
$a \leq 5$, the consumption and saving gaps are both
4.26e-04. The fine grid is only an accuracy check.

## Results

The consumption policy is increasing and concave in assets. Higher income shifts consumption up because IID income enters cash on hand. Slopes are steep near the borrowing limit, where households consume most extra resources. The dashed fine-grid policy overlaps the main grid on the plotted range.

<img src="figures/consumption-policy.png" alt="Consumption policy with fine-grid EGP reference" width="80%">

Net saving separates low and high income states. Low-income households draw down assets and hit the borrowing limit near zero wealth. High-income households rebuild the buffer. The zero crossings are not steady states because IID income keeps households moving across asset states.

<img src="figures/savings-policy.png" alt="Net saving policy with fine-grid EGP reference" width="80%">

The endogenous grid shows the asset level that makes each candidate $a_i'$ optimal after the lowest income draw. The curve lies above the 45-degree line because low-income households want to draw down assets. The first endogenous point, $a^{\mathrm{endo}}_{1,1}=0.261$, marks the constraint threshold.

<img src="figures/endogenous-grid.png" alt="Endogenous current asset grid for the low income state" width="80%">

Forward simulation gives a right-skewed wealth distribution. Mean assets are $\bar a=0.39$, and 5.1\% of households are at the borrowing limit. The scale is modest because income is IID and $\beta R<1$.

<img src="figures/wealth-distribution.png" alt="Simulated terminal wealth distribution" width="80%">

The MPC distribution is high near the constraint and low for wealthy households. The average MPC out of a 0.10 transfer is 0.228. The dotted line marks the perfect-foresight limit, $\kappa^{\ast}\approx0.041$.

<img src="figures/mpc-distribution.png" alt="Distribution of marginal propensities to consume" width="80%">

The table reports cross-section statistics and the fine-grid policy gaps. Wealth inequality and MPCs are economic outputs. The last two gap rows are numerical accuracy checks.

**Simulation and Accuracy Summary**

| Statistic                            | Value    |
|:-------------------------------------|:---------|
| Mean assets                          | 0.388    |
| Mean consumption                     | 1.012    |
| Wealth Gini                          | 0.389    |
| Average MPC, 0.10 transfer           | 0.228    |
| Average local MPC                    | 0.252    |
| Fraction at borrowing limit          | 5.1%     |
| Consumption gap vs fine grid, a <= 5 | 4.26e-04 |
| Savings gap vs fine grid, a <= 5     | 4.26e-04 |
| Perfect-foresight MPC limit          | 0.0413   |

## Takeaway

EGP solves the same buffer-stock household problem while avoiding an inner search over next assets.

At this calibration, the policy converges in 103 iterations and matches the fine-grid reference within 4e-04.

The simulated cross section has a 0.389 asset Gini and high MPCs near the borrowing limit. Those outcomes come from income risk, impatience, and the constraint, not from the grid reversal itself.

## References

- Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.
- Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.
- Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.
- Kaplan, G. and Violante, G. L. (2022). The Marginal Propensity to Consume in Heterogeneous Agent Models. *Annual Review of Economics*, 14, 747-775.
