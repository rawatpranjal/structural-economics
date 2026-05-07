# Buffer-Stock Saving by Endogenous Grid Points

> Euler-equation inversion for a partial-equilibrium income-risk household problem.

## Overview

An impatient household with CRRA preferences faces IID labor-income risk and a no-borrowing constraint $\underline a = 0$. The economic content is the buffer-stock logic of Deaton (1991) and Carroll (1997): assets are held purely to self-insure against bad income draws, and the constraint binds with strictly positive probability, so the Euler equation holds with complementary slackness. The [buffer-stock VFI tutorial](../../dynamic-programming/consumption-savings/) solves the persistent-income version by grid maximization over $a'$. The exercise here is to compute the same kind of policy without that inner maximization.

The trick is Carroll's (2006) endogenous-grid trade. VFI iterates on $V(a,y)$ by asking, at each current $a$, which $a'$ delivers the highest $u(c) + \beta\,\mathbb{E}V(a',y')$. EGP fixes a grid for *next-period* assets $a'$ instead, evaluates the Euler equation pointwise to recover the consumption that rationalizes each $a'$, and reads the implied current asset level off the budget identity. The map $a'\mapsto a$ is then inverted by interpolation to deliver $g_a(a,y)$ on the original grid. No argmax, no gradient, just one expectation, one inverse marginal utility, and one univariate interpolation per income state per iteration.

The reversal pays off whenever the household block is solved repeatedly inside an outer loop. The [Aiyagari tutorial](../../dynamic-programming/aiyagari/) bisects on $r$ around the household problem; [Huggett](../huggett-incomplete-markets/) does the same for a bond economy in continuous time. The neighbouring [envelope-equation iteration](../envelope-equation-iteration/) tutorial keeps the same Euler discipline but iterates on $W_a(a)$ rather than the consumption policy itself, and shows that grid VFI, EGP, and EEI all coincide on the buffer-stock policy at the resolution used here.

## Equations

The household enters the period with assets $a$ and observes income $y_j$ drawn
IID from $\{y_1,\dots,y_{n_y}\}$ with probabilities $\pi_j$. With gross return
$R=1+r$, it chooses next-period assets $a'=g(a,y_j)$, consumes the residual,
and faces a non-borrowing constraint:

$$
V(a,y_j) = \max_{a'\geq \underline a}
  [\,u(R a + y_j - a') + \beta\,\sum_{\ell=1}^{n_y}\pi_\ell\, V(a',y_\ell)\,],
\qquad c(a,y_j) = R a + y_j - g(a,y_j).
$$

Because income is IID, the continuation $\mathbb{E}V(a',y')$ depends only on
$a'$, which is what makes EGP especially clean here. Preferences are CRRA, so
the marginal utility map and its analytic inverse are

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

When the borrowing limit binds, $g(a,y_j)=\underline a$ and the Euler condition
holds as an inequality,
$u'(c(a,y_j)) \geq \beta R \sum_\ell \pi_\ell u'(c(\underline a,y_\ell))$.
This Kuhn-Tucker margin is what generates large MPCs at low wealth: a marginal
dollar of cash relaxes a slack constraint dollar-for-dollar, so $\partial c/\partial a$
can be close to $R$ rather than the small perfect-foresight value
$1-(\beta R)^{1/\gamma}/R$.

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

**The key trade.** VFI on this problem maximizes $u((1+r)a+y_j-a')+\beta\,
\mathbb{E} V(a',y')$ over $a'$ at every state, paying a one-dimensional search
per grid point per iteration. EGP holds the grid $\{a_i'\}_{i=1}^{N_a}$ fixed
in the *next-period* assets, evaluates the Euler equation pointwise to recover
the consumption that is consistent with stepping to each $a_i'$, and reads the
implied current asset off the budget line:

$$
c_i = (u')^{-1}\!(\beta R \sum_{\ell} \pi_\ell\, u'(c_n(a_i', y_\ell))),
\qquad
a^{\text{endo}}_{ij} = \frac{c_i + a_i' - y_j}{R}.
$$

Because $c$ is strictly increasing in cash on hand and $u'$ is strictly
decreasing, the map $a_i' \mapsto a^{\text{endo}}_{ij}$ is monotone for each
$y_j$. Inverting it is therefore a single sorted interpolation onto the
exogenous grid $A$. The borrowing constraint enters as a left-tail boundary
correction: any $a < a^{\text{endo}}_{1j}$ cannot rationalize an interior
saving choice given $y_j$, so the policy is pinned at $g_a = \underline a$.

```text
Algorithm: EGP for IID-income buffer-stock saving
Inputs    grid {a_i'} (also serves as the exogenous current-asset grid),
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

Three observations help in practice. First, EGP inherits the geometric
contraction rate of the underlying Bellman operator, so iteration counts scale
with $\beta$, not with $N_a$. Second, the interpolation is over a sorted
sequence; using `np.interp` is fine and the extrapolation branch on the right
end matters only if the grid bound $\bar a$ is set aggressively low. Third,
when income is persistent (not the case here) the endogenous-current-asset
grid depends on $y_j$, and the inversion has to be done income state by income
state — the IID simplification used in this tutorial is a clean expository
benchmark, not a structural assumption.

**Convergence and accuracy.** The 120-point grid converged in
**103 EGP iterations** with a consumption sup-norm
residual of 9.77e-07. To audit the discretization, the
same EGP solve was rerun on a 900-point grid at the identical
calibration and the two policies were compared on $a \leq 5$,
the asset range that holds essentially all of the simulated mass. The maximum
consumption-policy gap is 4.26e-04 and the next-asset gap is
4.26e-04; both are pure grid-and-interpolation wedges with no
economic content. The fine grid is not used in the simulation — it appears
only as the dashed reference in the policy plots and as the diagnostic row in
the summary table.

## Results

The first figure shows the EGP consumption policy at five income states (the two extreme states bolded, the three interior states in grey) with the dashed fine-grid reference overlaid for the lowest and highest $y_j$. The shape is the same buffer-stock policy that VFI delivers in the [persistent-income tutorial](../../dynamic-programming/consumption-savings/): concave, increasing in $a$, and shifted vertically by income because IID $y_j$ enters cash on hand directly. Slopes near $\underline a$ are close to the 45-degree reference $c=Ra+y_j$, the certainty-equivalent rule for a constrained agent who consumes everything; far from the constraint the slope falls toward the perfect-foresight limit $\kappa^{\ast} \approx 0.041$ derived from $c_{t+1}/c_t=(\beta R)^{1/\gamma}$. The coarse and fine-grid policies are visually indistinguishable on the plotted range, which is the discretization audit.

<img src="figures/consumption-policy.png" alt="Consumption policy with fine-grid EGP reference" width="80%">

Net saving $g_a(a,y_j)-a$ separates income states more cleanly than consumption does. After a low income draw the household decumulates to consume more than $Ra+y_j$ for $a$ above the constraint, and rolls onto $g_a=\underline a$ once cash on hand can no longer support an interior Euler-equation choice — that is the discrete kink at the left end of the lowest-income curve. A high draw flips the sign and rebuilds the buffer. The zero crossings are not steady states: with IID income, the household keeps cycling across asset states as draws arrive, and the simulated cross section averages over those cycles.

<img src="figures/savings-policy.png" alt="Net saving policy with fine-grid EGP reference" width="80%">

This third figure makes the EGP construction visible. For each $a_i'$ on the exogenous grid, the Euler inversion fixes $c_i = (u')^{-1}(\beta R\,\mathbb{E}u'(c_n(a_i',y')))$, and the budget identity $a^{\mathrm{endo}} = (c_i + a_i' - y_j)/R$ then pins down the current asset level that would have rationalized stepping to $a_i'$ after observing the lowest income draw $y_1=0.61$. The 45-degree line is the static no-saving rule $a^{\mathrm{endo}}=a'$; the policy curve sits above it because the household with the lowest current income wants to draw down assets, so a given $a'$ requires a larger current $a$ to finance. The first endogenous point $a^{\mathrm{endo}}_{1,1}=0.261$ is the kink threshold: any current $a$ below it would force negative interior consumption, so the borrowing constraint supplies $g_a=\underline a$ on that left tail.

<img src="figures/endogenous-grid.png" alt="Endogenous current asset grid for the low income state" width="80%">

Forward-iterating $g_a$ for 550 periods on 50,000 households gives the cross section in the fourth figure. The distribution is right-skewed with mean $\bar a=0.39$ and a small mass exactly at the constraint (5.1\% of agents); the spike at zero is the Kuhn-Tucker margin showing up in the marginal distribution. The scale is modest because income is IID — there is no persistence to amplify good histories — and because $\beta R<1$ rules out a drifting asset target. Replacing IID income with the persistent [Rouwenhorst chain](../../dynamic-programming/shock-discretization/) and closing the model with capital-market clearing produces the much wider Aiyagari cross section in the [Aiyagari tutorial](../../dynamic-programming/aiyagari/).

<img src="figures/wealth-distribution.png" alt="Simulated terminal wealth distribution" width="80%">

The fifth figure plots the cross-sectional distribution of MPCs out of a small transfer of 0.10 — about ten percent of mean income. The average MPC is 0.228, an order of magnitude above the perfect-foresight limit $\kappa^{\ast}\approx0.041$ marked by the dotted line. The high values come from constrained or near-constrained households, for whom an extra dollar of cash is spent dollar-for-dollar; the right tail near 1 is exactly the Kuhn-Tucker margin from the equations section made empirical. The low-MPC mode near $\kappa^{\ast}$ is the wealth-rich subpopulation for whom the constraint is slack and the Euler equation pins the consumption response. This bimodality is the proximate reason heterogeneous-agent models can deliver aggregate consumption responses to fiscal transfers far above what a representative-agent PIH model implies — see the discussion in Kaplan and Violante (2022).

<img src="figures/mpc-distribution.png" alt="Distribution of marginal propensities to consume" width="80%">

The summary table separates the economic outputs of the cross section (mean wealth, Gini, average MPCs, mass at the constraint) from the discretization diagnostics (consumption and savings gaps against the fine-grid solve). The Gini and the high average MPC are model results that depend on $\gamma$, $\beta R$, and the income process; the policy gaps are pure numerical wedges that shrink toward zero as $N_a\to\infty$.

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

Carroll's grid reversal is a workhorse precisely because it is not a new model. The same buffer-stock policy that VFI computes by maximizing over $a'$ at every state falls out of EGP in 103 iterations of one expectation, one analytic inverse marginal utility, and one univariate interpolation per income state. The two solutions agree to 4e-04 on the asset range that holds the simulated mass — a discretization wedge, not a different economic object.

The economic content stays put: a Gini of 0.389 on assets alone, average MPCs many times the perfect-foresight benchmark, and a non-trivial fraction of agents pinned at the borrowing limit. What EGP buys is the ability to use that household block as the inner step of a general-equilibrium fixed point or a structural estimation loop without the inner search becoming the binding cost — exactly the role it plays in the [Aiyagari](../../dynamic-programming/aiyagari/) and [Huggett](../huggett-incomplete-markets/) computations next door.

## References

- Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.
- Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.
- Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.
- Kaplan, G. and Violante, G. L. (2022). The Marginal Propensity to Consume in Heterogeneous Agent Models. *Annual Review of Economics*, 14, 747-775.
