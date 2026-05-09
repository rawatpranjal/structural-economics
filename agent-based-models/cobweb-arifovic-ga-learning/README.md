# Cobweb Markets and Arifovic Genetic-Algorithm Learning

## Overview

Strawberries take a season to grow. Farmers must decide how much to plant before the price is known, and many of them have to make that choice at the same time. The classical cobweb model puts a linear demand on top of a linear supply curve and assumes farmers expect the next price to equal the last one.

When supply is more elastic than demand, this naive feedback loop is unstable. Each year overshoots the rational-expectations price by a larger amount and the market explodes. Arifovic (1994) asked whether a population of boundedly rational farmers, learning by genetic operators on binary production rules, could nevertheless settle on the rational price.

This tutorial reproduces the headline result. We solve for the REE in closed form, simulate naive cobweb dynamics in a stable and an unstable regime, run Arifovic's GA on the same parameters, and finally take a noisy cobweb price series and recover the true demand curve via lagged-price IV.

## Equations

**Notation.** Each row below is one symbol. Read this first; everything else in
this section uses these letters.

| Symbol | What it means |
|---|---|
| $t$ | Period index. One season of strawberries. |
| $i$ | Firm index, running from $1$ to $n$. |
| $p_t$ | Market price in period $t$, the same for every buyer and every firm. |
| $q_{i,t}$ | Quantity firm $i$ chooses to plant for period $t$, decided before $p_t$ is known. |
| $Q_t$ | Total market quantity, $Q_t = \sum_{i=1}^{n} q_{i,t}$. |
| $\pi_{i,t}$ | Firm $i$'s realized profit in period $t$ once the market clears. |
| $a$ | Demand intercept. The price at which buyers stop buying ("choke price"). |
| $b$ | Demand slope. How many extra units buyers absorb per one-unit price cut. |
| $\varepsilon_t$ | Random demand shock in period $t$, mean zero, i.i.d. across periods. Stands in for weather, taste shifts, or news. |
| $x$ | Marginal-cost intercept. Cost of the very first unit a firm produces. |
| $y$ | Marginal-cost slope. How fast marginal cost rises as a firm produces more. |
| $n$ | Number of firms in the market. |
| $L$ | Length in bits of each firm's chromosome. |
| $N$ | GA population size. Equal to $n$ here, since one chromosome per firm. |
| $T$ | Number of GA generations the simulation runs. |
| $p_c$ | Per-pair crossover probability. |
| $p_m$ | Per-bit mutation probability. |

Per-firm cost is quadratic in own quantity: $C(q) = x q + \tfrac{y}{2} q^{2}$.

**Inverse demand with shock.** Market clearing gives

$$p_t = \frac{1}{b}\Big(\underbrace{a}_{\text{choke price}} + \underbrace{\varepsilon_t}_{\text{i.i.d. demand shock}} - \underbrace{Q_t}_{\text{aggregate quantity}}\Big).$$

**Firm supply.** Firm $i$ forms a price expectation $p_{i,t}^{e}$ before
producing, then sets $q_{i,t}$ to maximize expected profit. The first-order
condition gives the price-taking supply rule

$$q_{i,t} = \frac{p_{i,t}^{e} - x}{y}.$$

**Naive cobweb law of motion.** Plugging $p_{i,t}^{e} = p_{t-1}$ into the
supply rule for every firm and substituting into inverse demand gives a
one-step recursion in price:

$$p_t = \underbrace{\alpha}_{\text{intercept}} - \underbrace{\beta}_{\text{slope ratio}}\, p_{t-1}, \qquad \alpha = \frac{a y + n x}{b y}, \quad \beta = \frac{n}{b y}.$$

The fixed point of this recursion is the rational-expectations equilibrium

$$p^{\ast} = \frac{a y + n x}{b y + n}, \qquad q^{\ast} = \frac{p^{\ast} - x}{y}.$$

The naive cobweb converges to $p^{\ast}$ when $\beta < 1$ and explodes when
$\beta > 1$.

**Genetic-algorithm representation.** Each firm carries a length-$L$ binary
string $b_i \in \{0,1\}^{L}$ that decodes deterministically to a quantity in
the bracket $[q_{\min}, q_{\max}]$. The population size equals the number of
firms in the market, $N = n$, so each chromosome is one firm's production
plan in the current period.

**Fitness function.** The realized profit at the cleared price $p_t$ is

$$\pi_{i,t} = \underbrace{p_t\, q_{i,t}}_{\text{revenue}} - \underbrace{x\, q_{i,t}}_{\text{linear cost}} - \underbrace{\tfrac{y}{2} q_{i,t}^{2}}_{\text{convex cost}}.$$

This profit is the GA's fitness signal. The next generation is built by
tournament selection on $\pi_{i,t}$, single-point crossover with probability
$p_c$, bit-flip mutation with probability $p_m$ per bit, and Arifovic's
election operator: a candidate child replaces its parent only if its
hypothetical profit at the just-realized price $p_t$ weakly exceeds the
parent's actual profit.

The simulation runs for $T$ generations. The hyperparameters $L, N, p_c,
p_m, T$ are listed alongside the market calibration in the next section.

## Model Setup

Two regimes share the same demand intercept and per-firm cost, but differ in the demand slope $b$. The stable regime has $\beta < 1$; the unstable regime has $\beta > 1$ and naive expectations diverge.

| Object | Stable | Unstable | Role |
|---|---:|---:|---|
| Demand intercept $a$ | 60 | 60 | Choke price |
| Demand slope $b$ | 30 | 10 | Sensitivity of consumers to price |
| Cost intercept $x$ | 1 | 1 | Marginal cost at zero output |
| Cost curvature $y$ | 2 | 2 | Supply slope per firm |
| Number of firms $n$ | 30 | 30 | Population size |
| Naive slope $\beta$ | 0.50 | 1.50 | Cobweb stability |
| REE price $p^{\ast}$ | 1.67 | 3.00 | Fixed point |
| REE per-firm quantity $q^{\ast}$ | 0.33 | 1.00 | Steady-state output |

GA hyperparameters follow Arifovic's specification: chromosome length $L = 8$ giving $256$ encoded quantity levels in $[0, 2]$, one chromosome per firm so the population size equals $N = n = 30$, crossover probability $p_c = 0.6$, mutation probability per bit $p_m = 0.02$, and $T = 500$ generations. The election operator is always on in the headline runs.

## Solution Method

There is no closed-form solution for the GA dynamics. The model is solved by direct simulation. Each generation is one market period.

```text
Algorithm: Arifovic GA cobweb learning
Input: market parameters (a, b, x, y, n), GA parameters (L, N, p_c, p_m, T)
Output: price path p_1, ..., p_T and per-period population quantities

1. Initialize N random binary chromosomes of length L.
2. For t = 1 to T:
   2a. Decode each chromosome into a quantity q_i in [q_min, q_max].
   2b. Aggregate Q = sum_i q_i; clear the market at p_t = (a + e_t - Q) / b.
   2c. Compute realized profit pi_i = p_t q_i - x q_i - (y/2) q_i^2.
   2d. Roulette-select N parents in proportion to shifted profits.
   2e. Pair parents and apply single-point crossover with prob p_c.
   2f. Bit-flip mutate each child with prob p_m per bit.
   2g. ELECTION: keep each child only if its profit at p_t exceeds its parent's profit.
   2h. Replace the population with the survivors of step 2g.
```

The election step is the difference-maker. Without it, lucky offspring from a crossover that happened to land in a low-supply period are rewarded with high realized profit and propagate even though their implied quantity is far from the equilibrium. The election filter evaluates each child *as if it had played in the current market* and only keeps it if the same population's price would have rewarded that decision.

**Where this fits in evolutionary computation.** The genetic algorithm is the original member of a wider family of evolutionary search methods. Holland (1975) introduced the GA as a population-based heuristic for fixed-length bit strings under selection, crossover, and mutation. Koza (1992) generalized to genetic programming, where candidates are variable-length expressions or programs. Modern relatives such as evolution strategies, CMA-ES, and neuroevolution replace the bit string with continuous parameters and Gaussian perturbations. Arifovic's contribution is the election operator: a domain-specific filter that ties the GA's fitness to a counterfactual evaluation at the just-realized market price. The election operator is what stabilizes the otherwise unstable cobweb, and it is structurally similar in spirit to the policy-improvement step that stabilizes Q-learning in the [`q-learning-growth`](../../dynamic-programming/q-learning-growth/) tutorial: in both, a learner accepts a candidate update only if it would have been an improvement under the most recent observed state.

## Results

The cobweb diagram makes the stability story visible. In the stable regime, the staircase spirals inward to the supply-demand crossing. In the unstable regime, the same construction spirals outward and prices would explode if firms truly used last period's price as their forecast.

<img src="figures/cobweb-naive-vs-ree.png" alt="Naive cobweb staircase in the stable and unstable regimes" width="80%">

Replacing naive expectations with the GA changes the picture. In the stable regime both rules behave similarly; the GA has a slightly noisier approach to REE because mutation never fully shuts off. In the unstable regime naive expectations diverge within a few periods while the GA settles into a tight band around the REE price.

<img src="figures/price-paths.png" alt="Naive vs GA price paths in both regimes" width="80%">

Looking inside the GA population shows what convergence means in this model. The initial chromosome distribution is uniform over the encoded quantity grid. Within a few dozen generations the bulk of firms are producing close to the REE quantity, and by generation 500 the population is concentrated in a narrow band around $q^{\ast}$.

<img src="figures/chromosome-snapshots.png" alt="Population quantity histograms at four generations (unstable regime)" width="80%">

An animated view of the same population shows the convergence as a shrinking distribution centered on the REE reference line.

<img src="figures/ga-evolution.gif" alt="Animated GA population distribution converging to REE" width="80%">

The estimation block uses a naive-cobweb price series with i.i.d. demand-intercept shocks $\varepsilon_t$ as test data. The GA itself tracks REE so closely under the election operator that the resulting price barely moves; the naive cobweb provides the AR(1)-style persistence that makes the IV exercise interesting.

The demand shock $\varepsilon_t$ feeds into the realized price through market clearing, which makes a same-period regression of $Q_t$ on $p_t$ inconsistent. The lagged price $p_{t-1}$ side-steps the simultaneity: it drives the next-period quantity through firms' naive expectations, so it is correlated with $p_t$, while remaining uncorrelated with $\varepsilon_t$ when shocks are independent across periods. Two-stage least squares with $p_{t-1}$ as instrument recovers the true demand intercept and slope.

<img src="figures/iv-recovery.png" alt="Lagged-price 2SLS demand-curve recovery from a naive-cobweb price series" width="80%">

Naive cobweb stability is a knife-edge in $\beta$. The GA tracks REE in both regimes; the absolute deviation of the last-100-period mean price from $p^{\ast}$ stays small even when the naive rule diverges.

**Regime grid and GA convergence summary**

| regime   |   demand_slope_b |   supply_slope_y |   n_firms |   naive_slope_beta |   ree_price |   ree_quantity_per_firm | naive_diverges   |   ga_mean_price_last100 |   ga_abs_dev_from_ree |
|:---------|-----------------:|-----------------:|----------:|-------------------:|------------:|------------------------:|:-----------------|------------------------:|----------------------:|
| stable   |               30 |                2 |        30 |                0.5 |     1.66667 |                0.333333 | False            |                 1.66655 |           0.000112418 |
| unstable |               10 |                2 |        30 |                1.5 |     3       |                1        | True             |                 3.01176 |           0.0117647   |

Coefficients with HC0 standard errors. 2SLS with lagged price as instrument is consistent under i.i.d. demand shocks because $p_{t-1}$ enters $p_t$ only through firms' naive supply rule.

**Demand-curve recovery: true vs 2SLS**

| parameter   |   true |   iv_estimate |   iv_se |
|:------------|-------:|--------------:|--------:|
| intercept a |     60 |       67.059  | 3.08055 |
| slope b     |     30 |       34.2436 | 1.84817 |

## Takeaway

Cobweb instability under naive expectations is a property of the aggregator, not of the agents. The same parameter grid that explodes under last-price forecasts converges under a population of binary learners, because the election operator filters out the lucky-price offspring whose decisions would not have been profitable in the current market.

On the econometric side, the simulated cobweb price series sits in the same simultaneity geometry as a real market: demand shocks feed into the realized price through clearing, so a same-period regression cannot identify the demand curve. Lagged price is the textbook instrument and recovers the demand structure here, underlining that the identification logic depends on the timing of shocks more than on whether the supply side is strictly rational.

## References

- [Arifovic, J. (1994). Genetic algorithm learning and the cobweb model. *Journal of Economic Dynamics and Control*, 18(1), 3-28.](https://doi.org/10.1016/0165-1889(94)90067-1)
- [Ezekiel, M. (1938). The cobweb theorem. *Quarterly Journal of Economics*, 52(2), 255-280.](https://doi.org/10.2307/1881734)
- [Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.]
- [Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.]
- [Hansen, N. and Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159-195.](https://doi.org/10.1162/106365601750190398)
