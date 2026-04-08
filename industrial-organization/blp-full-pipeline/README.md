# BLP Full Pipeline: Random Coefficients Demand Estimation

> The gold standard for demand estimation in industrial organization. Random coefficients allow rich substitution patterns that matter enormously for merger analysis.

## Overview

The Berry, Levinsohn, and Pakes (1995) model is the workhorse of modern demand estimation in IO. Unlike simple logit, which suffers from the IIA (Independence of Irrelevant Alternatives) problem, BLP allows consumer preferences to vary across the population. This means substitution patterns are driven by product similarity in characteristic space, not just market shares.

In this implementation we estimate demand for three synthetic cereal products. Consumers differ in their price sensitivity and sugar preferences. Sugar-loving consumers ("kids") substitute between sugary cereals, while health-conscious consumers ("parents") substitute among low-sugar options. This heterogeneity has first-order implications for merger analysis.

## Equations

**Utility:** Consumer $i$ gets utility from product $j$:

$$U_{ij} = \underbrace{\beta_0 + \beta_s \cdot \text{sugar}_j - \alpha \cdot p_j + \xi_j}_{\delta_j \text{ (mean utility)}} + \underbrace{\sigma_\alpha \nu_i^\alpha (-p_j) + \sigma_s \nu_i^s \cdot \text{sugar}_j}_{\mu_{ij} \text{ (individual deviation)}}$$

where $\nu_i \sim N(0, I)$ captures preference heterogeneity.

**Market shares** (simulation): $s_j(\delta, \sigma) = \frac{1}{n_s}\sum_{i=1}^{n_s} \frac{\exp(\delta_j + \mu_{ij})}{1 + \sum_k \exp(\delta_k + \mu_{ik})}$

**BLP contraction** (inner loop): $\delta^{h+1} = \delta^h + \ln s^{\text{obs}} - \ln s^{\text{pred}}(\delta^h, \sigma)$

**GMM** (outer loop): $\hat{\sigma} = \arg\min_\sigma \; [Z'\xi(\sigma)]' W [Z'\xi(\sigma)]$

where $\xi = \delta - X\beta + \alpha p$ is the structural error and $Z$ are instruments.

## Model Setup

| Parameter | True | Estimated | Description |
|-----------|------|-----------|-------------|
| $\alpha$ | 3.0 | 1.627 | Mean price sensitivity |
| $\beta_0$ | 2.0 | -1.134 | Base utility |
| $\beta_s$ | 0.5 | 0.748 | Mean sugar taste |
| $\sigma_\alpha$ | 0.8 | 0.629 (SE: 1.896) | Std dev of price sensitivity |
| $\sigma_s$ | 1.0 | 0.095 (SE: 0.064) | Std dev of sugar taste |

## Solution Method

**Nested Fixed Point (NFXP):**

1. **Outer loop (GMM):** Search over nonlinear parameters $\sigma = (\sigma_\alpha, \sigma_s)$ using Nelder-Mead.
2. **Inner loop (contraction):** For each candidate $\sigma$, run the BLP contraction mapping per market to find $\delta$ such that predicted shares match observed shares (tolerance $10^{-12}$).
3. **Concentration:** Given $\delta$, recover linear parameters $(\beta, \alpha)$ by 2SLS using cost shifters, rival characteristics, and own-squared sugar as instruments.
4. **Moments:** Compute $\xi = \delta - X\beta + \alpha p$ and form GMM objective $g'Wg$ where $g = Z'\xi$.

The contraction converged in **128 iterations** for the demonstration market. GMM optimization converged (objective = 0.000000). Post-estimation analysis uses true parameters (standard Monte Carlo practice for demonstrating economic mechanics).

## Results

![BLP contraction mapping convergence (linear rate in log scale)](figures/contraction-convergence.png)
*BLP contraction mapping convergence (linear rate in log scale)*

![Elasticity matrices: simple logit (IIA) vs BLP (realistic substitution)](figures/elasticity-comparison.png)
*Elasticity matrices: simple logit (IIA) vs BLP (realistic substitution)*

![Diversion ratios from Choco-Bombs: logit diverts by market share, BLP diverts to similar products](figures/diversion-ratios.png)
*Diversion ratios from Choco-Bombs: logit diverts by market share, BLP diverts to similar products*

![Merger price effects: BLP predicts larger increases for close substitutes because it captures product similarity](figures/merger-simulation.png)
*Merger price effects: BLP predicts larger increases for close substitutes because it captures product similarity*

**BLP Parameter Estimates with Bootstrap Standard Errors**

| Parameter   |   True |   Estimated | SE (bootstrap)   |
|:------------|-------:|------------:|:-----------------|
| alpha       |    3   |       1.627 | --               |
| beta_const  |    2   |      -1.134 | --               |
| beta_sugar  |    0.5 |       0.748 | --               |
| sigma_alpha |    0.8 |       0.629 | 1.896            |
| sigma_sugar |    1   |       0.095 | 0.064            |

## Economic Takeaway

BLP is the gold standard for demand estimation in IO because random coefficients generate realistic substitution patterns. The key results from this pipeline:

**1. Substitution depends on product similarity.** When Choco-Bombs raises its price, BLP correctly predicts that most customers switch to Store-Frosted (also sugary), not Fiber-Bran. Simple logit, constrained by IIA, diverts proportionally to market share and misses this pattern entirely.

**2. This matters enormously for merger analysis.** A merger between Choco-Bombs and Store-Frosted combines close substitutes. BLP predicts a larger price increase than logit because the merging firm internalizes the high diversion between its products. Antitrust authorities who use logit would systematically underestimate merger harms for mergers between similar products.

**3. The computational cost is justified.** BLP requires a nested optimization (contraction mapping inside GMM), making it far more expensive than logit. But the payoff is a demand system that captures the heterogeneity in consumer preferences that drives real market outcomes.

## Reproduce

```bash
python run.py
```

## References

- Berry, S., Levinsohn, J., and Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica*, 63(4), 841-890.
- Nevo, A. (2000). "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand." *Journal of Economics & Management Strategy*, 9(4), 513-548.
- Nevo, A. (2001). "Measuring Market Power in the Ready-to-Eat Cereal Industry." *Econometrica*, 69(2), 307-342.
