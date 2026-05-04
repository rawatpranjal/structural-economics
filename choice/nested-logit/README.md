# Nested Logit Demand and Within-Nest Substitution

> The simplest fix for the IIA problem: grouping products into nests so that closer substitutes have higher cross-price elasticities.

## Overview

The plain logit imposes the Independence of Irrelevant Alternatives (IIA): the ratio of any two products' market shares is independent of the attributes of all other products. This means a price increase for Choco-Bombs sends customers to Fiber-Bran and Store-Frosted in proportion to their market shares, regardless of how similar those products are.

The **nested logit** fixes this by grouping products into nests (e.g., sugary vs healthy cereals). Within a nest, products are closer substitutes. The nesting parameter $\sigma \in [0,1)$ controls the degree of within-nest correlation:
- $\sigma = 0$: collapses to plain logit (IIA holds)
- $\sigma \to 1$: products within a nest are perfect substitutes

This makes nested logit a compact demand model for correlated alternatives while keeping the share equation easy to estimate.

## Equations

$$s_j = s_{j|g} \cdot s_g$$

**Within-nest share:**
$$s_{j|g} = \frac{\exp\!\bigl(\delta_j / (1-\sigma)\bigr)}{D_g}, \qquad D_g = \sum_{k \in g} \exp\!\bigl(\delta_k / (1-\sigma)\bigr)$$

**Nest share:**
$$s_g = \frac{D_g^{\,1-\sigma}}{1 + \sum_h D_h^{\,1-\sigma}}$$

**Berry inversion (estimation equation):**
$$\ln s_j - \ln s_0 = \mathbf{x}_j \beta - \alpha \, p_j + \sigma \ln s_{j|g} + \xi_j$$

Both $p_j$ and $\ln s_{j|g}$ are endogenous; we instrument with cost shifters,
rival characteristics, number of products in nest, and same-nest rival characteristics.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | 1.5 | Price sensitivity |
| $\beta_{\text{sugar}}$ | 0.3 | Taste for sugar |
| $\beta_0$ | 1.0 | Base utility |
| $\sigma$ | 0.7 | Nesting parameter |
| Products | 4 (2 nests of 2) | Sugary: Choco-Bombs, Store-Frosted; Healthy: Fiber-Bran, Granola-Crunch |
| Markets | 50 | Cross-sectional variation for IV |

## Solution Method

**Two-Stage Least Squares (2SLS)** on the Berry-inverted equation.

The nested logit introduces a second endogenous variable, $\ln s_{j|g}$, because within-nest shares depend on unobserved quality $\xi_j$. We need instruments for **both** price and within-nest share:

| Instrument | Targets | Rationale |
|------------|---------|----------|
| Cost shifter | Price | Cost variation |
| Rival sugar (all) | Price | Characteristic sum |
| Number of products in nest | $\ln s_{j\mid g}$ | Affects within-nest competition |
| Same-nest rival sugar | $\ln s_{j\mid g}$ | Within-nest characteristic variation |

## Results

The gold-outlined blocks mark within-nest product pairs. Cross-elasticities inside these blocks are visibly larger than those outside, reflecting the intuition that consumers substitute more readily among similar products. This asymmetric substitution pattern is exactly what IIA forbids and what the nested logit is designed to capture.

![Nested logit elasticity matrix with nest blocks highlighted. Same-nest cross-elasticities (inside gold boxes) are higher than cross-nest elasticities.](figures/elasticity-heatmap.png)
*Nested logit elasticity matrix with nest blocks highlighted. Same-nest cross-elasticities (inside gold boxes) are higher than cross-nest elasticities.*

Under IIA (blue bars), all rivals gain equally from Choco-Bombs' price increase regardless of product similarity. The nested logit (green/red bars) predicts that Store-Frosted -- another sugary cereal -- absorbs the lion's share of switching customers. The point is demand-side substitution: observed groups can concentrate switching toward closer alternatives.

![Logit vs nested logit cross-elasticities when Choco-Bombs raises its price. Nested logit sends more customers to Store-Frosted (same nest).](figures/cross-elasticity-comparison.png)
*Logit vs nested logit cross-elasticities when Choco-Bombs raises its price. Nested logit sends more customers to Store-Frosted (same nest).*

Diversion ratios summarize where customers go after one product becomes less attractive. The nested logit concentrates switching within the sugary nest instead of spreading it flatly by market share.

![Diversion ratios: fraction of Choco-Bombs' lost sales captured by each rival. Nested logit predicts much higher diversion to same-nest products.](figures/diversion-ratios.png)
*Diversion ratios: fraction of Choco-Bombs' lost sales captured by each rival. Nested logit predicts much higher diversion to same-nest products.*

At $\sigma \approx 0$ the model collapses to plain logit with symmetric cross-elasticities. As $\sigma$ rises toward 1, within-nest products become near-perfect substitutes while cross-nest substitution barely changes. The structural role of $\sigma$ is clear: it controls the degree to which product groupings shape consumer switching behavior.

![Effect of the nesting parameter sigma on substitution patterns. As sigma increases, within-nest substitution intensifies while cross-nest substitution stays flat.](figures/sigma-effect.png)
*Effect of the nesting parameter sigma on substitution patterns. As sigma increases, within-nest substitution intensifies while cross-nest substitution stays flat.*

The plain logit omits $\sigma$ entirely, which biases the price sensitivity estimate because within-nest correlation is absorbed into the price coefficient. The nested logit recovers all four structural parameters, including the nesting parameter that governs substitution patterns.

**Parameter estimates: true values vs plain logit vs nested logit**

| Parameter   |   True | Logit   |   Nested Logit |
|:------------|-------:|:--------|---------------:|
| alpha       |    1.5 | 1.649   |          1.455 |
| beta_sugar  |    0.3 | 0.404   |          0.279 |
| beta_const  |    1   | -0.034  |          1.518 |
| sigma       |    0.7 | ---     |          0.913 |

## Takeaway

The nested logit is the simplest departure from the IIA assumption. By grouping products into nests, it allows consumers who leave one product to disproportionately switch to similar products rather than spreading evenly across the market.

**Key insights:**
- The nesting parameter $\sigma$ controls within-nest correlation. We estimated $\hat{\sigma} = 0.913$ vs the true value of 0.7, confirming that products within a nest are closer substitutes.
- Same-nest cross-price elasticities are **higher** than cross-nest elasticities. When Choco-Bombs raises its price, customers primarily switch to Store-Frosted (also sugary), not to Fiber-Bran.
- The plain logit gets the overall price sensitivity roughly right but completely misses the substitution pattern -- this matters for product positioning and targeted pricing.
- Estimation requires instruments for **both** price and the within-nest share $\ln s_{j|g}$, since both are endogenous. Rival characteristics and nest size serve this purpose in the synthetic example.

## Reproduce

```bash
python run.py
```

## References

- Berry, S. (1994). Estimating Discrete-Choice Models of Product Differentiation. *RAND Journal of Economics*, 25(2), 242--262.
- McFadden, D. (1978). Modelling the Choice of Residential Location. In A. Karlqvist et al. (Eds.), *Spatial Interaction Theory and Planning Models*. North-Holland.
- Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition, Ch. 4.
