# Cereal Demand with Nested Logit Substitution

> Estimate where buyers go after a cereal price increase when some products are closer substitutes than others.

## Overview

A supermarket raises the price of Choco-Bombs. Fewer shoppers choose it. Some switch to Store-Frosted, and some leave the cereal category.

The object is the diversion pattern. Nested logit groups Choco-Bombs with Store-Frosted and groups the two healthier cereals. The nesting parameter $\sigma$ controls how much substitution stays inside a group.

The computation starts from shares, prices, nests, and instruments. Berry inversion gives a linear equation. 2SLS estimates price sensitivity and $\sigma$. Elasticities and diversion ratios then show where lost Choco-Bombs demand goes.

## Equations

Products $j=1,\ldots,J$ appear in markets $t=1,\ldots,T$. Product $j$ belongs
to nest $g(j)$, and $s_{0t}$ is the outside-good share. Mean utility combines a
common inside-good term, sugar content, price, and unobserved product quality:
$$\delta_{jt}=\beta_0+\beta_{\text{sugar}}\text{sugar}_j-\alpha p_{jt}+\xi_j, \qquad \alpha>0 .$$

The inclusive-value denominator aggregates the products inside one nest:
$$D_{gt}=\sum_{k:g(k)=g}\exp\left(\frac{\delta_{kt}}{1-\sigma}\right), \qquad 0\leq \sigma<1 .$$

Total share factors into a conditional share inside the nest and the nest's
overall market share:
$$s_{j|g,t}= \frac{\exp\left(\delta_{jt}/(1-\sigma)\right)}{D_{g(j)t}}, \qquad s_{gt}= \frac{D_{gt}^{1-\sigma}}{1+\sum_h D_{ht}^{1-\sigma}}, \qquad s_{jt}=s_{j|g,t}s_{g(j)t}.$$

The Berry inversion turns observed shares into a linear estimating equation:
$$\ln s_{jt}-\ln s_{0t} = \beta_0+\beta_{\text{sugar}}\text{sugar}_j-\alpha p_{jt} +\sigma\ln s_{j|g,t}+\xi_j .$$
Both $p_{jt}$ and $\ln s_{j|g,t}$ are endogenous in this regression.

After estimation, the substitution object is the elasticity matrix. Rows are
products whose shares change; columns are products whose prices change. For
market $t$,
$$\eta_{jk,t}=\frac{\partial\ln s_{jt}}{\partial\ln p_{kt}}= \begin{cases} -\alpha p_{jt}\left[\dfrac{1}{1-\sigma} -\dfrac{\sigma}{1-\sigma}s_{j|g,t}-s_{jt}\right], & j=k,\\[1.0em] \alpha p_{kt}\left[\dfrac{\sigma}{1-\sigma}s_{k|g,t}+s_{kt}\right], & j\neq k,\ g(j)=g(k),\\[1.0em] \alpha p_{kt}s_{kt}, & g(j)\neq g(k). \end{cases}$$
Diversion ratios convert elasticities into the share loss from product $k$ that
goes to product $j$:
$$D_{j\leftarrow k}= -\frac{\partial s_{jt}/\partial p_{kt}}{\partial s_{kt}/\partial p_{kt}} = \frac{\eta_{jk,t}s_{jt}}{|\eta_{kk,t}|s_{kt}} .$$

## Model Setup

The synthetic panel has a small cereal category across many markets. Prices move with a cost shifter. Shares come from the nested-logit model. The estimator observes prices, sugar, shares, nests, and excluded shifters.

| Object | Value | Role |
|---|---:|---|
| Markets $T$ | 50 | Cross-market price and cost variation |
| Inside products $J$ | 4 | Two sugary and two healthy cereals |
| Outside good | Included | Pins down the Berry share ratio |
| True $\alpha$ | 1.5 | Price sensitivity in the data-generating model |
| True $\beta_{\text{sugar}}$ | 0.3 | Taste for sugar content |
| True $\beta_0$ | 1.0 | Common inside-good utility shifter |
| True $\sigma$ | 0.7 | Extra same-nest substitution |
| Nests | Sugary, healthy | Maintained grouping used by nested logit |

## Solution Method

Nested logit has closed-form shares. Inclusive values summarize each product group. The estimation step uses 2SLS on the Berry-inverted regression. Price and within-nest share both move with unobserved product quality. Plain logit is estimated as the IIA benchmark.

```text
Algorithm: nested-logit IV demand
Input: markets t, products j, nests g(j), shares s_jt, outside shares s_0t
Output: IV estimates, elasticity matrix, and diversion ratios

1. For each market, compute within-nest shares s_{j|g,t} from observed shares.
2. Form y_jt = log(s_jt) - log(s_0t) and w_jt = log(s_{j|g,t}).
3. First stage: project price p_jt and w_jt on sugar and instruments Z_jt.
4. Second stage: regress y_jt on sugar, fitted price, and fitted w_jt.
5. Read alpha from the negative price coefficient and sigma from w_jt.
6. Compute eta_jk,t and D_{j<-k}; compare plain logit, fitted nested logit,
   and the true synthetic nested-logit benchmark.
```

The instruments match the two endogenous variables. Cost variation moves prices. Rival characteristics and nest composition predict $\ln s_{j|g,t}$.

| Instrument | Targets | Rationale |
|---|---|---|
| Cost shifter | Price | Moves marginal cost without entering utility directly |
| Rival sugar, all products | Price | Summarizes rival characteristics in the market |
| Number of products in nest | $\ln s_{j\mid g,t}$ | Changes the local competitive set |
| Same-nest rival sugar | $\ln s_{j\mid g,t}$ | Moves the attractiveness of close substitutes |

## Results

Rows are products whose shares respond. Columns are prices that move. Gold blocks mark the nests. The Choco-Bombs column is largest for Store-Frosted, so substitution follows product similarity.

<img src="figures/elasticity-heatmap.png" alt="Nested logit elasticity matrix with nest blocks highlighted. Same-nest responses are higher than cross-nest responses." width="80%">

The blue bars show the plain-logit restriction. Cross responses follow product shares without product closeness. The green and red bars use 2SLS nested-logit estimates. The hatched bars show the true synthetic model. It ranks Store-Frosted as the close substitute.

<img src="figures/cross-elasticity-comparison.png" alt="Cross-elasticities when Choco-Bombs raises its price: plain logit, fitted nested logit, and the true synthetic model." width="80%">

Diversion ratios convert elasticities back into share derivatives. Plain logit sends lost Choco-Bombs demand toward larger rivals. Nested logit shifts more diversion to Store-Frosted. The remaining lost demand goes to the outside good.

<img src="figures/diversion-ratios.png" alt="Product diversion ratios from a Choco-Bombs price increase." width="80%">

The table checks whether estimation recovers the parameters used to generate the synthetic shares. Plain logit cannot estimate $\sigma$. Nested logit recovers the signs and the same-nest ranking.

**Parameter estimates: true values vs plain logit vs nested logit**

| Parameter   |   True | Logit   |   Nested Logit |
|:------------|-------:|:--------|---------------:|
| alpha       |    1.5 | 1.649   |          1.455 |
| beta_sugar  |    0.3 | 0.404   |          0.279 |
| beta_const  |    1   | -0.034  |          1.518 |
| sigma       |    0.7 | ---     |          0.913 |

## Takeaway

Nested logit is useful when product groups are defensible. Here a Choco-Bombs price increase mainly sends buyers to Store-Frosted. The nests do real work. The diversion matrix is only as credible as the grouping.

## References

- Berry, S. (1994). Estimating Discrete-Choice Models of Product Differentiation. *RAND Journal of Economics*, 25(2), 242--262.
- McFadden, D. (1978). Modelling the Choice of Residential Location. In A. Karlqvist et al. (Eds.), *Spatial Interaction Theory and Planning Models*. North-Holland.
- Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition, Ch. 4.
