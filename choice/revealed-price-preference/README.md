# Revealed Price Preference

> Testing rationalizability of price vectors rather than chosen bundles.

## Overview

Standard revealed preference asks whether chosen bundles can be rationalized by utility maximization. Revealed price preference asks a dual question: do the price vectors themselves behave as if there is a consistent preference over prices? This is useful when the empirical object is a price regime, tax schedule, or tariff vector rather than only a consumed bundle.

The key distinction is simple. GARP compares bundles at the prices faced when a bundle was chosen. GAPP fixes an observed chosen bundle and asks whether another price vector would have made that same bundle weakly cheaper. Cycles in these price comparisons reject rationalizability of prices, even in examples where standard bundle GARP may pass.

## Equations

Let $p^s$ be price vector $s$ and $x^t$ be the bundle chosen under price vector $t$.
Price $s$ is directly revealed preferred to price $t$ when it makes bundle $x^t$ weakly cheaper:

$$R_p[s,t] = 1\{p^s \cdot x^t \le p^t \cdot x^t\}.$$

The strict relation is

$$P_p[s,t] = 1\{p^s \cdot x^t < p^t \cdot x^t\}.$$

GAPP fails when price $s$ is indirectly revealed preferred to price $t$, but $t$ is strictly directly revealed preferred to $s$:

$$R_p^{*}[s,t] = 1 \quad \text{and} \quad P_p[t,s] = 1.$$

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Price vectors | 3 | Candidate price regimes |
| Goods | 3 | Bundles are costed under each price vector |
| Focal example | Case A | GARP passes but GAPP fails |
| Focal price violations | 2 | Cyclic price-preference contradictions |

## Solution Method

The script first runs the usual GARP test on bundles. It then transposes the economic comparison: for each observed bundle, compare how costly that same bundle would have been under every observed price vector. Those comparisons form a price preference graph. Warshall closure gives indirect price preference, and GAPP checks whether any indirect price ranking is contradicted by a strict reverse comparison.

The four deterministic examples show that bundle rationalizability and price rationalizability are distinct restrictions rather than two names for the same test.

## Results

The examples occupy all four pass/fail cells. This is the main conceptual point: testing choices over bundles and testing choices over prices are different revealed-preference exercises.

**Bundle GARP and Price GAPP Diagnostics**

| Case   | Economic comparison                 | GARP   | GAPP   |   Bundle violations |   Price violations |
|:-------|:------------------------------------|:-------|:-------|--------------------:|-------------------:|
| A      | Bundle-rational, price-inconsistent | pass   | fail   |                   0 |                  2 |
| B      | Both restrictions pass              | pass   | pass   |                   0 |                  0 |
| C      | Bundle-inconsistent, price-rational | fail   | pass   |                   4 |                  0 |
| D      | Both restrictions fail              | fail   | fail   |                   2 |                  2 |

Rows are candidate price vectors and columns are observed bundles. Entries below one mean the row price vector makes that column's bundle cheaper than its observed price vector did.

<img src="figures/price-cost-ratios.png" alt="Cost ratios used to reveal preferences over price vectors." width="80%">
*Cost ratios used to reveal preferences over price vectors.*

The arrows are not preferences over bundles. They are revealed comparisons among price vectors obtained by holding observed bundles fixed.

<img src="figures/price-preference-graph.png" alt="A cycle in the price-preference graph rejects GAPP." width="80%">
*A cycle in the price-preference graph rejects GAPP.*

Because GARP and GAPP ask different economic questions, a dataset can pass one test and fail the other.

<img src="figures/garp-vs-gapp-cases.png" alt="GARP and GAPP classify the same datasets differently." width="80%">
*GARP and GAPP classify the same datasets differently.*

## Takeaway

Revealed price preference is not a cosmetic relabeling of GARP. GARP tests whether chosen bundles can be rationalized by utility maximization. GAPP tests whether price vectors are themselves consistently ranked by revealed cost comparisons. The same price-quantity data can pass one test and fail the other, so the right diagnostic depends on whether the empirical question is about bundles or price regimes.

## Reproduce

```bash
python run.py
```

## References

- Deb, R., Kitamura, Y., Quah, J. K. H., & Stoye, J. (2023). Revealed price preference: Theory and empirical analysis. Review of Economic Studies, 90(2), 707-743.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
- Chambers, C. P., & Echenique, F. (2016). Revealed Preference Theory. Cambridge University Press.
