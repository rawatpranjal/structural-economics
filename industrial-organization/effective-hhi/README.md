# HHI, Effective Firms, and Merger Screens

> Market concentration is a useful antitrust screen, but it is not a model of competitive effects.

## Overview

HHI answers a narrow but important question: how concentrated is control of sales in a relevant market? It is cheap to compute, transparent to explain, and therefore useful as an early merger screen. The same simplicity is also its limitation. HHI knows ownership shares, not diversion ratios, entry, efficiencies, or demand curvature.

This tutorial keeps that distinction explicit. The first part works through the index arithmetic, including the effective number of equal-sized firms implied by a given HHI. The second part puts the same ownership change inside a four-product Bertrand model. When products are segmented, HHI rises but prices do not move. When products substitute, common ownership changes the pricing FOC. For the fuller counterfactual exercise, compare the neighboring [merger simulation](../merger-simulation/) and [logit supply-side](../logit-supply-side/) tutorials.

## Equations

Let firms be indexed by $f=1,\ldots,F$, with market shares $s_f$ measured as
fractions that sum to one. The Herfindahl-Hirschman Index is

$$
\text{HHI}=10{,}000\sum_{f=1}^{F}s_f^2.
$$

The associated effective number of equal-sized firms is

$$
N_{\text{eff}}=\frac{1}{\sum_f s_f^2}=\frac{10{,}000}{\text{HHI}}.
$$

Thus a market with HHI 2,000 has the same concentration as five symmetric
firms, even if the actual firm count is different.

If firms $a$ and $b$ merge while all quantities are held fixed, the arithmetic
change is

$$
\Delta\text{HHI}
=10{,}000[(s_a+s_b)^2-s_a^2-s_b^2]
=20{,}000 s_a s_b.
$$

For product-level data, product $j$ belongs to firm $f(j)$ and sells quantity
$q_j$. Firm shares aggregate product quantities:

$$
s_f=\frac{\sum_{j:f(j)=f}q_j}{\sum_{\ell}q_{\ell}}.
$$

The small structural comparison uses linear differentiated-products demand,

$$
q(p)=a+Dp,\qquad D_{jj}=\alpha<0,\quad D_{jk}=\beta\geq 0\ (j\neq k).
$$

Let $\Omega_{jk}=1$ if products $j$ and $k$ are commonly owned. Bertrand-Nash
prices satisfy

$$
q(p)+(\Omega\circ D^\top)(p-c)=0.
$$

The 2023 DOJ/FTC Merger Guidelines treat HHI above 1,800 as highly
concentrated and an HHI increase above 100 points as significant for the
structural presumption. The tutorial also reports the familiar category scale:
below 1,000 is unconcentrated, 1,000 to 1,800 is moderately concentrated, and
above 1,800 is highly concentrated.

## Model Setup

The index calculations use share vectors chosen to isolate firm count from asymmetry. The Bertrand comparison uses four products. Initially each product is owned by a separate firm; the merger puts products 1 and 2 under common ownership.

| Object | Value | Role |
|--------|-------|------|
| $F$ | 1 to 100 | Firm counts for the symmetric HHI benchmark |
| $s_f$ | Several share vectors | Firm shares used in the concentration table |
| Products | 4 | Two high-demand products and two lower-demand products |
| $\alpha$ | -1.0 | Own-price slope in linear demand |
| $\beta_{\text{seg}}$ | 0.0 | No cross-price substitution |
| $\beta_{\text{diff}}$ | 0.1 | Positive cross-price substitution |
| Merger | products 1 and 2 | Same ownership change in both demand environments |

## Solution Method

The concentration part is exact arithmetic. The only equilibrium computation is the four-product pricing problem, where the ownership matrix changes the markup equation.

```text
Inputs: firm shares s, product quantities q, costs c, demand slopes D, ownership f(j)
Outputs: HHI, effective firm count, delta-HHI, equilibrium prices

1. For each market, compute HHI = 10000 * sum_f s_f^2.
2. Report N_eff = 10000 / HHI to put asymmetric markets on a symmetric scale.
3. For a candidate merger (a,b), compute delta-HHI = 20000 * s_a * s_b.
4. In product data, aggregate q_j to firm shares using the ownership map f(j).
5. Build Omega_jk = 1[f(j) = f(k)].
6. Solve q(p) + (Omega .* D') (p - c) = 0 for Bertrand-Nash prices.
7. Recompute firm shares and HHI under the post-merger ownership map.
```

Step 6 is solved by root finding. In this linear example the root is a numerical way to solve a small system of first-order conditions, not the economic point of the tutorial. The economic point is that HHI is an ownership screen, while the pricing effect appears only through substitution and the Bertrand FOC.

## Results

The screen and the pricing model give different objects. In the segmented case, the merged products are independent, so prices and total quantity are unchanged. HHI still jumps because the two product shares are now counted under one owner. With positive cross-price substitution, common ownership also changes the pricing problem, so the merged products become more expensive.

| Demand environment | HHI before | HHI after | $\Delta$HHI | Merged-price change | Total-output change |
|---|---:|---:|---:|---:|---:|
| Segmented ($\beta=0.0$) | 3125 | 5937 | 2812 | 0.00% | 0.00% |
| Differentiated ($\beta=0.1$) | 2600 | 4288 | 1688 | 1.69% | -2.61% |

For symmetric firms, HHI is exactly $10{,}000/N$. Moving from monopoly to five equal firms does most of the work: HHI falls from 10,000 to 2,000. The highly concentrated threshold of 1,800 corresponds to about 5.6 equal-sized firms, while the unconcentrated threshold of 1,000 corresponds to ten equal firms.

<img src="figures/hhi-vs-nfirms.png" alt="HHI equals 10000/N for equal-sized firms, with DOJ/FTC threshold regions shaded" width="80%">

The merger bars are pure index arithmetic. The same formula, $20{,}000 s_a s_b$, makes a 40-30 merger much larger than a merger of two small firms. That is why HHI is informative as a first screen, even before estimating demand.

<img src="figures/merger-delta-hhi.png" alt="HHI before and after merger of the two largest firms across market structures" width="80%">

The Lorenz curves show what the scalar index compresses. Equal firms stay on the diagonal. A dominant firm bends the curve away from the diagonal because many firms account for little output while one firm accounts for most of it. The table below turns the same share vectors into HHI and effective firm counts.

<img src="figures/lorenz-curves.png" alt="Lorenz curves: more bowed curves indicate greater concentration and higher HHI" width="80%">

The effective firm count makes asymmetry visible. A 70-10-10-10 market has four firms, but its HHI of 5,200 is equivalent to fewer than two equal-sized firms. Firm count alone would miss that concentration.

**HHI for Example Market Structures**

| Market Structure                |   N Firms |   Top Share (%) |   HHI |   Effective N | Classification          |
|:--------------------------------|----------:|----------------:|------:|--------------:|:------------------------|
| Perfect competition (100 firms) |       100 |               1 |   100 |        100    | Unconcentrated          |
| 10 equal firms                  |        10 |              10 |  1000 |         10    | Moderately Concentrated |
| 5 equal firms                   |         5 |              20 |  2000 |          5    | Highly Concentrated     |
| Asymmetric (40-30-20-10)        |         4 |              40 |  3000 |          3.33 | Highly Concentrated     |
| Duopoly (50-50)                 |         2 |              50 |  5000 |          2    | Highly Concentrated     |
| Dominant firm (70-10-10-10)     |         4 |              70 |  5200 |          1.92 | Highly Concentrated     |
| Near-monopoly (90-5-5)          |         3 |              90 |  8150 |          1.23 | Highly Concentrated     |
| Monopoly                        |         1 |             100 | 10000 |          1    | Highly Concentrated     |

## Takeaway

HHI is valuable because it is transparent: it converts shares into a concentration number and gives a closed-form delta for mergers. But the segmented-product example is the warning label. Ownership aggregation can raise HHI even when the maintained demand model implies no price effect. Once products substitute, the same ownership change works through the Bertrand FOC and prices move. In applied work, HHI should start the antitrust conversation, not end it.

## References

- U.S. Department of Justice & Federal Trade Commission (2023). *Merger Guidelines*.
- Werden, G. (1991). "A Robust Test for Consumer Welfare Enhancing Mergers Among Sellers of Differentiated Products." *Journal of Industrial Economics*, 39(4).
- Farrell, J. and Shapiro, C. (1990). "Horizontal Mergers: An Equilibrium Analysis." *American Economic Review*, 80(1), 107-126.
- Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 5.
