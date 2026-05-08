# Differentiated-Products Merger Pricing with Logit Demand

> Ownership, diversion, and Bertrand-Nash price effects in a four-product market.

## Overview

Products 1 and 2 begin as separate firms in a four-product category. A merger puts both products under one owner.

The object is the post-merger Bertrand price vector. When Product 1 raises price, some lost sales go to Product 2. Common ownership makes those sales count for the same firm.

The computation calibrates logit demand and marginal costs from prices, shares, and one margin. It then solves the pricing FOCs after changing ownership and costs.

## Equations

There are $J$ inside products and an outside good. Product $j$ has price $p_j$,
marginal cost $c_j$, and mean non-price utility $\xi_j$. With
$\alpha<0$, mean utility is
$$\delta_j(p)=\xi_j+\alpha p_j.$$

Logit demand gives inside share
$$
s_j(p)=
\frac{\exp(\delta_j(p))}
{1+\sum_{\ell=1}^J \exp(\delta_\ell(p))},
\qquad
s_0(p)=
\frac{1}
{1+\sum_{\ell=1}^J \exp(\delta_\ell(p))}.
$$

The demand derivative used by the pricing equation is
$$
\frac{\partial s_k}{\partial p_j}
=\alpha s_k(\mathbf 1\{j=k\}-s_j).
$$

Let $\Omega_{jk}=1$ when products $j$ and $k$ are controlled by the same firm.
It is zero otherwise. Bertrand-Nash pricing satisfies one condition per product.
$$
0=s_j(p)+\sum_{k=1}^J
\Omega_{jk}(p_k-c_k)\frac{\partial s_k(p)}{\partial p_j}.
$$
Let $\Delta_{jk}=\partial s_j/\partial p_k$. The markup equation is
$$
p-c=-(\Omega\circ \Delta')^{-1}s.
$$

The diversion ratio records where a lost sale goes. For $j\neq k$,
$$
D_{j\to k}=\frac{s_k}{1-s_j},\qquad j\neq k.
$$
Under simple logit this depends only on product $k$'s share and the outside
option.

## Model Setup

The calibration uses one small category. Products 1 and 2 are merger partners. Products 3 and 4 stay outside the deal. The numbers show how shares and prices imply markups.

| Object | Value | Role |
|--------|-------|------|
| Inside products | 4 | Four single-product firms before the merger |
| Inside shares | [0.15, 0.15, 0.30, 0.30] | Observed product shares |
| Outside share | 0.10 | No-purchase option |
| Prices | [1.00, 1.00, 1.00, 1.00] | Pre-merger prices |
| Product 1 margin | 0.50 | Pins down the logit price coefficient |
| $\alpha$ | -2.3529 | Calibrated price sensitivity |
| Marginal costs | [0.50, 0.50, 0.39, 0.39] | Recovered from the pre-merger FOCs |
| Scenarios | merger 1+2, merger 1+2 with lower costs | Ownership and cost experiments |

## Solution Method

The algorithm makes observed pre-merger prices an equilibrium of the calibrated model. It then changes the ownership matrix and resolves the same pricing equations. Each price changes every product's share, so the system is nonlinear.

```text
Inputs: pre-merger prices p, shares s, firm labels f(j),
        one observed margin, and new ownership labels
Outputs: calibrated demand/costs and equilibrium outcomes by scenario

1. Compute the outside share: s0 = 1 - sum_j s_j.
2. Infer alpha from Product 1's observed margin and single-product FOC.
3. Back out mean utilities: xi_j = log(s_j / s0) - alpha p_j.
4. Form Delta(p), the logit demand Jacobian at observed prices.
5. Invert the pre-merger markup equation to recover marginal costs c.
6. Replace Omega and c when ownership or efficiencies change.
7. Use root finding on the Bertrand FOCs F(p; Omega, c) = 0.
8. Report prices, shares, outside share, and FOC residuals.
```

The pre-merger FOC residual after calibration is 0.00e+00. The post-merger prices solve the same FOCs under new ownership.

## Results

After Products 1 and 2 merge, the common owner raises both prices. A lost sale to the partner product is no longer fully lost. The 10 percent cost reduction mutes the increase but does not erase it.

<img src="figures/price-comparison.png" alt="Equilibrium prices under alternative ownership" width="80%">

Volume leaves the merged products after their price increases. Some sales move to rival inside products, and some leave the inside market through the outside good. The outside option limits how much price pressure any owner can internalize.

<img src="figures/share-comparison.png" alt="Market shares and the outside option" width="80%">

Each row is the product losing a marginal sale; each column is the product that receives it. Under logit, larger-share products absorb more diverted sales from every other product.

<img src="figures/diversion-ratios.png" alt="Diversion ratios between products" width="80%">

The table reports average prices and shares by scenario. The FOC residual checks that the reported prices solve the post-merger pricing equations.

**Merger simulation outcomes**

| Scenario                |   Avg Price |   Price Change (%) |   Inside Share |   Outside Share |   FOC Residual |
|:------------------------|------------:|-------------------:|---------------:|----------------:|---------------:|
| Pre-merger              |      1      |               0    |         0.9    |          0.1    |        0       |
| Merger 1+2              |      1.0456 |               4.56 |         0.8928 |          0.1072 |        9.2e-15 |
| Merger 1+2, lower costs |      1.0241 |               2.41 |         0.8962 |          0.1038 |        1.2e-15 |

## Takeaway

Merger simulation here is an ownership-matrix exercise inside a pricing FOC. Diversion gives the merged firm an upward pricing incentive. Cost reductions push in the other direction.

## References

- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics*, 25(2).
- Werden, G. and Froeb, L. (1994). "The Effects of Mergers in Differentiated Products Industries." *Journal of Law, Economics, & Organization*, 10(2).
