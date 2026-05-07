# Differentiated-Products Merger Pricing with Logit Demand

> Ownership, diversion, and Bertrand-Nash price effects in a four-product market.

## Overview

Take a four-product category with four single-product firms. Products 1 and 2 are candidates for common ownership. If Product 1 raises its price before the merger, some consumers leave Product 1 for Product 2, but Firm 1 does not earn those sales. After the merger, that same diversion becomes revenue inside the merged firm. Pricing incentives change even if demand and marginal costs are unchanged.

To quantify the price effect, the tutorial builds the demand and supply objects used in merger simulation. It calibrates a simple logit model to pre-merger prices, shares, and one observed margin, recovers marginal costs from the Bertrand first-order conditions, then solves those conditions again under new ownership and under a cost-efficiency case. The core computation is a nonlinear root-finding problem: find prices that make every product's pricing condition hold for a given ownership matrix. [BLP random coefficients](../blp-random-coefficients/) relaxes the logit substitution pattern; [merger simulation across demand systems](../merger-simulation/) compares the consequences of that modeling choice.

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

Let $\Omega_{jk}=1$ when products $j$ and $k$ are controlled by the same firm
and zero otherwise. Bertrand-Nash pricing satisfies, for each product $j$,
$$
0=s_j(p)+\sum_{k=1}^J
\Omega_{jk}(p_k-c_k)\frac{\partial s_k(p)}{\partial p_j}.
$$
If $\Delta_{jk}=\partial s_j/\partial p_k$, the markup equation is
$$
p-c=-(\Omega\circ \Delta')^{-1}s.
$$

The diversion ratio from product $j$ to product $k$ is
$$
D_{j\to k}=\frac{s_k}{1-s_j},\qquad j\neq k.
$$
Under simple logit this depends only on product $k$'s share and the outside
option. That is the IIA restriction: substitution is not allowed to depend on
which products are objectively closer substitutes.

For welfare comparisons, the logit consumer-surplus index is
$$
CS(p)=\frac{1}{-\alpha}
\log\left(1+\sum_{j=1}^J \exp(\xi_j+\alpha p_j)\right),
$$
up to the usual income constant.

## Model Setup

The calibration is deliberately small. Products 1 and 2 are candidate merger partners in one category; products 3 and 4 remain rival products. The numbers are not estimated from a dataset. They make the mapping from shares and prices to markups visible.

| Object | Value | Role |
|--------|-------|------|
| Inside products | 4 | Four single-product firms before the merger |
| Inside shares | [0.15, 0.15, 0.30, 0.30] | Observed product shares |
| Outside share | 0.10 | No-purchase option |
| Prices | [1.00, 1.00, 1.00, 1.00] | Pre-merger prices |
| Product 1 margin | 0.50 | Pins down the logit price coefficient |
| $\alpha$ | -2.3529 | Calibrated price sensitivity |
| Marginal costs | [0.50, 0.50, 0.39, 0.39] | Recovered from the pre-merger FOCs |
| Counterfactuals | merger 1+2, merger 1+2 with lower costs, common ownership | Ownership and cost experiments |

## Solution Method

The algorithm first makes the observed pre-merger prices an equilibrium of the calibrated model. It then asks how the same demand system prices the category when the ownership matrix changes. Because each price changes shares for every product, the post-merger calculation is a coupled nonlinear system rather than four separate markup calculations.

```text
Inputs: pre-merger prices p, shares s, firm labels f(j),
        one observed margin, and counterfactual ownership labels
Outputs: calibrated demand/costs and equilibrium outcomes by scenario

1. Compute the outside share: s0 = 1 - sum_j s_j.
2. Infer alpha from Product 1's observed margin and single-product FOC.
3. Back out mean utilities: xi_j = log(s_j / s0) - alpha p_j.
4. Form Delta(p), the logit demand Jacobian at observed prices.
5. Invert the pre-merger markup equation to recover marginal costs c.
6. Replace Omega and c when ownership or efficiencies change.
7. Use root finding on the Bertrand FOCs F(p; Omega, c) = 0.
8. Report prices, shares, outside share, consumer surplus, HHI, and residuals.
```

The pre-merger FOC residual after calibration is 0.00e+00. The post-merger solutions below come from the same pricing equations, not from a reduced-form pass-through rule.

## Results

After Products 1 and 2 merge, the common owner raises both prices because a lost sale to the partner product is no longer fully lost. Products 3 and 4 also rise because prices are strategic complements in this demand system. The 10 percent cost reduction mutes the increase but does not erase it.

<img src="figures/price-comparison.png" alt="Equilibrium prices under alternative ownership" width="80%">

Volume leaves the merged products after their price increases. Some sales move to rival inside products, and some leave the inside market through the outside good. The outside option limits how much price pressure any owner can internalize.

<img src="figures/share-comparison.png" alt="Market shares and the outside option" width="80%">

Each row is the product losing a marginal sale; each column is the product that receives it. Under logit, larger-share products absorb more diverted sales from every other product. This makes the first merger calculation transparent, but it also shows why richer demand systems matter when product closeness drives the case.

<img src="figures/diversion-ratios.png" alt="Diversion ratios between products" width="80%">

HHI is computed on inside-good firm shares, so it captures the ownership change rather than the outside option. Consumer surplus is reported as a change from the pre-merger calibration. The FOC residuals check that the reported prices solve the post-merger pricing equations.

**Merger simulation outcomes**

| Scenario                |   Avg Price |   Price Change (%) |   Inside Share |   Outside Share |   CS Change |   HHI |   FOC Residual |
|:------------------------|------------:|-------------------:|---------------:|----------------:|------------:|------:|---------------:|
| Pre-merger              |      1      |               0    |         0.9    |          0.1    |      0      |  2778 |        0       |
| Merger 1+2              |      1.0456 |               4.56 |         0.8928 |          0.1072 |     -0.0296 |  3351 |        9.2e-15 |
| Merger 1+2, lower costs |      1.0241 |               2.41 |         0.8962 |          0.1038 |     -0.016  |  3338 |        1.2e-15 |
| Common ownership        |      1.6808 |              68.08 |         0.6557 |          0.3443 |     -0.5254 | 10000 |        8.9e-16 |

## Takeaway

Merger simulation here is an ownership-matrix exercise inside a pricing first-order condition. The same calibrated demand and costs support separate firms, common ownership, cost efficiencies, and monopoly. Diversion gives the merged firm its upward pricing incentive, while cost reductions push in the other direction. Simple logit makes diversion transparent, but its IIA pattern also limits what this benchmark can say about product closeness.

## References

- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics*, 25(2).
- Werden, G. and Froeb, L. (1994). "The Effects of Mergers in Differentiated Products Industries." *Journal of Law, Economics, & Organization*, 10(2).
- Nevo, A. (2000). "Mergers with Differentiated Products: The Case of the Ready-to-Eat Cereal Industry." *RAND Journal of Economics*, 31(3).
