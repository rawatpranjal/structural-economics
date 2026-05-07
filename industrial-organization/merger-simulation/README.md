# Merger Pricing in Differentiated-Products Markets

> GUPPI screens and Bertrand-Nash counterfactuals with calibrated substitution.

## Overview

A merger between close substitutes changes pricing incentives even if marginal costs do not move. Before the merger, Firm 1 treats sales lost to Firm 2 as business lost to a rival. After the merger, some of those consumers stay inside the same portfolio. That internalized diversion is the unilateral price effect an antitrust merger simulation tries to measure.

The missing object is the substitution matrix. Shares, prices, and margins describe the observed market, but they do not directly reveal how many consumers leave Product 1 for Product 3 when Product 1 becomes more expensive. This tutorial fixes a six-product market, calibrates logit, linear, and log-linear demand to the same observables, changes only ownership, and solves the post-merger Bertrand-Nash pricing conditions.

GUPPI and CMCR summarize local upward pricing pressure at observed prices. The full counterfactual asks what happens after every firm reacts and after claimed cost efficiencies shift marginal costs. The logit-only [Bertrand pricing](../bertrand-logit-demand/) tutorial isolates the ownership matrix in one demand model. The [BLP random coefficients](../blp-random-coefficients/) tutorial shows how richer substitution patterns can be estimated. This page focuses on the gap between first-order screens and a solved post-merger equilibrium.

## Equations

The target object is the counterfactual price vector after ownership changes.
There are $J$ inside products. Product $j$ has price $p_j$, marginal cost
$c_j$, quantity or share $q_j(p)$, and owner $f(j)$. The ownership matrix is

$$
\Omega_{jk}=\mathbf 1\{f(j)=f(k)\}.
$$

For a multi-product Bertrand firm, the pricing equation is

$$
0=q_j(p)+\sum_{k=1}^J
\Omega_{jk}(p_k-c_k)\frac{\partial q_k(p)}{\partial p_j},
\qquad j=1,\ldots,J .
$$

With $\Delta_{kj}(p)=\partial q_k(p)/\partial p_j$, the vector equation is

$$
q(p)+(\Omega\circ \Delta(p)') (p-c)=0.
$$

The three demand systems are calibrated to the same observed market:

$$
s_j^{L}(p)=
\frac{\exp(\xi_j+\alpha p_j)}
{1+\sum_{\ell=1}^J \exp(\xi_\ell+\alpha p_\ell)},
\qquad \alpha<0,
$$

$$
q_j^{A}(p)=a_j-\sum_{k=1}^J B_{jk}p_k,
$$

and

$$
\log q_j^{E}(p)=a_j^E+\sum_{k=1}^J E_{jk}\log p_k .
$$

The local diversion ratio from product $j$ to product $k$ is

$$
D_{j\to k}=
-\frac{\partial q_k(p)/\partial p_j}
{\partial q_j(p)/\partial p_j}, \qquad j\neq k.
$$

For products that become newly co-owned after the merger,

$$
UPP_j=\sum_{k:\Omega^{post}_{jk}=1,\ \Omega^{pre}_{jk}=0}
D_{j\to k}(p_k-c_k),
$$

with

$$
GUPPI_j=\frac{UPP_j}{p_j},
\qquad
CMCR_j=\frac{UPP_j}{c_j}.
$$

GUPPI is a first-order screen for upward pricing pressure. CMCR reports the
product-level marginal-cost reduction that would offset that pressure at the
observed price vector. The simulation then solves the full pricing system under
post-merger ownership.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| Products $J$ | 6 | 3 firms, 2 products each |
| Shares | [0.12, 0.10, 0.15, 0.13, 0.08, 0.07] | Pre-merger inside shares |
| Prices | [1.00, 1.20, 0.90, 1.10, 1.30, 1.40] | Pre-merger prices |
| Margins | [0.40, 0.35, 0.45, 0.40, 0.30, 0.28] | Price-cost margins |
| Outside share | 0.35 | Outside option in the logit demand system |
| $\alpha$ (logit) | -3.1611 | Calibrated price coefficient |
| Linear cross-slope ratio | 0.10 | Cross-slope relative to geometric mean own-slope |
| Log-linear cross elasticity | 0.15 | Maintained symmetric cross-price elasticity |
| Merger | Firm 1 buys Firm 2 | Products 1-4 move under common ownership |
| Benchmark | Post-merger Bertrand-Nash FOC | Equilibrium used to judge first-order screens |

## Solution Method

Calibration and simulation do different jobs. Calibration makes each demand system pass through the observed market. Simulation keeps those demand primitives fixed, replaces the pre-merger ownership matrix with the post-merger one, and searches for a new Bertrand-Nash price vector.

```text
Algorithm: calibrated merger simulation
Input: observed shares q, prices p, margins m, pre- and post-merger owners f(j)
Output: screening metrics, post-merger prices, and welfare changes
Build Omega_pre and Omega_post from owner labels
for each demand system d in {logit, linear, log-linear}:
    choose demand parameters so q_d(p_obs) matches observed shares
    recover marginal costs c_d from the pre-merger pricing FOC
    evaluate Delta_d(p_obs) and diversion ratios D_d
    compute UPP, GUPPI, and CMCR for newly co-owned products
    solve q_d(p) + (Omega_post .* Delta_d(p)') (p - c_d) = 0
    compare solved price changes with the first-order screens
    compute changes in consumer surplus, producer surplus, and total surplus
for a grid of merger efficiencies:
    reduce costs on the merging products, re-solve the post-merger FOC,
    and interpolate the cost reduction where average merged-product prices stop rising
```

The residual check confirms that each calibration reproduces the observed pricing conditions (logit 1.4e-17, linear 2.8e-17, log-linear 2.8e-17). The results below compare first-order screens with the solved post-merger equilibrium, rather than comparing three separately estimated demand models.

## Results

Products 1-4 move inside the same portfolio after the merger, so the merged firm internalizes diversion among them. The logit and log-linear systems give roughly double-digit average increases for those products. The linear system is more muted under this cross-slope calibration.

<img src="figures/price-comparison.png" alt="Pre- vs post-merger prices across three demand systems. Merging products (1-4) see larger price increases, with magnitudes shaped by the calibrated substitution pattern." width="80%">

The welfare bars separate consumer surplus, producer surplus, and their sum. Consumers lose in every demand system here. Producer surplus rises, but the gain does not offset the consumer loss, so total surplus falls under this calibration.

<img src="figures/welfare-decomposition.png" alt="Welfare decomposition across demand systems: consumers lose, producers may gain, and the net effect depends on the demand model." width="80%">

GUPPI uses observed margins and diversion before the counterfactual prices are solved. The left panel treats the solved post-merger price increase as the benchmark. The gap comes from pass-through, demand curvature, and rival price responses.

<img src="figures/upp-guppi.png" alt="GUPPI screen versus solved equilibrium price effects, with product-level UPP for the products that become newly co-owned." width="80%">

The efficiency curve lowers marginal costs for products 1-4 and re-solves the post-merger pricing problem. The zero markers are interpolated from a fine efficiency grid, so they approximate the solved-equilibrium break-even point. Below the zero line, efficiencies are large enough to reverse the average price increase on the merged products.

<img src="figures/efficiency-frontier.png" alt="How much marginal cost reduction is needed to offset the merger price increase? The break-even point differs substantially across demand models." width="80%">

The table puts local screens and solved counterfactuals side by side. Average actual price increases come from the post-merger FOC solution. GUPPI and CMCR are local screens. The break-even efficiency column comes from re-solving the pricing equilibrium on a finer cost-reduction grid.

**Merger Price Effects and Screens**

| Demand Model   |   Avg Actual Price Inc. (%) |   Max Price Change (%) |   Avg GUPPI Screen (%) |   Screen Gap (pp) |   Avg CMCR Screen (%) |   Break-even Eff. (%) |   Delta CS |   Delta PS |   Delta W |   Post FOC Residual |
|:---------------|----------------------------:|-----------------------:|-----------------------:|------------------:|----------------------:|----------------------:|-----------:|-----------:|----------:|--------------------:|
| Logit          |                       11.15 |                  13.34 |                  11.59 |             -0.44 |                 19.76 |                 34.38 |    -0.0537 |     0.0204 |   -0.0333 |             2.4e-10 |
| Linear         |                        5.13 |                   5.59 |                   7.27 |             -2.15 |                 12.15 |                 16.66 |    -0.0272 |     0.0078 |   -0.0193 |             5.6e-17 |
| Log-linear     |                       10.27 |                  13.57 |                   4.56 |              5.7  |                  7.61 |                  9.29 |    -0.0489 |     0.0065 |   -0.0425 |             3e-12   |

## Takeaway

Merger simulation turns a change in control into an equilibrium price calculation. The ownership matrix is easy to change. The economic content sits in substitution and cost pass-through, so different calibrated demand systems can give different counterfactuals in the same observed market.

UPP, GUPPI, and CMCR are useful triage tools. They point to products with strong internalized diversion, while the FOC solve accounts for every firm's reaction, demand curvature, and cost-efficiency claims. In this calibration, all three systems raise prices and lower consumer surplus, but they disagree on magnitudes and on the efficiency needed to offset the merger.

## References

- Werden, G. and Froeb, L. (1994). "The Effects of Mergers in Differentiated Products Industries: Logit Demand and Merger Policy." *Journal of Law, Economics, & Organization*, 10(2).
- Farrell, J. and Shapiro, C. (2010). "Antitrust Evaluation of Horizontal Mergers: An Economic Alternative to Market Definition." *The B.E. Journal of Theoretical Economics*, 10(1).
- Werden, G. (1996). "A Robust Test for Consumer Welfare Enhancing Mergers Among Sellers of Differentiated Products." *Journal of Industrial Economics*, 44(4).
- Nevo, A. (2000). "Mergers with Differentiated Products: The Case of the Ready-to-Eat Cereal Industry." *RAND Journal of Economics*, 31(3).
- Berry, S., Levinsohn, J., and Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica*, 63(4).
