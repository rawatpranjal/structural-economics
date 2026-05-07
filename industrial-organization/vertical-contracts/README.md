# Vending Assortments Under Vertical Contracts

> Rebates, slotting fees, and exact assortment choice in a capacity-constrained vending channel.

## Overview

Imagine a vending operator with seven slots and twelve snack products. Consumers only see products that make it into the machine, so a manufacturer can change demand without changing the posted retail price directly: it can make its own products more profitable to stock. An all-unit discount lowers the wholesale price once enough Mars products are carried. A slotting fee pays the operator for placement.

The model turns that business problem into an assortment choice. Each selected product has linear retail demand and a wholesale cost, then the retailer prices the products it decided to carry. The computation is needed because each possible assortment can change wholesale terms, fixed transfers, quantities, and upstream profit. Exact enumeration lets the tutorial compare the retailer's preferred assortment under each contract.

## Equations

Let $\mathcal J$ be the product catalog and let $K$ be the number of vending
slots. Product $j$ has demand intercept $a_j$, marginal cost $c_j$, and
manufacturer label $m(j)$. Retail demand is separable:
$$
q_j(p_j)=\max\{a_j-bp_j,0\}.
$$

Given wholesale price $w_j$, the retailer sets the product price by
$$
p_j^{\ast}(w_j)
=\arg\max_{p_j\geq w_j} (p_j-w_j)q_j(p_j).
$$
For an interior product, this gives
$$
p_j^{\ast}(w_j)=\frac{a_j+bw_j}{2b}.
$$

Contract $C$ maps an assortment $A$ into a wholesale price
$w_j^C(A)$ and a fixed transfer $F_j^C(A)$ paid by the upstream side to the
retailer. The retailer chooses
$$
A_C^{\ast}
=\arg\max_{A\subset\mathcal J:\ |A|=K}
\sum_{j\in A}\left[(p_j^{\ast}-w_j^C(A))q_j(p_j^{\ast})+F_j^C(A)\right].
$$

The upstream payoff reported in the results is
$$
\Pi_C^U(A)=
\sum_{j\in A}\left[(w_j^C(A)-c_j)q_j(p_j^{\ast})-F_j^C(A)\right].
$$

In the all-unit discount case, the retailer pays a lower wholesale price on
Mars products only if the selected assortment contains at least $\tau$ Mars
products:
$$
w_j^C(A)=c_j+\mu-d\,\mathbf 1\{m(j)=\text{Mars}\}\mathbf 1\{M(A)\geq\tau\},
\quad
M(A)=\sum_{j\in A}\mathbf 1\{m(j)=\text{Mars}\}.
$$
Slotting fees instead leave wholesale margins unchanged and work through
$F_j^C(A)$.

## Model Setup

One machine can hold seven of twelve products. Products differ in demand intercepts and costs; Mars controls five products and rivals control seven. The calibration is stylized, but it keeps the assortment tradeoff transparent. The retailer selects the assortment and prices selected products. Upstream profit is shown separately because discounts and slotting fees shift money between the two sides of the vertical relationship.

| Object | Value |
|--------|-------|
| Products | 12 candy and snack alternatives |
| Machine capacity | 7 slots |
| Demand | Product-specific intercepts with common slope $b=4$ |
| Wholesale-only contract | Per-unit margin $\mu=0.42$, no fixed transfers |
| All-unit discount | Mars margin falls by $d=0.18$ once the shelf target is met |
| Slotting-fee contract | Fixed payments of 1.10 for Mars products and 0.35 for rival products |

## Solution Method

The assortment optimum is a finite subset search. There are only
$\binom{12}{7}=792$ feasible assortments, so the script can evaluate every
candidate exactly. A heuristic is unnecessary, and the plotted choices are
the finite-catalog optimum rather than a simulation approximation.

```text
Inputs: product catalog J, capacity K, contract C, rebate threshold tau
Output: optimal assortment A*_C and payoff diagnostics

for each feasible assortment A with |A| = K:
    count Mars products M(A)
    for each product j in A:
        set wholesale price w_j^C(A)
        set fixed transfer F_j^C(A)
        solve the retail pricing first-order condition for p_j*(w_j)
        compute q_j, retailer payoff, and upstream payoff
    add product-level payoffs to get Pi^D_C(A) and Pi^U_C(A)

choose A*_C in argmax_A Pi^D_C(A)
repeat over rebate thresholds tau to see where the all-unit discount binds
```

Enumeration has a useful economic role here. When the contract changes the payoff
from one Mars product, it can also activate a discount on all selected Mars products.
The algorithm keeps that discrete shelf-space response visible.

## Results

The heat map reads the assortment outcome directly. Wholesale-only pricing leaves the retailer with four Mars products. The rebate and slotting-fee contracts both move one more scarce slot toward Mars, even though retail prices are still chosen product by product.

<img src="figures/assortment-selection.png" alt="Assortment selected under each vertical contract" width="80%">

The incidence plot separates the retailer's objective from upstream profit. Slotting fees raise the retailer's payoff because they enter the downstream objective as transfers. Upstream profit falls in this calibration, which is the cost of buying placement.

<img src="figures/profit-incidence.png" alt="Retailer and upstream payoffs by contract" width="80%">

Rebate design is not monotone in the target. A low target gives away margin on products the retailer would have stocked anyway. A high target may fail to move the assortment. The useful region is where the threshold changes the exact assortment optimum relative to the wholesale-only benchmark.

<img src="figures/rebate-thresholds.png" alt="Mars slots and upstream profit as the all-unit discount threshold changes" width="80%">

The table collects the objects behind the figures. Average retail prices move little compared with product availability, which is why the assortment choice is the object of interest.

**Contract outcomes**

| Contract          |   Retailer objective |   Retailer variable profit |   Upstream profit |   Fixed fees |   Average price |   Total quantity |   Mars slots |
|:------------------|---------------------:|---------------------------:|------------------:|-------------:|----------------:|-----------------:|-------------:|
| Wholesale only    |                24.93 |                      24.93 |             10.92 |          0   |            1.91 |            25.99 |            4 |
| All-unit discount |                28.15 |                      28.15 |              7.94 |          0   |            1.82 |            27.59 |            5 |
| Slotting fees     |                30.87 |                      24.67 |              4.63 |          6.2 |            1.88 |            25.79 |            5 |

## Takeaway

The main lesson is about availability. In this calibration, all-unit discounts and slotting fees change which products get scarce slots while average retail prices move only a little. Empirical work on vertical contracts therefore needs product availability and transfers, not only posted prices.

## References

- Conlon, C., and Mortimer, J. (2021). JPE article on vertical contracts in vending-machine markets.
- Hristakeva, S. (2022). JPE article on vertical contracts and product selection in retail markets.
- Lecture 9 Slides 2023: Vertical contracts, vending assortments, and slotting fees.
