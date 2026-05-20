# Vending Assortments Under Vertical Contracts

## Overview

Vending space is scarce. A retailer has seven slots and twelve snack products. A manufacturer can use contract terms to make its products more attractive to stock.

The object is the retailer's assortment. Each selected product has linear demand, a wholesale price, and a retail price chosen after stocking. The contracts are wholesale pricing, an all-unit discount, and slotting fees.

Exact enumeration checks every feasible seven-product subset. One extra Mars item can trigger a rebate on all Mars items. The script compares the retailer's best assortment under each contract.

## Equations

Let $\mathcal J$ be the product catalog and let $K$ be the number of vending
slots. Product $j$ has demand intercept $a_j$, marginal cost $c_j$, and
manufacturer label $m(j)$. Retail demand is separable:

$$
q_j(p_j)=\max\{a_j-bp_j,0\}.
$$

Here $b$ is the common demand slope. Given wholesale price $w_j$, the retailer sets the product price by

$$
p_j^{\ast}(w_j)
=\arg\max_{p_j\geq w_j} (p_j-w_j)q_j(p_j).
$$

For an interior product, the unconstrained optimum is $(a_j+bw_j)/(2b)$. The
retailer never prices below a small markup over wholesale, so the price used
is

$$
p_j^{\ast}(w_j)=\max\lbrace (a_j+bw_j)/(2b),\ w_j+\epsilon\rbrace,
\quad \epsilon=0.05.
$$

The margin floor $\epsilon=0.05$ is a numerical guard; it never binds under
the calibration in this tutorial.

Contract $C$ maps assortment $A$ into wholesale prices and fixed transfers.
The upstream side pays $F_j^C(A)$ to the retailer. The retailer chooses

$$
A_C^{\ast}
=\arg\max_{A\subset\mathcal J:\ |A|=K}
\sum_{j\in A}\left[(p_j^{\ast}-w_j^C(A))q_j(p_j^{\ast})+F_j^C(A)\right].
$$

Write $\Pi^D_C(A)$ for the sum inside the argmax, the retailer's total payoff under contract $C$.

The upstream payoff reported in the results is

$$
\Pi_C^U(A)=
\sum_{j\in A}\left[(w_j^C(A)-c_j)q_j(p_j^{\ast})-F_j^C(A)\right].
$$

In the all-unit discount case, Mars products can receive a lower wholesale
price. The discount applies only if the assortment contains at least $\tau$
Mars products, with $\tau=4$ in this tutorial:

$$
w_j^C(A)=c_j+\mu-d\,\mathbf 1\{m(j)=\text{Mars}\}\mathbf 1\{M(A)\geq\tau\},
\quad
M(A)=\sum_{j\in A}\mathbf 1\{m(j)=\text{Mars}\}.
$$

Slotting fees instead leave wholesale margins unchanged and work through
$F_j^C(A)$.

## Model Setup

One machine can hold seven of twelve products. Mars controls five products, and rivals control seven. Demand intercepts and costs differ by product. The retailer chooses the assortment and then sets selected product prices. Upstream profit is reported because transfers move surplus across the channel.

| Object | Value |
|--------|-------|
| Products | 12 candy and snack alternatives |
| Machine capacity | 7 slots |
| Demand | Product-specific intercepts with common slope $b=4$ |
| Wholesale-only contract | Per-unit margin $\mu=0.42$, no fixed transfers |
| All-unit discount | Mars margin falls by $d=0.18$ once the shelf holds at least $\tau=4$ Mars products |
| Slotting-fee contract | Fixed payments of 1.10 for Mars products and 0.35 for rival products |

## Solution Method

The assortment problem is a finite subset search. With twelve products and
seven slots, there are 792 feasible assortments. The script evaluates each
subset exactly and keeps the one with the highest retailer objective.

```text
Inputs: product catalog J, capacity K, contract C
Output: optimal assortment A*_C and payoffs

for each feasible assortment A with |A| = K:
    count Mars products M(A)
    for each product j in A:
        set wholesale price w_j^C(A)
        set fixed transfer F_j^C(A)
        solve the retail pricing first-order condition for p_j*(w_j)
        compute q_j, retailer payoff, and upstream payoff
    add product-level payoffs to get Pi^D_C(A) and Pi^U_C(A)

choose A*_C in argmax_A Pi^D_C(A)
```

Enumeration keeps the shelf-space margin visible. One Mars slot can activate
lower wholesale prices on other Mars products. No simulation approximation is
needed.

## Results

The heat map shows selected products. Wholesale pricing leaves the retailer with four Mars items. The rebate and slotting fee each add one Mars slot. Average retail prices change little, so availability carries the response.

<img src="figures/assortment-selection.png" alt="Assortment selected under each vertical contract" width="80%">

The table gives the payoff objects behind the assortment choice. The retailer objective includes fixed fees. Upstream profit subtracts those transfers.

**Contract outcomes**

| Contract          |   Retailer objective |   Retailer variable profit |   Upstream profit |   Fixed fees |   Average price |   Total quantity |   Mars slots |
|:------------------|---------------------:|---------------------------:|------------------:|-------------:|----------------:|-----------------:|-------------:|
| Wholesale only    |                24.93 |                      24.93 |             10.92 |          0   |            1.91 |            25.99 |            4 |
| All-unit discount |                28.15 |                      28.15 |              7.94 |          0   |            1.82 |            27.59 |            5 |
| Slotting fees     |                30.87 |                      24.67 |              4.63 |          6.2 |            1.88 |            25.79 |            5 |

## Takeaway

Vertical contracts can change availability without large retail price movement. In this example, rebates and slotting fees move scarce slots toward Mars. Empirical work on vertical contracts needs product availability and transfers.

## References

- Conlon, C., and Mortimer, J. (2021). JPE article on vertical contracts in vending-machine markets.
- Hristakeva, S. (2022). JPE article on vertical contracts and product selection in retail markets.
- Lecture 9 Slides 2023: Vertical contracts, vending assortments, and slotting fees.
