# Vertical Contracts and Vending Assortments

> All-unit discounts, slotting fees, and assortment choice in a capacity-constrained vending channel.

## Overview

Vending and retail contracts often combine per-unit wholesale prices with fixed payments, rebates, or slotting fees. Those terms matter because the downstream firm chooses which products receive scarce shelf or machine space. A contract can change assortment even when retail prices show little variation.

The tutorial builds a small vending-machine problem with seven confection slots. The retailer chooses the assortment that maximizes its own payoff. The manufacturer can use an all-unit discount or fixed payments to shift which products are stocked.

## Equations

For product $j$, demand is:
$$q_j(p_j) = a_j - bp_j$$

Given wholesale price $w_j$, the retailer chooses:
$$p_j^*(w_j) = \frac{a_j + bw_j}{2b}$$

The retailer chooses an assortment $A$ with capacity $K$:
$$A^* = \arg\max_{A: |A|=K} \sum_{j\in A}(p_j-w_j)q_j + F_j(A)$$

All-unit discounts make $w_j$ depend on whether the manufacturer's shelf target is met.
Slotting fees enter through fixed payments $F_j(A)$.

## Model Setup

| Object | Value |
|--------|-------|
| Products | 12 candy and snack alternatives |
| Machine capacity | 7 slots |
| Wholesale only | Linear wholesale prices, no fixed payments |
| All-unit discount | Mars wholesale price falls if enough Mars products are stocked |
| Slotting fees | Fixed payments reward placement, separate from per-unit margins |

## Solution Method

The assortment problem is solved by enumerating all feasible seven-product subsets. For each subset and contract, the script computes retail prices, quantities, downstream variable profit, manufacturer profit, and fixed transfers. Enumeration keeps the economics transparent and is fast because the capacity problem is small.

## Results

The all-unit discount and slotting-fee contracts shift scarce machine slots toward the manufacturer's products. The contract changes product availability, not just transfers between firms.

![Assortment selected under each vertical contract](figures/assortment-selection.png)
*Assortment selected under each vertical contract*

Fixed payments can make the retailer willing to stock products that are not selected under wholesale-only incentives. The manufacturer pays for placement when the incremental product-level margin is worth it.

![Retailer and manufacturer payoffs by contract](figures/profit-incidence.png)
*Retailer and manufacturer payoffs by contract*

A target that is too low gives away margin without changing behavior. A target that is too high may be unreachable. The interesting region is where the threshold changes the retailer's assortment choice.

![Mars slots and manufacturer profit as the all-unit discount threshold changes](figures/rebate-thresholds.png)
*Mars slots and manufacturer profit as the all-unit discount threshold changes*

**Contract outcomes**

| Contract          |   Retailer objective |   Retailer variable profit |   Manufacturer profit |   Fixed fees |   Average price |   Total quantity |   Mars slots |
|:------------------|---------------------:|---------------------------:|----------------------:|-------------:|----------------:|-----------------:|-------------:|
| Wholesale only    |                24.93 |                      24.93 |                 10.92 |          0   |            1.91 |            25.99 |            4 |
| All-unit discount |                28.15 |                      28.15 |                  7.94 |          0   |            1.82 |            27.59 |            5 |
| Slotting fees     |                30.87 |                      24.67 |                  4.63 |          6.2 |            1.88 |            25.79 |            5 |

## Takeaway

Vertical contracts are identification problems as much as pricing problems. A fixed payment does not show up as product-level price variation, yet it can explain why a product is present in the assortment. That is why vending and slotting-fee papers model assortment and transfers together.

## Reproduce

```bash
python run.py
```

## References

- Conlon, C., and Mortimer, J. (2021). JPE article on vertical contracts in vending-machine markets.
- Hristakeva, S. (2022). JPE article on vertical contracts and product selection in retail markets.
- Lecture 9 Slides 2023: Vertical contracts, vending assortments, and slotting fees.
