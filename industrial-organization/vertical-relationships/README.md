# Vertical Relationships and Double Marginalization

> Wholesale pricing, retail pricing, two-part tariffs, and simple vertical restraints.

## Overview

A vertically separated manufacturer and retailer can both have market power. With a linear wholesale price, the retailer treats the wholesale price as a marginal cost and adds its own margin on top. The result is double marginalization: price is too high and quantity is too low relative to the joint-profit benchmark.

Vertical contracts can repair this distortion. A two-part tariff sets the per-unit wholesale price close to marginal cost and uses a fixed fee to transfer surplus. Resale price maintenance can also target the retail price, though its welfare effect depends on the environment.

## Equations

Retail demand is linear:
$$q(p) = a - bp$$

Given wholesale price $w$, the retailer chooses:
$$p_R(w) = \frac{a + b(w+c_R)}{2b}$$

The manufacturer anticipates that response and chooses:
$$w^* = \frac{a/b - c_R + c_M}{2}$$

The integrated channel sets:
$$p^I = \frac{a + b(c_M+c_R)}{2b}$$

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $a$ | 20.0 | Demand intercept |
| $b$ | 2.0 | Demand slope |
| $c_M$ | 2.0 | Manufacturer marginal cost |
| $c_R$ | 1.0 | Retail service cost |
| Contracts | 4 | Integration, linear wholesale, two-part tariff, RPM |

## Solution Method

The model is solved by backward induction. The retailer first solves its static pricing problem for any wholesale price. The manufacturer then chooses the wholesale price anticipating the retailer response. Counterfactual contracts change either the marginal wholesale price, the fixed transfer, or the retail price target.

## Results

The linear wholesale contract produces the highest retail price. The two-part tariff restores the integrated-channel price by removing the manufacturer's margin from the retailer's marginal cost.

<img src="figures/price-quantity.png" alt="Price and quantity by vertical contract" width="80%">
*Price and quantity by vertical contract*

Double marginalization destroys surplus. A two-part tariff improves channel profit and consumer surplus in this simple environment because it lowers the marginal wholesale price.

<img src="figures/surplus-decomposition.png" alt="Consumer surplus and channel profit by contract" width="80%">
*Consumer surplus and channel profit by contract*

A higher wholesale price shifts the retailer's perceived marginal cost. Only part of the increase is passed through to price, but quantity still falls.

<img src="figures/wholesale-pass-through.png" alt="Retail pass-through as wholesale price changes" width="80%">
*Retail pass-through as wholesale price changes*

**Contract outcomes**

| Contract              |   Retail price |   Wholesale price |   Quantity |   Manufacturer profit |   Retailer profit |   Channel profit |   Consumer surplus |   Total surplus |
|:----------------------|---------------:|------------------:|-----------:|----------------------:|------------------:|-----------------:|-------------------:|----------------:|
| Integrated channel    |           6.5  |               2   |        7   |                  0    |             24.5  |            24.5  |              12.25 |           36.75 |
| Linear wholesale      |           8.25 |               5.5 |        3.5 |                 12.25 |              6.12 |            18.38 |               3.06 |           21.44 |
| Two-part tariff       |           6.5  |               2   |        7   |                 24.5  |              0    |            24.5  |              12.25 |           36.75 |
| RPM at monopoly price |           6.5  |               5.5 |        7   |                 24.5  |              0    |            24.5  |              12.25 |           36.75 |

## Takeaway

The central IO lesson is marginal incentives. A fixed fee only transfers surplus, while a high wholesale price changes the retailer's marginal pricing decision. That is why nonlinear pricing can eliminate double marginalization even when the manufacturer still extracts profit.

## Reproduce

```bash
python run.py
```

## References

- Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press.
- Asker, J. (2016). Diagnosing Foreclosure Due to Exclusive Dealing. *Journal of Industrial Economics*, 64(3), 375-410.
- Lecture 7 Slides 2023: Vertical relationships, double marginalization, and restraints.
