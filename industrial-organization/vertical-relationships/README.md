# Double Marginalization in Vertical Supply Chains

> A manufacturer-retailer pricing game solved by backward induction.

## Overview

A manufacturer sells through an independent retailer. The manufacturer sets wholesale terms. The retailer sets the shelf price. Consumers buy from the retailer.

The object is double marginalization. A linear wholesale price makes the retailer treat the upstream markup as marginal cost. The retailer then adds its own markup, so the channel charges too much and sells too little.

The computation solves the integrated channel and the separated game. Backward induction gives the wholesale price, retail price, quantity, and profits under each contract.

## Equations

Demand follows
$$q(p)=a-bp,\qquad p\leq \bar p\equiv a/b,$$
where $p$ is the retail price. The choke price is $\bar p$. Costs are $c_M$
upstream and $c_R$ downstream.

The integrated channel solves
$$\Pi^I(p)=(p-c_M-c_R)q(p),$$
so the joint-profit price is
$$p^I=\frac{\bar p+c_M+c_R}{2}.$$

Under a linear wholesale price $w$, the retailer solves
$$\max_p\ (p-w-c_R)q(p).$$
Its best response is
$$p_R(w)=\frac{\bar p+w+c_R}{2}.$$

The manufacturer chooses $w$ while anticipating that response:
$$\max_w\ (w-c_M)q(p_R(w)),$$
which gives
$$w^{DM}=\frac{\bar p-c_R+c_M}{2}.$$

Because $w^{DM}>c_M$, the retailer acts as if marginal cost is too high.

A two-part tariff sets
$$w^{TPT}=c_M$$
and uses the fixed fee
$$F=(p^I-c_M-c_R)q(p^I)$$
to transfer operating profit upstream.

The fee changes the profit split without changing the retailer's margin.

## Model Setup

The calibration is small enough to solve analytically. Each number uses the same unit. The integrated channel is a benchmark, not an ownership assumption.

| Parameter | Value | Description |
|-----------|-------|-------------|
| $a$ | 20.0 | Demand intercept |
| $b$ | 2.0 | Demand slope |
| $\bar p=a/b$ | 10.0 | Choke price |
| $c_M$ | 2.0 | Manufacturer marginal cost |
| $c_R$ | 1.0 | Retail service cost |
| Contracts | 3 | Integrated benchmark, linear wholesale, two-part tariff |

## Solution Method

The solution follows the order of moves. Each step uses the same demand curve, so price and quantity are comparable across contracts.

```text
Inputs: demand q(p)=a-bp, costs c_M and c_R

1. Integrated channel
    p_I = (a/b + c_M + c_R) / 2
    q_I = q(p_I)

2. Linear wholesale game
    For a candidate wholesale price w:
        retailer sets p_R(w) = (a/b + w + c_R) / 2
    Manufacturer chooses w_DM to maximize (w-c_M) q(p_R(w))
    Evaluate p_R(w_DM), q(p_R(w_DM)), profits, and surplus

3. Two-part tariff
    Two-part tariff: set w=c_M, p=p_I, and fixed fee F=(p_I-c_M-c_R)q_I

Outputs: contract outcomes and pass-through curve p_R(w)
```

The comparison treats the fixed fee as a transfer. It changes the profit split, not the retailer's marginal cost.

## Results

The integrated channel charges $6.50$ and sells 7.0 units. Linear wholesale pricing raises the retail price to $8.25$ and cuts quantity to 3.5. The two-part tariff returns price and quantity to the integrated line.

<img src="figures/price-quantity.png" alt="Price and quantity by vertical contract" width="80%">

The wholesale-price sweep varies only $w$. The retailer's best response price rises with $w$. At $w^{DM}$, quantity falls below the integrated benchmark.

<img src="figures/wholesale-pass-through.png" alt="Retail pass-through as wholesale price changes" width="80%">

The table reports the same comparison in numbers. Channel profit and consumer surplus fall only when quantity falls. The fixed fee moves operating profit upstream.

**Contract outcomes**

| Contract             |   Retail price |   Wholesale price |   Fixed fee |   Quantity |   Channel profit |   Consumer surplus |   Total surplus |   Manufacturer profit |   Retailer profit |
|:---------------------|---------------:|------------------:|------------:|-----------:|-----------------:|-------------------:|----------------:|----------------------:|------------------:|
| Integrated benchmark |           6.5  |               2   |         0   |        7   |            24.5  |              12.25 |           36.75 |                  0    |             24.5  |
| Linear wholesale     |           8.25 |               5.5 |         0   |        3.5 |            18.38 |               3.06 |           21.44 |                 12.25 |              6.12 |
| Two-part tariff      |           6.5  |               2   |        24.5 |        7   |            24.5  |              12.25 |           36.75 |                 24.5  |              0    |

## Takeaway

Double marginalization comes from the retailer's perceived marginal cost. A high wholesale price raises that cost and lowers quantity. A two-part tariff sets $w=c_M$, so the retailer chooses the integrated price. The fixed fee then allocates profit.

## References

- Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press.
