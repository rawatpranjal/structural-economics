# Double Marginalization in Vertical Supply Chains

> A manufacturer-retailer pricing game solved by backward induction.

## Overview

A manufacturer often sells through an independent retailer. The manufacturer chooses wholesale terms, the retailer chooses the shelf price, and consumers respond to the final price. Both firms may prefer the channel to behave like one monopolist, but a linear wholesale price puts the upstream markup inside the retailer's marginal cost. The retailer then adds its own markup, and the channel sells too little.

The example keeps demand linear so the contract comparison stays visible. We compute the integrated benchmark, solve the manufacturer-retailer game by backward induction, and then evaluate contracts that separate marginal incentives from profit transfers. A two-part tariff restores the retail pricing incentive with a low per-unit price and uses a fixed fee to allocate profit. Resale price maintenance reaches the same retail price by controlling the downstream choice directly.

## Equations

Let demand be summarized by the linear quantity rule
$$q(p)=a-bp,\qquad p\leq \bar p\equiv a/b,$$
where $p$ is the retail price, $a$ is market size, $b$ is the demand slope, and
$\bar p$ is the choke price. The upstream marginal cost is $c_M$ and the
retailer's own marginal selling cost is $c_R$.

The integrated channel chooses $p$ to maximize
$$\Pi^I(p)=(p-c_M-c_R)q(p),$$
so the joint-profit price is
$$p^I=\frac{\bar p+c_M+c_R}{2}.$$

Under a linear wholesale contract, the manufacturer sets a per-unit wholesale
price $w$ and the retailer solves
$$\max_p\ (p-w-c_R)q(p).$$
The retailer's best response is
$$p_R(w)=\frac{\bar p+w+c_R}{2}.$$

The manufacturer anticipates that response and solves
$$\max_w\ (w-c_M)q(p_R(w)),$$
which gives
$$w^{DM}=\frac{\bar p-c_R+c_M}{2}.$$
Because $w^{DM}>c_M$, the retailer prices as if marginal cost is higher than
the true channel cost.

A two-part tariff replaces the high marginal wholesale price with
$$w^{TPT}=c_M$$
and lets the fixed fee
$$F=(p^I-c_M-c_R)q(p^I)$$
transfer the retailer's operating profit upstream. Resale price maintenance
sets the retail price at $p^I$ directly; in this calibration the wholesale price
can then be chosen so the downstream participation constraint binds.

## Model Setup

The calibration is small enough to solve analytically. All dollar amounts below are in the same arbitrary unit, and the integrated channel is a benchmark rather than an observed ownership structure.

| Parameter | Value | Description |
|-----------|-------|-------------|
| $a$ | 20.0 | Demand intercept |
| $b$ | 2.0 | Demand slope |
| $\bar p=a/b$ | 10.0 | Choke price |
| $c_M$ | 2.0 | Manufacturer marginal cost |
| $c_R$ | 1.0 | Retail service cost |
| Contracts | 4 | Integrated benchmark, linear wholesale, two-part tariff, resale price maintenance |

## Solution Method

Each contract changes which first-order condition determines the retail price. The computation solves those conditions in the order the firms move, then puts every contract on the same demand curve. That lets the tutorial separate an efficiency loss, lower quantity, from a transfer, the fixed fee or wholesale margin.

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

3. Contract counterfactuals
    Two-part tariff: set w=c_M, p=p_I, and fixed fee F=(p_I-c_M-c_R)q_I
    Resale price maintenance: set p=p_I and choose w so retailer profit is zero

Outputs: contract outcomes and pass-through curve p_R(w)
```

The integrated solution is the benchmark for the channel's joint-profit problem. The backward-induction solution shows how the separated linear contract departs from that benchmark, and the counterfactual contracts show which part of the departure comes from marginal incentives.

## Results

The integrated channel charges $6.50$ and sells 7.0 units. The separated linear contract chooses a wholesale price of $5.50$, which pushes the retail price to $8.25$ and cuts quantity to 3.5. The two-part tariff and resale price maintenance return the retail price to the integrated-channel line.

<img src="figures/price-quantity.png" alt="Price and quantity by vertical contract" width="80%">

The wholesale margin reallocates profit inside the channel, but the welfare loss comes from the smaller quantity sold under the linear contract. Once the marginal wholesale price is neutralized, consumer surplus and channel profit return to the integrated benchmark in this single-product environment.

<img src="figures/surplus-decomposition.png" alt="Consumer surplus and channel profit by contract" width="80%">

The pass-through exercise varies only the per-unit wholesale price. Moving from $w=c_M$ to $w^{DM}$ traces the double-marginalization mechanism: the retailer's best response price rises with perceived marginal cost, and the quantity gap opens relative to the integrated benchmark.

<img src="figures/wholesale-pass-through.png" alt="Retail pass-through as wholesale price changes" width="80%">

The table separates efficiency from incidence. Channel profit and consumer surplus describe the real allocation. Manufacturer and retailer profits show how each contract allocates that profit. For the integrated benchmark, the internal split is only a transfer convention.

**Contract outcomes**

| Contract                 |   Retail price |   Wholesale price |   Fixed fee |   Quantity |   Channel profit |   Consumer surplus |   Total surplus |   Manufacturer profit |   Retailer profit |
|:-------------------------|---------------:|------------------:|------------:|-----------:|-----------------:|-------------------:|----------------:|----------------------:|------------------:|
| Integrated benchmark     |           6.5  |               2   |         0   |        7   |            24.5  |              12.25 |           36.75 |                  0    |             24.5  |
| Linear wholesale         |           8.25 |               5.5 |         0   |        3.5 |            18.38 |               3.06 |           21.44 |                 12.25 |              6.12 |
| Two-part tariff          |           6.5  |               2   |        24.5 |        7   |            24.5  |              12.25 |           36.75 |                 24.5  |              0    |
| Resale price maintenance |           6.5  |               5.5 |         0   |        7   |            24.5  |              12.25 |           36.75 |                 24.5  |              0    |

## Takeaway

Vertical contracts matter because they decide which margins affect the retail price. A high wholesale price changes the downstream first-order condition and creates a real quantity distortion. A fixed fee moves profit without changing the retailer's marginal cost. Nonlinear pricing can remove double marginalization while still letting the upstream firm collect the channel's profit.

## References

- Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press.
- Asker, J. (2016). Diagnosing Foreclosure Due to Exclusive Dealing. *Journal of Industrial Economics*, 64(3), 375-410.
- Lecture 7 Slides 2023: Vertical relationships, double marginalization, and restraints.
