# Firm Boundaries, Hold-Up, and Vertical Integration

> Relationship-specific investment, ownership, and hierarchy cost.

## Overview

A supplier may invest in tooling useful only for one buyer. The asset raises joint surplus. After the asset is sunk, bargaining may leave the supplier with too little of the return. The supplier then invests for private surplus.

The object is a governance choice. Asset specificity $s\in[0,1]$ measures how hard the asset is to redeploy. The choices are spot exchange, a long-term contract, and vertical integration. Each form changes investment incentives and governance cost.

For each $s$, the code computes investment and surplus under each form. A grid search then selects the surplus-maximizing boundary choice. The output is a set of specificity thresholds.

## Equations

Let $s$ denote asset specificity.
Let $g\in\mathcal G$ index spot exchange, a long-term contract, and vertical
integration.
Relationship-specific investment $x$ creates gross value
$$V(x) = \theta x - \frac{1}{2}x^2$$

First-best investment solves $V'(x)=0$, so
$$x^{\ast} = \theta$$

Regime $g$ lets the investor capture share $b_g(s)$ of marginal value.
The private first-order condition is
$$b_g(s)\theta - x = 0,$$
which gives
$$x_g(s) = b_g(s)\theta$$

Total surplus subtracts governance cost $F_g(s)$:
$$W_g(s) = \theta x_g(s) - \frac{1}{2}x_g(s)^2 - F_g(s)$$

The incentive schedules are
$$b_{\text{spot}}(s)=0.72-0.55s,\quad
b_{\text{contract}}(s)=0.72-0.25s,\quad
b_{\text{integration}}(s)=0.74-0.03s.$$

Governance costs are
$$F_{\text{spot}}(s)=0.02+0.04s,\quad
F_{\text{contract}}(s)=0.38+0.03s,\quad
F_{\text{integration}}(s)=1.05-0.35s.$$

The selected governance form is
$$g^{\ast}(s)=\arg\max_{g\in\mathcal G} W_g(s).$$

The first-best surplus benchmark is
$$W^{\ast}=\frac{1}{2}\theta^2$$

## Model Setup

The calibration is illustrative. Higher specificity weakens market incentives. Contracts protect some returns at a cost. Integration gives control rights but carries hierarchy cost.

| Object | Interpretation |
|--------|----------------|
| $s\in[0,1]$ | Asset specificity, with higher $s$ meaning weaker redeployability outside the relationship |
| $\theta=4$ | Marginal productivity scale, so the first-best investment is $x^{\ast}=4$ |
| $b_g(s)$ | Share of the marginal investment return captured by the investor under governance $g$ |
| $F_g(s)$ | Drafting, monitoring, bureaucracy, and adaptation cost under governance $g$ |
| Spot contract | Low fixed governance cost, but incentives fall sharply as specificity rises |
| Long-term contract | More protection against hold-up, with moderate contracting cost |
| Vertical integration | Stronger residual control rights, with higher internal governance cost |

## Solution Method

Given $b_g(s)$, private investment has the closed form $x_g(s)=b_g(s)\theta$. The only numerical step is a grid comparison over $s$. At each point, the code evaluates $W_g(s)$ for the three governance forms. It keeps the form with the largest surplus.

```text
Inputs: specificity grid S, regimes G, productivity theta,
        incentive schedules b_g(s), governance costs F_g(s)

First-best benchmark:
    x_star = theta
    W_star = 0.5 * theta^2

For each s in S:
    For each governance regime g in G:
        x_g(s) = b_g(s) * theta
        W_g(s) = theta * x_g(s) - 0.5 * x_g(s)^2 - F_g(s)
    Choose g_star(s) = argmax_g W_g(s)

Outputs: investment schedules, surplus schedules, and governance regions
```

In this calibration, spot exchange wins for $s\lesssim 0.21$. Long-term contracts win for $0.21\lesssim s\lesssim 0.37$. Vertical integration wins for $s\gtrsim 0.37$. These thresholds come from surplus comparisons.

## Results

The dashed line is the first-best investment. Spot exchange falls quickly as specificity rises. Integration stays close to first best. Surplus decides whether higher investment is worth hierarchy cost.

<img src="figures/investment-incentives.png" alt="Relationship-specific investment by governance regime" width="80%">

Surplus changes with incentives and governance cost. Spot contracts win when redeployment is easy. Vertical integration wins when hold-up losses exceed hierarchy cost.

<img src="figures/surplus-by-regime.png" alt="Surplus by governance regime" width="80%">

These regions plot the surplus winner for each specificity value. Long-term contracts occupy the middle. They protect investment at lower cost than integration.

<img src="figures/governance-regions.png" alt="Governance regions over asset specificity" width="80%">

The table shows the accounting at four specificity values. Low specificity favors market exchange. Middle values favor a contract. High specificity favors integration.

**Governance comparison at selected levels of asset specificity**

|   Specificity | Regime               |   Incentive share |   Investment |   Surplus | Efficiency ratio   | Chosen regime   |
|--------------:|:---------------------|------------------:|-------------:|----------:|:-------------------|:----------------|
|           0   | Spot contract        |              0.72 |         2.88 |      7.35 | 91.9%              | yes             |
|           0   | Long-term contract   |              0.72 |         2.88 |      6.99 | 87.4%              |                 |
|           0   | Vertical integration |              0.74 |         2.96 |      6.41 | 80.1%              |                 |
|           0.3 | Spot contract        |              0.55 |         2.22 |      6.38 | 79.8%              |                 |
|           0.3 | Long-term contract   |              0.65 |         2.58 |      6.6  | 82.5%              | yes             |
|           0.3 | Vertical integration |              0.73 |         2.92 |      6.48 | 81.0%              |                 |
|           0.5 | Spot contract        |              0.44 |         1.78 |      5.5  | 68.7%              |                 |
|           0.5 | Long-term contract   |              0.59 |         2.38 |      6.29 | 78.7%              |                 |
|           0.5 | Vertical integration |              0.72 |         2.9  |      6.52 | 81.5%              | yes             |
|           1   | Spot contract        |              0.17 |         0.68 |      2.43 | 30.4%              |                 |
|           1   | Long-term contract   |              0.47 |         1.88 |      5.34 | 66.8%              |                 |
|           1   | Vertical integration |              0.71 |         2.84 |      6.63 | 82.8%              | yes             |

## Takeaway

Firm boundaries follow the hold-up tradeoff. Market exchange works when assets are easy to redeploy. Integration pays when stronger control rights offset hierarchy cost. Long-term contracts fill the middle range.

## References

- Williamson, O. (1975). *Markets and Hierarchies*. Free Press.
- Grossman, S., and Hart, O. (1986). The Costs and Benefits of Ownership. *Journal of Political Economy*, 94(4), 691-719.
- Hart, O., and Moore, J. (1990). Property Rights and the Nature of the Firm. *Journal of Political Economy*, 98(6), 1119-1158.
- Lecture 6 Slides 2023: Theory of the Firm and incomplete contracts.
