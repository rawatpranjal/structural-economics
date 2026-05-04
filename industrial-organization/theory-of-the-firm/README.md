# Theory of the Firm: Incomplete Contracts and Hold-Up

> Asset specificity, relationship-specific investment, and the vertical integration boundary.

## Overview

Incomplete contracts matter when parties must invest before all future contingencies can be written down and enforced. If an investment is specific to one trading relationship, the investing party expects to be held up in ex-post bargaining and invests too little. Vertical integration reallocates residual control rights. It can strengthen investment incentives, but it also brings governance and adaptation costs.

This tutorial turns that logic into a small numerical model. The state is asset specificity: low values describe transactions that can move through spot markets, high values describe assets whose value is tied to one buyer-seller pair.

## Equations

Investment $x$ raises relationship value:
$$V(x) = \theta x - \frac{1}{2}x^2$$

The efficient investment satisfies:
$$x^* = \theta$$

Under regime $g$, the investor captures only share $b_g(s)$ of marginal returns:
$$x_g(s) = b_g(s)\theta$$

Total surplus subtracts governance cost $F_g(s)$:
$$W_g(s) = \theta x_g(s) - \frac{1}{2}x_g(s)^2 - F_g(s)$$

## Model Setup

| Object | Interpretation |
|--------|----------------|
| Specificity $s$ | Share of investment value locked into one relationship |
| Spot contract | Low governance cost, weak protection against hold-up |
| Long-term contract | Better protection, moderate drafting and monitoring cost |
| Vertical integration | Stronger control rights, higher bureaucracy cost |
| $\theta=4$ | Efficient investment benchmark $x^*=4$ |

## Solution Method

For each value of specificity, the script computes the private investment implied by each governance regime and then evaluates total surplus. The preferred regime is the one with the highest surplus, not the one with the largest investment mechanically.

## Results

Spot contracts provide weak protection when assets are highly specific. Integration keeps investment closer to the efficient benchmark at high specificity, but it is not free.

![Relationship-specific investment by governance regime](figures/investment-incentives.png)
*Relationship-specific investment by governance regime*

The best governance form changes with specificity because incentive benefits and governance costs move in opposite directions.

![Surplus by governance regime](figures/surplus-by-regime.png)
*Surplus by governance regime*

The model reproduces the Williamson/GHM comparative static: integration is most attractive where relationship-specific investment is hardest to protect with ordinary contracts.

![Governance regions over asset specificity](figures/governance-regions.png)
*Governance regions over asset specificity*

**Governance comparison at low, medium, and high asset specificity**

|   Specificity | Regime               |   Incentive share |   Investment |   Surplus | Efficiency ratio   |
|--------------:|:---------------------|------------------:|-------------:|----------:|:-------------------|
|           0   | Spot contract        |              0.72 |         2.88 |      7.35 | 91.9%              |
|           0.5 | Spot contract        |              0.44 |         1.78 |      5.5  | 68.7%              |
|           1   | Spot contract        |              0.17 |         0.68 |      2.43 | 30.4%              |
|           0   | Long-term contract   |              0.72 |         2.88 |      6.99 | 87.4%              |
|           0.5 | Long-term contract   |              0.59 |         2.38 |      6.29 | 78.7%              |
|           1   | Long-term contract   |              0.47 |         1.88 |      5.34 | 66.8%              |
|           0   | Vertical integration |              0.74 |         2.96 |      6.41 | 80.1%              |
|           0.5 | Vertical integration |              0.72 |         2.9  |      6.52 | 81.5%              |
|           1   | Vertical integration |              0.71 |         2.84 |      6.63 | 82.8%              |

## Takeaway

The firm boundary is not a generic preference for hierarchy. Integration is useful when incomplete contracts and asset specificity make hold-up severe. When assets are easy to redeploy, ordinary market contracts or longer-term contracts can dominate because they avoid the internal governance cost.

## Reproduce

```bash
python run.py
```

## References

- Williamson, O. (1975). *Markets and Hierarchies*. Free Press.
- Grossman, S., and Hart, O. (1986). The Costs and Benefits of Ownership. *Journal of Political Economy*, 94(4), 691-719.
- Hart, O., and Moore, J. (1990). Property Rights and the Nature of the Firm. *Journal of Political Economy*, 98(6), 1119-1158.
- Lecture 6 Slides 2023: Theory of the Firm and incomplete contracts.
