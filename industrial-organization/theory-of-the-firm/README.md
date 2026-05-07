# Firm Boundaries, Hold-Up, and Vertical Integration

> Relationship-specific investment, asset ownership, and the surplus cost of hierarchy.

## Overview

An automaker may ask a supplier to buy tooling that has little value outside their relationship. The tooling raises joint surplus, but the supplier pays for it before the buyer and supplier bargain over delivery terms. If a court cannot verify the relevant investment or adaptation effort, the supplier invests for its expected bargaining payoff rather than for total surplus. That gap is the hold-up problem.

The tutorial studies how ownership and contracting change that investment incentive. Asset specificity $s\in[0,1]$ indexes how hard the asset is to redeploy. Spot exchange has low governance cost and weak protection at high $s$. A long-term contract preserves more of the investment return, but it uses resources in drafting and monitoring. Vertical integration gives stronger residual control rights while adding the internal cost of hierarchy.

For each value of $s$, the code computes private investment under each governance form and then compares total surplus net of governance cost. The calculation turns the property-rights tradeoff into threshold regions for market exchange, contracting, and integration.

## Equations

Let $s$ denote asset specificity and let $g\in\mathcal G$ index governance
regimes: spot exchange, a long-term contract, and vertical integration.
Relationship-specific investment $x$ creates gross value
$$V(x) = \theta x - \frac{1}{2}x^2$$

so the first-best investment, before contracting frictions, is
$$x^{\ast} = \theta$$

Under regime $g$, the investor internalizes only share $b_g(s)$ of the marginal
return. The private first-order condition is
$$b_g(s)\theta - x = 0,$$
which gives the regime-specific investment rule
$$x_g(s) = b_g(s)\theta$$

Total surplus nets out the governance cost $F_g(s)$:
$$W_g(s) = \theta x_g(s) - \frac{1}{2}x_g(s)^2 - F_g(s)$$

The calibration uses simple schedules for the share of marginal returns that
the investor captures:
$$b_{\text{spot}}(s)=0.72-0.55s,\quad
b_{\text{contract}}(s)=0.72-0.25s,\quad
b_{\text{integration}}(s)=0.74-0.03s.$$

Governance costs are
$$F_{\text{spot}}(s)=0.02+0.04s,\quad
F_{\text{contract}}(s)=0.38+0.03s,\quad
F_{\text{integration}}(s)=1.05-0.35s.$$

The selected governance form is
$$g^{\ast}(s)=\arg\max_{g\in\mathcal G} W_g(s).$$

The first-best surplus line used in the figures is
$$W^{\ast}=\frac{1}{2}\theta^2,$$
which is a benchmark, not an attainable governance regime once hold-up and
governance costs are present.

## Model Setup

The calibration is a transparent comparative static, not an estimate of a particular industry. The schedules make the Williamson/Grossman-Hart-Moore logic visible: higher specificity weakens market incentives, contracting protects some returns at a cost, and integration becomes cheaper relative to market governance when redeployment is poor.

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

The private investment problem has a closed-form first-order condition once $b_g(s)$ is fixed. The remaining numerical step is a grid comparison over asset specificity. At each grid point, the algorithm computes the investment chosen under each governance form, evaluates total surplus, and records the surplus-maximizing regime. This separation matters because integration can raise investment and still lose if hierarchy costs absorb the gain.

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

In this calibration the approximate surplus-maximizing regions are: Spot contract for $s\lesssim 0.21$; Long-term contract for $0.21\lesssim s\lesssim 0.37$; Vertical integration for $s\gtrsim 0.37$. The switch points are not parameters of the model; they come from comparing stronger investment incentives with the resource costs of writing contracts or running hierarchy.

## Results

The dashed line is the first-best investment $x^{\ast}=\theta$. Spot exchange loses investment incentives as specificity rises because the investor expects more bargaining over quasi-rents after the asset is sunk. Integration keeps investment close to the benchmark, but surplus also depends on the cost of organizing the relationship inside the firm.

<img src="figures/investment-incentives.png" alt="Relationship-specific investment by governance regime" width="80%">

The surplus ranking changes because each governance form moves two objects at once: investment incentives and governance cost. Spot contracts are best when assets are easy to redeploy. Vertical integration becomes attractive after the hold-up cost of market exchange dominates the internal cost of hierarchy.

<img src="figures/surplus-by-regime.png" alt="Surplus by governance regime" width="80%">

The governance regions summarize the same comparison without the surplus levels. In the middle interval, a long-term contract can dominate both spot exchange and integration when it protects enough investment without bringing the full internal governance cost.

<img src="figures/governance-regions.png" alt="Governance regions over asset specificity" width="80%">

Four values of $s$ show the accounting behind the regions. At low specificity, cheap market exchange wins despite underinvestment. Around the middle of the grid, a long-term contract can protect enough investment without the full hierarchy cost. At higher specificity, integration's incentive effect is large enough to offset its governance cost.

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

The firm boundary is not a generic preference for hierarchy. Integration is valuable when noncontractible, relationship-specific investment is important enough that stronger control rights pay for themselves. When assets are easy to redeploy, market exchange can dominate because it avoids the internal costs of hierarchy. Between those cases, a long-term contract is often the surplus-maximizing compromise.

## References

- Williamson, O. (1975). *Markets and Hierarchies*. Free Press.
- Grossman, S., and Hart, O. (1986). The Costs and Benefits of Ownership. *Journal of Political Economy*, 94(4), 691-719.
- Hart, O., and Moore, J. (1990). Property Rights and the Nature of the Firm. *Journal of Political Economy*, 98(6), 1119-1158.
- Lecture 6 Slides 2023: Theory of the Firm and incomplete contracts.
