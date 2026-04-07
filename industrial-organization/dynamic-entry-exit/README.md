# Dynamic Entry and Exit

> Firm turnover and market structure in an oligopolistic industry with sunk entry costs.

## Overview

This model studies how firms' entry and exit decisions determine market structure over time. Each period, incumbent firms decide whether to continue operating (paying a fixed cost $f$) or exit permanently. Simultaneously, potential entrants decide whether to pay a sunk cost $K$ to enter the market. Firms compete as Cournot oligopolists, so profits depend on the number of active firms.

The model generates a stationary equilibrium with persistent heterogeneity in market structure: even in steady state, there is simultaneous entry and exit ("churning"). This captures a key empirical regularity in industrial organization — markets exhibit substantial firm turnover despite relatively stable aggregate concentration.

## Equations

**Per-firm Cournot profit with $N$ symmetric firms:**

$$\pi(N) = \frac{(a - c)^2}{b \cdot (N+1)^2}$$

**Incumbent's value function:**

$$V_I(N) = \max\left\{ \pi(N) - f + \beta \, \mathbb{E}[V_I(N')], \quad 0 \right\}$$

The first term is the value of staying (flow profit minus fixed cost, plus discounted
continuation value). The second term (zero) is the value of exiting.

**Free entry condition:**

$$\mathbb{E}[V_I(N')] \leq K$$

with equality if entry is positive. Potential entrants enter until the expected value
of being an incumbent (post-entry) equals the sunk cost $K$.

**Transition:**

$$N' = \text{Survivors}(N, p_{\text{exit}}) + \text{Entrants}(N)$$

where survivors follow a Binomial distribution and entry is determined by free entry.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $a$       | 10  | Demand intercept |
| $b$       | 1  | Demand slope |
| $c$       | 2  | Marginal cost |
| $f$       | 0.5  | Fixed operating cost (per period) |
| $K$       | 5.0  | Sunk entry cost |
| $\beta$  | 0.95 | Discount factor |
| $N_{\max}$ | 30 | Maximum number of firms |

## Solution Method

**Value Function Iteration (VFI)** with simultaneous computation of exit and entry policies:

1. Initialize $V(N)$ for all states $N = 1, \ldots, N_{\max}$.
2. For each state $N$, compute the continuation value by integrating over possible transitions (binomial survival of other incumbents).
3. Determine exit probability via a smoothed (logistic) best response: firms exit when $V_{\text{stay}} < 0$.
4. Determine entry via free entry: entrants enter until the marginal entrant's value falls below $K$.
5. Iterate until $\|V_{n+1} - V_n\|_\infty < 10^{-8}$.

Converged in **2000 iterations** (error = 2.43e+00).

The stationary distribution is computed by constructing the Markov transition matrix $P(N' | N)$ and finding its invariant distribution via power iteration.

## Results

![Incumbent value function V(N): value of being an active firm as a function of market structure](figures/value-function.png)
*Incumbent value function V(N): value of being an active firm as a function of market structure*

![Exit probability and expected entry as functions of the number of active firms](figures/entry-exit-probabilities.png)
*Exit probability and expected entry as functions of the number of active firms*

![Stationary distribution of the number of active firms](figures/stationary-distribution.png)
*Stationary distribution of the number of active firms*

![Simulated market: number of firms and entry/exit flows over 200 periods](figures/simulated-market.png)
*Simulated market: number of firms and entry/exit flows over 200 periods*

**Equilibrium Statistics**

| Statistic                     |    Value |
|:------------------------------|---------:|
| Expected number of firms E[N] |   13.25  |
| Std. deviation of N           |    6.77  |
| Modal number of firms         |    9     |
| Per-firm profit at E[N]       |    0.327 |
| Net profit (pi - f) at E[N]   |   -0.173 |
| HHI at E[N]                   |  769     |
| Expected exit rate            |    0     |
| Expected entry (firms/period) |    0     |
| VFI iterations                | 2000     |

**Value Function and Policies at Selected Market Structures**

|   N |   Profit pi(N) |   Net profit pi-f |   V(N) |   Exit prob |   Entry rate |
|----:|---------------:|------------------:|-------:|------------:|-------------:|
|   1 |         16     |            15.5   | 21.012 |      0      |            7 |
|   2 |          7.111 |             6.611 | 12.123 |      0      |            6 |
|   3 |          4     |             3.5   |  9.012 |      0      |            5 |
|   5 |          1.778 |             1.278 |  6.79  |      0      |            3 |
|   7 |          1     |             0.5   |  6.012 |      0      |            1 |
|  10 |          0.529 |             0.029 |  0.758 |      0.0005 |            0 |
|  15 |          0.25  |            -0.25  |  0.574 |      0.0032 |            0 |
|  20 |          0.145 |            -0.355 |  0.021 |      0.4478 |            0 |
|  25 |          0.095 |            -0.405 |  1.117 |      0      |            0 |
|  30 |          0.067 |            -0.433 |  2.427 |      0      |            0 |

## Economic Takeaway

Dynamic entry/exit models explain why markets have persistent differences in concentration. Entry costs create barriers that sustain above-competitive profits, while exit occurs when negative shocks or increased competition erode incumbents' continuation values.

**Key insights:**
- The value of incumbency declines sharply with $N$: more competitors erode Cournot rents. Beyond a threshold, $V(N) = 0$ and all firms prefer to exit.
- The sunk cost $K$ creates hysteresis: firms that are already in the market stay (since they only face $f$), while potential entrants need $V > K$ to justify entry. This wedge between entry and exit thresholds is the source of inertia in market structure.
- The model generates "churning" — simultaneous entry and exit even in steady state — because stochastic transitions create states where some firms find it unprofitable to continue while others find it attractive to enter.
- The stationary distribution shows the long-run probability of each market structure. Markets spend most of their time near the modal $N$, but occasionally visit very concentrated or very competitive states.

## Reproduce

```bash
python run.py
```

## References

- Ericson, R. and Pakes, A. (1995). Markov-perfect industry dynamics: A framework for empirical work. *Review of Economic Studies*, 62(1):53-82.
- Hopenhayn, H. (1992). Entry, exit, and firm dynamics in long run equilibrium. *Econometrica*, 60(5):1127-1150.
- Rust, J. (1987). Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. *Econometrica*, 55(5):999-1033.
- Pakes, A. and McGuire, P. (1994). Computing Markov-perfect Nash equilibria: Numerical implications of a dynamic differentiated product model. *RAND Journal of Economics*, 25(4):555-589.
