# Cournot Quantity Competition and Best-Response Iteration

> A quantity-setting oligopoly solved by Nash first-order conditions and checked by fixed-point iteration.

## Overview

Two firms sell a homogeneous good, such as cement in one local market. Each firm chooses output before the market price clears. Producing more raises the firm's own sales, but it also lowers the price paid on every unit. Cournot equilibrium asks where those incentives balance when each firm treats the rival's quantity as given.

This linear duopoly has a closed-form Nash quantity, so the equilibrium condition is easy to see. The computation treats the same condition as a fixed point of best responses. That numerical view is useful because most oligopoly games lose the one-line formula once demand, costs, or the number of firms become richer.

## Equations

Two firms choose quantities $q_1$ and $q_2$ simultaneously. Total output is
$Q=q_1+q_2$, inverse demand is

$$
P(Q)=a-bQ,
$$

and firm $i$ has constant marginal cost $c$. Given $q_j$, firm $i$ solves

$$
\max_{q_i \geq 0}\ (a-b(q_i+q_j)-c)q_i.
$$

The interior first-order condition gives the best response

$$
BR_i(q_j)=\frac{a-c-bq_j}{2b}.
$$

A symmetric Nash equilibrium satisfies $q_i=q_j=q^{\ast}$ and
$q^{\ast}=BR_i(q^{\ast})$, so

$$
q^{\ast}=\frac{a-c}{3b},\qquad
P^{\ast}=a-2bq^{\ast}.
$$

The comparison points are also useful:

$$
Q^{M}=\frac{a-c}{2b},\qquad
Q^{C}=\frac{a-c}{b},
$$

where $Q^{M}$ is monopoly output and $Q^{C}$ is the competitive output at
price equal to marginal cost.

## Model Setup

| Object | Value | Meaning |
|---|---:|---|
| $a$ | 10.0 | Demand intercept |
| $b$ | 1.0 | Demand slope |
| $c$ | 2.0 | Marginal cost |
| $q^{\ast}$ | 2.667 | Nash output per firm |
| $P^{\ast}$ | 4.667 | Nash price |
| $\pi^{\ast}$ | 7.111 | Nash profit per firm |
| Damping $\lambda$ | 0.65 | Weight on each new best response |

## Solution Method

The analytic solution solves the two first-order conditions directly. The numerical calculation keeps the economic object the same: a Nash quantity pair where both firms best respond. It searches for a fixed point of the map $BR(q_1,q_2)=(BR_1(q_2),BR_2(q_1))$.

```text
Algorithm: damped Cournot best-response iteration
Inputs: demand parameters a, b, marginal cost c, start q_0, damping lambda
Output: quantity path q_t and fixed-point residuals

1. Start from a candidate pair q_t = (q_{1t}, q_{2t}).
2. Compute each firm's best response to the other firm's current output.
3. Update q_{t+1} = (1-lambda) q_t + lambda BR(q_t).
4. Repeat until max_i |q_{it} - BR_i(q_{-i,t})| is near zero.
5. Compare the numerical fixed point with q* = (a-c)/(3b).
```

The residual links the iteration back to game theory. A path can look stable on a figure while still leaving a firm with a profitable output change. A small residual checks the no-deviation condition directly.

## Results

The best-response curves cross at the Nash quantity. The joint-monopoly split sits below that crossing. If one firm expected the rival to stay at the collusive quantity, it would expand output and gain profit. The damped paths show the same Nash condition reached from several starting quantities.

<img src="figures/cournot-best-response.png" alt="Cournot best-response curves and damped iteration paths" width="80%">

The residual falls quickly because damping stabilizes this linear best-response map. The final residual answers a game-theoretic question: at the computed quantities, how large is the best remaining unilateral output adjustment?

<img src="figures/residuals.png" alt="Fixed-point residuals for damped best-response iteration" width="80%">

The output comparison gives the equilibrium an economic interpretation. Cournot output lies between monopoly and perfect competition, and so does the price. The exact levels come from the calibration, while the ranking comes from the strategic output effect.

<img src="figures/welfare-analysis.png" alt="Monopoly, Cournot, and competitive output benchmarks" width="80%">

**Best-Response Convergence**

| Initial q   |   Final q1 |   Final q2 |   Residual |
|:------------|-----------:|-----------:|-----------:|
| (0.5, 7.0)  |     2.6667 |     2.6667 |   5.6e-06  |
| (7.0, 0.5)  |     2.6667 |     2.6667 |   5.6e-06  |
| (7.0, 7.0)  |     2.6667 |     2.6667 |   4.44e-16 |
| Closed form |     2.6667 |     2.6667 |   4.44e-16 |

**Cournot Benchmarks**

| Market structure    |   Total output |   Price |   Profit per firm |
|:--------------------|---------------:|--------:|------------------:|
| Monopoly            |          4     |   6     |            16     |
| Cournot duopoly     |          5.333 |   4.667 |             7.111 |
| Perfect competition |          8     |   2     |             0     |

## Takeaway

Cournot equilibrium is a fixed point with economic content: each firm is already choosing its profit-maximizing quantity given the rival's output. The closed form makes that condition transparent in a linear duopoly. Best-response iteration turns the same idea into a numerical procedure, and the residual checks whether the computed quantities satisfy Nash incentives.

## References

- Cournot, A. A. (1838/1897). *Researches into the Mathematical Principles of the Theory of Wealth*. English translation.
- Fudenberg, D. and Tirole, J. (1991). *Game Theory*. MIT Press.
- Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 5.
- Vives, X. (1999). *Oligopoly Pricing: Old Ideas and New Tools*. MIT Press.
