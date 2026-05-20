# Cournot Quantity Competition and Best-Response Iteration

## Overview

Two firms sell a homogeneous good in one market. Each firm chooses output before the market price clears. Extra output raises own sales and lowers the price on every unit.

Cournot equilibrium is a quantity pair. Each firm must maximize profit given the rival's output.

The linear game has a closed-form Nash quantity. Best-response iteration treats the same condition as a fixed point. The residual checks whether a firm still wants to change output.

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

The interior first-order condition gives $(a-c-bq_j)/(2b)$, and clipping at
the non-negativity constraint $q_i \geq 0$ gives the best response

$$
BR_i(q_j)=\max\lbrace 0,\ \tfrac{a-c-bq_j}{2b} \rbrace.
$$

A symmetric Nash equilibrium satisfies $q_i=q_j=q^{\ast}$ and
$q^{\ast}=BR_i(q^{\ast})$, so

$$
q^{\ast}=\frac{a-c}{3b},\qquad
P^{\ast}=a-2bq^{\ast}.
$$

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

The first-order conditions solve the linear game directly. The numerical check keeps the same economic object. It searches for a fixed point of the map $BR(q_1,q_2)=(BR_1(q_2),BR_2(q_1))$.

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

The residual links the iteration back to game theory. A path can look stable while still leaving a profitable output change. A small residual checks the no-deviation condition directly.

## Results

The best-response curves cross at the Nash quantity. Each damped path moves toward that crossing. Different starting quantities produce the same Nash condition.

<img src="figures/cournot-best-response.png" alt="Cournot best-response curves and damped iteration paths" width="80%">

The residual falls quickly because damping stabilizes this linear best-response map. The final residual measures the largest remaining unilateral output adjustment.

<img src="figures/residuals.png" alt="Fixed-point residuals for damped best-response iteration" width="80%">

**Best-Response Convergence**

| Initial q   |   Final q1 |   Final q2 |   Residual |
|:------------|-----------:|-----------:|-----------:|
| (0.5, 7.0)  |     2.6667 |     2.6667 |   5.6e-06  |
| (7.0, 0.5)  |     2.6667 |     2.6667 |   5.6e-06  |
| (7.0, 7.0)  |     2.6667 |     2.6667 |   4.44e-16 |
| Closed form |     2.6667 |     2.6667 |   4.44e-16 |

## Takeaway

Cournot equilibrium is a best-response fixed point. Each firm already chooses its profit-maximizing quantity given the rival's output.

The closed form makes this condition transparent. Best-response iteration provides a numerical check. The residual tests Nash incentives.

## References

- Cournot, A. A. (1838/1897). *Researches into the Mathematical Principles of the Theory of Wealth*. English translation.
- Fudenberg, D. and Tirole, J. (1991). *Game Theory*. MIT Press.
- Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 5.
- Vives, X. (1999). *Oligopoly Pricing: Old Ideas and New Tools*. MIT Press.
