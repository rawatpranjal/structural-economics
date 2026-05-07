# Market Entry with Quantal Response Equilibrium

> How payoff-sensitive entry mistakes trace a fixed-point path toward mixed Nash.

## Overview

Two firms are deciding whether a small market can support entry. One entrant earns positive profit. If both firms enter, competition makes entry unprofitable. Exact Nash equilibrium predicts either one entrant for sure or a symmetric mixed strategy that makes each firm indifferent.

Quantal response equilibrium asks a nearby behavioral question. Suppose firms still compare expected payoffs, but they make payoff-sensitive mistakes. More profitable actions are chosen more often, yet lower-payoff actions can still occur. The calculation below follows the symmetric logit-QRE branch in this entry game and compares each finite-precision entry probability with the exact mixed Nash benchmark. The computation is a fixed-point problem because each firm's noisy entry probability must agree with the noisy response induced by the rival's probability.

## Equations

Each player chooses $E$ (Enter) or $O$ (Stay Out). Let $p_i$ be player $i$'s
probability of entry. If the rival enters with probability $q$, the expected
payoff difference between entering and staying out is

$$
\Delta(q)
= E[u_i(E,a_{-i})]-E[u_i(O,a_{-i})]
= 2(1-q)-q
= 2-3q.
$$

The exact symmetric mixed Nash equilibrium sets this difference to zero:

$$
p^{N} = \frac{2}{3}.
$$

Logit QRE replaces the discontinuous best response with

$$
QBR(q;\lambda)
=
\frac{\exp(\lambda \Delta(q))}
     {1+\exp(\lambda \Delta(q))}
=
\left[1+\exp(-\lambda(2-3q))\right]^{-1}.
$$

A symmetric logit-QRE is a fixed point

$$
p = QBR(p;\lambda).
$$

The precision parameter $\lambda \geq 0$ controls how strongly payoff gaps move
choice probabilities. At $\lambda=0$, both actions receive probability one half.
As $\lambda$ rises, the symmetric QRE branch $p(\lambda)$ moves toward the mixed
Nash probability $p^{N}=2/3$.

## Model Setup

The payoff table gives the economic tension directly. Entry pays when the rival stays out. Joint entry destroys profits for both firms.

| | Column Enter | Column Stay Out |
|---|---:|---:|
| **Row Enter** | -1, -1 | 2, 0 |
| **Row Stay Out** | 0, 2 | 0, 0 |

| Object | Value | Role |
|---|---:|---|
| Exact pure Nash profiles | $(E,O)$ and $(O,E)$ | One entrant serves the market |
| Symmetric mixed Nash $p^N$ | 0.6667 | Exact benchmark for symmetric entry |
| Precision grid | 0 to 32 | Strength of payoff sensitivity |
| Focal fixed-point plot | $\lambda=4.0$ | One logit response map |

## Solution Method

The symmetric branch turns the equilibrium calculation into a one-dimensional root search at each precision value. For a candidate entry probability $p$, define $G_{\lambda}(p)=p-QBR(p;\lambda)$. A QRE sets this residual to zero. In this entry game, $G_{\lambda}$ is strictly increasing on $[0,1]$ and changes sign between the endpoints, so bisection finds the fixed point without tuning a step size.

```text
Algorithm: symmetric logit-QRE path in the entry game
Inputs: precision grid Lambda, payoff gap Delta(p)=2-3p, tolerance epsilon
Outputs: QRE entry probabilities p(lambda), residuals, gaps to p^N

1. Compute the exact symmetric mixed Nash benchmark p^N from Delta(p^N)=0.
2. For each lambda in Lambda, define QBR(p;lambda) = [1+exp(-lambda Delta(p))]^{-1}.
3. Set the initial bracket [low, high] = [0, 1].
4. Bisect the bracket on G_lambda(p)=p-QBR(p;lambda).
5. Stop when |G_lambda(p)| or the bracket width is below epsilon.
6. Report p(lambda), |G_lambda(p(lambda))|, and p(lambda)-p^N.
```

For larger normal-form games, QRE becomes a system of fixed-point equations over mixed strategies. The one-dimensional version here keeps the entry object visible: a noisy entry probability that must be consistent with the noisy response it gives the other firm.

## Results

At zero precision, firms ignore payoffs and enter with probability one half. As precision rises, the symmetric QRE entry probability moves upward. The reason is economic: entry has positive expected payoff whenever the rival enters with probability below $2/3$. The dotted line comes from the exact indifference condition, not from the QRE root search.

<img src="figures/qre-path.png" alt="Symmetric logit-QRE entry probability and exact mixed Nash benchmark" width="80%">

At $\lambda=4.0$, the noisy best-response curve is smooth but still strategic. A higher rival entry probability lowers the payoff from entry, so the response curve slopes down. The QRE is the crossing with the 45-degree line. The exact mixed Nash benchmark sits to the right because finite precision still puts weight on the lower-payoff action.

<img src="figures/fixed-point-map.png" alt="Noisy best-response map and symmetric QRE fixed point" width="80%">

The residual column is numerical root-finding error. The gap to Nash is economic: it is the distance between finite-precision behavior and the exact symmetric mixed equilibrium.

**QRE Path Summary**

|   Precision lambda |   QRE Pr(Enter) |   Mixed Nash Pr(Enter) |   Gap to Nash |   Iterations |   Residual |
|-------------------:|----------------:|-----------------------:|--------------:|-------------:|-----------:|
|                  0 |          0.5    |                 0.6667 |       -0.1667 |            1 |   0        |
|                  1 |          0.5712 |                 0.6667 |       -0.0955 |           36 |   5.06e-13 |
|                  2 |          0.5995 |                 0.6667 |       -0.0672 |           41 |   3.84e-14 |
|                  4 |          0.6243 |                 0.6667 |       -0.0423 |           39 |   9.94e-13 |
|                  8 |          0.6423 |                 0.6667 |       -0.0244 |           41 |   9.18e-13 |
|                 16 |          0.6535 |                 0.6667 |       -0.0132 |           41 |   1.79e-12 |
|                 32 |          0.6598 |                 0.6667 |       -0.0069 |           41 |   4.45e-13 |

The high-precision endpoint is close to, but still below, the mixed Nash limit. That distinction matters: QRE at a finite precision is a behavioral model, not a failed Nash computation.

**High-Precision Diagnostic**

|   Precision lambda |   QRE Pr(Enter) |   Mixed Nash Pr(Enter) |   Absolute gap |   Iterations |   Fixed-point residual |
|-------------------:|----------------:|-----------------------:|---------------:|-------------:|-----------------------:|
|                 32 |        0.659768 |               0.666667 |         0.0069 |           41 |               4.45e-13 |

## Takeaway

QRE keeps mutual consistency but softens the exact best-response rule. In this entry game, higher precision moves the symmetric QRE toward the mixed Nash probability. The fixed-point residual checks the computation, while the gap to Nash measures finite-precision behavior. Those are different objects, and the tables keep them separate.

## References

- [McKelvey, R. D. and Palfrey, T. R. (1995). Quantal Response Equilibria for Normal Form Games. *Games and Economic Behavior*, 10(1), 6-38.](https://doi.org/10.1006/game.1995.1023)
- [Goeree, J. K., Holt, C. A., and Palfrey, T. R. (2016). *Quantal Response Equilibrium: A Stochastic Theory of Games*. Princeton University Press.](https://doi.org/10.23943/princeton/9780691124230.001.0001)
