# Entry Game QRE and Noisy Best Responses

> How payoff-sensitive mistakes trace a smooth path toward mixed Nash.

## Overview

Entry games are a good place to see why quantal response equilibrium is not just a numerical trick. Exact Nash behavior says a firm enters whenever entry is a best response and mixes only when it is exactly indifferent. QRE keeps the same strategic payoff comparison, but turns the sharp best response into a smooth choice probability: better actions are chosen more often, not with probability one.

The example below follows the symmetric logit-QRE branch in a two-player entry game. The exact game has two asymmetric pure Nash equilibria and one symmetric mixed Nash equilibrium. That mixed equilibrium is the benchmark for the symmetric QRE path, which continues the residual logic in [normal-form games](../normal-form-games/): equilibrium is still a fixed point of best responses, but the response map is probabilistic.

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

The precision parameter $\lambda \geq 0$ governs how sharply payoff gaps affect
behavior. At $\lambda=0$, the entry probability is one half regardless of
payoffs. Along the symmetric branch, $p(\lambda)$ approaches $p^{N}=2/3$ as
$\lambda$ becomes large.

## Model Setup

The payoff table is intentionally small. Entry is profitable when the other player stays out, but competition or congestion makes joint entry costly.

| | Column Enter | Column Stay Out |
|---|---:|---:|
| **Row Enter** | -1, -1 | 2, 0 |
| **Row Stay Out** | 0, 2 | 0, 0 |

| Object | Value | Role |
|---|---:|---|
| Exact pure Nash profiles | $(E,O)$ and $(O,E)$ | One entrant serves the market |
| Symmetric mixed Nash $p^N$ | 0.6667 | Exact benchmark for the symmetric branch |
| Precision grid | 0 to 32 | How strongly choices react to payoff gaps |
| Focal fixed-point plot | $\lambda=4.0$ | One noisy best-response map |

## Solution Method

Tracking the symmetric branch reduces the problem to one dimension at each precision value. Define the residual $G_{\lambda}(p)=p-QBR(p;\lambda)$. In this entry game the residual is strictly increasing on $[0,1]$, with opposite signs at the endpoints, so bisection gives a transparent fixed-point solver.

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

For larger normal-form games, QRE is a system of fixed-point equations. The one-dimensional version here is narrow so the economic object stays visible: a noisy entry probability that must be consistent with the noisy response it induces in the other player.

## Results

At zero precision, behavior ignores payoffs and both players enter with probability one half. As precision rises, the symmetric QRE entry probability moves upward because entry has positive expected payoff whenever the rival enters with probability below $2/3$. The dotted line is not estimated by the QRE solver; it is the exact mixed Nash probability from the indifference condition.

<img src="figures/qre-path.png" alt="Symmetric logit-QRE entry probability and exact mixed Nash benchmark" width="80%">

At $\lambda=4.0$, the noisy best-response curve is smooth but still strategic. A higher rival entry probability lowers the payoff from entering, so the response curve slopes down. The QRE is the crossing with the 45-degree line. The exact mixed Nash benchmark sits to the right because finite precision still puts weight on the lower-payoff action.

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

The high-precision endpoint is close to, but still below, the mixed Nash limit. That distinction matters: QRE at a finite precision is a behavioral model, not a failed computation of Nash.

**High-Precision Diagnostic**

|   Precision lambda |   QRE Pr(Enter) |   Mixed Nash Pr(Enter) |   Absolute gap |   Iterations |   Fixed-point residual |
|-------------------:|----------------:|-----------------------:|---------------:|-------------:|-----------------------:|
|                 32 |        0.659768 |               0.666667 |         0.0069 |           41 |               4.45e-13 |

## Takeaway

QRE keeps the equilibrium discipline of mutual consistency but relaxes the knife-edge best-response rule. In this entry game, higher precision moves the symmetric QRE toward the exact mixed Nash probability, while the fixed-point residual verifies that each reported probability is internally consistent with logit best response. The lesson is not the bisection routine itself; it is the separation between numerical error, measured by the residual, and behavioral smoothing, measured by the gap to Nash.

## References

- [McKelvey, R. D. and Palfrey, T. R. (1995). Quantal Response Equilibria for Normal Form Games. *Games and Economic Behavior*, 10(1), 6-38.](https://doi.org/10.1006/game.1995.1023)
- [Goeree, J. K., Holt, C. A., and Palfrey, T. R. (2016). *Quantal Response Equilibrium: A Stochastic Theory of Games*. Princeton University Press.](https://doi.org/10.23943/princeton/9780691124230.001.0001)
