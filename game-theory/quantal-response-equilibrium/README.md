# Quantal Response Equilibrium

> Noisy best responses solved as a fixed point.

## Overview

Quantal response equilibrium relaxes the exact best-response assumption. Players are more likely to choose actions with higher expected payoffs, but they can still make mistakes. The logit precision parameter controls how sharply choice probabilities respond to payoff differences.

## Equations

For each action $a_i$, player $i$ assigns logit probability
$$
\sigma_i(a_i) =
\frac{\exp(\lambda E[u_i(a_i, a_{-i})])}
{\sum_{a'_i}\exp(\lambda E[u_i(a'_i, a_{-i})])}.
$$

A logit quantal response equilibrium is a fixed point:
$$
\sigma_i = QBR_i(\sigma_{-i}; \lambda)
\qquad \text{for each player } i.
$$

As $\lambda \to 0$, choices approach uniform randomization. As $\lambda$ rises,
choices put more weight on higher-payoff actions.

## Model Setup

The example is a two-player entry game. Each player chooses Enter or Stay Out.

| | Column Enter | Column Stay Out |
|---|---:|---:|
| **Row Enter** | -1, -1 | 2, 0 |
| **Row Stay Out** | 0, 2 | 0, 0 |

The exact game also has two asymmetric pure Nash equilibria. Its symmetric mixed Nash equilibrium has each player entering with probability $2/3$.

## Solution Method

For each precision value $\lambda$, follow the symmetric QRE branch by solving $p = QBR(p; \lambda)$ by bisection on the residual $p - QBR(p; \lambda)$. This keeps the implementation low-code while avoiding the cycling that naive iteration can produce at high precision.

## Results

At zero precision, players randomize 50-50. As precision rises, the logit fixed point moves toward the symmetric mixed Nash entry probability.

<img src="figures/qre-path.png" alt="QRE entry probabilities approach the symmetric mixed Nash benchmark" width="80%">
*QRE entry probabilities approach the symmetric mixed Nash benchmark*

The fixed point is where the noisy best-response curve crosses the 45-degree line.

<img src="figures/fixed-point-map.png" alt="Logit QRE is a fixed point of noisy best responses" width="80%">
*Logit QRE is a fixed point of noisy best responses*

**QRE Summary**

|   Precision lambda |   Row Pr(Enter) |   Column Pr(Enter) |   Iterations |   Residual |
|-------------------:|----------------:|-------------------:|-------------:|-----------:|
|                  0 |          0.5    |             0.5    |            1 |   0        |
|                  1 |          0.5712 |             0.5712 |           36 |   5.06e-13 |
|                  2 |          0.5995 |             0.5995 |           41 |   3.84e-14 |
|                  4 |          0.6243 |             0.6243 |           39 |   9.94e-13 |
|                  8 |          0.6423 |             0.6423 |           41 |   9.18e-13 |

**Final Fixed-Point Diagnostic**

|   Precision lambda |   Row Pr(Enter) |   Column Pr(Enter) |   Fixed-point residual |
|-------------------:|----------------:|-------------------:|-----------------------:|
|                  8 |         0.64228 |            0.64228 |               9.18e-13 |

## Takeaway

QRE is useful when exact best response is too sharp or behavior is noisy. Computationally, it is just another fixed-point problem: probabilities must equal the logit best responses to the probabilities chosen by opponents. This makes it a low-code bridge between finite games and stochastic choice models.

## Reproduce

```bash
python run.py
```

## References

- [McKelvey, R. D. and Palfrey, T. R. (1995). Quantal Response Equilibria for Normal Form Games. *Games and Economic Behavior*, 10(1), 6-38.](https://doi.org/10.1006/game.1995.1023)
- [Goeree, J. K., Holt, C. A., and Palfrey, T. R. (2016). *Quantal Response Equilibrium: A Stochastic Theory of Games*. Princeton University Press.](https://doi.org/10.23943/princeton/9780691124230.001.0001)
