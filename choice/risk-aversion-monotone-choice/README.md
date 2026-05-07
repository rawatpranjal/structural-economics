# Lottery Risk Aversion with Monotone Choice

> Estimate risk aversion from a lottery ladder while enforcing ordered choice probabilities.

## Overview

A Holt-Laury lottery ladder asks subjects to choose between a safer lottery and a riskier lottery across a sequence of rows. Only the probability of the high payoff changes. As that probability rises, the risky option becomes more attractive under expected utility, even for a risk-averse subject.

Finite samples can blur that ordering. One row may happen to draw fewer risky choices than the row before it, and a saturated row-by-row logit will treat the wiggle as behavior. This tutorial estimates the same lottery panel three ways: raw row logits, a fixed-scale CRRA logit for the risk-aversion parameter, and a monotone logit that keeps flexible row probabilities but requires the risky-choice curve to rise along the ladder.

## Equations

At row $j$, the high-payoff probability is $p_j$. The subject chooses between a
safer lottery $A$ and a riskier lottery $B$:

$$
A(p)=(2.00 \text{ with } p,\ 1.60 \text{ otherwise}),\qquad
B(p)=(3.85 \text{ with } p,\ 0.10 \text{ otherwise}).
$$

For CRRA utility,

$$
u(c;\rho)=\frac{c^{1-\rho}-1}{1-\rho},
\qquad \rho\neq 1.
$$

The expected-utility index for choosing the risky lottery is

$$
\Delta EU(p;\rho)=E[u(B(p);\rho)]-E[u(A(p);\rho)].
$$

The fixed-scale stochastic-choice model is

$$
\Pr(d=1\mid p;\rho)
= \lambda + (1-2\lambda)\frac{1}{1+\exp[-s\,\Delta EU(p;\rho)]}.
$$

The monotone task-logit model estimates row-specific logits $\alpha_j$ from
binomial counts $y_j$ out of $N_j$ choices:

$$
\ell(\alpha)=\sum_j y_j\log\Lambda(\alpha_j) + (N_j-y_j)\log[1-\Lambda(\alpha_j)].
$$

The shape restriction is

$$
\alpha_{j+1}\geq \alpha_j
\quad \text{for all adjacent rows }j.
$$

Since $\Lambda$ is monotone, this is equivalent to requiring
$\Pr(d=1\mid p_{j+1})\geq \Pr(d=1\mid p_j)$.

## Model Setup

| Primitive | Value | Economic role |
|--------|-------|------|
| Lottery rows | 10 | Probability ladder from 0.10 to 1.00 |
| Choices per row | 80 | Binomial observations for each lottery pair |
| True risk aversion | 0.45 | Data-generating CRRA curvature |
| True scale | 5.00 | Maps utility differences into stochastic choice |
| Lapse rate | 0.02 | Symmetric lower and upper error floor |
| Fixed-scale estimator | scale = 5.00 | Recovers $\rho$ from the payoff index |
| Shape restriction | nondecreasing | Risky-choice probability cannot fall as $p$ rises |

## Solution Method

The calculation compares a fully flexible fit with two ways to add economic discipline. The CRRA logit imposes a one-parameter utility index. The monotone logit solves a finite-dimensional constrained likelihood problem, so it can stay close to row shares when the data are clear and pool adjacent rows when sampling noise reverses the ladder.

```text
Algorithm: monotone lottery-choice estimation
Input: rows (p_j, y_j, N_j), fixed scale s, lapse rate lambda
1. Convert each observed row share y_j/N_j into a saturated logit alpha_j.
2. For the structural fit, search over rho and evaluate Delta EU(p_j; rho).
3. Map Delta EU into choice probabilities and maximize the binomial likelihood.
4. For the monotone fit, choose all alpha_j jointly subject to alpha_{j+1} >= alpha_j.
5. Convert logits to probabilities and compare likelihood loss, violations, and rho recovery.
Output: fitted risky-choice curves and diagnostics for each specification
```

The fixed-scale model is the tight structural specification. The monotone logit keeps less theory in the model, but it still uses the economics of the experiment: a better chance at the high payoff should not lower the probability of choosing the risky lottery.

## Results

At low high-payoff probabilities, only a few subjects choose the risky lottery. In this sample, the row with $p=0.30$ draws a lower risky share than the row with $p=0.20$, even though the risky payoff became more likely. The unconstrained logit repeats that downward step. The monotone estimator pools the affected rows, while the fixed-scale CRRA logit traces a smooth payoff-based curve.

The constrained curve removes the downward step without forcing the full CRRA shape.

<img src="figures/risky-choice-fits.png" alt="Observed and fitted risky-choice probabilities" width="80%">

The CRRA likelihood uses the payoff model rather than row labels. With the stochastic scale fixed at 5.0, the maximizer is **0.451**, close to the true value **0.450**. In an empirical application, this number would summarize risk aversion. Here the known data-generating value lets us see the target directly.

Fixing the stochastic scale turns lottery choices into a one-dimensional likelihood for risk aversion.

<img src="figures/rho-likelihood.png" alt="Likelihood over CRRA risk aversion" width="80%">

For the true $\rho$, the expected-utility difference rises with the high-payoff probability and crosses zero once. The crossing marks the point where expected utility switches from the safer lottery to the riskier lottery. The monotone estimator borrows this ordering implication without requiring every row to lie on the CRRA-logit curve.

The risky lottery becomes more attractive as the high-payoff probability rises.

<img src="figures/eu-index.png" alt="Expected-utility difference for the true CRRA parameter" width="80%">

Equal adjacent monotone fits show where the inequality constraint binds.

**Lottery-row fit diagnostics**

|   Row |   High-payoff probability |   Risky count |   Trials |   Observed share |   True probability |   Unconstrained fit |   Fixed-scale fit |   Monotone fit |
|------:|--------------------------:|--------------:|---------:|-----------------:|-------------------:|--------------------:|------------------:|---------------:|
|     1 |                       0.1 |             1 |       80 |           0.0125 |             0.0204 |              0.0125 |            0.0204 |         0.0125 |
|     2 |                       0.2 |             4 |       80 |           0.05   |             0.0219 |              0.05   |            0.0219 |         0.0313 |
|     3 |                       0.3 |             1 |       80 |           0.0125 |             0.0285 |              0.0125 |            0.0285 |         0.0313 |
|     4 |                       0.4 |             8 |       80 |           0.1    |             0.057  |              0.1    |            0.0568 |         0.1    |
|     5 |                       0.5 |            12 |       80 |           0.15   |             0.1659 |              0.15   |            0.1653 |         0.15   |
|     6 |                       0.6 |            29 |       80 |           0.3625 |             0.4471 |              0.3625 |            0.4461 |         0.3625 |
|     7 |                       0.7 |            69 |       80 |           0.8625 |             0.7707 |              0.8625 |            0.77   |         0.8625 |
|     8 |                       0.8 |            72 |       80 |           0.9    |             0.9237 |              0.9    |            0.9234 |         0.9    |
|     9 |                       0.9 |            77 |       80 |           0.9625 |             0.9668 |              0.9625 |            0.9667 |         0.9625 |
|    10 |                       1   |            79 |       80 |           0.9875 |             0.977  |              0.9875 |            0.977  |         0.9875 |

The monotone model gives up 0.99 log-likelihood points relative to the saturated fit. The fixed-scale model gives up more fit because it asks one payoff-based curve to explain all rows.

**Estimator comparison**

| Model                    |   Log likelihood |   Monotonicity violations |   Max probability error |   Estimated rho |   Rho error |   LL loss vs saturated |
|:-------------------------|-----------------:|--------------------------:|------------------------:|----------------:|------------:|-----------------------:|
| Unconstrained task logit |         -215.051 |                         1 |                 0.09185 |       nan       |   nan       |                0       |
| Fixed-scale CRRA logit   |         -221.83  |                         0 |                 0.001   |         0.45074 |     0.00074 |                6.77875 |
| Monotone task logit      |         -216.044 |                         0 |                 0.09185 |       nan       |   nan       |                0.99276 |

The diagnostics check convergence and the adjacent-logit inequality.

**Solver and constraint diagnostics**

| Diagnostic                       |        Value |
|:---------------------------------|-------------:|
| Fixed-scale optimizer success    |  1           |
| Monotone optimizer success       |  1           |
| Monotone iterations              |  8           |
| Minimum adjacent logit spacing   | -4.44089e-16 |
| Observed monotonicity violations |  1           |
| Monotone fit violations          |  0           |

## Takeaway

Risk-aversion experiments ask for a preference parameter, but small panels also raise a numerical question about how much noise to allow in the choice curve. A saturated logit describes row shares. A fixed-scale CRRA logit converts payoffs into a one-dimensional likelihood for $\rho$. A monotone logit enforces the economically natural ordering with inequality constraints. The reusable idea is to put trusted shape restrictions into the likelihood when a full structural model feels too tight.

## References

- [Holt, C. A. and Laury, S. K. (2002). Risk Aversion and Incentive Effects. *American Economic Review*, 92(5), 1644-1655.](https://doi.org/10.1257/000282802762024700)
- [Apesteguia, J. and Ballester, M. A. (2018). Monotone Stochastic Choice Models: The Case of Risk and Time Preferences. *Journal of Political Economy*, 126(1), 74-106.](https://doi.org/10.1086/695504)
