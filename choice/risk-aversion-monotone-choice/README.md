# Risk Aversion and Monotone Stochastic Choice

> Constrained stochastic-choice estimation on a Holt-Laury lottery ladder.

## Overview

A Holt-Laury lottery ladder is designed so the risky option becomes more attractive as the probability of the high payoff rises. A risk-averse expected utility model therefore predicts a monotone risky-choice probability across the rows of the experiment.

The computational lesson is shape-restricted estimation. The saturated task logit fits each row separately and can inherit sampling noise. A fixed-scale CRRA logit imposes an economic index and estimates one risk-aversion parameter. A monotone task logit keeps row-level flexibility but constrains the estimated choice probabilities to be nondecreasing along the lottery ladder.

## Equations

Lottery $A$ is safer and lottery $B$ is riskier:

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

The monotone task-logit model estimates row-specific logits $\alpha_j$ subject
to the shape restriction

$$
\alpha_{j+1}\geq \alpha_j
\quad \text{for all adjacent rows }j.
$$

Since the logistic map is monotone, this is equivalent to requiring
$\Pr(d=1\mid p_{j+1})\geq \Pr(d=1\mid p_j)$.

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| Lottery rows | 10 | Probability ladder from 0.10 to 1.00 |
| Choices per row | 80 | Binomial observations used for estimation |
| True risk aversion | 0.45 | Data-generating CRRA curvature |
| True scale | 5.00 | Maps utility differences into choice probabilities |
| Lapse rate | 0.02 | Symmetric stochastic error floor |
| Fixed-scale estimator | scale = 5.00 | Identifies $\rho$ without estimating noise scale |
| Shape restriction | nondecreasing | Risky choices cannot fall as $p$ rises |

## Solution Method

The three estimators answer different questions about the same binomial counts.

```text
Algorithm: lottery choice estimators
Input: risky counts y_j out of N_j choices at probability rows p_j
Unconstrained logit:
  Estimate alpha_j separately, so expit(alpha_j)=y_j/N_j
Fixed-scale CRRA logit:
  For each candidate rho, compute Delta EU(p_j; rho)
  Evaluate the binomial likelihood with fixed scale s
  Choose rho that maximizes the likelihood
Monotone task logit:
  Estimate alpha_j jointly
  Maximize the binomial likelihood subject to alpha_{j+1} >= alpha_j
Diagnostics:
  Report likelihood loss, monotonicity violations, parameter recovery, and fit to truth
```

The fixed-scale model is the tight structural specification. The monotone logit is a weaker but still economic restriction: it does not impose CRRA curvature, but it rules out fitted stochastic choice patterns that contradict the ordering of the lottery rows.

## Results

The observed shares are noisy enough to violate monotonicity once: the risky share falls between two adjacent low-probability rows. The unconstrained task logit inherits that violation. The monotone estimator pools exactly where needed, while the fixed-scale CRRA logit traces a smooth structural curve.

The constrained fit removes a sampling artifact without forcing the full CRRA index.

<img src="figures/risky-choice-fits.png" alt="Observed and fitted risky-choice probabilities" width="80%">

The structural likelihood is sharply centered near the data-generating risk aversion parameter. With the scale fixed at 5.0, the estimate is **0.451**, close to the true value **0.450**.

Fixing the stochastic scale turns the lottery panel into a one-dimensional risk-aversion likelihood.

<img src="figures/rho-likelihood.png" alt="Likelihood over CRRA risk aversion" width="80%">

The underlying expected-utility index crosses zero once. That single crossing is why a structural model predicts ordered choice probabilities across the ladder. Shape restrictions use the same economic ordering without insisting that the whole curve has exactly the CRRA-logit form.

The risky lottery becomes more attractive as the high-payoff probability rises.

<img src="figures/eu-index.png" alt="Expected-utility difference for the true CRRA parameter" width="80%">

The row table shows where the monotone constraint binds: adjacent fitted probabilities are pooled when sampling noise breaks the ordering.

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

The constrained model gives up a small amount of saturated likelihood in exchange for an economically ordered choice curve. The fixed-scale model is more restrictive but recovers the known risk-aversion parameter.

**Estimator comparison**

| Model                    |   Log likelihood |   Monotonicity violations |   Max probability error |   Estimated rho |   Rho error |   LL loss vs saturated |
|:-------------------------|-----------------:|--------------------------:|------------------------:|----------------:|------------:|-----------------------:|
| Unconstrained task logit |         -215.051 |                         1 |                 0.09185 |       nan       |   nan       |                0       |
| Fixed-scale CRRA logit   |         -221.83  |                         0 |                 0.001   |         0.45074 |     0.00074 |                6.77875 |
| Monotone task logit      |         -216.044 |                         0 |                 0.09185 |       nan       |   nan       |                0.99276 |

The minimum adjacent spacing is the numerical slack in the monotonicity constraints.

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

The saturated logit is descriptive, but it can treat sampling noise as behavior. The fixed-scale CRRA logit is parsimonious and identifies risk aversion when the scale is fixed. The monotone specification sits between them: it allows flexible row effects while enforcing the economic shape restriction implied by the lottery ordering. This is the core computational lesson: inequality constraints can encode economic discipline without fully specifying a parametric utility model.

## References

- [Holt, C. A. and Laury, S. K. (2002). Risk Aversion and Incentive Effects. *American Economic Review*, 92(5), 1644-1655.](https://doi.org/10.1257/000282802762024700)
- [Apesteguia, J. and Ballester, M. A. (2018). Monotone Stochastic Choice Models: The Case of Risk and Time Preferences. *Journal of Political Economy*, 126(1), 74-106.](https://doi.org/10.1086/695504)
