# Urn Choices and Latent Decision Rules

> Bayesian learning benchmarks and EM mixtures for repeated urn classifications.

## Overview

Consider a lab task in which a subject sees a small sample from one of two urns and must decide whether the urn is the high-red state. The setting is attractive for studying belief-based choice because the Bayesian benchmark is fully pinned down. For any red count and sample size, Bayes' rule gives the posterior probability of the high state.

Repeated choices make the inference problem more interesting. The researcher observes a sequence of high-urn choices, not the rule each subject used. The code first turns each signal into a likelihood-ratio state variable. It then estimates a finite mixture by EM, treating each person's decision rule as a latent class. That gives the researcher two outputs: population shares for the rules and subject-level probabilities over candidate behavioral types.

## Equations

Let $H$ denote the high-red urn and $L$ the low-red urn. A task draws $n$ balls
and observes $k$ red balls. The likelihood-ratio statistic is

$$
\Lambda(k,n)
= \log \frac{\Pr(k\mid H,n)}{\Pr(k\mid L,n)}
= k\log\frac{p_H}{p_L} + (n-k)\log\frac{1-p_H}{1-p_L}.
$$

With prior $\pi_0=\Pr(H)$, Bayes' rule is

$$
\Pr(H\mid k,n)
= \frac{1}{1+\exp[-\{\log(\pi_0/(1-\pi_0))+\Lambda(k,n)\}]}.
$$

Rule $m$ maps the sufficient statistic and counts into a choice probability
$q_m(k,n)$. With subject $i$'s choices $d_{it}\in\{0,1\}$, the panel likelihood
under rule $m$ is

$$
L_{im}
= \prod_t q_m(k_t,n_t)^{d_{it}}
[1-q_m(k_t,n_t)]^{1-d_{it}}.
$$

The finite-mixture likelihood is

$$
\ell(w)=\sum_i \log\left[\sum_m w_m L_{im}\right],
\qquad \sum_m w_m=1,\quad w_m\geq 0.
$$

The posterior probability that subject $i$ follows rule $m$ is

$$
\tau_{im}
= \frac{w_m L_{im}}{\sum_h w_h L_{ih}}.
$$

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| Subjects | 600 | Repeated-choice panel units |
| Tasks per subject | 60 | Variation used to classify latent rules |
| Prior high urn | 0.45 | Baseline probability of state $H$ |
| Red probability under $H$ | 0.72 | Signal distribution for high urn |
| Red probability under $L$ | 0.32 | Signal distribution for low urn |
| Draw counts | 3, 4, 5, 6, 7, 8, 9, 12 | Signal-size variation separates Bayes and cutoff rules |
| Bayes-conservative separating tasks | 6 | Tasks with posterior between the two decision cutoffs |
| Tremble rate | 0.06 | Symmetric error around each deterministic rule |
| Latent rules | 4 | Bayesian and cutoff decision types |

## Solution Method

The calculation has two layers. First, each urn task is reduced to a likelihood ratio, so the Bayesian benchmark depends on $(k,n)$ rather than the full signal history. Second, EM estimates the population shares of the candidate rules. Each E step computes the probability that a subject followed each rule, and each M step averages those probabilities into new mixture weights.

```text
Algorithm: EM for latent decision rules
Input: repeated choices d_it, task counts (k_t, n_t), candidate rules m=1,...,M
1. For each task, compute Lambda(k_t,n_t)
2. For each rule, compute q_m(k_t,n_t), the probability of choosing high
3. Initialize weights w_m = 1/M
4. Repeat until the log likelihood changes by less than the tolerance:
   E step: tau_im = w_m L_im / sum_h w_h L_ih
   M step: w_m = mean_i tau_im
5. Assign each subject to argmax_m tau_im for diagnostics
Output: mixture shares, posterior responsibilities, allocation accuracy
```

Because the rule-specific choice probabilities are fixed here, the M step is just a normalized average of responsibilities. With unknown cutoff points or response noise, the same likelihood would optimize those rule-specific parameters inside the M step.

## Results

The likelihood ratio is the sufficient statistic for the Bayesian state classification. With 5 draws, three red balls put the posterior above one half but below the conservative 0.75 cutoff. Those middle signals help the repeated-choice panel distinguish exact Bayesian updating from a stricter decision rule.

Exact Bayesian updating converts the signal count into a posterior belief before classification.

<img src="figures/bayes-likelihood-ratio.png" alt="Bayesian posterior as a function of the likelihood ratio" width="80%">

The EM estimator recovers the population shares of the fixed candidate rules. The L1 distance between estimated and true weights is **0.028**. The exercise keeps the candidate rules fixed, so the estimate answers a concrete heterogeneity question: what share of subjects behave like each rule?

Mixture weights are recovered from repeated choices, not from observing rule labels.

<img src="figures/mixture-weights.png" alt="True and estimated latent rule shares" width="80%">

Posterior responsibilities measure subject-level classification confidence. A subject with choices that several rules can explain receives diffuse responsibilities, even if the aggregate mixture weights are accurate. The hard allocation accuracy in this run is **0.998**. In the simulated task menu, Bayes differs from the conservative rule on 6 tasks, from the red-share rule on 4 tasks, and from the raw-count rule on 10 tasks.

The distribution of max responsibilities separates confident rule assignments from ambiguous choice histories.

<img src="figures/classification-confidence.png" alt="Posterior confidence in assigned latent rule" width="80%">

The EM likelihood converged in 6 iterations with log likelihood -8816.71.

**Latent rule weight recovery**

| Rule         | Definition                                                                        |   True weight |   Estimated weight |   Error |
|:-------------|:----------------------------------------------------------------------------------|--------------:|-------------------:|--------:|
| Bayes        | Choose high if the posterior probability of the high urn is at least one half.    |          0.46 |             0.4575 | -0.0025 |
| Conservative | Choose high only when the posterior probability of the high urn is at least 0.75. |          0.24 |             0.2523 |  0.0123 |
| Share cutoff | Choose high when at least half of sampled balls are red.                          |          0.2  |             0.2019 |  0.0019 |
| Count cutoff | Choose high when at least four sampled balls are red, ignoring sample size.       |          0.1  |             0.0883 | -0.0117 |

Rows are true simulated rules and columns are posterior-modal assignments. The labels are known only because this is a Monte Carlo tutorial.

**True versus assigned latent rule counts**

| True rule    |   Bayes |   Conservative |   Share cutoff |   Count cutoff |
|:-------------|--------:|---------------:|---------------:|---------------:|
| Bayes        |     275 |              0 |              0 |              0 |
| Conservative |       0 |            151 |              0 |              0 |
| Share cutoff |       1 |              0 |            120 |              0 |
| Count cutoff |       0 |              0 |              0 |             53 |

The exact Bayes classifier is correct on 86.7% of the task states in this simulated set. The table keeps the likelihood-ratio statistic visible because it is the state variable for the later rule estimator.

**Bayesian classifier diagnostics for the first twelve tasks**

|   Task |   Draws |   Red count |   True high urn |   Log likelihood ratio |   Posterior high |   Bayes choice high | Bayes correct   |
|-------:|--------:|------------:|----------------:|-----------------------:|-----------------:|--------------------:|:----------------|
|      1 |      12 |           5 |               0 |                -2.1565 |           0.0865 |                   0 | True            |
|      2 |       6 |           5 |               1 |                 3.1673 |           0.951  |                   1 | True            |
|      3 |      12 |           8 |               1 |                 2.9382 |           0.9392 |                   1 | True            |
|      4 |       5 |           5 |               1 |                 4.0547 |           0.9792 |                   1 | True            |
|      5 |       5 |           2 |               0 |                -1.04   |           0.2243 |                   0 | True            |
|      6 |       3 |           0 |               0 |                -2.6619 |           0.054  |                   0 | True            |
|      7 |       5 |           5 |               1 |                 4.0547 |           0.9792 |                   1 | True            |
|      8 |       5 |           1 |               0 |                -2.7383 |           0.0503 |                   0 | True            |
|      9 |      12 |           5 |               0 |                -2.1565 |           0.0865 |                   0 | True            |
|     10 |       5 |           2 |               0 |                -1.04   |           0.2243 |                   0 | True            |
|     11 |       6 |           4 |               1 |                 1.4691 |           0.7805 |                   1 | True            |
|     12 |       7 |           2 |               0 |                -2.8147 |           0.0467 |                   0 | True            |

The diagnostics separate aggregate share recovery from individual-level type classification.

**Estimator and known-truth diagnostics**

| Diagnostic                        |    Value |
|:----------------------------------|---------:|
| EM converged                      | 1        |
| EM iterations                     | 6        |
| Mixture weight L1 error           | 0.028395 |
| Hard type allocation accuracy     | 0.998333 |
| Mean max posterior responsibility | 0.996831 |
| Bayes task-state accuracy         | 0.866667 |

## Takeaway

Likelihood ratios turn each urn signal into a belief benchmark. The EM mixture then turns repeated choices into estimates of latent behavioral-rule shares. This is the reusable lesson for nearby choice data: when several simple rules are economically meaningful, a finite mixture can estimate population shares and show which individuals are hard to classify.

## References

- [El-Gamal, M. A. and Grether, D. M. (1995). Are People Bayesian? Uncovering Behavioral Strategies. *Journal of the American Statistical Association*, 90(432), 1137-1145.](https://doi.org/10.1080/01621459.1995.10476622)
- [McLachlan, G. and Peel, D. (2000). *Finite Mixture Models*. Wiley.](https://doi.org/10.1002/0471721182)
