# Bayesian Urn Classification and Behavioral Mixtures

> Likelihood-ratio sufficient statistics and latent decision rules in repeated urn choices.

## Overview

The urn experiment is useful because the normative classifier is exactly known. A subject sees a finite sample of red and blue balls and chooses whether it came from a high-red or low-red urn. Bayes' rule compresses the whole signal history into one likelihood-ratio statistic.

The computational method is not another economic setting. It is a finite mixture estimator for unobserved decision rules. Subjects are observed over repeated urn tasks, and each subject is assigned probabilistically to one of several candidate rules: exact Bayesian updating, a conservative Bayesian cutoff, a red-share rule, or a raw red-count rule. The known data-generating mixture lets the tutorial show classification error and parameter recovery.

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
| Tasks per subject | 30 | Variation used to classify latent rules |
| Prior high urn | 0.45 | Baseline probability of state $H$ |
| Red probability under $H$ | 0.72 | Signal distribution for high urn |
| Red probability under $L$ | 0.32 | Signal distribution for low urn |
| Draw counts | 4, 6, 8 | Sample-size variation separates Bayes and cutoff rules |
| Tremble rate | 0.06 | Symmetric error around each deterministic rule |
| Latent rules | 4 | Bayesian and cutoff decision types |

## Solution Method

The Bayes classifier is computed directly from the likelihood ratio. The mixture estimator then treats the rule label as missing data and updates posterior rule responsibilities until the likelihood stops changing.

```text
Algorithm: finite mixture of behavioral rules
Input: repeated choices d_it, task counts (k_t, n_t), candidate rules m=1,...,M
For each task, compute the sufficient statistic Lambda(k_t,n_t)
For each rule, compute q_m(k_t,n_t), the probability of choosing high
Initialize weights w_m = 1/M
Repeat until convergence:
  E-step: tau_im = w_m L_im / sum_h w_h L_ih
  M-step: w_m = mean_i tau_im
Assign each subject to argmax_m tau_im for diagnostics
Output: mixture shares, posterior responsibilities, allocation accuracy
```

Because the rule-specific choice probabilities are fixed here, the M-step is just a normalized average of responsibilities. In richer models, the same likelihood architecture can add rule-specific parameters and optimize them inside the M-step.

## Results

The likelihood ratio is the sufficient statistic. For a fixed sample size, moving from fewer to more red balls moves the posterior monotonically. The labels on the curve are red counts out of eight draws. Cutoff rules can look close to Bayes in some tasks but disagree when sample size changes.

Exact Bayesian updating converts the signal count into a posterior belief before classification.

<img src="figures/bayes-likelihood-ratio.png" alt="Bayesian posterior as a function of the likelihood ratio" width="80%">

The EM estimator recovers the population shares of the fixed candidate rules. The L1 distance between estimated and true weights is **0.229**. The exercise is deliberately finite: the rules are specified before estimation, so the statistical problem is classifying unobserved heterogeneity rather than searching over arbitrary decision trees.

Mixture weights are recovered from repeated choices, not from observing rule labels.

<img src="figures/mixture-weights.png" alt="True and estimated latent rule shares" width="80%">

Posterior responsibilities also say when classification is weak. A subject with choices that several rules can explain receives diffuse responsibilities, even if the aggregate mixture weights are accurate. The hard allocation accuracy in this run is **0.740**.

The distribution of max responsibilities separates confident rule assignments from ambiguous choice histories.

<img src="figures/classification-confidence.png" alt="Posterior confidence in assigned latent rule" width="80%">

The EM likelihood converged in 8 iterations with log likelihood -4516.07.

**Latent rule weight recovery**

| Rule         | Definition                                                                        |   True weight |   Estimated weight |   Error |
|:-------------|:----------------------------------------------------------------------------------|--------------:|-------------------:|--------:|
| Bayes        | Choose high if the posterior probability of the high urn is at least one half.    |          0.46 |             0.3544 | -0.1056 |
| Conservative | Choose high only when the posterior probability of the high urn is at least 0.75. |          0.24 |             0.3544 |  0.1144 |
| Share cutoff | Choose high when at least half of sampled balls are red.                          |          0.2  |             0.1991 | -0.0009 |
| Count cutoff | Choose high when at least four sampled balls are red, ignoring sample size.       |          0.1  |             0.0922 | -0.0078 |

Rows are true simulated rules and columns are posterior-modal assignments. The labels are known only because this is a Monte Carlo tutorial.

**True versus assigned latent rule counts**

| True rule    |   Bayes |   Conservative |   Share cutoff |   Count cutoff |
|:-------------|--------:|---------------:|---------------:|---------------:|
| Bayes        |     272 |              0 |              0 |              3 |
| Conservative |     150 |              0 |              0 |              3 |
| Share cutoff |       0 |              0 |            118 |              0 |
| Count cutoff |       0 |              0 |              0 |             54 |

The exact Bayes classifier is correct on 90.0% of the task states in this simulated set. The table keeps the likelihood-ratio statistic visible because it is the state variable for the later rule estimator.

**Bayesian classifier diagnostics for the first twelve tasks**

|   Task |   Draws |   Red count |   True high urn |   Log likelihood ratio |   Posterior high |   Bayes choice high | Bayes correct   |
|-------:|--------:|------------:|----------------:|-----------------------:|-----------------:|--------------------:|:----------------|
|      1 |       8 |           7 |               1 |                 4.7892 |           0.9899 |                   1 | True            |
|      2 |       6 |           3 |               0 |                -0.2291 |           0.3942 |                   0 | True            |
|      3 |       8 |           5 |               1 |                 1.3927 |           0.7671 |                   1 | True            |
|      4 |       4 |           2 |               1 |                -0.1527 |           0.4126 |                   0 | False           |
|      5 |       4 |           0 |               0 |                -3.5492 |           0.023  |                   0 | True            |
|      6 |       4 |           3 |               1 |                 1.5455 |           0.7933 |                   1 | True            |
|      7 |       4 |           1 |               0 |                -1.851  |           0.1139 |                   0 | True            |
|      8 |       4 |           3 |               1 |                 1.5455 |           0.7933 |                   1 | True            |
|      9 |       8 |           2 |               0 |                -3.702  |           0.0198 |                   0 | True            |
|     10 |       4 |           1 |               0 |                -1.851  |           0.1139 |                   0 | True            |
|     11 |       6 |           2 |               1 |                -1.9274 |           0.1064 |                   0 | False           |
|     12 |       6 |           0 |               0 |                -5.3238 |           0.004  |                   0 | True            |

The diagnostics separate aggregate share recovery from individual-level type classification.

**Estimator and known-truth diagnostics**

| Diagnostic                        |    Value |
|:----------------------------------|---------:|
| EM converged                      | 1        |
| EM iterations                     | 8        |
| Mixture weight L1 error           | 0.228745 |
| Hard type allocation accuracy     | 0.74     |
| Mean max posterior responsibility | 0.636672 |
| Bayes task-state accuracy         | 0.9      |

## Takeaway

Bayesian classification supplies the economically disciplined state variable: the likelihood ratio. The finite-mixture estimator then uses repeated choices to recover unobserved heterogeneity in decision rules. The point is methodological: instead of fitting one average rule, the likelihood treats behavioral types as latent classes and reports both population shares and individual responsibilities.

## References

- [El-Gamal, M. A. and Grether, D. M. (1995). Are People Bayesian? Uncovering Behavioral Strategies. *Journal of the American Statistical Association*, 90(432), 1137-1145.](https://doi.org/10.1080/01621459.1995.10476622)
- [McLachlan, G. and Peel, D. (2000). *Finite Mixture Models*. Wiley.](https://doi.org/10.1002/0471721182)
