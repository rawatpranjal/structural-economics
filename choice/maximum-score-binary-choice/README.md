# Binary Participation with Maximum Score

> Recover a binary participation index with a nonsmooth semiparametric estimator.

## Overview

Suppose an applied microeconomist observes whether people enroll in a job-training program after seeing wage gains, commuting costs, and other covariates. A binary choice model describes the sign of latent net surplus, but the researcher may not want to assume a correctly specified logit error. Maximum score estimates the direction of that surplus index from the side of the boundary each observation falls on.

The computation is awkward because the estimator maximizes the share of choices classified correctly. A small change in the slope can leave all classifications unchanged, then another change can flip several observations at once. This tutorial uses direct grid search for Manski's original score and a smoothed Horowitz-style score to compute the same median-choice target.

## Equations

The simulated decision is a participation rule:

$$
y_i = 1\{x^B_i+\beta x^C_i+\varepsilon_i \geq 0\}.
$$

Here $x^B_i$ is a benefit shifter, $x^C_i$ is a cost shifter, and
$\beta<0$ makes high costs reduce participation. Only the direction of the
index is identified, so the coefficient on $x^B_i$ is normalized to one.
Manski's maximum-score estimator solves

$$
\hat\beta
= \arg\max_b \frac{1}{n}\sum_i
\left[y_i 1\{x^B_i+b x^C_i\geq 0\} + (1-y_i)1\{x^B_i+b x^C_i<0\}\right].
$$

The smoothed version replaces the hard indicator with a normal CDF:

$$
S_h(b)=\frac{1}{n}\sum_i
\left[y_i \Phi((x^B_i+b x^C_i)/h)
+(1-y_i)\{1-\Phi((x^B_i+b x^C_i)/h)\}\right].
$$

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| Observations | 2,500 | Simulated participation decisions |
| Normalized coefficient | 1 | Benefit shifter weight |
| True $\beta$ | -0.85 | Cost shifter weight |
| Error distribution | heteroskedastic logistic | Median zero, logit likelihood misspecified |
| Grid points | 501 | Direct search over nonsmooth objective |
| Smoothing bandwidth | 0.25 | Smooth boundary approximation |
| Bootstrap draws | 80 | Finite-sample check for smoothed estimate |

## Solution Method

Because the score is a step function, derivative methods do not see the objective's jumps. The code first evaluates every candidate cost weight on a grid. It then replaces the indicator with Phi((x^B_i+b x^C_i)/h) and optimizes the smooth approximation. The bandwidth controls how sharply observations near the boundary switch classifications.

```text
Algorithm: estimate a binary participation index
Input: choices y_i, benefit shifter x^B_i, cost shifter x^C_i, grid B, bandwidth h
Normalize the benefit-shifter coefficient to one
For each b in B:
  classify i as a participant when x^B_i + b x^C_i >= 0
  record the share of choices classified correctly
Choose the grid value with the largest score
For the smoothed estimate:
  replace the hard classification rule with Phi((x^B_i+b x^C_i)/h)
  maximize the smooth score over b
Bootstrap observations and repeat the smoothed estimate
Output: normalized cost weight, classification score, and bootstrap interval
```

The normalization matters for interpretation. If every coefficient were multiplied by a positive constant, the sign of latent surplus would not change. The estimate is a cost weight relative to the normalized benefit shifter, not an absolute utility scale.

## Results

The raw score is flat across ranges of beta because a boundary that does not cross any observation classifies the same people. The smoothed curve peaks at **-0.831**, close to the true normalized cost weight **-0.850**. In this sample, maximum score recovers a negative cost effect even though the logit likelihood is misspecified.

Both objectives point to a similar participation boundary; smoothing mainly makes the search surface easier to optimize.

<img src="figures/score-objectives.png" alt="Maximum-score and smoothed-score objective functions" width="80%">

The scatterplot shows why the estimator is a boundary problem. People with larger benefit shifters are more likely to participate, while large cost shifters push against participation when beta is negative. Noise means the regions overlap, so the best boundary cannot classify everyone correctly.

The estimated median-surplus boundary tracks the simulated boundary despite heteroskedastic choice noise.

<img src="figures/classification-boundary.png" alt="Observed participation choices and estimated surplus boundary" width="80%">

The nonparametric bootstrap interval is **[-0.957, -0.679]**. It gives a finite-sample read on how stable the smoothed estimate is. It is not a replacement for the full nonstandard asymptotic theory; here it keeps uncertainty connected to the computation the tutorial just ran.

The bootstrap distribution summarizes finite-sample uncertainty for the smoothed estimator.

<img src="figures/bootstrap-estimates.png" alt="Bootstrap distribution of smoothed maximum-score estimates" width="80%">

The logit coefficient vector is normalized by the benefit-shifter coefficient so its cost slope can be compared with maximum score.

**Estimator comparison**

| Estimator                |   Normalized beta |    Error |   Classification score |
|:-------------------------|------------------:|---------:|-----------------------:|
| True participation index |          -0.85    |  0       |                 0.6872 |
| Grid maximum score       |          -0.88    | -0.03    |                 0.69   |
| Smoothed maximum score   |          -0.83084 |  0.01916 |                 0.6884 |
| Misspecified logit ratio |          -0.66013 |  0.18987 |                 0.6816 |

The score is the share of choices classified by the sign of the normalized surplus index.

**Score and bootstrap diagnostics**

| Diagnostic                     |      Value |
|:-------------------------------|-----------:|
| Choice-one share               |  0.4908    |
| Grid maximum score             |  0.69      |
| Smoothed score                 |  0.678026  |
| Smoothed optimizer success     |  1         |
| Smoothed optimizer evaluations | 22         |
| Bootstrap mean                 | -0.826309  |
| Bootstrap standard deviation   |  0.0683461 |
| Bootstrap lower 95             | -0.95708   |
| Bootstrap upper 95             | -0.679272  |

## Takeaway

For binary participation models, maximum score separates the median-surplus boundary from a full probability model. The computation maximizes correct classifications, so the objective is nonsmooth and inference is less automatic than in logit MLE. Smoothing turns the boundary search into a continuous problem while preserving the normalized index interpretation.

## References

- [Manski, C. F. (1975). Maximum Score Estimation of the Stochastic Utility Model of Choice. *Journal of Econometrics*, 3(3), 205-228.](https://doi.org/10.1016/0304-4076(75)90032-9)
- [Horowitz, J. L. (1992). A Smoothed Maximum Score Estimator for the Binary Response Model. *Econometrica*, 60(3), 505-531.](https://doi.org/10.2307/2951573)
