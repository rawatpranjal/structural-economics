# Binary Participation with Maximum Score

## Overview

A worker enrolls in training when expected gains exceed travel and time costs. The econometrician observes participation and covariates, not latent surplus.

The object is the sign boundary of a binary choice index. We normalize the benefit coefficient to one and estimate the relative cost weight.

Maximum score searches for the index that classifies the most choices correctly. Its objective is flat and jumpy, so smoothing gives a cleaner numerical target.

## Equations

The simulated decision is a participation rule:

$$y_i = 1\{x^B_i+\beta x^C_i+\varepsilon_i \geq 0\}.$$

Here $x^B_i$ is the benefit shifter, and $x^C_i$ is the cost shifter.
A negative $\beta$ means higher costs lower participation.
Only the index direction is identified.
The coefficient on $x^B_i$ is normalized to one.
Manski's maximum-score estimator solves

$$\hat\beta = \arg\max_b \frac{1}{n}\sum_i \left[y_i 1\{x^B_i+b x^C_i\geq 0\} + (1-y_i)1\{x^B_i+b x^C_i<0\}\right].$$

Smoothing replaces the hard indicator with a normal CDF:

$$S_h(b)=\frac{1}{n}\sum_i \left[y_i \Phi((x^B_i+b x^C_i)/h) +(1-y_i)\{1-\Phi((x^B_i+b x^C_i)/h)\}\right].$$

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

The score is a step function, so local derivatives miss the jumps. The code first evaluates candidate cost weights on a grid. It then replaces the indicator with Phi((x^B_i+b x^C_i)/h) and optimizes the smooth approximation. The bandwidth controls how sharply points near the boundary switch classifications.

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

The normalization fixes the scale. Multiplying every coefficient by a positive constant leaves the surplus sign unchanged. The estimate is a cost weight relative to the benefit shifter.

## Results

The raw score is flat when moving the boundary changes no classifications. The smoothed curve peaks at **-0.831**, close to the true cost weight **-0.850**. The estimate keeps the negative cost effect without using the logit likelihood.

Smoothing keeps the same boundary target while making the search surface easier to optimize.

<img src="figures/score-objectives.png" alt="Maximum-score and smoothed-score objective functions" width="80%">

The scatterplot shows the boundary problem. Benefit shifters raise participation, while cost shifters move choices the other way. Noise leaves overlap, so no linear boundary classifies everyone correctly.

The estimated median-surplus boundary tracks the simulated boundary despite heteroskedastic noise.

<img src="figures/classification-boundary.png" alt="Observed participation choices and estimated surplus boundary" width="80%">

The nonparametric bootstrap interval is **[-0.957, -0.679]**. It summarizes how much the smoothed estimate moves across resampled data.

The bootstrap distribution shows finite-sample uncertainty for the smoothed estimator.

<img src="figures/bootstrap-estimates.png" alt="Bootstrap distribution of smoothed maximum-score estimates" width="80%">

**Estimator comparison**

| Estimator                |   Normalized beta |    Error |   Classification score |
|:-------------------------|------------------:|---------:|-----------------------:|
| True participation index |          -0.85    |  0       |                 0.6872 |
| Grid maximum score       |          -0.88    | -0.03    |                 0.69   |
| Smoothed maximum score   |          -0.83084 |  0.01916 |                 0.6884 |
| Misspecified logit ratio |          -0.66013 |  0.18987 |                 0.6816 |

The score is the share of choices classified by the normalized index.

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

Maximum score estimates the median-surplus boundary without a full probability model. The objective counts correct classifications, so it is nonsmooth. Smoothing gives a continuous search target while preserving the normalized-index interpretation.

## References

- [Manski, C. F. (1975). Maximum Score Estimation of the Stochastic Utility Model of Choice. *Journal of Econometrics*, 3(3), 205-228.](https://doi.org/10.1016/0304-4076(75)90032-9)
- [Horowitz, J. L. (1992). A Smoothed Maximum Score Estimator for the Binary Response Model. *Econometrica*, 60(3), 505-531.](https://doi.org/10.2307/2951573)
