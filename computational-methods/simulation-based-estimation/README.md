# Simulation-Based Estimation: MSM and Indirect Inference

> Estimate one stochastic search DGP by direct moment matching and auxiliary-model matching.

## Overview

Many structural models are easy to simulate but hard to write as a closed-form likelihood. This tutorial uses a small search environment: each worker receives a log wage offer and accepts it with a smooth probability that rises above a reservation log wage. The econometrician sees offers and acceptances.

The same simulated data are estimated two ways. Method of simulated moments (MSM) matches economic moments such as acceptance rates and accepted wages. Indirect inference instead fits an auxiliary acceptance model to both observed and simulated data and matches the auxiliary coefficients. Common random numbers keep criterion surfaces smooth enough to inspect.

## Equations

The structural data-generating process is

$$
\log w_i = \mu + \sigma z_i,\qquad z_i\sim N(0,1),
$$

and the acceptance decision is stochastic:

$$
\Pr(d_i=1\mid w_i;\theta)
= \frac{1}{1+\exp[-(\log w_i-r)/s]}.
$$

The parameter vector is $\theta=(\mu,\sigma,r)$, while $s$ is fixed. MSM chooses
$\theta$ to match a vector of economic moments:

$$
\hat\theta_{MSM}
= \arg\min_\theta
\left[m_{sim}(\theta)-m_{obs}\right]'
W_m
\left[m_{sim}(\theta)-m_{obs}\right].
$$

Indirect inference fits an auxiliary model $a(d_i,\log w_i)$ and matches its
estimated statistics:

$$
\hat\theta_{II}
= \arg\min_\theta
\left[b_{sim}(\theta)-b_{obs}\right]'
W_b
\left[b_{sim}(\theta)-b_{obs}\right].
$$

Here the auxiliary model is a linear probability regression of acceptance on
log wages, augmented with offer-distribution and acceptance statistics.

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| True $\mu$ | 3.00 | Mean log offer wage |
| True $\sigma$ | 0.45 | Dispersion of log offers |
| True $r$ | 3.15 | Reservation log wage |
| Choice scale $s$ | 0.18 | Smoothness of acceptance rule |
| Observed sample | 5,000 | Synthetic data generated once from the DGP |
| Simulation draws | 30,000 | Common random numbers used in both criteria |
| MSM targets | 5 | Acceptance and wage moments |
| II targets | 6 | Auxiliary coefficients and auxiliary moments |

## Solution Method

Both estimators use the same simulator and the same fixed simulation draws. That keeps objective changes attributable to parameters rather than new Monte Carlo noise at each trial value.

```text
Algorithm: simulation-based estimation with common random numbers
Input: observed data, simulator S(theta, eps), fixed simulation shocks eps_sim
Observed targets:
  MSM: economic moments m_obs
  II: auxiliary statistics b_obs from a simple acceptance model
For each candidate theta:
  Simulate log wages and acceptances using eps_sim
  Compute m_sim(theta) or b_sim(theta)
  Minimize the scaled quadratic distance from observed targets
Report parameter recovery, criterion values, residuals, and criterion surfaces
```

MSM makes moment choice explicit. Indirect inference moves that choice into an auxiliary model: if the auxiliary regression summarizes the behavior that matters, matching its coefficients can be more informative than matching raw moments alone.

## Results

The criterion surfaces show the key simulation-estimation object: a smooth enough mapping from parameters to simulated targets. Both surfaces are drawn at the true offer dispersion. The valleys slope because a higher offer mean and a higher reservation wage can partly offset each other in acceptance behavior.

Common random numbers make the surfaces interpretable rather than dominated by fresh simulation noise.

<img src="figures/criterion-surfaces.png" alt="MSM and indirect-inference criterion surfaces" width="80%">

The observed sample has an acceptance rate of **0.401**. Accepted wages are drawn from the upper tail, but stochastic choice leaves overlap around the reservation wage. That overlap is why the smooth simulator is useful.

The reservation wage is not observed directly; it is inferred from the acceptance pattern around the offer distribution.

<img src="figures/search-data.png" alt="Offer and accepted-wage distributions" width="80%">

Both simulation estimators recover the acceptance rule because their targets use variation around the threshold. The remaining differences are finite-sample and simulation error, not a change in the DGP.

The estimated reservation wages map directly into the smooth acceptance curves.

<img src="figures/acceptance-rule.png" alt="True and estimated acceptance probabilities" width="80%">

Both estimators use the same simulated model but target different summaries of the data.

**Known-truth parameter recovery**

| Parameter            |   True |   MSM estimate |   MSM error |   Indirect inference estimate |   Indirect inference error |
|:---------------------|-------:|---------------:|------------:|------------------------------:|---------------------------:|
| Offer mean mu        |   3    |        3.00719 |     0.00719 |                       3.01486 |                    0.01486 |
| Offer sd sigma       |   0.45 |        0.44364 |    -0.00636 |                       0.44714 |                   -0.00286 |
| Reservation log wage |   3.15 |        3.14907 |    -0.00093 |                       3.15368 |                    0.00368 |

MSM residuals are scaled by the target magnitudes used in the quadratic criterion.

**MSM moment residuals**

| Statistic              |   Observed target |   Simulated at estimate |   Scaled residual |
|:-----------------------|------------------:|------------------------:|------------------:|
| Acceptance rate        |           0.4012  |                 0.40027 |          -0.00233 |
| Mean log wage          |           3.00239 |                 3.00751 |           0.00171 |
| SD log wage            |           0.45058 |                 0.44322 |          -0.01634 |
| Mean accepted log wage |           3.36538 |                 3.35825 |          -0.00212 |
| SD accepted log wage   |           0.31928 |                 0.32516 |           0.01842 |

Indirect inference matches auxiliary regression coefficients and auxiliary distribution statistics.

**Indirect-inference auxiliary residuals**

| Statistic              |   Observed target |   Simulated at estimate |   Scaled residual |
|:-----------------------|------------------:|------------------------:|------------------:|
| LPM intercept          |          -1.75243 |                -1.74529 |           0.00408 |
| LPM slope              |           0.71731 |                 0.7125  |          -0.0067  |
| Mean log wage          |           3.00239 |                 3.01519 |           0.00426 |
| SD log wage            |           0.45058 |                 0.44672 |          -0.00858 |
| Acceptance rate        |           0.4012  |                 0.40303 |           0.00457 |
| Mean accepted log wage |           3.36538 |                 3.36797 |           0.00077 |

The RMSE is computed against the known DGP parameters, which are available only in the simulation exercise.

**Solver and estimator diagnostics**

| Estimator          |   Criterion | Success   |   Iterations |   Parameter RMSE |
|:-------------------|------------:|:----------|-------------:|-----------------:|
| MSM                | 0.000619286 | True      |           64 |       0.00557007 |
| Indirect inference | 0.000174795 | True      |           84 |       0.00899372 |

## Takeaway

Simulation-based estimation replaces closed-form likelihood evaluation with a simulator and a distance metric. MSM asks the researcher to choose economic moments directly. Indirect inference asks the researcher to choose an auxiliary model whose fitted coefficients summarize the behavior to match. In both cases, common random numbers and residual diagnostics are not cosmetic: they determine whether the criterion surface is informative enough to trust.

## References

- [McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. *Econometrica*, 57(5), 995-1026.](https://doi.org/10.2307/1913621)
- [Gourieroux, C., Monfort, A., and Renault, E. (1993). Indirect Inference. *Journal of Applied Econometrics*, 8(S1), S85-S118.](https://doi.org/10.1002/jae.3950080507)
