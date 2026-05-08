# Estimating a Search Acceptance Rule by Simulation

> Recover offer-distribution and reservation-wage parameters with MSM and indirect inference.

## Overview

Suppose a researcher observes wage offers and worker acceptances but does not observe the reservation wage that turns an offer into a job. The object of interest is an acceptance rule: how the distribution of offers and the latent reservation wage shape employment decisions.

The model is easy to simulate. Draw an offer, pass it through a smooth acceptance rule, and record whether the worker takes the job. Writing and maximizing the exact likelihood is not the point here. The computation replaces that likelihood with simulated data and asks which parameter values make the simulated search economy look like the observed one.

The tutorial estimates the same search environment two ways. Method of simulated moments (MSM) matches economic summaries such as acceptance rates and accepted wages. Indirect inference fits a simple auxiliary acceptance model to observed and simulated samples, then matches the fitted coefficients. Fixed simulation draws make the criterion surfaces readable, so the reader can see where the parameters are identified and where tradeoffs remain.

## Equations

Worker $i$ receives a log wage offer,

$$
\log w_i = \mu + \sigma z_i,\qquad z_i\sim N(0,1),
$$

and accepts with probability

$$
\Pr(d_i=1\mid w_i;\theta)
= \frac{1}{1+\exp[-(\log w_i-r)/s]}.
$$

The parameter vector is $\theta=(\mu,\sigma,r)$. The mean and dispersion of
offers are $\mu$ and $\sigma$, the reservation log wage is $r$, and $s$ fixes
how sharply acceptance changes around the reservation wage.

MSM chooses $\theta$ to match a vector of economic moments:

$$
\hat\theta_{MSM}
= \arg\min_\theta
\left[m_{sim}(\theta)-m_{obs}\right]'
W_m
\left[m_{sim}(\theta)-m_{obs}\right].
$$

Indirect inference takes a different route. It fits an auxiliary model
$a(d_i,\log w_i)$ and matches its estimated statistics:

$$
\hat\theta_{II}
= \arg\min_\theta
\left[b_{sim}(\theta)-b_{obs}\right]'
W_b
\left[b_{sim}(\theta)-b_{obs}\right].
$$

Here the auxiliary model is a linear probability regression of acceptance on
log wages, augmented with offer-distribution and acceptance statistics. It is
not the structural model. It is a low-dimensional description of the acceptance
pattern that the structural simulator has to reproduce.

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| True $\mu$ | 3.00 | Mean of the latent log offer distribution |
| True $\sigma$ | 0.45 | Dispersion of latent log offers |
| True $r$ | 3.15 | Latent reservation log wage |
| Choice scale $s$ | 0.18 | Smoothness of acceptance rule |
| Observed sample | 5,000 | Synthetic data generated once from the DGP |
| Simulation draws | 30,000 | Common random numbers used in both criteria |
| MSM targets | 5 | Acceptance rate and offer-wage moments |
| II targets | 6 | Auxiliary acceptance coefficients and moments |

## Solution Method

The computation uses one simulator and two choices of summary statistics. For a candidate parameter vector, the code simulates a large search panel with fixed random draws, computes the requested summaries, scales their differences from the observed targets, and minimizes the resulting quadratic distance. Reusing the same draws at every trial value keeps changes in the objective tied to parameters rather than fresh Monte Carlo noise.

```text
Algorithm: estimate the search rule by simulation
Input: observed offers and decisions, simulator S(theta, eps), fixed shocks eps_sim
Observed targets
  MSM: m_obs, acceptance and wage moments
  II: b_obs, auxiliary acceptance-model statistics
For each candidate theta:
  Draw simulated offers and acceptances using eps_sim
  Compute m_sim(theta) for MSM or b_sim(theta) for indirect inference
  Evaluate the scaled quadratic distance from the observed targets
Choose the theta with the smallest distance
Output: estimated offer distribution, reservation wage, residuals, and surfaces
```

MSM keeps the economist's moment choice in the foreground. Indirect inference moves some of that choice into an auxiliary regression. If the regression slope captures how acceptance changes with wages, matching it gives the structural estimator information about the reservation rule.

## Results

The criterion surfaces show how the simulated search economy maps parameters into target statistics. Both surfaces hold offer dispersion at its true value. The valleys tilt because a higher offer mean and a higher reservation wage can partly offset each other in the acceptance rate.

Common random numbers make the surfaces interpretable rather than dominated by fresh simulation noise.

<img src="figures/criterion-surfaces.png" alt="MSM and indirect-inference criterion surfaces" width="80%">

The observed sample has an acceptance rate of **0.401**. Accepted wages come mostly from the upper tail of the offer distribution, but stochastic choice leaves overlap around the reservation wage. That overlap is the evidence the estimator uses to locate the latent threshold.

The reservation wage is not observed directly; it is inferred from the acceptance pattern around the offer distribution.

<img src="figures/search-data.png" alt="Offer and accepted-wage distributions" width="80%">

Both simulation estimators recover the acceptance rule because their targets use variation near the threshold. The remaining differences come from the observed sample and the finite simulation, not from a different search model.

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

Simulation-based estimation lets a researcher estimate a structural search rule when simulation is easier than likelihood evaluation. MSM asks the researcher to choose economic moments directly. Indirect inference asks for an auxiliary model whose fitted coefficients summarize the same behavior. In this example, both routes recover the offer distribution and reservation wage because their targets discipline the acceptance pattern around the latent threshold.

## References

- [McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. *Econometrica*, 57(5), 995-1026.](https://doi.org/10.2307/1913621)
- [Gourieroux, C., Monfort, A., and Renault, E. (1993). Indirect Inference. *Journal of Applied Econometrics*, 8(S1), S85-S118.](https://doi.org/10.1002/jae.3950080507)
