# GMM Foundations: Moment Conditions, Identification, and Optimal Weighting

## Overview

GMM estimates a structural parameter by minimising a weighted distance between sample moments and their population counterparts. When the model implies more moments than parameters, the researcher must choose which moments to target and how to weight them.

The setting here is a scalar location parameter that shifts a non-symmetric error distribution. Hansen's two-step procedure replaces an arbitrary weighting matrix with the inverse moment covariance. A Monte Carlo measures the efficiency gain.

Dense tutorials in the catalog assume this material. The Method-of-Simulated-Moments criterion in [`computational-methods/simulation-based-estimation/`](../../computational-methods/simulation-based-estimation/) and the SMM step in [`agent-based-models/brock-hommes-asset-pricing/`](../../agent-based-models/brock-hommes-asset-pricing/) both apply a weighting matrix derived here. The adversarial estimator in [`structural-econometrics/adversarial-estimation/`](../../structural-econometrics/adversarial-estimation/) is benchmarked against optimally-weighted SMM, with the comparison anchored here.

## Preliminary readings

- [`computational-methods/numerical-optimization/`](../../computational-methods/numerical-optimization/)

## Equations

Let $`x_1, \ldots, x_n`$ be an i.i.d. sample from a distribution indexed by an unknown parameter $`\theta_0 \in \Theta \subseteq \mathbb{R}^p`$. A moment condition is a vector-valued function $`g(\theta, x) \in \mathbb{R}^q`$ that the population satisfies at the truth,

```math
\mathbb{E}[g(\theta_0, x)] = 0.
```

Here $`q`$ counts moments and $`p`$ counts parameters. The model is *just-identified* when $`q = p`$. It is *over-identified* when $`q > p`$. Over-identification is the case of interest. No $`\theta`$ matches every moment exactly in a finite sample, so the estimator must trade them off.

The sample analogue of the moment condition is the empirical mean

```math
\bar g(\theta) = \frac{1}{n} \sum_{i=1}^{n} g(\theta, x_i).
```

GMM defines the estimator as the minimiser of a weighted quadratic form in $`\bar g`$. Let $`W \in \mathbb{R}^{q \times q}`$ be a symmetric positive-definite weighting matrix. The GMM criterion is

```math
Q(\theta) = \bar g(\theta)^{\top} W \, \bar g(\theta),
\qquad
\hat\theta = \arg\min_{\theta} Q(\theta).
```

Different $`W`$ yield consistent estimators with different finite-sample precision. Two-step weighting picks the $`W`$ that minimises asymptotic variance.

Under standard regularity conditions the GMM estimator is asymptotically normal,

```math
\sqrt{n} \, (\hat\theta - \theta_0) \to_d \mathcal{N}\!\left(0, \;
(G^{\top} W G)^{-1} G^{\top} W \Omega W G (G^{\top} W G)^{-1}
\right),
```

where $`G = \mathbb{E}[\partial g(\theta_0, x) / \partial \theta^{\top}] \in \mathbb{R}^{q \times p}`$ is the moment Jacobian and $`\Omega = \mathrm{Var}[g(\theta_0, x)] \in \mathbb{R}^{q \times q}`$ is the moment covariance.

The sandwich variance is minimised when $`W \propto \Omega^{-1}`$. The efficient asymptotic variance collapses to

```math
\mathrm{AVar}(\hat\theta_{\mathrm{eff}}) = (G^{\top} \Omega^{-1} G)^{-1}.
```

A moment in the span of existing ones leaves $`G^{\top} \Omega^{-1} G`$ unchanged after optimal reweighting, so it brings zero efficiency gain. A moment with information orthogonal to existing ones shrinks the asymptotic variance strictly.

Hansen's two-step procedure makes optimal weighting feasible without prior knowledge of $`\Omega`$. A first-step estimator $`\hat\theta^{(1)}`$ uses an arbitrary positive-definite weight, typically $`W = I_q`$. First-step moment vectors estimate the covariance,

```math
\hat\Omega = \frac{1}{n} \sum_{i=1}^{n}
g(\hat\theta^{(1)}, x_i) \, g(\hat\theta^{(1)}, x_i)^{\top}
\;-\; \bar g(\hat\theta^{(1)}) \, \bar g(\hat\theta^{(1)})^{\top}.
```

The second-step estimator re-minimises the criterion with $`W = \hat\Omega^{-1}`$. The asymptotic variance attains the efficient bound.

The simulated method of moments replaces population moments by simulated moments when the analytic moment is unavailable. Let $`\tilde m(\theta)`$ denote the moment computed on simulated data at $`\theta`$, and $`m_{\mathrm{data}}`$ the same moment on observed data. The SMM moment vector is

```math
\hat m(\theta) = \tilde m(\theta) - m_{\mathrm{data}},
```

and the SMM estimator minimises $`\hat m(\theta)^{\top} W \hat m(\theta)`$. The optimal SMM weighting is the same $`\Omega^{-1}`$ derived above, up to a simulation-noise inflation factor. The dense tutorials [`computational-methods/simulation-based-estimation/`](../../computational-methods/simulation-based-estimation/) and [`agent-based-models/brock-hommes-asset-pricing/`](../../agent-based-models/brock-hommes-asset-pricing/) build on this identity.

The J statistic tests over-identification. At the second-step estimator under optimal weighting,

```math
J = n \cdot Q(\hat\theta) = n \, \bar g(\hat\theta)^{\top} \hat\Omega^{-1} \bar g(\hat\theta).
```

Under correct specification $`J \to_d \chi^2_{q - p}`$. A large $`J`$ relative to the chi-square reference flags a violated moment condition, so the model is misspecified rather than estimated poorly.

## Model Setup

The illustration uses a scalar location model

```math
x_i = \theta + \varepsilon_i,
\qquad
\varepsilon_i \sim 0.85 \cdot \mathcal{N}(-0.30, 0.70^2) \;+\; 0.15 \cdot \mathcal{N}(1.70, 1.20^2).
```

The mixture weights and means are chosen so the error has zero population mean. The positive third central moment makes higher moments informative beyond the first. The unknown parameter is $`\theta_0 = 1`$.

| Object | Symbol | Role |
|---|---|---|
| Parameter | $`\theta`$ | location to be estimated [from `computational-methods/simulation-based-estimation/`; `agent-based-models/brock-hommes-asset-pricing/` uses $`\beta`$ as its intensity-of-choice parameter in the same role] |
| Moment vector | $`g(\theta, x)`$ | population-zero residual vector evaluated at sample [Hayashi notation; dense tutorials use $`m`$ for the same object and adopt $`g`$ on cut] |
| Jacobian | $`G`$ | $`\mathbb{E}[\partial g / \partial \theta^{\top}]`$ at the truth [prelim introduces this name] |
| Weighting matrix | $`W`$ | symmetric positive-definite, sets relative moment importance [from `agent-based-models/brock-hommes-asset-pricing/` step 4 of SMM pseudocode, where it is the silent matrix this prelim derives] |
| Moment covariance | $`\Omega`$ | $`\mathrm{Var}[g(\theta_0, x)]`$, the optimal-weighting target [prelim introduces this; dense tutorials adopt the name on cut] |
| SMM moment | $`\hat m(\theta) = \tilde m(\theta) - m_{\mathrm{data}}`$ | simulated-minus-observed summary [from `computational-methods/simulation-based-estimation/` (`m_sim - m_obs`), `agent-based-models/brock-hommes-asset-pricing/` (`m(beta) - m_data`)] |
| J statistic | $`J = n \cdot Q(\hat\theta)`$ | over-identification test under optimal weighting [Hansen (1982)] |
| Sample size | $`n`$ | 500 observations per Monte Carlo replication |
| Monte Carlo replications | $`M`$ | 400 |

The three moment specifications differ in $`q`$:

| Set | Moments | Identification |
|---|---|---|
| 1 moment | mean residual | just-identified ($`q = p = 1`$) |
| 2 moments | mean residual, centred squared residual minus population variance | over-identified by 1 |
| 5 moments | mean, variance, third central moment, 0.05-quantile check, 0.95-quantile check | over-identified by 4 |

The quantile moments use the check-function identity $`\mathbb{E}[\mathbf{1}\{\varepsilon \le q_{\alpha}\} - \alpha] = 0`$, evaluated at the population quantiles of the error. Population quantiles are computed once on a four-million-draw reference sample so the moment set is comparable across replications.

The misspecified DGP adds a state-dependent shift $`0.20 \cdot (\varepsilon_i^2 - \mathrm{Var}(\varepsilon))`$ to the error. This preserves the first moment but violates moments two through five. It feeds the J-statistic exhibit.

## Solution Method

A Nelder-Mead minimiser searches over the scalar $`\theta`$. The criterion $`Q(\theta) = \bar g(\theta)^{\top} W \bar g(\theta)`$ is recomputed at each candidate $`\theta`$. The weighting matrix $`W`$ is set in two passes for the over-identified specifications.

```text
Algorithm: two-step GMM for the location model
Inputs   sample x, moment function g(theta, x), number of moments q
Output   theta_hat, W_optimal

1. Set W = I_q (identity weighting).
2. Solve theta_hat_1 = argmin_theta gbar(theta)' W gbar(theta) by Nelder-Mead.
3. If q = p (just-identified), return theta_hat_1; the weight is irrelevant.
4. Compute Omega_hat as the sample covariance of g(theta_hat_1, x_i).
5. Set W_optimal to the inverse of (Omega_hat + 1e-10 * I_q).
6. Solve theta_hat = argmin_theta gbar(theta)' W_optimal gbar(theta)
   starting from theta_hat_1.
7. Return theta_hat, W_optimal.
```

The ridge term $`10^{-10} \cdot I_q`$ guards against ill-conditioned $`\hat\Omega`$ when two moments are nearly collinear. Steps 4 and 6 carry the content: step 4 measures relative informativeness across moments, and step 6 reweights so a mismatch in an informative moment costs more than a mismatch in a redundant one.

The Monte Carlo wraps the algorithm in an outer loop over $`M = 400`$ samples with deterministic seeds. For each moment set the loop records identity-weighted and optimally-weighted estimates. For the five-moment specification it also records the J statistic at the second-step estimator under $`W_{\mathrm{opt}}`$. A second Monte Carlo with a misspecified DGP repeats the J-statistic exhibit under the alternative.

## Results

The sampling-distribution figure puts the three moment specifications side by side. Histograms show $`\hat\theta`$ under identity and optimal weighting. All centre on the truth, so consistency is not at issue. Dispersion tells the efficiency story.

<img src="figures/sampling-distribution.png" alt="Sampling distribution of theta-hat by moment count and weighting choice" width="95%">

Optimal weighting on two moments cuts variance by 38%, from 0.00242 to 0.00151. The one-moment case is just-identified, so identity and optimal weightings coincide. The five-moment case shows a sharper contrast in the other direction. Identity weighting on five moments is worse than identity weighting on one. The third moment and the two quantile checks have very different scales, so the criterion is dominated by whichever residual is largest in magnitude. Optimal weighting on the same five moments brings variance back down to 0.00201, below the five-moment-identity figure but above the two-moment-optimal figure. The lesson is two-sided. Adding moments without reweighting hurts. Adding moments with reweighting helps at fixed moment count, but finite-sample noise in $`\hat\Omega`$ can break strict monotonicity across moment counts.

The efficiency-gain figure plots Monte Carlo variance of $`\hat\theta`$ against moment count.

<img src="figures/efficiency-gain.png" alt="Monte Carlo variance against moment count for identity and optimal weighting" width="80%">

*At any over-identified moment set, optimally-weighted GMM beats identity-weighted GMM, and the gap widens with the moment count.* The blue identity curve is non-monotone in $`q`$ and rises sharply between two and five moments. The red optimal curve lies strictly below the blue curve everywhere. Asymptotically, optimal weighting is variance-non-increasing in the moment set whenever $`\hat\Omega`$ is consistent. In finite samples the picture is nuanced. Estimating $`\hat\Omega`$ from the same data injects noise that grows with $`q`$. The red curve can tick up between adjacent moment counts at $`n = 500`$ even while staying below identity.

The J-statistic figure tests the moment conditions themselves. Under correct specification the five-moment $`J`$ should follow a chi-square with $`q - p = 4`$ degrees of freedom. Under misspecification it should drift to the right.

<img src="figures/j-statistic.png" alt="J statistic histogram under correct and misspecified data" width="80%">

The misspecified histogram shifts hard to the right, mean near 35; a researcher rejects at any conventional level. The blue histogram tracks the chi-square reference, with mean 4.16 against a chi-square mean of 4. Read $`J`$ alongside any two-step GMM estimate. A small second-step criterion is reassuring. A large $`J`$ relative to $`\chi^2_{q - p}`$ flags a moment condition the model cannot match.

## Takeaway

GMM converts moment conditions into a point estimate. Two-step weighting makes the optimal matrix feasible without prior knowledge of the moment covariance. Adding moments helps only when each new moment carries information orthogonal to existing ones. The J statistic turns the criterion value into a specification test for over-identified models.

The machinery extends to settings where moments are not analytic. SMM replaces $`\bar g(\theta)`$ with a simulated counterpart and inherits the same optimal-weighting derivation. That extension is the subject of [`computational-methods/simulation-based-estimation/`](../../computational-methods/simulation-based-estimation/) and [`agent-based-models/brock-hommes-asset-pricing/`](../../agent-based-models/brock-hommes-asset-pricing/). Conditional moments built from instruments give the IV-GMM estimator that underpins modern empirical micro and macro.

## References

- [Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054.](https://doi.org/10.2307/1912775) Equations 3.1-3.7 derive the two-step weighting matrix and the efficiency bound.
- Hayashi, F. (2000). *Econometrics*, Princeton University Press, Chapter 3. The pedagogical anchor; develops GMM as a generalisation of OLS and instrumental variables.
- Hall, A. R. (2005). *Generalized Method of Moments*, Oxford University Press. Book-length treatment of identification, testing, and finite-sample behaviour.
- [McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. *Econometrica*, 57(5), 995-1026.](https://doi.org/10.2307/1913621) Bridge to SMM.
- [Pakes, A. and Pollard, D. (1989). Simulation and the Asymptotics of Optimization Estimators. *Econometrica*, 57(5), 1027-1057.](https://doi.org/10.2307/1913622) Paired SMM paper with the asymptotic theory.

**See also.** The optimal-weighting matrix glossed over in MSM step 4 of [`computational-methods/simulation-based-estimation/`](../../computational-methods/simulation-based-estimation/) is the $`W = \Omega^{-1}`$ derived here. The silent weighting matrix $`W`$ in step 4 of the SMM pseudocode in [`agent-based-models/brock-hommes-asset-pricing/`](../../agent-based-models/brock-hommes-asset-pricing/) is the same matrix. The optimally-weighted SMM benchmark against which [`structural-econometrics/adversarial-estimation/`](../../structural-econometrics/adversarial-estimation/) compares its discriminator-based estimator uses this two-step procedure.
