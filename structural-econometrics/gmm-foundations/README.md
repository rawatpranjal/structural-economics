# GMM Foundations: Moment Conditions, Identification, and Optimal Weighting

## Overview

The Generalized Method of Moments, or GMM, is an estimator. It takes a structural model that predicts the population values of certain summary statistics, called moments, and turns those predictions into a parameter estimate. Examples of moments are the mean, the variance, or the fraction of observations below a threshold.

The hard case arises when the model predicts more moments than it has parameters. No single parameter value can match every moment exactly in a finite sample, so the estimator has to combine the moments into one criterion, typically a weighted sum of squared mismatches. The remaining question is which weights.

Hansen's two-step procedure answers that. The first step uses any reasonable weights to get a preliminary estimate. The second step uses that estimate to learn how noisy each moment is, then reweights so noisier moments count less.

The setting here is a scalar location parameter that shifts a non-symmetric error distribution. A Monte Carlo measures the efficiency gain from two-step weighting against identity weighting.

Dense tutorials in the catalog assume this material. The Method-of-Simulated-Moments criterion in [`computational-methods/simulation-based-estimation/`](../../computational-methods/simulation-based-estimation/) and the SMM step in [`agent-based-models/brock-hommes-asset-pricing/`](../../agent-based-models/brock-hommes-asset-pricing/) both apply a weighting matrix derived here. The adversarial estimator in [`structural-econometrics/adversarial-estimation/`](../../structural-econometrics/adversarial-estimation/) is benchmarked against optimally-weighted SMM, with the comparison anchored here.

## Preliminary readings

- [`computational-methods/numerical-optimization/`](../../computational-methods/numerical-optimization/)

## Equations

A *moment* is a population expectation of some function of the data, such as the mean, variance, or probability of lying below a threshold. A *moment condition* is an equation that says some chosen function of the data has expectation zero at the true parameter. The structural model tells us which expectations should vanish at the truth.

Let $`x_1, \ldots, x_n`$ be an i.i.d. sample from a distribution indexed by an unknown parameter $`\theta_0 \in \Theta \subseteq \mathbb{R}^p`$. Let $`g(\theta, x) \in \mathbb{R}^q`$ be the vector of moment functions. The moment condition states

```math
\mathbb{E}[g(\theta_0, x)] = 0.
```

Here $`q`$ counts moments and $`p`$ counts parameters. The model is *just-identified* when $`q = p`$ and *over-identified* when $`q > p`$. Over-identification is the case of interest. No $`\theta`$ matches every moment exactly in a finite sample, so the estimator has to combine the mismatches into a single number to minimise.

The population expectation is not observed. The sample analogue replaces it with the empirical mean,

```math
\bar g(\theta) = \frac{1}{n} \sum_{i=1}^{n} g(\theta, x_i).
```

By the law of large numbers $`\bar g(\theta)`$ converges to $`\mathbb{E}[g(\theta, x)]`$, which is zero at the truth and nonzero elsewhere.

To pick a single estimate, the estimator turns the $`q`$-vector $`\bar g(\theta)`$ into a scalar by taking a weighted sum of squared entries. Let $`W \in \mathbb{R}^{q \times q}`$ be a symmetric positive-definite weighting matrix. The GMM criterion is

```math
Q(\theta) = \bar g(\theta)^{\top} W \, \bar g(\theta),
\qquad
\hat\theta = \arg\min_{\theta} Q(\theta).
```

Every positive-definite $`W`$ gives a consistent estimator in the limit. The choice of $`W`$ only affects finite-sample precision. The next step asks which $`W`$ minimises the variance of $`\hat\theta`$.

Under standard regularity conditions the GMM estimator is asymptotically normal,

```math
\sqrt{n} \, (\hat\theta - \theta_0) \to_d \mathcal{N}\!\left(0, \;
(G^{\top} W G)^{-1} G^{\top} W \Omega W G (G^{\top} W G)^{-1}
\right),
```

where $`G = \mathbb{E}[\partial g(\theta_0, x) / \partial \theta^{\top}] \in \mathbb{R}^{q \times p}`$ is the moment Jacobian, and $`\Omega = \mathrm{Var}[g(\theta_0, x)] \in \mathbb{R}^{q \times q}`$ is the covariance of the moment functions at the truth. The covariance of this limit distribution is the *asymptotic variance* of $`\hat\theta`$; the smaller, the more precise the estimator.

The formula above is called a *sandwich* because $`\Omega`$ sits between two copies of $`W`$ in the inner expression. The question is which $`W`$ minimises it. A direct calculation gives the answer: $`W \propto \Omega^{-1}`$. The efficient asymptotic variance, meaning the smallest achievable by any GMM estimator using these moments, collapses to

```math
\mathrm{AVar}(\hat\theta_{\mathrm{eff}}) = (G^{\top} \Omega^{-1} G)^{-1}.
```

Adding a moment to an optimally weighted set can only weakly improve precision. If the new moment is a linear combination of the existing ones at the truth, $`G^{\top} \Omega^{-1} G`$ does not change and the variance stays the same. If the new moment carries identifying variation that the existing ones lack, the variance strictly shrinks.

The optimal weight $`\Omega^{-1}`$ depends on the unknown $`\theta_0`$. Hansen's two-step procedure makes optimal weighting feasible by estimating $`\Omega`$ in a first pass. A first-step estimator $`\hat\theta^{(1)}`$ uses an arbitrary positive-definite weight, typically $`W = I_q`$. First-step moment vectors then estimate the covariance,

```math
\hat\Omega = \frac{1}{n} \sum_{i=1}^{n}
g(\hat\theta^{(1)}, x_i) \, g(\hat\theta^{(1)}, x_i)^{\top}
\;-\; \bar g(\hat\theta^{(1)}) \, \bar g(\hat\theta^{(1)})^{\top}.
```

The second-step estimator re-minimises the criterion with $`W = \hat\Omega^{-1}`$. The asymptotic variance attains the efficient bound.

When the model has no analytic formula for $`\mathbb{E}[g(\theta, x)]`$, the simulated method of moments replaces it by an average over a simulated dataset drawn at $`\theta`$. Let $`\tilde m(\theta)`$ denote the simulated moment, and $`m_{\mathrm{data}}`$ the same summary on observed data. The SMM moment vector is the gap between the two,

```math
\hat m(\theta) = \tilde m(\theta) - m_{\mathrm{data}},
```

and the SMM estimator minimises $`\hat m(\theta)^{\top} W \hat m(\theta)`$. The optimal SMM weighting is the same $`\Omega^{-1}`$ derived above, up to a simulation-noise inflation factor. The dense tutorials [`computational-methods/simulation-based-estimation/`](../../computational-methods/simulation-based-estimation/) and [`agent-based-models/brock-hommes-asset-pricing/`](../../agent-based-models/brock-hommes-asset-pricing/) build on this identity.

A final question is whether the moment conditions themselves are correct. If the model is misspecified, no $`\theta`$ satisfies all of them. The J statistic turns the criterion value at the second-step estimator into a formal test,

```math
J = n \cdot Q(\hat\theta) = n \, \bar g(\hat\theta)^{\top} \hat\Omega^{-1} \bar g(\hat\theta).
```

Under correct specification $`J \to_d \chi^2_{q - p}`$, a chi-squared distribution with degrees of freedom equal to the number of over-identifying moments. A large $`J`$ relative to that reference means at least one moment condition is violated at the estimator. The model is misspecified; the issue is not estimation noise but a wrong moment condition.

## Worked Numerical Example

To see the criterion at work, take a just-identified scalar mean estimator with $`p = q = 1`$, single moment function $`g(\theta, x) = x - \theta`$, and the four observations $`x = \{0.5, 1.2, 0.9, 1.4\}`$.

The sample moment is

```math
\bar g(\theta) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \theta) = \bar x - \theta,
\qquad
\bar x = \frac{0.5 + 1.2 + 0.9 + 1.4}{4} = \frac{4.0}{4} = 1.0.
```

Because $`q = p = 1`$, any positive weight $`W > 0`$ yields the same minimiser, so set $`W = 1`$. The criterion is

```math
Q(\theta) = (\bar x - \theta)^{2},
\qquad
\frac{dQ}{d\theta} = -2(\bar x - \theta) = 0
\implies \hat\theta = \bar x = 1.0.
```

For the standard error, the moment covariance at the truth is $`\Omega = \mathrm{Var}(x - \theta_0) = \mathrm{Var}(x)`$, estimated by the sample variance

```math
\hat\Omega = s^{2} = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \hat\theta)^{2}
           = \frac{0.25 + 0.04 + 0.01 + 0.16}{3} = \frac{0.46}{3} = 0.1533.
```

The Jacobian is $`G = \mathbb{E}[\partial g / \partial \theta] = -1`$, so the asymptotic variance formula $`(G^{\top} \Omega^{-1} G)^{-1} = \hat\Omega`$ gives

```math
\mathrm{SE}(\hat\theta) = \sqrt{\hat\Omega / n} = \frac{s}{\sqrt{n}}
                       = \frac{\sqrt{0.1533}}{\sqrt{4}} = \frac{0.392}{2}
\implies \boxed{\hat\theta = 1.0, \quad \mathrm{SE}(\hat\theta) \approx 0.196}.
```

The $`t`$-statistic for $`H_0: \theta_0 = 0`$ is $`t = 1.0 / 0.196 = 5.10`$, well past the usual rejection threshold. The just-identified case collapses to method-of-moments: the GMM machinery only starts paying off when $`q > p`$ forces the weighting choice in $`W`$ to matter.

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

The ridge term $`10^{-10} \cdot I_q`$ guards against ill-conditioned $`\hat\Omega`$ when two moments are nearly collinear. Steps 4 and 6 carry the content. Step 4 estimates the variances and covariances of the moment functions. Step 6 reweights so that a precise (low-variance) moment gets a large weight and a noisy (high-variance) moment gets a small one.

The Monte Carlo wraps the algorithm in an outer loop over $`M = 400`$ samples with deterministic seeds. For each moment set the loop records identity-weighted and optimally-weighted estimates. For the five-moment specification it also records the J statistic at the second-step estimator under $`W_{\mathrm{opt}}`$. A second Monte Carlo with a misspecified DGP repeats the J-statistic exhibit under the alternative.

## Results

The sampling-distribution figure puts the three moment specifications side by side. Histograms show $`\hat\theta`$ under identity and optimal weighting. All centre on the truth, so consistency is not at issue. Dispersion tells the efficiency story.

<img src="figures/sampling-distribution.png" alt="Sampling distribution of theta-hat by moment count and weighting choice" width="95%">

Optimal weighting on two moments cuts variance by 38%, from 0.00242 to 0.00151. The one-moment case is just-identified, so identity and optimal weightings coincide. The five-moment case shows a sharper contrast in the other direction. Identity weighting on five moments is worse than identity weighting on one. The reason is scale: with identity weighting every moment is weighted by 1, but the third central moment and the two quantile checks live on very different numerical scales, so the criterion ends up tracking whichever residual is largest in raw magnitude. Optimal weighting on the same five moments brings variance back down to 0.00201, below the five-moment-identity figure but above the two-moment-optimal figure. The lesson is two-sided. Adding moments without reweighting hurts. Adding moments with reweighting helps at fixed moment count, but variance need not fall monotonically across moment counts because estimating $`\hat\Omega`$ from the same data injects sampling noise.

The efficiency-gain figure plots Monte Carlo variance of $`\hat\theta`$ against moment count.

<img src="figures/efficiency-gain.png" alt="Monte Carlo variance against moment count for identity and optimal weighting" width="80%">

*At any over-identified moment set, optimally-weighted GMM beats identity-weighted GMM, and the gap widens with the moment count.* The blue identity curve is non-monotone in $`q`$ and rises sharply between two and five moments. The red optimal curve lies strictly below the blue curve everywhere. In the limit as $`n \to \infty`$, adding a moment under optimal weighting can never raise the variance, provided $`\hat\Omega`$ is consistent. In finite samples, estimating $`\hat\Omega`$ from the same data injects sampling noise that grows with $`q`$. The red curve can tick up between adjacent moment counts at $`n = 500`$ even while staying below identity.

The J-statistic figure tests the moment conditions themselves. Under correct specification the five-moment $`J`$ should follow a chi-square with $`q - p = 4`$ degrees of freedom. Under misspecification it should drift to the right.

<img src="figures/j-statistic.png" alt="J statistic histogram under correct and misspecified data" width="80%">

The misspecified histogram shifts hard to the right, mean near 35. A researcher rejects at any conventional level. The blue histogram tracks the chi-square reference, with mean 4.16 against a chi-square mean of 4. Read $`J`$ alongside any two-step GMM estimate. A small second-step criterion is reassuring. A large $`J`$ relative to $`\chi^2_{q - p}`$ does not signal a noisy estimate; it says the moment conditions themselves do not hold at the estimator, so the model is misspecified.

## Takeaway

GMM converts moment conditions into a point estimate. Two-step weighting makes the optimal matrix feasible without prior knowledge of the moment covariance. Adding moments helps only when each new moment carries identifying variation that the existing moments do not already provide. The J statistic turns the criterion value into a specification test for over-identified models.

The machinery extends to settings where moments are not analytic. SMM replaces $`\bar g(\theta)`$ with a simulated counterpart and inherits the same optimal-weighting derivation. That extension is the subject of [`computational-methods/simulation-based-estimation/`](../../computational-methods/simulation-based-estimation/) and [`agent-based-models/brock-hommes-asset-pricing/`](../../agent-based-models/brock-hommes-asset-pricing/). Conditional moments built from instruments give the IV-GMM estimator that underpins modern empirical micro and macro.

## References

- [Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054.](https://doi.org/10.2307/1912775) Equations 3.1-3.7 derive the two-step weighting matrix and the efficiency bound.
- Hayashi, F. (2000). *Econometrics*, Princeton University Press, Chapter 3. The pedagogical anchor; develops GMM as a generalisation of OLS and instrumental variables.
- Hall, A. R. (2005). *Generalized Method of Moments*, Oxford University Press. Book-length treatment of identification, testing, and finite-sample behaviour.
- [McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. *Econometrica*, 57(5), 995-1026.](https://doi.org/10.2307/1913621) Bridge to SMM.
- [Pakes, A. and Pollard, D. (1989). Simulation and the Asymptotics of Optimization Estimators. *Econometrica*, 57(5), 1027-1057.](https://doi.org/10.2307/1913622) Paired SMM paper with the asymptotic theory.

**See also.** The optimal-weighting matrix glossed over in MSM step 4 of [`computational-methods/simulation-based-estimation/`](../../computational-methods/simulation-based-estimation/) is the $`W = \Omega^{-1}`$ derived here. The silent weighting matrix $`W`$ in step 4 of the SMM pseudocode in [`agent-based-models/brock-hommes-asset-pricing/`](../../agent-based-models/brock-hommes-asset-pricing/) is the same matrix. The optimally-weighted SMM benchmark against which [`structural-econometrics/adversarial-estimation/`](../../structural-econometrics/adversarial-estimation/) compares its discriminator-based estimator uses this two-step procedure.
