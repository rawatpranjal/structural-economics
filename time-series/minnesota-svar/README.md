# Monetary Policy SVARs with Minnesota Priors

## Overview

A quarterly macro VAR tries to summarize the joint dynamics of output, inflation, and the policy rate. The object of interest is a monetary policy shock: an unexpected policy tightening after current output and inflation have been accounted for.

The problem is that even a small VAR can be noisy in a short macro sample. Four lags of three variables already give each equation thirteen coefficients including the intercept. Unrestricted OLS can fit accidental lag patterns and then produce unstable forecasts or impulse responses.

The economic object is still a reduced-form forecasting system plus an identifying assumption for shocks. Shrinkage regularizes the reduced form; the recursive SVAR ordering is a separate step that gives one residual innovation a monetary-policy interpretation.

The Minnesota prior is ridge-like shrinkage for dynamic systems. It puts prior mass on persistent own first lags, pulls most other coefficients toward zero, and tightens the prior for cross-variable and distant-lag effects. The shrinkage is soft, so the data can still move coefficients away from the prior when the sample is informative.

## Preliminary readings

- [`time-series/ar-processes/`](../../time-series/ar-processes/)
- [`time-series/reduced-form-var/`](../../time-series/reduced-form-var/)
- [`bayesian-methods/bayesian-foundations/`](../../bayesian-methods/bayesian-foundations/)

## Equations

The reduced-form VAR setup, equation-by-equation OLS, and companion form are the same ones used in [`time-series/reduced-form-var/`](../../time-series/reduced-form-var/). Here $`y_t = (x_t, \pi_t, i_t)'`$ collects the output gap, inflation, and the policy rate, and the equation-by-equation stack writes equation $`i`$ as $`y_i = X \beta_i + e_i`$ with $`e_i \sim N(0, \sigma_i^2 I_T)`$, where $`X`$ contains an intercept and $`p`$ lagged values of $`y_t`$.

The Gaussian-Gaussian conjugate regression update used here (posterior precision $`V_i^{-1}`$ as the sum of prior precision and data precision; posterior mean as the precision-weighted average of the prior mean and the ordinary-least-squares estimate) is derived in [`bayesian-methods/bayesian-foundations/`](../../bayesian-methods/bayesian-foundations/). The Minnesota-specific content is the structured choice of prior mean $`b_i^0`$ and prior covariance $`V_i^0`$ below.

The Minnesota prior is Gaussian, $`\beta_i \sim N(b_i^0, V_i^0)`$. For variable $`j`$ and lag $`\ell`$, the prior mean is persistent only for the own first lag:

```math
b_{i,j,\ell}^0 =
\begin{cases}
\rho_0, & i=j \ \mathrm{and}\ \ell=1,\\
0, & \mathrm{otherwise}.
\end{cases}
```

The prior variance is larger for own lags and smaller for cross lags and distant
lags:

```math
v_{i,j,\ell} =
\left(\frac{\lambda}{\ell^d}\right)^2
\left(\frac{\sigma_i}{\sigma_j}\right)^2
\theta_{ij}^2,
\qquad
\theta_{ij}=1 \ \mathrm{for}\ i=j,\quad
\theta_{ij}=\theta \ \mathrm{for}\ i\ne j.
```

Here $`\lambda`$ is the overall tightness, $`d`$ is the lag-decay exponent, and $`\theta`$ is the base cross-variable tightness.

This tutorial plugs in $`\hat\sigma_i^2`$ from OLS residuals. Conditional on that
plug-in scale, conjugacy gives a Gaussian posterior:

```math
V_i^{-1} =
\frac{X'X}{\hat\sigma_i^2} + (V_i^0)^{-1}.
```

```math
b_i =
V_i\left(
\frac{X'y_i}{\hat\sigma_i^2} + (V_i^0)^{-1}b_i^0
\right).
```

Coefficient uncertainty comes from the posterior covariance:

```math
\mathrm{sd}(\beta_{im}\mid y_i) =
\sqrt{(V_i)_{mm}},
\qquad
\beta_{im}\approx b_{im}\pm 1.96\sqrt{(V_i)_{mm}}.
```

Recursive SVAR identification of the BVAR follows [`time-series/reduced-form-var/`](../../time-series/reduced-form-var/): factor the posterior-mean residual covariance as $`\Sigma_u = P P'`$ with $`P`$ lower triangular, and read each column of $`P`$ as the impact effect of one orthogonal structural shock $`\varepsilon_t = P^{-1} u_t`$. With ordering output gap, inflation, policy rate, the policy shock is the third column. It has zero impact effect on output and inflation because the third column of $`P`$ is zero in those rows. It can still affect them after one or more quarters through the lag matrices. The policy rate can react on impact to output and inflation shocks through the off-diagonal entries $`p_{31}`$ and $`p_{32}`$ in the third row of $`P`$.

The plotted shock is scaled to move the policy rate by $`\tau=0.25`$ on impact:

```math
q =
\tau \frac{P e_3}{e_3'P e_3}.
```

Impulse responses then propagate the scaled impact vector $`q`$ through the posterior-mean VAR dynamics using the companion-form recursion $`\Phi_j = J F^j J' P`$ derived in [`time-series/reduced-form-var/`](../../time-series/reduced-form-var/).

## Worked Numerical Example

Compute the Minnesota prior standard deviation for three coefficients in the output-gap equation, using the calibration from Model Setup: $`\lambda = 0.18`$, $`d = 1.4`$, $`\theta = 0.35`$. Assume homoscedastic residual scales $`\sigma_i = \sigma_j = 1`$ to isolate the shrinkage geometry.

The variance formula from Equations is

```math
v_{i,j,\ell} = \left(\frac{\lambda}{\ell^d}\right)^2 \left(\frac{\sigma_i}{\sigma_j}\right)^2 \theta_{ij}^2.
```

The own-lag-1 coefficient ($`i=j`$, $`\ell=1`$, $`\theta_{ii}=1`$):

```math
v_{i,i,1} = \left(\frac{0.18}{1^{1.4}}\right)^2 (1)^2 (1)^2 = (0.18)^2 = 0.0324,
\qquad \mathrm{sd} = 0.180.
```

The own-lag-2 coefficient ($`i=j`$, $`\ell=2`$). Compute $`2^{1.4} = 2 \cdot 2^{0.4} \approx 2.639`$, so

```math
v_{i,i,2} = \left(\frac{0.18}{2.639}\right)^2 (1)^2 (1)^2 = (0.0682)^2 = 0.00465,
\qquad \mathrm{sd} = 0.0682.
```

The cross-equation lag-1 coefficient ($`i \neq j`$, $`\ell=1`$, $`\theta_{ij}=\theta=0.35`$):

```math
v_{i,j,1} = \left(\frac{0.18}{1^{1.4}}\right)^2 (1)^2 (0.35)^2 = (0.18 \cdot 0.35)^2 = (0.063)^2 = 0.00397,
\qquad \mathrm{sd} = 0.063.
```

Collecting the three prior standard deviations,

```math
\boxed{\mathrm{sd}_{\text{own},1} = 0.180, \quad \mathrm{sd}_{\text{own},2} = 0.0682, \quad \mathrm{sd}_{\text{cross},1} = 0.063.}
```

Own lag-1 carries the widest prior, so the data move it most freely; this matches the posterior persistence estimates near $`0.80`$ in Results, far from the prior mean $`\rho_0 = 0.85`$ in absolute units but well inside the $`\pm 1.96 \cdot 0.08`$ band. Lag decay shrinks own lag-2 to roughly $`38\%`$ of the own lag-1 width, and the cross-equation factor $`\theta`$ pulls cross-variable slopes another notch tighter at lag 1. That ordering is exactly the heatmap pattern in the prior-standard-deviations figure: bright diagonal at short lags, fading rapidly into the off-diagonal and deeper-lag cells.

## Model Setup

| Object | Value | Role |
|---|---:|---|
| Variables | 3 | Output gap, inflation, and policy rate |
| Simulated quarters | 132 | Short macro panel after burn-in |
| VAR lag order $`p`$ | 4 | Quarterly dynamics with one year of lags |
| Training observations | 88 | Sample used to estimate each VAR |
| Test observations | 40 | Held-out quarters for one-step forecasts |
| Coefficients per equation | 13 | Intercept plus lagged variables |
| Structural ordering | y, pi, i | Policy shock ordered last |
| Policy shock scale | 0.25 | Impact rise in the policy rate |

## Solution Method

The tutorial estimates the same reduced-form VAR in two ways. OLS treats all lag coefficients as free; that estimator is documented in [`time-series/reduced-form-var/`](../../time-series/reduced-form-var/). The Minnesota BVAR treats the OLS residual scales as fixed, then computes the Gaussian posterior for each equation. That makes this an empirical-Bayes shrinkage estimator: posterior means and posterior covariance matrices are available without running an MCMC sampler.

There are two stages. First, estimate stable reduced-form dynamics with the Minnesota prior. Second, take the reduced-form residual covariance and impose a Cholesky ordering to name the policy shock. The prior controls coefficient noise; the ordering controls the shock interpretation.

```text
Procedure: Minnesota-prior monetary policy SVAR
Inputs: quarterly series y_t, lag order p, prior hyperparameters
Output: posterior coefficients, forecasts, and policy-shock impulse responses

1. Run the reduced-form pipeline from time-series/reduced-form-var/:
   build X from an intercept and p lags, fit the unrestricted OLS VAR,
   and record residual scales sigma_i.
2. Construct the Minnesota prior mean b_i^0 and diagonal covariance V_i^0.
3. For each equation i:
   precision_i <- X'X / sigma_i^2 + inv(V_i^0)
   covariance_i <- inv(precision_i)
   mean_i <- covariance_i * (X'y_i / sigma_i^2 + inv(V_i^0) b_i^0)
4. Report selected posterior means and 1.96 posterior-sd intervals.
5. Take the BVAR residual covariance, recursive-Cholesky factor as in
   the prelim, and pick the third structural shock because policy is
   ordered last.
6. Scale that shock to raise the policy rate by 25 bp on impact.
7. Propagate impulse responses through the posterior-mean VAR dynamics.
```

## Results

The shaded region is the held-out forecast block. The data are stationary deviations from a macro steady state, so the policy rate should be read as a rate gap rather than a literal nominal level.

<img src="figures/simulated-macro-series.png" alt="Simulated output, inflation, and policy-rate series" width="80%">

The Minnesota BVAR has an overall test RMSE of 0.391, compared with 0.434 for the unrestricted OLS VAR. The gain comes from accepting small bias in exchange for lower coefficient variance.

<img src="figures/forecast-comparison.png" alt="OLS VAR and Minnesota BVAR one-step forecasts" width="80%">

The policy shock is ordered last, so output and inflation do not jump on impact. That zero-impact pattern is the recursive identifying restriction, not a coefficient estimate. The BVAR response is smoother than the OLS response. In this run, output reaches -0.037 and inflation reaches -0.018 after the tightening.

<img src="figures/policy-shock-irfs.png" alt="Impulse responses to a recursively identified policy-rate shock" width="80%">

The heatmap shows the prior before seeing the VAR coefficients. Own first lags have the loosest priors. Cross-variable lags and later lags are pulled more tightly toward zero.

<img src="figures/minnesota-prior-heatmap.png" alt="Prior standard deviations by equation and lagged regressor" width="80%">

The coefficient posterior plot shows where the data move important VAR slopes relative to the Minnesota prior. The intervals condition on the OLS plug-in residual scales, so they are empirical-Bayes coefficient intervals rather than full posterior draws over all hyperparameters.

<img src="figures/coefficient-posteriors.png" alt="Prior means, posterior means, and approximate intervals for selected coefficients" width="80%">

The RMSE table reports one-step forecast errors on quarters not used for estimation. Values below one in the ratio column favor the Minnesota BVAR.

**Forecast RMSE comparison**

| Variable      |   OLS VAR RMSE |   Minnesota BVAR RMSE |   BVAR / OLS |
|:--------------|---------------:|----------------------:|-------------:|
| Output gap    |          0.486 |                 0.449 |        0.924 |
| Inflation     |          0.376 |                 0.319 |        0.849 |
| Policy rate   |          0.434 |                 0.395 |        0.909 |
| All variables |          0.434 |                 0.391 |        0.901 |

The stability radius is the spectral radius of the VAR companion matrix; values below one imply a stationary system. The shrinkage ratio compares the lag-coefficient norm of the Minnesota BVAR with that of the OLS VAR. These are the metrics quoted in the Takeaway.

**Companion-matrix stability and shrinkage metrics**

| Metric                                             |   Value |
|:---------------------------------------------------|--------:|
| Companion-matrix stability radius (OLS VAR)        |   0.875 |
| Companion-matrix stability radius (Minnesota BVAR) |   0.765 |
| BVAR coefficient norm relative to OLS norm         |   0.698 |

These hyperparameters encode the economic belief that macro variables are persistent, but that distant and cross-variable lags should need strong evidence before receiving large coefficients.

**Minnesota prior hyperparameters**

| Hyperparameter           |   Value | Role                                                             |
|:-------------------------|--------:|:-----------------------------------------------------------------|
| Own first-lag mean       |    0.85 | Pulls each variable toward persistent own-lag dynamics           |
| Overall tightness        |    0.18 | Controls how strongly coefficients are shrunk toward prior means |
| Cross-variable tightness |    0.35 | Shrinks lags of other variables more tightly than own lags       |
| Lag decay                |    1.4  | Makes distant lags less important a priori                       |
| Intercept prior sd       |    3    | Leaves equation intercepts weakly regularized                    |

The posterior table reports the same selected slopes as the interval plot. Own first lags measure persistence, while the policy-rate slopes in the output and inflation equations summarize the first dynamic transmission channel.

**Selected coefficient posterior summaries**

| Coefficient                     | Equation    | Regressor   |   Prior mean |   Posterior mean |   Posterior sd | Approx 95% interval   |
|:--------------------------------|:------------|:------------|-------------:|-----------------:|---------------:|:----------------------|
| Output persistence              | Output gap  | y(t-1)      |         0.85 |            0.799 |          0.079 | [0.643, 0.954]        |
| Inflation persistence           | Inflation   | pi(t-1)     |         0.85 |            0.84  |          0.08  | [0.683, 0.996]        |
| Policy-rate persistence         | Policy rate | i(t-1)      |         0.85 |            0.758 |          0.075 | [0.611, 0.905]        |
| Policy rate effect on output    | Output gap  | i(t-1)      |         0    |           -0.073 |          0.067 | [-0.205, 0.059]       |
| Policy rate effect on inflation | Inflation   | i(t-1)      |         0    |           -0.033 |          0.043 | [-0.117, 0.051]       |

The identification table separates estimated VAR objects from the extra restrictions used to name a residual innovation as a monetary policy shock.

**Recursive shock-identification assumptions**

| Object                  | Meaning                                 | Assumption                                              |
|:------------------------|:----------------------------------------|:--------------------------------------------------------|
| Reduced-form innovation | Forecast error $`u_t`$ from the VAR       | Estimated covariance can be non-diagonal                |
| Structural shock        | Orthogonal shock $`\varepsilon_t`$        | Unit variance and no cross-shock correlation            |
| Recursive ordering      | Output gap, inflation, then policy rate | Policy can react within quarter to output and inflation |
| Policy shock            | Third Cholesky innovation               | No impact effect on output or inflation                 |
| Shock scale             | Impact rise in policy rate              | Normalized to 0.25 rate points                          |

The impulse-response table summarizes the BVAR response to a policy shock scaled to raise the policy rate by 25 basis points on impact.

**Selected BVAR policy-shock responses**

| Variable    |   Impact |   After 4 quarters |   After 8 quarters |   Selected response | Response type   |   Quarter |
|:------------|---------:|-------------------:|-------------------:|--------------------:|:----------------|----------:|
| Output gap  |     0    |             -0.037 |             -0.015 |              -0.037 | Trough          |         4 |
| Inflation   |     0    |             -0.018 |             -0.01  |              -0.018 | Trough          |         4 |
| Policy rate |     0.25 |              0.019 |             -0.016 |               0.25  | Peak            |         0 |

## Takeaway

The Minnesota prior does not replace the VAR with a theory model. It regularizes the reduced form toward simple own-lag dynamics while still letting the data estimate monetary-policy transmission. The posterior coefficient intervals show which slopes are pulled close to the prior and which remain informed by the sample. The policy-shock responses also depend on the recursive ordering, so they should be read as conditional responses under that timing assumption. In a short macro sample, the shrinkage lowers the coefficient norm to 0.70 of the OLS norm and gives smoother policy-shock responses. The stability radius falls from 0.88 under OLS to 0.76 under the Minnesota BVAR.

## References

- Doan, Thomas, Robert Litterman, and Christopher Sims (1984). "Forecasting and Conditional Projection Using Realistic Prior Distributions." Econometric Reviews, 3(1), 1-100.
- Litterman, Robert B. (1986). "Forecasting with Bayesian Vector Autoregressions: Five Years of Experience." Journal of Business & Economic Statistics, 4(1), 25-38.
- Banbura, Marta, Domenico Giannone, and Lucrezia Reichlin (2010). "Large Bayesian Vector Auto Regressions." Journal of Applied Econometrics, 25(1), 71-92.
