# Monetary Policy SVARs with Minnesota Priors

## Overview

A quarterly macro VAR tries to summarize the joint dynamics of output, inflation, and the policy rate. The object of interest is a monetary policy shock: an unexpected policy tightening after current output and inflation have been accounted for.

The problem is that even a small VAR can be noisy in a short macro sample. Four lags of three variables give each equation thirteen coefficients including the intercept. Unrestricted OLS can fit accidental lag patterns and then produce unstable forecasts or impulse responses.

Litterman (1986) identified the gap: macro time series are short, so unrestricted VARs overfit. His answer was a prior that encodes the belief that each variable is best predicted by its own recent past. That prior pulls the coefficient matrix toward a diagonal structure without discarding off-diagonal information when the data are informative.

The economic object is still a reduced-form forecasting system plus an identifying assumption for shocks. Shrinkage regularizes the reduced form. The recursive SVAR ordering is a separate step that gives one residual innovation a monetary-policy interpretation.

## Read before

- [Autoregressive Processes](../ar-processes/README.md)
- [Reduced-Form VARs](../reduced-form-var/README.md)
- [Bayesian Foundations](../../bayesian-methods/bayesian-foundations/README.md)

## Equations

Let $`y_t = (x_t, \pi_t, i_t)'`$ collect the output gap, inflation, and the policy rate. The reduced-form VAR stacks $`p`$ lags of $`y_t`$ into a design matrix $`X`$ with an intercept, and writes equation $`i`$ as

```math
y_i = X \beta_i + e_i, \qquad e_i \sim N(0, \sigma_i^2 I_T).
```

The Minnesota prior is Gaussian, $`\beta_i \sim N(b_i^0, V_i^0)`$. The prior mean is persistent only for the own first lag:

```math
b_{i,j,\ell}^0 =
\begin{cases}
\rho_0, & i=j \ \mathrm{and}\ \ell=1,\\
0, & \mathrm{otherwise}.
\end{cases}
```

The prior variance is larger for own lags and smaller for cross lags and distant lags:

```math
v_{i,j,\ell} =
\left(\frac{\lambda}{\ell^d}\right)^2
\left(\frac{\sigma_i}{\sigma_j}\right)^2
\theta_{ij}^2,
\qquad
\theta_{ij}=1 \ \mathrm{for}\ i=j,\quad
\theta_{ij}=\theta \ \mathrm{for}\ i\ne j.
```

Here $`\lambda`$ is the overall tightness, $`d`$ is the lag-decay exponent, and $`\theta`$ is the cross-variable tightness. The tutorial plugs in $`\hat\sigma_i^2`$ from OLS residuals. Conditional on that plug-in scale, conjugacy gives a Gaussian posterior with precision and mean:

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

Recursive SVAR identification follows the convention in the reduced-form-var tutorial. Factor the posterior-mean residual covariance as $`\Sigma_u = P P'`$ with $`P`$ lower triangular, and read each column of $`P`$ as the impact effect of one orthogonal structural shock $`\varepsilon_t = P^{-1} u_t`$. With ordering output gap, inflation, policy rate, the policy shock is the third column.

The plotted shock is scaled to move the policy rate by $`\tau=0.25`$ on impact:

```math
q =
\tau \frac{P e_3}{e_3'P e_3}.
```

Impulse responses propagate the scaled impact vector $`q`$ through the posterior-mean VAR dynamics using the companion-form recursion derived in the reduced-form-var tutorial.

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

Own lag-1 carries the widest prior, so the data move it most freely. Lag decay shrinks own lag-2 to roughly 38% of the own lag-1 width. The cross-equation factor $`\theta`$ pulls cross-variable slopes another notch tighter at lag 1. That ordering is exactly the heatmap pattern in the prior-standard-deviations figure: bright diagonal at short lags, fading rapidly into the off-diagonal and deeper-lag cells.

## Model Setup

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Variables $`k`$ | 3 | VAR lag order $`p`$ | 4 |
| Simulated quarters | 132 | Training observations | 88 |
| Test observations | 40 | Coefficients per equation | 13 |
| Policy shock scale $`\tau`$ | 0.25 | Structural ordering | y, pi, i |
| Own-lag prior mean $`\rho_0`$ | 0.85 | Overall tightness $`\lambda`$ | 0.18 |
| Cross-variable tightness $`\theta`$ | 0.35 | Lag-decay exponent $`d`$ | 1.4 |
| Intercept prior sd | 3.0 | IRF horizon (quarters) | 20 |

## Solution Method

The tutorial estimates the same reduced-form VAR in two ways. OLS treats all lag coefficients as free. The Minnesota BVAR treats the OLS residual scales as fixed, then computes the Gaussian posterior for each equation equation-by-equation. That makes this an empirical-Bayes shrinkage estimator: posterior means and covariance matrices are available without MCMC.

There are two stages. First, estimate stable reduced-form dynamics with the Minnesota prior. Second, take the reduced-form residual covariance and impose a Cholesky ordering to name the policy shock.

```
        quarterly data y_t, Minnesota prior (lambda, d, theta), lag order p
                                        |
                                        v
  +------------- Bayesian VAR (Minnesota-prior BVAR) ---------------+
  |                                                                 |
  |   y_t, p  -->  [ OLS reduced-form VAR ]  -->  sigma_i (scales)  |
  |                                                                 |
  |   sigma_i, hyperparameters  -->  [ Minnesota prior ]            |
  |                     -->  prior mean b_i^0, prior variance V_i^0 |
  |                                                                 |
  |   (b_i^0, V_i^0, sigma_i)  -->  [ Gaussian posterior update ]   |
  |                     -->  posterior mean b_i, posterior cov V_i  |
  |                                                                 |
  |   b_i  -->  [ Cholesky SVAR ]  -->  impact matrix P             |
  |   P, b_i  -->  [ IRF recursion ]  -->  IRF(h)                   |
  |                                                                 |
  +------------------------------------------------------------------+
                                        |
                                        v
                     b_i, V_i, forecasts, IRF(h), FEVD
```

## Results

The shaded region is the held-out forecast block. The data are stationary deviations from a macro steady state, so the policy rate should be read as a rate gap rather than a literal nominal level.

![Simulated output, inflation, and policy-rate series](figures/simulated-macro-series.png)

The Minnesota BVAR has lower test RMSE than the unrestricted OLS VAR. The gain comes from accepting small bias in exchange for lower coefficient variance.

![OLS VAR and Minnesota BVAR one-step forecasts](figures/forecast-comparison.png)

The policy shock is ordered last, so output and inflation do not jump on impact. That zero-impact pattern is the recursive identifying restriction, not a coefficient estimate. The BVAR response is smoother than the OLS response.

![Impulse responses to a recursively identified policy-rate shock](figures/policy-shock-irfs.png)

The heatmap shows the prior before seeing the VAR coefficients. Own first lags have the loosest priors. Cross-variable lags and later lags are pulled more tightly toward zero.

![Prior standard deviations by equation and lagged regressor](figures/minnesota-prior-heatmap.png)

The coefficient posterior plot shows where the data move important VAR slopes relative to the Minnesota prior. The intervals condition on the OLS plug-in residual scales, so they are empirical-Bayes coefficient intervals rather than full posterior draws over all hyperparameters.

![Prior means, posterior means, and approximate intervals for selected coefficients](figures/coefficient-posteriors.png)

### Forecast and Stability Diagnostics

The RMSE table reports one-step forecast errors on quarters not used for estimation. Values below one in the ratio column favor the Minnesota BVAR.

| Variable | OLS VAR RMSE | Minnesota BVAR RMSE | BVAR / OLS |
|:---|---:|---:|---:|
| Output gap | 0.486 | 0.449 | 0.924 |
| Inflation | 0.376 | 0.319 | 0.849 |
| Policy rate | 0.434 | 0.395 | 0.909 |
| All variables | 0.434 | 0.391 | 0.901 |

The stability radius is the spectral radius of the VAR companion matrix. Values below one imply a stationary system. The shrinkage ratio compares the lag-coefficient norm of the Minnesota BVAR with that of the OLS VAR.

| Metric | OLS VAR | Minnesota BVAR |
|:---|---:|---:|
| Companion-matrix stability radius | 0.875 | 0.765 |
| Lag-coefficient norm (relative to OLS) | 1.000 | 0.698 |

These hyperparameters encode the economic belief that macro variables are persistent, but that distant and cross-variable lags should need strong evidence before receiving large coefficients.

| Hyperparameter | Value | Role |
|:---|---:|:---|
| Own first-lag mean | 0.85 | Pulls each variable toward persistent own-lag dynamics |
| Overall tightness | 0.18 | Controls how strongly coefficients are shrunk |
| Cross-variable tightness | 0.35 | Shrinks lags of other variables more tightly than own lags |
| Lag decay | 1.4 | Makes distant lags less important a priori |
| Intercept prior sd | 3 | Leaves equation intercepts weakly regularized |

The posterior table reports the same selected slopes as the interval plot. Own first lags measure persistence. The policy-rate slopes in the output and inflation equations summarize the first dynamic transmission channel.

| Coefficient | Equation | Regressor | Prior mean | Posterior mean | Posterior sd | Approx 95% interval |
|:---|:---|:---|---:|---:|---:|:---|
| Output persistence | Output gap | y(t-1) | 0.85 | 0.799 | 0.079 | [0.643, 0.954] |
| Inflation persistence | Inflation | pi(t-1) | 0.85 | 0.840 | 0.080 | [0.683, 0.996] |
| Policy-rate persistence | Policy rate | i(t-1) | 0.85 | 0.758 | 0.075 | [0.611, 0.905] |
| Policy rate on output | Output gap | i(t-1) | 0 | -0.073 | 0.067 | [-0.205, 0.059] |
| Policy rate on inflation | Inflation | i(t-1) | 0 | -0.033 | 0.043 | [-0.117, 0.051] |

The identification table separates estimated VAR objects from the extra restrictions used to name a residual innovation as a monetary policy shock.

| Object | Meaning | Assumption |
|:---|:---|:---|
| Reduced-form innovation | Forecast error $`u_t`$ from the VAR | Estimated covariance can be non-diagonal |
| Structural shock | Orthogonal shock $`\varepsilon_t`$ | Unit variance and no cross-shock correlation |
| Recursive ordering | Output gap, inflation, then policy rate | Policy can react within quarter to output and inflation |
| Policy shock | Third Cholesky innovation | No impact effect on output or inflation |
| Shock scale | Impact rise in policy rate | Normalized to 0.25 rate points |

The impulse-response table summarizes the BVAR response to a policy shock scaled to raise the policy rate by 25 basis points on impact.

| Variable | Impact | After 4 quarters | After 8 quarters | Selected response | Response type | Quarter |
|:---|---:|---:|---:|---:|:---|---:|
| Output gap | 0.000 | -0.037 | -0.015 | -0.037 | Trough | 4 |
| Inflation | 0.000 | -0.018 | -0.010 | -0.018 | Trough | 4 |
| Policy rate | 0.250 | 0.019 | -0.016 | 0.250 | Peak | 0 |

## Takeaway

*Shrinkage* is the central contribution of the Minnesota prior. Litterman (1986) showed that encoding the prior belief that each variable follows a persistent own-lag process could outperform unrestricted OLS forecasts in real macro data. That result was surprising because it said a dogmatic prior about coefficient structure regularly beats a flexible estimator. The legacy is substantial: Minnesota-style priors became the standard regularization device for empirical macro VARs and are built into modern large-scale BVAR systems. The policy-shock responses also depend on the recursive ordering, so they should be read as conditional responses under that timing assumption. The prior does not replace theory. It regularizes the reduced form toward simple own-lag dynamics while still letting the data estimate monetary-policy transmission.

## See also

- [Reduced-Form VARs](../reduced-form-var/README.md)
- [Autoregressive Processes](../ar-processes/README.md)

## References

- Litterman, Robert B. (1986). Forecasting with Bayesian Vector Autoregressions: Five Years of Experience. *Journal of Business and Economic Statistics*, 4(1), 25-38.
- Doan, Thomas, Robert Litterman, and Christopher Sims (1984). Forecasting and Conditional Projection Using Realistic Prior Distributions. *Econometric Reviews*, 3(1), 1-100.
- Banbura, Marta, Domenico Giannone, and Lucrezia Reichlin (2010). Large Bayesian Vector Auto Regressions. *Journal of Applied Econometrics*, 25(1), 71-92.
