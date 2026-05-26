# Reduced-Form VARs: Estimation, Impulse Responses, and Cholesky Identification

## Overview

Before Sims (1980), applied macroeconomists imposed large blocks of zero restrictions on multi-equation models to achieve identification. The gap was that those restrictions were theory-driven and untestable, so the models could not be used to summarize what the data said on its own terms. Sims proposed treating every variable as potentially affected by all lags of all other variables, with no a-priori exclusions, and fitting by OLS.

The object is a *reduced-form vector autoregression*. Each variable is a linear function of its own lags and the lags of all other variables in the system, plus a serially uncorrelated forecast error. The coefficient matrices are estimated equation by equation by ordinary least squares.

The forecast errors are correlated across variables. Naming a shock to one variable requires an extra assumption about which errors are allowed to react contemporaneously to which others. The recursive choice is the simplest: pick an ordering and let the lower-triangular Cholesky factor of the error covariance map orthogonal structural shocks to reduced-form errors. A different ordering gives different impulse responses, so the ordering is an identifying assumption, not a property of the data.

This tutorial estimates a bivariate VAR(2) on a simulated output-gap-and-inflation panel, identifies shocks under two orderings, and shows how the estimated impulse responses approach the true ones as the sample grows. The same reduced-form setup feeds the Bayesian shrinkage version in [`time-series/minnesota-svar/`](../../time-series/minnesota-svar/) and the autoregressive backbone of the factor forecast in [`time-series/stock-watson/`](../../time-series/stock-watson/).

## Read before

- [Autoregressive Processes](../ar-processes/README.md)

## Equations

We need a vector-valued analogue of the univariate AR(p) model in [`time-series/ar-processes/`](../../time-series/ar-processes/). Let $`t`$ index time and let $`y_t \in \mathbb{R}^k`$ collect the $`k`$ endogenous variables at date $`t`$. The reduced-form vector autoregression of order $`p`$ writes each $`y_t`$ as an intercept plus a linear function of $`p`$ lags plus a multivariate forecast error:

```math
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t,
\qquad
u_t \sim (0, \Sigma_u).
```

Here $`A_1, \ldots, A_p`$ are $`k \times k`$ coefficient matrices, $`c \in \mathbb{R}^k`$ is the intercept, and $`u_t`$ is the reduced-form residual with covariance matrix $`\Sigma_u`$. The residuals are uncorrelated over time but generally correlated across equations, so $`\Sigma_u`$ need not be diagonal.

We need a one-step recursion that keeps track of all lags at once so that powers of one matrix generate impulse responses. Stacking the current and lagged values into a tall vector $`Y_t = (y_t', y_{t-1}', \ldots, y_{t-p+1}')'`$ of length $`kp`$ gives the companion form (a first-order rewrite of the higher-order VAR):

```math
Y_t = F  Y_{t-1} + e_t,
\qquad
F =
\begin{bmatrix}
A_1 & A_2 & \cdots & A_{p-1} & A_p \\
I_k & 0   & \cdots & 0       & 0   \\
0   & I_k & \cdots & 0       & 0   \\
\vdots & & \ddots & & \vdots \\
0   & 0   & \cdots & I_k     & 0
\end{bmatrix}.
```

The top $`k`$ rows reproduce the VAR equation. The remaining rows shift past observations down by one. The system is stable when the spectral radius of $`F`$ is less than one, in which case the unconditional variance of $`y_t`$ is finite.

We need an estimator for $`(c, A_1, \ldots, A_p)`$ that uses each equation's information without joint nonlinear optimization. Stack the $`p`$ lags and the constant into one design row $`x_t = (1, y_{t-1}', \ldots, y_{t-p}')'`$ of length $`1 + kp`$. The ordinary-least-squares estimator solves each equation separately:

```math
\hat\beta_i = \left(\sum_t x_t x_t'\right)^{-1} \sum_t x_t  y_{i,t}, \qquad i = 1, \ldots, k.
```

Equation-by-equation OLS coincides with the multivariate Gaussian maximum-likelihood estimator under standard assumptions, so for reduced-form purposes it is the default. With raw sample length $`T`$, the design uses $`T - p`$ rows (one row per period from $`p + 1`$ to $`T`$) and each equation has $`kp + 1`$ regressors, so the residual covariance is estimated as $`\hat\Sigma_u = (T - p - kp - 1)^{-1} \sum_t \hat u_t \hat u_t'`$.

We need a way to name shocks so that impulse responses describe how the system reacts to one orthogonal disturbance at a time. The recursive scheme factors $`\Sigma_u`$ into a lower-triangular matrix $`P`$ and its transpose, and treats the transformed residual as the structural shock vector:

```math
\Sigma_u = P  P',
\qquad
\varepsilon_t = P^{-1} u_t,
\qquad
E[\varepsilon_t \varepsilon_t'] = I_k.
```

Triangularity of $`P`$ is the identifying restriction. The first variable in the ordering reacts on impact only to its own structural shock. The second reacts on impact to its own shock and to the first. And so on. Reordering the variables before taking the Cholesky factor gives a different $`P`$ and a different set of impulse responses.

We need to turn $`P`$ and the dynamics in $`F`$ into the response of every variable to every shock at every horizon. The impulse response at horizon $`j`$ is the top $`k \times k`$ block of $`F^j`$ post-multiplied by $`P`$:

```math
\Phi_j = J  F^j  J'  P,
\qquad
J = \begin{bmatrix} I_k & 0_k & \cdots & 0_k \end{bmatrix} \in \mathbb{R}^{k \times kp}.
```

Entry $`(\Phi_j)_{ik}`$ reads as the response of variable $`i`$ to a unit-variance structural shock $`k`$, $`j`$ periods after the shock. Setting $`j = 0`$ recovers $`P`$, which encodes the impact restrictions; setting $`j > 0`$ propagates those restrictions through the estimated lag matrices.

## Worked Numerical Example

Take a bivariate VAR(1) ($`k = 2`$, $`p = 1`$, zero intercept) with $`y_t = (y_{1,t}, y_{2,t})'`$. The headline run uses $`p = 2`$; collapsing to one lag keeps the matrix algebra hand-sized while exercising the same companion-form and Cholesky machinery.

Set the lag matrix and the reduced-form covariance to

```math
A_1 = \begin{pmatrix} 0.5 & 0.2 \\ 0.1 & 0.6 \end{pmatrix},
\qquad
\Sigma_u = \begin{pmatrix} 1.0 & 0.2 \\ 0.2 & 0.5 \end{pmatrix}.
```

With $`p = 1`$ the companion matrix is $`F = A_1`$ and the selector is $`J = I_2`$. Conditioning on $`y_t = (1, 1)'`$, the one-step forecast is

```math
\mathbb{E}_t[y_{t+1}] = A_1 y_t
= \begin{pmatrix} 0.5(1) + 0.2(1) \\ 0.1(1) + 0.6(1) \end{pmatrix}
= \begin{pmatrix} 0.7 \\ 0.7 \end{pmatrix}.
```

The one-step forecast covariance equals the residual covariance, $`\mathrm{Var}_t(y_{t+1}) = \Sigma_u`$.

To check stability, compute the eigenvalues of $`F = A_1`$. The characteristic polynomial is

```math
\det(A_1 - \lambda I_2) = (0.5 - \lambda)(0.6 - \lambda) - (0.2)(0.1)
= \lambda^2 - 1.1\lambda + 0.28 = 0,
```

so $`\lambda = (1.1 \pm \sqrt{1.21 - 1.12})/2 = (1.1 \pm 0.3)/2 \in \{0.7,\ 0.4\}`$. Both lie inside the unit circle, so the system is stable.

The Cholesky factor of $`\Sigma_u`$ under the ordering $`(y_1, y_2)`$ is

```math
P = \begin{pmatrix} 1.0 & 0 \\ 0.2 & \sqrt{0.46} \end{pmatrix},
\qquad \sqrt{0.46} \approx 0.6782,
```

since $`P P' = \begin{pmatrix} 1 & 0.2 \\ 0.2 & 0.04 + 0.46 \end{pmatrix} = \Sigma_u`$. The first column of $`P`$ is the impact response to the structural shock $`\varepsilon_{1,t}`$: a unit shock raises $`y_1`$ by $`1.0`$ and $`y_2`$ by $`0.2`$ on impact. The one-period-ahead response to that same shock is the first column of $`\Phi_1 = F P`$:

```math
\Phi_1\, e_1 = A_1 \begin{pmatrix} 1.0 \\ 0.2 \end{pmatrix}
= \begin{pmatrix} 0.5(1.0) + 0.2(0.2) \\ 0.1(1.0) + 0.6(0.2) \end{pmatrix}
= \boxed{\begin{pmatrix} 0.54 \\ 0.22 \end{pmatrix}}.
```

The recursive ordering forces the impact response of $`y_1`$ to the second structural shock to be zero (top entry of column 2 of $`P`$). After one period, that zero leaks into $`y_1`$ through the lag matrix, which is how the impulse response builds up over horizons.

## Model Setup

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Endogenous variables $`k`$ | 2 | VAR lag order $`p`$ | 2 |
| Sample size $`T`$ | 400 | Burn-in | 200 |
| IRF horizon $`H`$ | 20 | Monte Carlo trials $`K`$ | 200 |
| MC sample sizes | 100 to 1600 | Ordering A | output gap first |
| Ordering B | inflation first | | |

## Solution Method

The procedure has three stages. The first two are mechanical. The third encodes the identifying restriction.

```
         data  y_1, ..., y_T,  lag order p,  horizon H
                           |
                           v
+------------ VAR estimation (OLS) -------------+
|                                               |
|   y  -->  [ design matrix X ]  -->  B_hat     |
|   B_hat  -->  [ residuals ]  -->  Sigma_hat   |
|                                               |
+------------ Cholesky identification ----------+
|                                               |
|   Sigma_hat  -->  [ permute, factor ]  -->  P |
|                                               |
+------------ IRF propagation ------------------+
|                                               |
|   B_hat, P  -->  [ companion F ]  -->  Phi_j  |
|                                               |
+-----------------------------------------------+
                           |
                           v
                  B_hat, Sigma_hat, P, Phi_0 ... Phi_H
```

Sign restrictions, narrative restrictions, and external-instrument identification keep the reduced-form layer unchanged and replace the Cholesky block with an alternative map from $`\Sigma_u`$ to a structural impact matrix. They are named here but not derived; the prelim's job is the reduced form plus the simplest identification scheme.

## Results

The headline run uses a bivariate VAR(2) on $`T = 400`$ simulated observations. The true companion matrix has spectral radius $`0.83`$, well inside the unit circle. The estimated radius is $`0.81`$, so the fitted system is also stable. Coefficient deviations are at most about $`0.1`$ in absolute value at this sample size.

The four panels show the impulse responses under each of the two possible orderings. Each panel reports four lines: the true response and the OLS estimate under the ordering that puts the output gap first, and the true response and the OLS estimate under the ordering that puts inflation first.

![Impulse responses to recursively identified shocks under two Cholesky orderings, with true and estimated paths](figures/irf-by-ordering.png)

The top-right and bottom-left panels make the identifying restriction visible. With the output gap ordered first, an inflation shock has zero impact on the output gap. With inflation ordered first, an output-gap shock has zero impact on inflation. Both restrictions are imposed by the lower-triangular Cholesky factor, not estimated from the data. The dashed estimated lines sit close to the solid true lines under both orderings, but the two true lines themselves are different objects. The choice of ordering changes which shock is interpretable on impact.

The scatter shows the OLS residuals together with the covariance structure and the two Cholesky axes.

![OLS residual scatter with 95 percent covariance ellipses and the Cholesky shock axes for two orderings](figures/residual-cholesky.png)

The estimated 95% covariance ellipse tracks the true ellipse closely at $`T = 400`$. The green arrows are the columns of the Cholesky factor under the ordering that puts the output gap first; the purple arrows are the columns of the factor under the reverse ordering. Each set of arrows describes a different decomposition of the same residual cloud into orthogonal shocks. In each set, one arrow lies on a coordinate axis: that arrow is the impact response to the shock to the second-ordered variable, and its zero on the first-ordered variable's axis is the recursive zero-impact restriction. The other arrow points freely into the residual cloud; it is the impact response to the first-ordered shock and tilts away from a coordinate axis whenever the two residuals are correlated.

The Monte Carlo sweep estimates the same VAR at five sample sizes between $`100`$ and $`1600`$ observations and reports the mean and one-standard-deviation band of the impulse-response RMSE across $`200`$ trials per sample size.

![Mean and one-standard-deviation band of IRF RMSE as a function of sample size, log-log axes with a sqrt-N reference line](figures/irf-rmse-by-n.png)

The mean error falls from roughly $`0.06`$ at small samples to around $`0.015`$ at large samples. The slope on log-log axes matches the dotted reference line of slope $`-1/2`$, which is the rate predicted by standard OLS asymptotics. Larger samples shrink the band as well as the mean.

### Estimation diagnostics

| Quantity | Ordering A | Quantity | Ordering B |
|:---|---:|:---|---:|
| Spectral radius (true) | 0.83 | Spectral radius (est.) | 0.81 |
| IRF RMSE at $`T=100`$ | ~0.06 | IRF RMSE at $`T=1600`$ | ~0.015 |
| Zero on impact, ordering A | inflation on output gap | Zero on impact, ordering B | output gap on inflation |

## Takeaway

*Recursive identification* is what turns a reduced-form VAR into something interpretable. The projection step is mechanical: each variable on its own lags and the others' lags, fitted by OLS. The identification step is not mechanical. It selects an ordering, factors the residual covariance, and treats the lower-triangular factor as the impact map from orthogonal shocks to forecast errors. Different orderings produce different impulse responses for the same reduced form. The ordering is part of the assumptions, not part of the data.

Sims (1980) showed that the data's own dynamics could discipline multi-equation macroeconomic models without imposing cross-equation exclusions. That was the surprise. The legacy is the reduced-form VAR as the standard diagnostic tool in applied macro: it shows what the data say about joint dynamics before a structural model is imposed.

## See also

- [Minnesota-Prior SVARs](../minnesota-svar/README.md)
- [Stock-Watson Factor Forecasts](../stock-watson/README.md)
- [Autoregressive Processes](../ar-processes/README.md)

## References

- Sims, C. A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1-48. Founding paper on reduced-form VARs and recursive identification.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press, Chapter 11. Textbook treatment of OLS VAR estimation, companion form, and recursive impulse responses.
- Stock, J. H. and Watson, M. W. (2001). "Vector Autoregressions." *Journal of Economic Perspectives*, 15(4), 101-115. Practitioner overview distinguishing reduced-form, recursive, and structural VARs.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer, Chapters 9-10. Reference for SVAR identification beyond the recursive case.
