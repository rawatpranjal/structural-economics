# Macro Forecasting with Stock-Watson Diffusion Indexes

## Overview

A forecaster wants next month's industrial production. The data are monthly indicators such as employment and prices. Each series is noisy, but they move together over the business cycle.

The object is a common macro factor. It is the shared state behind many observed indicators.

The panel has 100 series and 200 months. A forecast regression cannot use every series directly. PCA estimates the factor, then a small AR forecast adds it to own lags.

## Equations

Let $`X_t=(X_{1t},\ldots,X_{Nt})'`$ collect the macro panel at date $`t`$. The
static factor model writes each indicator as common movement plus series noise:

```math
X_{it}=\lambda_i'F_t+e_{it}, \qquad i=1,\ldots,N,\quad t=1,\ldots,T.
```

Here $`F_t\in\mathbb{R}^r`$ is the common macro factor. The loading
$`\lambda_i\in\mathbb{R}^r`$ measures exposure. The error $`e_{it}`$ is
series-specific noise. In this simulated panel, $`r=1`$ and

```math
F_t=\rho_F F_{t-1}+\eta_t,\qquad \eta_t\sim N(0,1), \qquad \lambda_i\sim N(1,0.5^2), \qquad e_{it}\sim N(0,\sigma_{e,i}^2).
```

Each series is standardized before PCA:

```math
Z_{it}=\frac{X_{it}-\bar X_i}{s_i}.
```

Here $`\bar X_i`$ and $`s_i`$ are the sample mean and standard deviation of series $`i`$. PCA uses the eigenvectors with the largest eigenvalues of $`T^{-1}Z'Z`$. The
estimated factor projects each date's standardized panel onto those directions:

```math
\hat F_t=(Z_t'v_1,\ldots,Z_t'v_r)'.
```

Here $`Z_t=(Z_{1t},\ldots,Z_{Nt})'`$ is the standardized panel vector at date $`t`$. Factors are identified only up to scale, sign, and rotation. The plots align
signs and compare standardized factors. The forecast regression adds the
estimated factor to own lags of a target series:

```math
y_{t+h} =\alpha+\sum_{\ell=1}^{p}\beta_\ell y_{t-\ell+1} +\gamma'\hat F_t+\varepsilon_{t+h}.
```

The AR benchmark sets $`\gamma=0`$. A true-factor benchmark replaces $`\hat F_t`$
with the simulated $`F_t`$.

## Worked Numerical Example

The 100-series panel is too large to compute by hand, but the PCA extraction step is the same on a toy $`N=2`$, $`T=4`$ panel. Take two zero-mean series at four dates:

```math
X_{1\cdot}=(-2,-1,1,2), \qquad X_{2\cdot}=(-1,-2,2,1).
```

Each series already has sample mean zero, so demeaning is a no-op. The sample variance with $`T-1=3`$ in the denominator is

```math
s_1^2=s_2^2=\frac{(-2)^2+(-1)^2+1^2+2^2}{3}=\frac{10}{3}, \qquad s_1=s_2=\sqrt{10/3}.
```

Standardize each row by its own $`s_i`$ to get $`Z_{it}=X_{it}/\sqrt{10/3}`$. The cross-product of the two deviation rows is

```math
\sum_{t=1}^{4} X_{1t}X_{2t}=(-2)(-1)+(-1)(-2)+(1)(2)+(2)(1)=8.
```

The sample correlation, equal to the (1,2) entry of $`(T-1)^{-1}Z'Z`$ here, is

```math
\rho=\frac{8}{3\cdot\sqrt{10/3}\cdot\sqrt{10/3}}=\frac{8}{10}=0.8.
```

The standardized covariance matrix is

```math
S=\begin{pmatrix}1 & 0.8 \\ 0.8 & 1\end{pmatrix},
```

whose eigenvalues solve $`(1-\lambda)^2=0.8^2`$, giving $`\lambda_1=1.8`$ and $`\lambda_2=0.2`$. The first eigenvector is $`v_1=(1,1)/\sqrt{2}`$, so the estimated factor at each date is the equal-weight average of standardized series,

```math
\hat F_t=v_1'Z_t=\tfrac{1}{\sqrt{2}}(Z_{1t}+Z_{2t}).
```

The variance share captured by PC1 is

```math
\frac{\lambda_1}{\lambda_1+\lambda_2}=\frac{1.8}{2.0}=\boxed{0.90}.
```

So with $`\rho=0.8`$ between two standardized series, the leading principal component already explains 90 percent of the cross-sectional variance. In the full 100-series run, $`\lambda_1`$ explains 57.2 percent because most of the 99 remaining components each pick up a small slice of idiosyncratic noise rather than common movement.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $`N`$ | 100 | Number of series (cross-section) |
| $`T`$ | 200 | Number of time periods |
| $`r`$ | 1 | True number of factors |
| $`\rho_F`$ | 0.8 | Factor AR(1) persistence |
| $`\lambda_i`$ | $`\sim N(1, 0.25)`$ | Factor loadings |
| $`\sigma_{e,i}`$ | $`\sim U(0.5, 1.5)`$ | Idiosyncratic std. deviations |
| AR lags ($`p`$) | 2 | Lags in forecasting equation |
| Horizon ($`h`$) | 1 | Forecast horizon |
| Initial training share | 60% of the usable evaluation window | Expanding-window forecast start |
| Target series | $`X_{1t}`$ | Representative observed macro variable |

## Solution Method

The computation has two steps. First, PCA estimates one common state from the standardized panel. Second, expanding-window regressions compare forecasts with and without that state.

The wide panel supplies repeated signals about the same business-cycle movement. The leading component averages through series-specific noise.

```text
Algorithm: Stock-Watson diffusion-index forecast
Inputs: panel X_it, target y_t, number of factors r, AR lag order p,
        forecast horizon h, initial training share q
Outputs: estimated factors Fhat_t, AR RMSE, PCA-factor RMSE, true-factor RMSE

1. Standardize each series: Z_it = (X_it - mean_i) / sd_i.
2. Form the cross-sectional covariance matrix S = T^(-1) Z'Z.
3. Extract the r largest eigenvectors v_1,...,v_r of S.
4. Set Fhat_t = (Z_t'v_1,...,Z_t'v_r) for each date t.
5. For each expanding-window forecast origin tau:
      fit AR(p): y[t+h] on 1, y_t,...,y[t-p+1]
      fit factor AR(p): add Fhat_t to the same regression
      fit true-factor AR(p): replace Fhat_t with the simulated F_t
      record each h-step forecast error
6. Compare RMSEs and cumulative squared errors over the evaluation window.
```

## Results

The first plot checks whether PCA measured the simulated state. Sign and scale are arbitrary, so the series are aligned before plotting. The estimate tracks the latent AR(1) factor closely. The sample correlation is 0.9970.

<img src="figures/factor-comparison.png" alt="True common factor vs PCA estimate (correlation = 0.9970). PCA recovers the latent factor up to a scale normalization." width="80%">

The scree plot checks factor count. PC1 explains 57.2% of standardized variance. Later components look small in this controlled one-factor panel.

<img src="figures/scree-plot.png" alt="Scree plot and cumulative variance explained. The sharp drop after the first eigenvalue indicates one dominant factor." width="80%">

The exposure plot shows which indicators carry the common state. The PCA exposure ranking almost matches the true ranking. The correlation is 0.9999.

<img src="figures/factor-loadings.png" alt="Standardized series-factor exposures sorted by the true exposure." width="80%">

The forecast plot compares one-step predictions. AR(2) uses only the target's own lags. The Stock-Watson regression adds the estimated factor. RMSE falls from 1.419 to 1.257. The true-factor forecast has RMSE 1.265.

<img src="figures/forecast-comparison.png" alt="Forecast comparison: the PCA factor forecast reduces RMSE by 11.4% relative to AR(2). Right panel shows cumulative squared errors." width="80%">

The eigenvalue table repeats the scree evidence. The large first eigenvalue is the simulated common factor. The remaining entries mostly reflect series-specific variation.

**Top five eigenvalues and variance explained**

| Component   |   Eigenvalue |   Var. Explained (%) |   Cumulative (%) |
|:------------|-------------:|---------------------:|-----------------:|
| PC1         |       57.198 |                57.2  |            57.2  |
| PC2         |        1.727 |                 1.73 |            58.93 |
| PC3         |        1.5   |                 1.5  |            60.43 |
| PC4         |        1.414 |                 1.41 |            61.84 |
| PC5         |        1.357 |                 1.36 |            63.2  |

The forecast table reports the same loss comparison. The estimated factor and true factor both beat AR(2). The close ordering should not be overinterpreted.

**Out-of-sample forecast comparison**

| Model             |   RMSE error |   Relative RMSE |
|:------------------|-------:|----------------:|
| AR(2)             | 1.4186 |          1      |
| PCA factor AR(2)  | 1.2572 |          0.8862 |
| True factor AR(2) | 1.2649 |          0.8917 |

## Takeaway

Stock-Watson diffusion indexes let a forecaster use many macro indicators without estimating one coefficient per series. In this run, PCA recovers the common state almost exactly. The factor forecast lowers one-step RMSE by 11.4% relative to AR(2). The practical lesson is simple: estimate the shared state first, then forecast with a small regression. The AR-with-factor forecast regression here is the univariate special case of the lag-stacking and OLS mechanics derived in [`time-series/reduced-form-var/`](../../time-series/reduced-form-var/), with the estimated factor entering as an additional regressor on the right-hand side.

## References

- Stock, J. and Watson, M. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.
