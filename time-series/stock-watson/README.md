# Macro Forecasting with Stock-Watson Diffusion Indexes

> Using a common business-cycle factor to forecast from a crowded macro panel.

## Overview

Suppose a researcher wants a one-month-ahead forecast of industrial production while watching employment, prices, interest rates, credit spreads, new orders, inventories, and many sectoral indicators. Those series are not separate stories. Much of their movement comes from a shared business-cycle state, but the panel is wide and the monthly sample is short.

Stock and Watson's diffusion-index approach measures that shared state before running the forecast. PCA extracts the common movement from the standardized panel, and a small forecast regression uses the estimated factor alongside the target's own lags. This synthetic example keeps the true factor visible, so the tutorial can check whether PCA recovers the latent state and whether the feasible factor forecast improves on an AR benchmark. The [FRED-style macro-data tutorial](../fred-macro-data/) builds the measurement intuition for a small macro panel, while [Persistent Shocks](../ar-processes/) isolates the AR(1) timing used for the common factor here.

## Equations

Let $X_t=(X_{1t},\ldots,X_{Nt})'$ collect the macro panel at date $t$. A static
factor model gives the shared state a precise form:

$$
X_{it}=\lambda_i'F_t+e_{it},
\qquad i=1,\ldots,N,\quad t=1,\ldots,T.
$$

Here $F_t\in\mathbb R^r$ is the common macro state, $\lambda_i\in\mathbb R^r$
is the exposure of series $i$ to that state, and $e_{it}$ is the series-specific
part. In the simulated panel used below, $r=1$ and

$$
F_t=\rho_F F_{t-1}+\eta_t,\qquad \eta_t\sim N(0,1),
\qquad
\lambda_i\sim N(1,0.5^2),
\qquad
e_{it}\sim N(0,\sigma_{e,i}^2).
$$

Before estimation, each series is standardized:

$$
Z_{it}=\frac{X_{it}-\bar X_i}{s_i}.
$$

Let $v_1,\ldots,v_r$ be the eigenvectors associated with the $r$ largest
eigenvalues of $T^{-1}Z'Z$. PCA estimates the factor by projecting each date's
standardized panel onto those directions:

$$
\hat F_t=(Z_t'v_1,\ldots,Z_t'v_r)'.
$$

Factors and loadings are identified only up to scale, sign, and rotation, so the
comparisons use normalized factors and standardized series-factor exposures. The
forecasting equation for a target series $y_t$ is

$$
y_{t+h}
=\alpha+\sum_{\ell=1}^{p}\beta_\ell y_{t-\ell+1}
+\gamma'\hat F_t+\varepsilon_{t+h}.
$$

The AR benchmark sets $\gamma=0$. The true-factor benchmark replaces $\hat F_t$
with the simulated $F_t$.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N$ | 100 | Number of series (cross-section) |
| $T$ | 200 | Number of time periods |
| $r$ | 1 | True number of factors |
| $\rho_F$ | 0.8 | Factor AR(1) persistence |
| $\lambda_i$ | $\sim N(1, 0.25)$ | Factor loadings |
| $\sigma_{e,i}$ | $\sim U(0.5, 1.5)$ | Idiosyncratic std. deviations |
| AR lags ($p$) | 2 | Lags in forecasting equation |
| Horizon ($h$) | 1 | Forecast horizon |
| Initial training share | 60% | Expanding-window forecast start |
| Target series | $X_{1t}$ | Representative observed macro variable |

## Solution Method

The method turns a high-dimensional forecasting problem into two small objects: an estimated common state and a short forecast regression. PCA works here because the cross-section contains repeated signals about the same aggregate movement. Averaging through the leading principal component filters out much of the idiosyncratic noise, while the forecast regression stays small enough for a short macro sample.

```text
Algorithm: Stock-Watson diffusion-index forecast
Inputs: panel X_it, target y_t, number of factors r, AR lag order p,
        forecast horizon h, initial training share q
Outputs: estimated factors Fhat_t, AR RMSE, PCA-factor RMSE, true-factor RMSE

1. Standardize each series: Z_it = (X_it - mean_i) / sd_i.
2. Form the cross-sectional covariance matrix S = T^{-1} Z'Z.
3. Extract the r largest eigenvectors v_1,...,v_r of S.
4. Set Fhat_t = (Z_t'v_1,...,Z_t'v_r) for each date t.
5. For each expanding-window forecast origin tau:
      fit AR(p): y_{t+h} on 1, y_t,...,y_{t-p+1}
      fit factor AR(p): add Fhat_t to the same regression
      fit true-factor AR(p): replace Fhat_t with the simulated F_t
      record each h-step forecast error
6. Compare RMSEs and cumulative squared errors over the evaluation window.
```

In this run, the first principal component explains 57.2% of standardized panel variance and has correlation 0.9970 with the true factor after sign alignment. The feasible factor forecast lowers RMSE by 28.0% relative to AR(2); the true-factor benchmark lowers it by 27.4%.

## Results

The first plot checks whether the measured common state is the one that generated the economy. Because a factor model is invariant to sign and scale, the PCA estimate is aligned and rescaled before plotting. After that normalization, it tracks the latent AR(1) state closely: the sample correlation is 0.9970. That matters for forecasting because one observed series is noisy; the cross-section filters some of that noise.

<img src="figures/factor-comparison.png" alt="True common factor vs PCA estimate (correlation = 0.9970). PCA recovers the latent factor up to a scale normalization." width="80%">

The scree plot asks how many aggregate states the panel is carrying. Here the answer is deliberately sharp: PC1 explains 57.2% of the standardized variance, and the next components look like residual cross-sectional noise. In an empirical FRED-MD application the choice would be less mechanical, but the same calculation keeps the factor dimension tied to the data.

<img src="figures/scree-plot.png" alt="Scree plot and cumulative variance explained. The sharp drop after the first eigenvalue correctly indicates one dominant factor." width="80%">

The exposure plot checks which indicators carry the common state. A high-exposure series is a clean signal of aggregate movement; a low-exposure series is mostly idiosyncratic. The PCA exposure ranking has correlation 0.9999 with the true ranking, so the estimator traces the time path of $F_t$ and also recovers which series are informative about it.

<img src="figures/factor-loadings.png" alt="Standardized series-factor exposures sorted by the true exposure." width="80%">

The forecast comparison asks whether the estimated state helps predict the target series. The AR(2) benchmark uses only the target's own lags. The feasible Stock-Watson regression adds the estimated diffusion index and cuts RMSE from 1.753 to 1.262. The true-factor line, which uses the simulated $F_t$, reaches 1.273. The two factor forecasts are close; the small finite-sample ordering should not be read as a structural ranking.

<img src="figures/forecast-comparison.png" alt="Forecast comparison: the PCA factor forecast reduces RMSE by 28.0% relative to AR(2). Right panel shows cumulative squared errors." width="80%">

The eigenvalue table is the numerical version of the scree plot. The large first eigenvalue is the simulated common state; the remaining small eigenvalues are mostly idiosyncratic variation that should not enter the forecast regression without evidence.

**Top five eigenvalues and variance explained**

| Component   |   Eigenvalue |   Var. Explained (%) |   Cumulative (%) |
|:------------|-------------:|---------------------:|-----------------:|
| PC1         |       57.198 |                57.2  |            57.2  |
| PC2         |        1.727 |                 1.73 |            58.93 |
| PC3         |        1.5   |                 1.5  |            60.43 |
| PC4         |        1.414 |                 1.41 |            61.84 |
| PC5         |        1.357 |                 1.36 |            63.2  |

The true-factor row is useful because the data-generating process is known. In this finite sample it is a benchmark, not a guaranteed lower bound on realized RMSE: the estimated factor is built from the same panel and can pick up small sample-specific target comovement. The main lesson is that both factor regressions beat the own-lag forecast.

**Out-of-sample forecast comparison**

| Model             |   RMSE |   Relative RMSE |
|:------------------|-------:|----------------:|
| AR(2)             | 1.753  |          1      |
| PCA factor AR(2)  | 1.2621 |          0.72   |
| True factor AR(2) | 1.2732 |          0.7263 |

## Takeaway

Stock-Watson diffusion indexes let a forecaster use a wide macro panel without estimating hundreds of time-series coefficients. In this controlled run, PCA recovers the common state almost exactly (factor correlation 0.9970) and lowers one-step RMSE by 28.0% relative to AR(2). The true-factor benchmark lowers RMSE by 27.4%, so the feasible diffusion index comes close to the forecast that observes the simulated common state.

The caveat is economic. PCA measures co-movement, not causality, and the number of factors is a modeling choice disciplined by the data. In a real macro panel, the researcher still has to choose transformations, vintages, forecast horizons, and evaluation windows before treating the factors as evidence.

## References

- Stock, J. and Watson, M. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.
- Bai, J. and Ng, S. (2002). "Determining the Number of Factors in Approximate Factor Models." *Econometrica*, 70(1), 191-221.
- Stock, J. and Watson, M. (2006). "Forecasting with Many Predictors." *Handbook of Economic Forecasting*, Vol. 1, Ch. 10.
- Bai, J. (2003). "Inferential Theory for Factor Models of Large Dimensions." *Econometrica*, 71(1), 135-171.
