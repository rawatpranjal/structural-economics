# Macro Forecasting with Stock-Watson Diffusion Indexes

## Overview

Before Stock and Watson (1999, 2002), practitioners either picked a handful of predictors by hand or ran horse races among individual series. Neither approach could use information spread across hundreds of correlated macro indicators simultaneously. Stock and Watson asked: can a single latent index, extracted from a large panel, systematically improve forecasts over a plain autoregression?

The object is a *diffusion index*. It is the dominant principal component of a standardized macro panel, treated as an additional regressor in an otherwise standard forecast equation.

The computation generates a 100-series panel with a known latent factor, extracts the factor via PCA, and compares one-step forecasts from three models. The three models are a plain AR(2), a factor-augmented AR(2), and an oracle AR(2) that observes the true latent state directly.

## Read before

- [Autoregressive Processes](../ar-processes/README.md)
- [Reduced-Form VARs](../reduced-form-var/README.md)

## Equations

Let $`X_t=(X_{1t},\ldots,X_{Nt})'`$ collect the macro panel at date $`t`$. The static factor model writes each indicator as common movement plus series noise:

```math
X_{it}=\lambda_i'F_t+e_{it}, \qquad i=1,\ldots,N,\quad t=1,\ldots,T.
```

Here $`F_t\in\mathbb{R}^r`$ is the common macro factor. The loading $`\lambda_i\in\mathbb{R}^r`$ measures exposure. The error $`e_{it}`$ is series-specific noise. In this simulated panel, $`r=1`$ and

```math
F_t=\rho_F F_{t-1}+\eta_t,\qquad \eta_t\sim N(0,1).
```

```math
\lambda_i\sim N(1,0.5^2), \qquad e_{it}\sim N(0,\sigma_{e,i}^2).
```

Each series is standardized before PCA:

```math
Z_{it}=\frac{X_{it}-\bar X_i}{s_i}.
```

Here $`\bar X_i`$ and $`s_i`$ are the sample mean and standard deviation of series $`i`$. PCA uses the eigenvectors with the largest eigenvalues of $`T^{-1}Z'Z`$. The estimated factor projects each date's standardized panel onto those directions:

```math
\hat F_t=(Z_t'v_1,\ldots,Z_t'v_r)'.
```

Here $`Z_t=(Z_{1t},\ldots,Z_{Nt})'`$ is the standardized panel vector at date $`t`$. Factors are identified only up to scale, sign, and rotation. The forecast regression adds the estimated factor to own lags of a target series:

```math
y_{t+h} =\alpha+\sum_{\ell=1}^{p}\beta_\ell y_{t-\ell+1} +\gamma'\hat F_t+\varepsilon_{t+h}.
```

The AR benchmark sets $`\gamma=0`$. A true-factor benchmark replaces $`\hat F_t`$ with the simulated $`F_t`$.

## Worked Numerical Example

The 100-series panel is too large to trace by hand, but the PCA extraction step is identical on a toy $`N=2`$, $`T=4`$ panel. Take two zero-mean series at four dates:

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

With $`\rho=0.8`$ between two standardized series, the leading principal component already explains 90 percent of the cross-sectional variance. In the full 100-series run, $`\lambda_1`$ explains 57.2 percent because the 99 remaining components each pick up a small slice of idiosyncratic noise rather than common movement.

## Model Setup

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| Series $`N`$ | 100 | Time periods $`T`$ | 200 |
| True factors $`r`$ | 1 | Factor persistence $`\rho_F`$ | 0.8 |
| Loadings $`\lambda_i`$ | $`\sim N(1,0.25)`$ | Idiosyncratic s.d. $`\sigma_{e,i}`$ | $`\sim U(0.5,1.5)`$ |
| AR lags $`p`$ | 2 | Forecast horizon $`h`$ | 1 |
| Initial training share | 60% of eval window | Target series | $`X_{1t}`$ |

## Solution Method

The computation has two steps. First, PCA extracts one common state from the standardized panel by taking the leading eigenvector of $`T^{-1}Z'Z`$. Second, expanding-window regressions compare forecasts with and without that state.

The wide panel supplies many noisy signals about the same business-cycle movement. The leading component averages through series-specific noise and recovers the latent factor up to an arbitrary sign.

```
          standardized panel Z, target y, horizon h
                          |
                          v
    +---------- PCA extraction ----------+
    |  Z  -->  [ T^{-1} Z'Z eigenvec ]  -->  F_hat  |
    +---------------------------------------------+
                          |
                          v
    +---------- factor-augmented forecast ----------+
    |  (y, F_hat) --> [ expanding OLS ] --> y_hat  |
    +----------------------------------------------+
                          |
                          v
              AR RMSE, factor RMSE, relative gain
```

```python
# PCA: extract the leading r eigenvectors of the cross-sectional covariance matrix.
def estimate_factors_pca(X, n_factors=1):
    T, N = X.shape
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    cov_matrix = Z.T @ Z / T
    eigenvalues, eigenvectors = eigh(cov_matrix)
    # Sort descending and project panel onto top eigenvectors.
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    F_hat = Z @ eigenvectors[:, :n_factors]
    return F_hat
```

The eigenvector computation is a one-shot $`O(N^2 T + N^3)`$ operation. No iteration is needed for factor extraction. The expanding-window loop then fits OLS at each forecast origin, adding one observation at a time.

## Results

The left panel tracks the latent AR(1) factor and the PCA estimate over time. The right panel scatters the two series against each other. The sample correlation is 0.9970, confirming that 100 noisy indicators are more than enough to pin down one common state.

![True common factor vs PCA estimate: time series and scatter](figures/factor-comparison.png)

The scree plot confirms one dominant component. PC1 explains 57.2 percent of standardized variance. The eigenvalue drops sharply after the first component and then levels off into the noise floor.

![Scree plot and cumulative variance explained](figures/scree-plot.png)

The left panel sorts each series by its true exposure to the common factor. The PCA exposures track the true ranking almost perfectly. The right panel scatters true against estimated exposures; the exposure correlation is 0.9999.

![Factor exposures sorted by true loading and exposure recovery scatter](figures/factor-loadings.png)

The forecast panel compares one-step predictions. The left panel shows realized and predicted values. The right panel tracks cumulative squared errors. The factor-augmented model separates from AR(2) after the first few out-of-sample periods and never catches back up.

![Forecast comparison and cumulative squared errors](figures/forecast-comparison.png)

### Forecast diagnostics

| Model | RMSE | Relative RMSE | Model | Relative RMSE |
|:---|---:|---:|:---|---:|
| AR(2) | 1.4186 | 1.000 | PCA factor AR(2) | 0.886 |
| True factor AR(2) | 1.2649 | 0.892 | PC1 variance share | 57.2% |

| Component | Eigenvalue | Var. explained (%) | Cumulative (%) |
|:---|---:|---:|---:|
| PC1 | 57.198 | 57.20 | 57.20 |
| PC2 | 1.727 | 1.73 | 58.93 |
| PC3 | 1.500 | 1.50 | 60.43 |
| PC4 | 1.414 | 1.41 | 61.84 |
| PC5 | 1.357 | 1.36 | 63.20 |

## Takeaway

*Diffusion indexes* resolve the degrees-of-freedom problem that blocked large-panel forecasting before Stock and Watson. The forecaster does not need to estimate one coefficient per series. A single latent index, extracted by PCA, captures most of the cross-sectional comovement and transfers it into the forecast equation at the cost of one extra parameter. The surprise in the 1999 and 2002 papers was how much improvement came from the first component alone: a small number of factors consistently beat autoregressive benchmarks across dozens of macro targets, with the marginal gain from adding more factors quickly diminishing. That finding made diffusion indexes a standard tool in central-bank nowcasting and short-horizon macro prediction.

## See also

- [Autoregressive Processes](../ar-processes/README.md)
- [Reduced-Form VARs](../reduced-form-var/README.md)
- [Minnesota-Prior SVARs](../minnesota-svar/README.md)

## References

- Stock, J. and Watson, M. (1999). "Forecasting Inflation." *Journal of Monetary Economics*, 44(2), 293-335.
- Stock, J. and Watson, M. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.
