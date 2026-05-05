# Fama-Bliss Forward-Rate Predictability

> Forward spreads, future yield changes, and the limits of a static Treasury CMT snapshot.

## Overview

Forward rates are useful because they turn the yield curve into a forecast-like object. If long rates mainly average expected future short rates, a steep forward curve should say something about later rate movements. If time-varying risk premia are important, the same spread can instead predict bond returns or compensation for bearing duration risk. The Fama-Bliss regressions sit exactly on that boundary: they are simple predictive regressions with an economic interpretation that depends on the maintained term-structure model.

This page is a teaching analogue, not a replication. The [Treasury yield-curve tutorial](../treasury-yield-curve/) works with the same offline 1990 Treasury CMT panel and explains the measurement object. Here we approximate forward-rate spreads from those par-yield nodes and ask whether they predict twenty-trading-day changes in longer yields. The short horizon and CMT measurement make the exercise diagnostic; they do not deliver the zero-coupon bond panel used by Fama and Bliss.

## Equations

Let $Y_t^1$ be the annual effective one-year CMT yield and $Y_t^n$ be the
annual effective CMT yield at maturity $n$. The code first maps each yield into
a continuously compounded rate,

$$
q_t^m = \log(1+Y_t^m).
$$

For $n>1$, the forward-rate analogue from year 1 to year $n$ is

$$
f_t^{1,n} =
\exp\left(\frac{n q_t^n - q_t^1}{n-1}\right)-1.
$$

The predictor is the forward-minus-short spread

$$
x_t^n = f_t^{1,n} - Y_t^1,
$$

and the dependent variable is the future yield change

$$
\Delta_h Y_{t+h}^n = Y_{t+h}^n - Y_t^n.
$$

The maturity-by-maturity predictive regression is

$$
\Delta_h Y_{t+h}^n = \alpha_n + \beta_n x_t^n + \varepsilon_{t+h}^n,
$$

with $h=20$ trading days. Because CMT rates are par-yield curve
nodes, $f_t^{1,n}$ is an approximation to the forward-rate object, not an
arbitrage-free zero-coupon forward rate.

## Model Setup

| Object | Value |
|--------|-------|
| Data | Static 1990 Treasury CMT panel |
| Date range | 1990-01-02 to 1990-12-31 |
| Usable observations | 230 per maturity after the lead |
| Short yield | 1-year CMT rate |
| Long maturities | 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 30 Yr |
| Forecast horizon | 20 trading days |
| Benchmark forecast | No yield change, $\Delta_h Y_{t+h}^n=0$ |
| Data limitation | CMT par yields, not a zero-coupon Fama-Bliss panel |

## Solution Method

The estimator is deliberately transparent: build the spread implied by the current curve, line it up with a future yield change, and run OLS separately by maturity. The important discipline is not the linear algebra. It is keeping the forecast horizon, maturity, and measurement object fixed when interpreting $\beta_n$.

```text
Algorithm: Fama-Bliss-style forward-spread regression
Input: daily yields Y_t^1 and Y_t^n, maturity set N, horizon h
Output: maturity-specific alpha_n, beta_n, R^2, and forecast errors
Sort observations by date
For each maturity n in N:
    convert yields to q_t^m = log(1 + Y_t^m)
    compute f_t^{1,n} = exp((n q_t^n - q_t^1) / (n - 1)) - 1
    form x_t^n = f_t^{1,n} - Y_t^1
    form Delta_h Y_{t+h}^n = Y_{t+h}^n - Y_t^n
    drop the last h dates, where the future yield is unavailable
    estimate Delta_h Y_{t+h}^n = alpha_n + beta_n x_t^n + epsilon_{t+h}^n by OLS
    compare fitted errors with the no-change forecast Delta_h Y_{t+h}^n = 0
Return the coefficient table and the ten-year fitted path
```

There is no ground-truth term premium in this dataset. The no-change forecast is included only as a modest benchmark: it asks whether the forward spread improves on a random-walk-style prediction for this short sample.

## Results

For the ten-year maturity, wider forward-minus-one-year spreads in this 1990 snapshot are associated with lower subsequent ten-year yields: the estimated slope is **-0.35** with $R^2=0.261$. The sign should not be turned into a structural claim. It is a short-horizon relationship in one CMT panel, useful because it shows how the Fama-Bliss object is constructed and how sensitive interpretation is to the data object.

The horizontal benchmark is a zero predicted yield change. The fitted line has visible slope, but the cloud also makes clear that this is a noisy predictive relationship rather than a pricing identity.

<img src="figures/forward-regression-10y.png" alt="Ten-year forward-spread predictability regression" width="80%">

The fitted ten-year series does better than a flat no-change forecast in this run, with RMSE **23.74 bp** versus **27.63 bp** for the benchmark, a **14.1%** reduction. That comparison is deliberately modest. It checks whether the spread has in-sample predictive content; it is not an out-of-sample trading rule.

The fitted line moves slowly because it is driven by the current curve, while realized twenty-day yield changes contain high-frequency surprises that the spread cannot absorb.

<img src="figures/fitted-vs-realized.png" alt="Realized versus fitted ten-year yield changes" width="80%">

Across maturities, the slopes are negative in this panel and the simple RMSE comparison favors OLS over the no-change benchmark. The pattern is suggestive, not definitive: all regressions use overlapping short-horizon changes from the same single-year CMT sample.

**Forward-regression coefficients by maturity**

| Maturity   |   Intercept (bp) |   Slope |   R-squared |   OLS RMSE (bp) |   No-change RMSE (bp) |   RMSE gain vs zero (%) |   Obs. |
|:-----------|-----------------:|--------:|------------:|----------------:|----------------------:|------------------------:|-------:|
| 2 Yr       |            60.92 |   -1.23 |       0.32  |           21.32 |                 26.7  |                    20.1 |    230 |
| 3 Yr       |            53.02 |   -1.07 |       0.4   |           20.3  |                 26.82 |                    24.3 |    230 |
| 5 Yr       |            31.77 |   -0.6  |       0.338 |           21.93 |                 27.16 |                    19.2 |    230 |
| 7 Yr       |            26.76 |   -0.4  |       0.279 |           22.31 |                 26.32 |                    15.2 |    230 |
| 10 Yr      |            23.77 |   -0.35 |       0.261 |           23.74 |                 27.63 |                    14.1 |    230 |
| 30 Yr      |            20.66 |   -0.29 |       0.212 |           24.84 |                 27.99 |                    11.3 |    230 |

## Takeaway

The forward spread is an economically meaningful summary of the term structure, not just a plotted difference between two rates. In this snapshot it predicts short-run yield changes better than a no-change benchmark, but the result inherits the limits of CMT par yields, overlapping horizons, and a single year of data. The reusable lesson is how to map a yield curve into a predictive regression while keeping the term-structure interpretation honest.

## References

- [Fama, E. F., and Bliss, R. R. (1987). The information in long-maturity forward rates. American Economic Review, 77(4), 680-692.](https://www.econbiz.de/Record/the-information-in-long-maturity-forward-rates-fama-eugene/10015130928)
- [Campbell, J. Y., and Shiller, R. J. (1991). Yield Spreads and Interest Rate Movements: A Bird's Eye View. Review of Economic Studies, 58(3), 495-514.](https://doi.org/10.2307/2298008)
- [Cochrane, J. H., and Piazzesi, M. (2005). Bond Risk Premia. American Economic Review, 95(1), 138-160.](https://www.aeaweb.org/articles?id=10.1257/0002828053828581)
