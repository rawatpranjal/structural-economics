# Weak-Form Efficient-Market Diagnostics

> Autocorrelation, variance-ratio, and predictive checks for Treasury yield changes.

## Overview

Weak-form efficiency asks whether information in past prices or returns predicts future returns. In this interest-rate setting, the return-like object is the daily change in the ten-year Treasury yield, so the diagnostics ask whether past yield changes forecast future yield changes.

The efficient-market interpretation must be modest. Yield changes are not fully measured holding-period bond returns, and every market-efficiency test is also a test of the maintained expected-return model, sampling choices, and measurement assumptions. The data are a static 1990 Treasury CMT snapshot.

## Equations

Define the daily yield change as

$$
\Delta y_t = y_t - y_{t-1}.
$$

A weak-form predictability regression is

$$
\Delta y_{t+1} = \alpha + \beta \Delta y_t + \epsilon_{t+1}.
$$

For independent increments, autocorrelations should be near zero. The variance
ratio compares a multi-period variance with scaled one-period variance:

$$
VR(q) = \frac{\operatorname{Var}(\Delta_q y_t)}
{q \operatorname{Var}(\Delta y_t)}.
$$

The random-walk benchmark has $VR(q) = 1$.

## Model Setup

| Object | Value |
|--------|-------|
| Series | Daily 10-year Treasury yield changes |
| Data | Static 1990 Treasury CMT snapshot |
| Autocorrelation lags | 1 to 20 |
| Variance-ratio horizons | 2, 5, 10, 20 days |
| Interpretation | Weak-form diagnostic, not a trading backtest |

## Solution Method

Daily yield changes, sample autocorrelations, simple variance ratios, and a one-lag predictive regression provide complementary diagnostics. The variance-ratio diagnostic follows the Lo-MacKinlay intuition, but the exercise does not implement the full heteroskedasticity-robust test statistic.

## Results

Weak-form tests ask whether past changes predict future changes. Sample autocorrelation is the first diagnostic, but isolated bars in a short sample should not be overread.

![Autocorrelations of daily ten-year yield changes](figures/autocorrelations.png)
*Autocorrelations of daily ten-year yield changes*

Under independent increments, q-period variance should scale roughly linearly with q. Deviations are evidence about serial dependence, not automatically evidence of exploitable profits.

![Variance ratios for yield changes](figures/variance-ratios.png)
*Variance ratios for yield changes*

**Weak-form diagnostic summary**

| Diagnostic             |   Value |   Null benchmark |
|:-----------------------|--------:|-----------------:|
| Lag-1 predictive slope |   0.118 |                0 |
| Predictive R-squared   |   0.014 |                0 |
| Variance ratio q=2     |   1.118 |                1 |
| Variance ratio q=5     |   1.161 |                1 |
| Variance ratio q=10    |   1.056 |                1 |
| Variance ratio q=20    |   1.029 |                1 |
| Mean daily change (bp) |   0.03  |                0 |

## Takeaway

Efficient-market tests should be read as disciplined diagnostics. Autocorrelation or variance-ratio deviations can reject a random-walk benchmark, but interpretation requires care because the null bundles market efficiency, expected returns, sampling, and measurement assumptions.

## Reproduce

```bash
python run.py
```

## References

- [Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance, 25(2), 383-417.](https://doi.org/10.2307/2325486)
- [Lo, A. W., and MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks. Review of Financial Studies, 1, 41-66.](https://web.mit.edu/~alo/www/Papers/lo-mackinlay-88.html)
