# Mean-Variance Portfolio Frontier

> Diversification, covariance, and the efficient frontier in a small portfolio model.

## Overview

Markowitz portfolio choice formalizes the risk-return tradeoff. Expected return is a weighted average of asset means, while risk depends on the full covariance matrix. Diversification comes from imperfect co-movement, not from low standalone volatility alone.

The inputs are stylized annual expected returns, volatilities, and correlations. Random long-only portfolios show the constrained feasible set; the analytic Markowitz frontier shows the unconstrained minimum-variance portfolio for each target return. The practical caveat is central: frontiers are highly sensitive to estimated means and covariances.

## Equations

For portfolio weights $w$, expected return is

$$
\mu_p = w^\top \mu.
$$

Portfolio variance is

$$
\sigma_p^2 = w^\top \Sigma w.
$$

The efficient frontier solves the minimum-variance problem for each target
return:

$$
\min_w w^\top \Sigma w
\quad \text{subject to} \quad
w^\top \mu = \mu_p,\quad w^\top \mathbf{1} = 1.
$$

A long-only constrained version adds $w_i \geq 0$ for every asset. The
unconstrained frontier is analytically convenient, but it can imply short or
levered positions.

## Model Setup

| Asset | Expected return | Volatility |
|-------|-----------------|------------|
| Bills | 2.5% | 1.0% |
| Bonds | 4.5% | 6.0% |
| Equity | 8.5% | 16.0% |
| Real assets | 6.5% | 12.0% |
| Risk-free rate | 2.0% | Used for Sharpe ratios |

## Solution Method

Random long-only portfolios are simulated from a Dirichlet distribution and assigned their mean, variance, and Sharpe ratio. The unconstrained Markowitz formulas then give the global minimum-variance portfolio, the tangency portfolio, and the continuous frontier.

## Results

Random portfolios fill the feasible long-only region. The analytic frontier shows the best risk-return tradeoff when short positions are allowed by the formula.

<img src="figures/frontier.png" alt="Simulated portfolios and analytic frontier" width="80%">
*Simulated portfolios and analytic frontier*

The unconstrained tangency portfolio can use negative or levered positions. That is a mathematical frontier object, not a recommendation.

<img src="figures/portfolio-weights.png" alt="Weights for selected portfolios" width="80%">
*Weights for selected portfolios*

**Selected portfolio summaries**

| Portfolio                       | Return   | Risk   |   Sharpe | Bills   | Bonds   | Equity   | Real assets   |
|:--------------------------------|:---------|:-------|---------:|:--------|:--------|:---------|:--------------|
| Global min variance             | 2.51%    | 1.00%  |     0.51 | 100.1%  | -0.7%   | -0.1%    | 0.8%          |
| Tangency                        | 2.97%    | 1.37%  |     0.71 | 85.8%   | 7.7%    | 2.6%     | 3.9%          |
| Best simulated long-only Sharpe | 2.96%    | 1.37%  |     0.7  | 87.0%   | 5.9%    | 2.9%     | 4.2%          |

## Takeaway

Markowitz's key insight is covariance. A portfolio is not just a weighted average of standalone risks, because assets move together. The practical caveat is equally important: frontiers are input-sensitive, especially to expected returns.

## Reproduce

```bash
python run.py
```

## References

- [Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.](https://doi.org/10.2307/2975974)
