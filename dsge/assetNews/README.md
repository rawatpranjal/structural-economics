# Lucas-Tree Dividend News and Asset Prices

> Price a tree claim when investors learn about future dividends before the cash flow arrives.

## Overview

An investor owns a claim to a dividend tree. Today she learns that next period's dividend will be higher. Today's dividend has not moved.

The object is a representative-agent Lucas tree. A surprise shock $z_t$ moves today's dividend. A news shock $n_t$ is observed today and moves tomorrow's dividend.

The computational task is a first-order pricing rule. It maps the dividend state and the news signal into today's asset price.

## Equations

Let $d_t$ be the tree dividend and the representative household's consumption,
and define $x_t=\log d_t$.

The accompanying `model.mod` spec writes the dividend process as

```text
d = exp(rho*log(d(-1)) + sigma1*n(-1) + sigma2*z)
```

or, in log deviations,

$$x_t = \rho x_{t-1} + \sigma_1 n_{t-1} + \sigma_2 z_t.$$

The surprise innovation $z_t$ is contemporaneous. The news innovation $n_t$ is
known at date $t$ but enters dividends at date $t+1$.

The asset-pricing equation is

$$p_t d_t^{-\gamma} = \beta \mathbb{E}_t\left[ d_{t+1}^{-\gamma}(p_{t+1}+d_{t+1}) \right],$$

which is equivalently

$$p_t = \mathbb{E}_t\left[ M_{t+1}(p_{t+1}+d_{t+1}) \right], \qquad M_{t+1}=\beta\left(\frac{d_{t+1}}{d_t}\right)^{-\gamma}.$$

At the deterministic steady state $d=1$,

$$p = \beta(p+1), \qquad p=\frac{\beta}{1-\beta}=99.00.$$

Write $q_t=\log(p_t/p)$. A first-order expansion of the Euler equation gives

$$q_t = \gamma x_t + \beta\mathbb{E}_t q_{t+1} +(1-\beta-\gamma)\mathbb{E}_t x_{t+1}.$$

Since $\mathbb{E}_t x_{t+1}=\rho x_t+\sigma_1 n_t$, the linear solution has
the form

$$q_t = A x_t + B n_t,$$

with

$$A=\frac{\gamma+\rho(1-\beta-\gamma)}{1-\beta\rho}, \qquad B=\sigma_1\left(\beta A+1-\beta-\gamma\right).$$

For this calibration, $A=1.917$ and $B=-0.009$.

## Model Setup

| Primitive | Value | Role |
|---|---:|---|
| $\beta$ | 0.99 | Quarterly discount factor |
| $\gamma$ | 2.0 | CRRA coefficient in marginal utility |
| $\rho$ | 0.9 | Persistence of log dividends |
| $\sigma_1$ | 0.1 | Effect of a unit news innovation on next period's log dividend |
| $\sigma_2$ | 0.1 | Effect of a unit surprise innovation on today's log dividend |
| IRF horizon | 40 quarters | Periods shown in the impulse-response figures |

| Steady-state object | Value |
|---|---:|
| Dividend $d$ | 1.000 |
| Asset price $p$ | 99.000 |
| Price-dividend ratio $p/d$ | 99.000 |
| Gross return $1/\beta$ | 1.0101 |

## Solution Method

First-order perturbation turns the Euler equation into coefficient matching. The state is $(x_t,n_t)$. The solution is $q_t=A x_t+B n_t$. The coefficient $B$ is the impact price of news.

A Klein QZ check solves the same system. It matches the hand-derived coefficients.

A nonlinear perfect-foresight path gives a scale check.

```text
Algorithm: Lucas-tree news and surprise IRFs
Inputs: beta, gamma, rho, sigma1, sigma2, shock type, horizon T
Outputs: x_t, q_t, p_t, and the price-dividend ratio

1. Compute the steady state d=1 and p=beta/(1-beta).
2. Linearize the Euler equation in x_t=log d_t and q_t=log(p_t/p).
3. Use E_t x_{t+1}=rho x_t + sigma1 n_t to solve q_t=A x_t+B n_t.
4. For a surprise shock, set x_0=sigma2 and n_t=0 for all t.
5. For a news shock, set n_0=1, x_0=0, and let x_1=sigma1.
6. Iterate x_t=rho x_{t-1} after the shock has entered dividends.
7. Recover the first-order price response q_t=A x_t+B n_t.
8. For the nonlinear benchmark, extend the same x_t path far into the
   future and solve p_t=beta(d_{t+1}/d_t)^(-gamma)(p_{t+1}+d_{t+1})
   backward from the terminal steady-state price.
```

Here $B<0$. Good dividend news slightly lowers today's price. The payoff arrives in a high-consumption state, where marginal utility is lower.

## Results

A surprise shock moves dividends immediately. The price response mainly reflects the current dividend state. The nonlinear path is close to the first-order rule.

A news shock has different timing. The dividend is still at steady state on impact. The price moves because agents know $x_1$ will be higher.

<img src="figures/irf-surprise-vs-news.png" alt="Dividend and asset-price impulse responses under surprise and news shocks" width="80%">

The date-0 news response separates three terms. Expected resale value raises today's price. The next dividend payoff adds a small positive term.

The stochastic discount factor moves the other way. Future dividends are paid in a high-consumption state. With $\gamma=2$, this term makes the net impact negative.

<img src="figures/price-dynamics.png" alt="Decomposition of the date-0 price response to a positive news shock" width="80%">

In the simulated path, prices track persistent dividends closely. News matters at signal dates. It enters the price rule through $B n_t$. One period later, it enters dividends through $\sigma_1 n_t$.

<img src="figures/simulated-paths.png" alt="Simulated first-order dividend and asset-price paths with surprise and news innovations" width="80%">

The impact table reports percent log deviations. The news experiment has zero dividend movement at date 0. The price-dividend ratio already moves.

The date-1 column shows the delayed cash-flow realization. The nonlinear path is close to the first-order solution.

**Impact and Realization Responses**

| Object                                   |   Surprise t=0 |   News t=0 |   News t=1 |
|:-----------------------------------------|---------------:|-----------:|-----------:|
| Dividend log deviation                   |         10     |      0     |     10     |
| Price log deviation, first order         |         19.174 |     -0.917 |     19.174 |
| Price log deviation, nonlinear benchmark |         19.191 |     -0.897 |     19.191 |
| Price-dividend ratio log deviation       |          9.174 |     -0.917 |      9.174 |

## Takeaway

News shocks change prices before cash flows arrive. The Lucas-tree Euler equation prices a future dividend with future marginal utility. Here positive dividend news moves the price on impact, and the sign is slightly negative.

## References

- Lucas, R. (1978). Asset Prices in an Exchange Economy. *Econometrica*, 46(6), 1429-1445.
- Cochrane, J. (2005). *Asset Pricing*. Princeton University Press.
- Beaudry, P. and Portier, F. (2006). Stock Prices, News, and Economic Fluctuations. *American Economic Review*, 96(4), 1293-1307.
- Schmitt-Grohe, S. and Uribe, M. (2012). What's News in Business Cycles. *Econometrica*, 80(6), 2733-2764.
- Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
