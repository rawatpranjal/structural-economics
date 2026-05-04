# Lucas-Tree News Shocks and Stochastic Discounting

> Anticipated dividend news in a representative-agent asset-pricing model where cash-flow news also changes marginal utility.

## Overview

A news shock separates two dates that are often collapsed in simple impulse responses: the date when agents learn something and the date when the cash flow actually changes. In this Lucas-tree example, a signal $n_t$ arrives today and shifts next period's dividend. A surprise shock $z_t$ instead moves today's dividend immediately.

The important economic wrinkle is that the dividend is also aggregate consumption. Good dividend news raises future payoffs, but it also says that future marginal utility will be lower. With the calibration in `model.mod`, $\gamma=2$ makes this discount-rate channel slightly stronger on impact than the cash-flow channel. The point of the tutorial is therefore sharper than "prices move before dividends": anticipated shocks are priced before they realize, and the sign depends on the stochastic discount factor.

## Equations

Let $d_t$ be the tree dividend and the representative household's consumption,
and define $x_t=\log d_t$. The Dynare file writes the dividend process as

```text
d = exp(rho*log(d(-1)) + sigma1*n(-1) + sigma2*z)
```

or, in log deviations,

$$
x_t = \rho x_{t-1} + \sigma_1 n_{t-1} + \sigma_2 z_t.
$$

The surprise innovation $z_t$ is contemporaneous. The news innovation $n_t$ is
known at date $t$ but enters dividends at date $t+1$. The asset-pricing equation is

$$
p_t d_t^{-\gamma}
=
\beta \mathbb{E}_t\left[
d_{t+1}^{-\gamma}(p_{t+1}+d_{t+1})
\right],
$$

which is equivalently

$$
p_t = \mathbb{E}_t\left[
M_{t+1}(p_{t+1}+d_{t+1})
\right],
\qquad
M_{t+1}=\beta\left(\frac{d_{t+1}}{d_t}\right)^{-\gamma}.
$$

At the deterministic steady state $d=1$,

$$
p = \beta(p+1), \qquad p=\frac{\beta}{1-\beta}=99.00.
$$

Write $q_t=\log(p_t/p)$. A first-order expansion of the Euler equation gives

$$
q_t =
\gamma x_t + \beta\mathbb{E}_t q_{t+1}
+(1-\beta-\gamma)\mathbb{E}_t x_{t+1}.
$$

Since $\mathbb{E}_t x_{t+1}=\rho x_t+\sigma_1 n_t$, the linear solution has
the form

$$
q_t = A x_t + B n_t,
$$

with

$$
A=\frac{\gamma+\rho(1-\beta-\gamma)}{1-\beta\rho},
\qquad
B=\sigma_1\left(\beta A+1-\beta-\gamma\right).
$$

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

The impulse responses use log deviations from steady state. The first-order solution is the closed-form pricing rule above. The comparison line is an exact nonlinear perfect-foresight transition for the same realized dividend path, computed by backward recursion on the level Euler equation. It is not a separate stochastic model; it is a local-solution check along the same one-shock experiment.

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

The sign of $B$ is the key diagnostic. Here $B<0$: a positive signal about future dividends slightly lowers today's price because the cash flow arrives in a future high-consumption state and is discounted at lower marginal utility.

## Results

A surprise shock moves dividends immediately, so the price response is mostly a magnified version of the current dividend state. The nonlinear benchmark is nearly indistinguishable from the first-order rule at this scale. A news shock does something different: the dividend is still at steady state on impact, but the price moves because agents already know $x_1$ will be higher. In this calibration that impact movement is slightly negative, not positive, because the marginal-utility effect dominates until the dividend actually realizes.

<img src="figures/irf-surprise-vs-news.png" alt="Dividend and asset-price impulse responses under surprise and news shocks" width="80%">

The date-0 news response can be read as three forces. Higher expected future prices raise today's value, and the next dividend payoff adds a small positive term. The stochastic discount factor moves the other way: future dividends are paid in a high-consumption state, where marginal utility is lower. With $\gamma=2$, that discounting term is just large enough to make the net impact negative.

<img src="figures/price-dynamics.png" alt="Decomposition of the date-0 price response to a positive news shock" width="80%">

In the simulated path, prices mostly track persistent dividends because the coefficient on the current dividend state is large. News still matters at the dates when signals arrive: it enters the price rule immediately through $B n_t$ and then enters the dividend process one period later through $\sigma_1 n_t$.

<img src="figures/simulated-paths.png" alt="Simulated first-order dividend and asset-price paths with surprise and news innovations" width="80%">

The impact table is in percent log deviations. The news experiment has zero dividend movement at date 0 by construction, yet the price and price-dividend ratio already move. The date-1 column shows the delayed cash-flow realization. The nonlinear benchmark is close to the first-order solution, so the sign change is economic rather than a plotting artifact.

**Impact and Realization Responses**

| Object                                   |   Surprise t=0 |   News t=0 |   News t=1 |
|:-----------------------------------------|---------------:|-----------:|-----------:|
| Dividend log deviation                   |         10     |      0     |     10     |
| Price log deviation, first order         |         19.174 |     -0.917 |     19.174 |
| Price log deviation, nonlinear benchmark |         19.191 |     -0.897 |     19.191 |
| Price-dividend ratio log deviation       |          9.174 |     -0.917 |      9.174 |

## Takeaway

News shocks are about information timing, not mechanically about higher prices. The Lucas-tree Euler equation prices a future dividend with the future marginal utility of consumption. If the dividend is paid in a state where consumption is high, the stochastic discount factor can offset the cash-flow effect. In this calibration, positive dividend news moves the price before the dividend, but the impact sign is slightly negative.

That makes this tutorial a useful companion to the [Lucas-tree dynamic-programming asset-pricing tutorial](../../dynamic-programming/asset-pricing/): both are about pricing payoffs with marginal utility, while this one isolates the timing distinction between surprise and anticipated shocks. The [Dynare RBC tutorial](../rbc/) uses the same local-solution logic for real quantities rather than asset prices.

## References

- Lucas, R. (1978). Asset Prices in an Exchange Economy. *Econometrica*, 46(6), 1429-1445.
- Cochrane, J. (2005). *Asset Pricing*. Princeton University Press.
- Beaudry, P. and Portier, F. (2006). Stock Prices, News, and Economic Fluctuations. *American Economic Review*, 96(4), 1293-1307.
- Schmitt-Grohe, S. and Uribe, M. (2012). What's News in Business Cycles. *Econometrica*, 80(6), 2733-2764.
