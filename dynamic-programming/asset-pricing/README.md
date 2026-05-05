# Lucas Tree Asset Prices and the Stochastic Discount Factor

> A representative-agent exchange economy where dividend risk is priced by marginal utility.

## Overview

This tutorial studies the Lucas tree, a setting in which asset prices are equilibrium objects rather than exogenous returns. A representative household owns a claim to the single tree. The tree pays the stochastic endowment $y_t$ each period, and because the household is representative, aggregate consumption must equal the dividend: $c_t=y_t$.

Market clearing pins down the quantity allocation, so all of the economics appears in the state-contingent price of the claim. The price must make the household willing to hold the tree after valuing tomorrow's payoff with the stochastic discount factor $\beta u'(y_{t+1})/u'(y_t)$. Compared with the [income-risk savings problem](../consumption-savings/), the state is still a persistent endowment process, but there is no saving policy to choose; prices absorb the intertemporal risk.

## Equations

Let $x_t=\log y_t$ follow
$$x_{t+1}=\rho x_t+\varepsilon_{t+1}, \qquad
\varepsilon_{t+1}\sim \mathcal{N}(0,\sigma^2),$$
and let the representative household have CRRA utility
$$u(c)=\frac{c^{1-\gamma}}{1-\gamma}, \qquad u'(c)=c^{-\gamma}.$$

In equilibrium $c_t=y_t$. A claim to the tree pays dividend $y_{t+1}$ and resale
value $p(y_{t+1})$ next period, so the Euler equation is
$$p(y_t)=\beta\,\mathbb{E}\left[
\frac{u'(y_{t+1})}{u'(y_t)}
\left(p(y_{t+1})+y_{t+1}\right)
\mid y_t
\right].$$

Define the marginal-utility-scaled price
$$f(y)=u'(y)p(y).$$
Multiplying the Euler equation by $u'(y_t)$ gives the linear fixed point
$$f(y)=\beta\,\mathbb{E}\left[f(y')+u'(y')y'\mid y\right].$$
After solving for $f$, recover the asset price from $p(y)=f(y)/u'(y)$ and the
price-dividend ratio from $p(y)/y$.

## Model Setup

| Primitive | Value | Role |
|---|---:|---|
| $\beta$ | 0.95 | Discount factor |
| $\rho$ | 0.90 | Persistence of log dividends |
| $\sigma$ | 0.10 | Innovation standard deviation in log dividends |
| $\gamma$ | 2.0 | Baseline CRRA risk aversion |
| Coarse grid | 120 log-endowment nodes | Tutorial solution |
| Quadrature | 21 Gauss-Hermite nodes | Conditional expectation |
| Benchmark | 900 grid nodes, 45 quadrature nodes | Fine-grid comparison |
| Stopping rule | $\|f_{n+1}-f_n\|_\infty < 10^{-9}$ | Fixed-point tolerance |

## Solution Method

The computation iterates directly on the scaled price $f$. This is not a household-choice Bellman equation: there is no policy function because market clearing sets $c=y$. The fixed point is still a contraction with modulus $\beta$, and the interpolation step only approximates the conditional expectation over next-period dividends.

```text
Inputs: beta, rho, sigma, gamma, log-endowment grid X, quadrature nodes eps_j, weights w_j
Initialize f_0(x_i) = 0 on X
For n = 0, 1, 2, ...:
    For each current state x_i:
        For each quadrature shock eps_j:
            x_ij' = rho x_i + eps_j
            y_ij' = exp(x_ij')
            interpolate f_n(x_ij') from the grid X
        Set f_{n+1}(x_i) = beta sum_j w_j [f_n(x_ij') + (y_ij')^(1-gamma)]
    Stop when max_i |f_{n+1}(x_i)-f_n(x_i)| is below tolerance
Output: p(exp(x_i)) = f(x_i) exp(gamma x_i)
```

The baseline solution converged in **405 iterations** with final sup-norm error **9.76e-10**. A finer grid and more quadrature nodes provide a numerical benchmark for the plotted state range.

## Results

The equilibrium price is increasing over the central dividend states. The dashed line is a fine-grid quadrature benchmark; the lower panel shows that the coarse tutorial solution stays within 0.011% of that benchmark on this range. The benchmark is numerical rather than analytic, but it is a useful check that the visible curvature is economic rather than a coarse-grid artifact.

<img src="figures/asset-price-function.png" alt="Lucas tree price function compared with a fine-grid benchmark" width="80%">

Along a simulated path, prices move with dividends because persistent high dividends raise expected future payoffs. The price series is more forward looking than the dividend itself: it capitalizes not only today's payment but also the continuation value implied by the Markov process.

<img src="figures/simulation-paths.png" alt="Simulated dividend and price path normalized to the initial period" width="80%">

Risk aversion changes the slope of the price-dividend ratio, not just its level. With log utility ($\gamma=1$), $p(y)/y=\beta/(1-\beta)$ is constant. When $\gamma>1$, high current dividends predict mean reversion toward lower future consumption, so future payoffs receive larger marginal-utility weights and the ratio rises with the current state.

<img src="figures/comparative-statics-gamma.png" alt="Price-dividend ratios under alternative CRRA risk aversion values" width="80%">

The selected states make the log-utility benchmark visible: the $\gamma=1$ price-dividend ratio is 19 at every $y$. Departures from log utility tilt this ratio across the state space because the stochastic discount factor interacts with mean reversion in dividends.

**Selected Price-Dividend Ratios**

|     y |   p(y), gamma=2 |   p/y, gamma=0.5 |   p/y, gamma=1 |   p/y, gamma=2 |   p/y, gamma=5 |
|------:|----------------:|-----------------:|---------------:|---------------:|---------------:|
| 0.504 |           6.221 |           24.302 |             19 |         12.334 |          5.208 |
| 0.624 |           8.815 |           22.534 |             19 |         14.137 |          8.09  |
| 0.786 |          12.949 |           20.772 |             19 |         16.477 |         14.075 |
| 0.99  |          19.105 |           19.167 |             19 |         19.29  |         26.323 |
| 1.248 |          28.308 |           17.706 |             19 |         22.679 |         52.398 |
| 1.573 |          42.114 |           16.373 |             19 |         26.771 |        109.637 |
| 1.983 |          62.894 |           15.157 |             19 |         31.724 |        238.292 |

## Takeaway

The Lucas tree turns a market-clearing allocation into a state-price problem. Once $c_t=y_t$, the asset price is the expected discounted payoff using marginal utility as the deflator. Solving for $f(y)=u'(y)p(y)$ makes the Euler equation linear and contractive. The price-dividend ratio is a valuation object: it records how persistence, discounting, and risk aversion transform a dividend process into an equilibrium claim price.

## References

- Lucas, R. (1978). "Asset Prices in an Exchange Economy." *Econometrica*, 46(6), 1429-1445.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 13.
- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press.
