# Lucas Tree Asset Prices and the Stochastic Discount Factor

> A representative-agent exchange economy where dividend risk is priced by marginal utility.

## Overview

A Lucas tree pays a stochastic dividend $y_t$ each period. A representative household owns the tree and consumes the dividend. Market clearing sets $c_t=y_t$, so there is no savings choice.

The equilibrium object is the price function $p(y)$. It makes the household willing to hold the tree after seeing today's dividend. The stochastic discount factor prices the next dividend and resale value.

The price satisfies an Euler equation with a conditional expectation. We solve a one-dimensional fixed point after scaling price by marginal utility. Gauss-Hermite quadrature evaluates the expectation at off-grid dividend states.

## Equations

**Endowment process.** Let $x_t=\log y_t$ follow

$$x_{t+1}=\rho x_t+\varepsilon_{t+1}, \qquad
\varepsilon_{t+1}\sim \mathcal{N}(0,\sigma^2),\qquad |\rho|<1.$$

The process is stationary with variance $\sigma^2/(1-\rho^2)$.
Persistence $\rho$ controls how fast dividends move back toward the mean.

**Preferences.** The representative household has CRRA utility

$$u(c)=\frac{c^{1-\gamma}}{1-\gamma}, \qquad u'(c)=c^{-\gamma},\qquad \gamma>0,$$

The log case is $u(c)=\log c$ as $\gamma\to 1$.

**Pricing equation.** Market clearing imposes $c_t=y_t$.
A claim pays $y_{t+1}$ plus resale value $p(y_{t+1})$.
Its price satisfies

$$p(y_t)=\mathbb{E}_t\left[M_{t+1}(y_{t+1}+p(y_{t+1}))\right],
\qquad M_{t+1}=\beta\left(\frac{y_{t+1}}{y_t}\right)^{-\gamma}.$$

Equivalently,

$$p(y_t)=\beta\,\mathbb{E}_t\left[
\frac{u'(y_{t+1})}{u'(y_t)}(p(y_{t+1})+y_{t+1})\right].$$

**Scaled price.** Define the marginal-utility-scaled price

$$f(y)\equiv u'(y)\,p(y).$$

Multiplying the Euler equation by $u'(y_t)$ gives

$$f(y)=\beta\,\mathbb{E}\left[f(y')+u'(y')\,y'\,\big|\,y\right].$$

This is a linear fixed point in $f$.
The price and price-dividend ratio recover from

$$p(y)=\frac{f(y)}{u'(y)},\qquad \frac{p(y)}{y}=\frac{f(y)}{y\,u'(y)}.$$

**Log-utility benchmark.** When $\gamma=1$, $u'(y)y=1$.
The recursion is $f=\beta(f+1)$ at every $y$.
It implies the constant ratio

$$\frac{p(y)}{y}=\frac{\beta}{1-\beta}.$$

The flat ratio gives a direct check on the numerical solution.

## Model Setup

| Primitive | Value | Role |
|---|---:|---|
| $\beta$ | 0.95 | Discount factor |
| $\rho$ | 0.90 | Persistence of log dividends |
| $\sigma$ | 0.10 | Innovation standard deviation in log dividends |
| Stationary $\mathrm{sd}(\log y)$ | 0.2294 | $\sigma/\sqrt{1-\rho^2}$ |
| $\gamma$ | 2.0 | Baseline CRRA risk aversion |
| Coarse grid | 120 log-endowment nodes on $[\pm 5\,\mathrm{sd}(\log y)]$ | Tutorial solution |
| Quadrature | 21 Gauss-Hermite nodes for $\varepsilon$ | Conditional expectation |
| Benchmark | 900 grid nodes, 45 quadrature nodes | Fine-grid check |
| Stopping rule | $\|f_{n+1}-f_n\|_\infty < 10^{-9}$ | Fixed-point tolerance |

## Solution Method

**Scaled-price iteration.** The iteration works with $f(y)=u'(y)p(y)$. This scaling removes current marginal utility from the denominator. The update maps a guessed scaled price into a new scaled price.

The update operator is

$$(Tf)(y)=\beta\,\mathbb{E}\\left[f(y')+u'(y')y'\,\big|\,y\right]$$

This operator is a $\beta$-contraction. The run stops when sup-norm changes fall below $10^{-9}$.

**Conditional expectation.** The state $x=\log y$ uses a uniform grid. At each grid point, the code forms quadrature nodes $x'=\rho x+\varepsilon_j$. It interpolates old $f$ at those nodes. It then averages continuation value plus $u'(y')y'$ with Gauss-Hermite weights.

```text
Algorithm  Lucas-tree fixed-point iteration on f = u'(y) p
Inputs   beta, rho, sigma, gamma; log-endowment grid X = {x_i};
           Gauss-Hermite nodes {eps_j}, weights {w_j};
           tolerance epsilon
Outputs  scaled price f(x_i), price p(y_i), price-dividend ratio p/y

Precompute   x'_{ij} <- rho * x_i + eps_j                  # next-state nodes
             y'_{ij} <- exp(x'_{ij})
             d_{ij}  <- (y'_{ij})^{1 - gamma}              # forcing term u'(y') y'
Initialise   f_0(x_i) <- 0
for n = 0, 1, 2, ...:
    for each x_i:
        f_hat_{ij}  <- interp(f_n, X, x'_{ij})              # off-grid continuation
        f_{n+1}(x_i) <- beta * sum_j w_j * (f_hat_{ij} + d_{ij})
    err <- max_i | f_{n+1}(x_i) - f_n(x_i) |
stop when err < epsilon
p(y_i)     <- f(x_i) * (y_i)^{gamma}
p(y_i)/y_i <- p(y_i) / y_i
```

A fine grid with 900 state nodes and 45 quadrature nodes checks interpolation and quadrature error. The baseline $\gamma=2.0$ solution converges in **405 iterations** to sup-norm residual **9.76e-10**. On the central $\pm 3\,\mathrm{sd}(\log y)$ region, the maximum relative error is **0.011%**.

## Results

The price rises with the dividend state. Persistence makes a high current dividend predict higher future dividends. Convexity reflects future cash flows and state-dependent discounting.

The lower panel compares the coarse grid with the fine-grid benchmark. The maximum central relative error is 0.011%, so visible curvature is not a grid artifact.

<img src="figures/asset-price-function.png" alt="Lucas tree price function compared with a fine-grid benchmark" width="80%">

In simulation, prices move closely with dividends. The price index is more volatile because it capitalizes the continuation stream.

The lower panel plots $p(y_t)/y_t$. Under $\gamma=2$, the ratio moves with the dividend state.

<img src="figures/simulation-paths.png" alt="Simulated dividend, tree price, and price-dividend ratio" width="80%">

Risk aversion changes the slope of the price-dividend ratio. Log utility gives the flat benchmark $\beta/(1-\beta)\approx 19.0$. The $\gamma=1$ curve overlaps the dashed line.

When $\gamma<1$, the ratio falls with current dividends. When $\gamma>1$, it rises. Dotted lines are fine-grid benchmarks.

<img src="figures/comparative-statics-gamma.png" alt="Price-dividend ratios under alternative CRRA risk aversion values" width="80%">

Rows compare dividend states. Near $y\approx 1$, all ratios are close to the log-utility benchmark $\beta/(1-\beta)=19.0$. Away from the mean, risk aversion changes how the SDF prices mean reversion.

**Price-dividend ratios at selected dividend states**

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

The Lucas tree has no household policy once market clearing sets $c=y$. The Euler equation is therefore a valuation equation for $p(y)$. Scaling by $u'(y)$ gives a linear fixed point. The price-dividend ratio shows how risk aversion prices dividend mean reversion.

## References

- Lucas, R. (1978). "Asset Prices in an Exchange Economy." *Econometrica*, 46(6), 1429-1445.
- Mehra, R. and Prescott, E. (1985). "The Equity Premium: A Puzzle." *Journal of Monetary Economics*, 15(2), 145-161.
- Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 13.
- Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press.
