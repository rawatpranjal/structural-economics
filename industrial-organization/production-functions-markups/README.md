# Production Functions and Markup Measurement

> Proxy-control production estimation and markup recovery from variable input cost shares.

## Overview

Production-function estimation is central to IO because productivity is observed by firms when they choose inputs but not by the econometrician. If more productive firms choose more labor or materials, a naive regression of output on inputs confounds technology with input choice.

The tutorial simulates a Cobb-Douglas panel, compares OLS with a simple investment-proxy control regression, and then uses the De Loecker-Warzynski markup formula: markup equals an output elasticity divided by the revenue share of that variable input.

## Equations

Cobb-Douglas production:
$$y_{it} = \beta_l l_{it} + \beta_k k_{it} + \beta_m m_{it} + \omega_{it} + \epsilon_{it}$$

Investment responds monotonically to productivity:
$$i_{it} = h(k_{it}, \omega_{it})$$

Proxy-control estimators invert this policy to control for productivity.

Markup from a variable input:
$$\mu_{it} = \frac{\theta^m_{it}}{\alpha^m_{it}}$$

where $\theta^m$ is the output elasticity of materials and $\alpha^m$ is the materials expenditure share in revenue.

## Model Setup

| Object | Value |
|--------|-------|
| Firms | 320 |
| Years | 6 |
| Technology | Cobb-Douglas in labor, capital, and materials |
| Productivity | Persistent AR(1), observed by firms before input choice |
| Proxy variable | Investment, increasing in productivity conditional on capital |
| Markup formula | Materials output elasticity divided by materials revenue share |

## Solution Method

OLS regresses log output on log inputs directly. The proxy-control regression uses the simulated investment policy to construct a noisy productivity proxy, which absorbs much of the productivity term that drives simultaneity. The estimated materials elasticity then enters the markup calculation for every firm-year observation.

## Results

OLS loads part of unobserved productivity onto flexible inputs. The proxy-control regression moves the input elasticities closer to the data-generating values.

<img src="figures/production-estimates.png" alt="True and estimated output elasticities" width="80%">
*True and estimated output elasticities*

Markup estimates inherit any error in the production elasticity and any noise in the expenditure share. The distribution is still informative about dispersion.

<img src="figures/markup-distribution.png" alt="True and estimated markup distributions" width="80%">
*True and estimated markup distributions*

The same production data can be used to study heterogeneity: high-productivity firms have lower material shares and therefore higher measured markups in this design.

<img src="figures/productivity-markups.png" alt="Estimated markups rise with productivity in the simulated panel" width="80%">
*Estimated markups rise with productivity in the simulated panel*

**Production function estimates**

| Input     |   True elasticity |   OLS |   Proxy-control |
|:----------|------------------:|------:|----------------:|
| Labor     |              0.32 | 0.452 |           0.333 |
| Capital   |              0.24 | 0.492 |           0.245 |
| Materials |              0.44 | 0.771 |           0.46  |

**Markup moments by productivity quintile**

| productivity_bin   |   mean_productivity |   mean_markup |   median_markup |
|:-------------------|--------------------:|--------------:|----------------:|
| Q1                 |              -0.45  |         0.901 |           0.894 |
| Q2                 |              -0.162 |         1.048 |           1.044 |
| Q3                 |               0.015 |         1.244 |           1.227 |
| Q4                 |               0.195 |         1.43  |           1.407 |
| Q5                 |               0.469 |         1.728 |           1.693 |

## Takeaway

Production-function estimates are not just technology parameters. Once combined with expenditure shares, they become markup estimates. That makes simultaneity, proxy assumptions, and revenue-vs-quantity measurement central to market-power claims.

## Reproduce

```bash
python run.py
```

## References

- Olley, S., and Pakes, A. (1996). The Dynamics of Productivity in the Telecommunications Equipment Industry. *Econometrica*, 64(6), 1263-1297.
- Levinsohn, J., and Petrin, A. (2003). Estimating Production Functions Using Inputs to Control for Unobservables. *Review of Economic Studies*, 70(2), 317-341.
- De Loecker, J., and Warzynski, F. (2012). Markups and Firm-Level Export Status. *American Economic Review*, 102(6), 2437-2471.
- Lectures 10-12 Slides 2023: Production functions, proxy methods, and markups.
