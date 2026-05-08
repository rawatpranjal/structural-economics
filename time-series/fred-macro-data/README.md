# Business-Cycle Moments from a FRED-Style Macro Panel

> HP filtering and moment measurement for a quarterly macro panel.

## Overview

Suppose you want to compare an RBC or New Keynesian model with business-cycle facts. The model reports cyclical movements in output, inflation, unemployment, and interest rates. The raw macro data arrive as quarterly series with their own units, trends, and sampling noise. Before the model can be judged, the researcher has to turn those series into moments defined on the same object.

This tutorial builds that object in a controlled setting. It simulates a small FRED-style panel, so the example stays reproducible without an API key or a changing data release. The calibration gives the panel familiar co-movement: output and unemployment move in opposite directions, inflation falls when slack rises, and the policy rate moves with inflation.

The computation applies the HP filter to each series, forms deviations from smooth trends, and summarizes the cyclical panel with volatility, GDP comovement, persistence, and an Okun slope. A long simulation from the same data-generating process gives a benchmark for reading the finite 50-year sample.

## Equations

Let

$$
y_t = (g_t,\pi_t,u_t,i_t)'
$$

collect GDP growth, CPI inflation, unemployment, and the federal funds rate,
all measured in percentage points. The synthetic data are generated from a
stationary vector process

$$
s_t = \rho \odot s_{t-1} + \sqrt{1-\rho^2}\odot \varepsilon_t,
\qquad
\varepsilon_t \sim N(0,C),
\qquad
y_t=\mu+\sigma\odot s_t.
$$

Here $\odot$ is element-by-element multiplication. The correlation matrix $C$
sets the contemporaneous macro relationships built into the example, while
$\rho_j$ controls how slowly each series adjusts after an innovation.

For each observed series $y_{j,t}$, the HP filter chooses a trend $\tau_{j,t}$
by solving

$$
\min_{\tau_j}
\sum_{t=1}^{T} (y_{j,t}-\tau_{j,t})^2 + \lambda\sum_{t=2}^{T-1}
[(\tau_{j,t+1}-\tau_{j,t})-(\tau_{j,t}-\tau_{j,t-1})]^2.
$$

The cycle is $c_{j,t}=y_{j,t}-\tau_{j,t}$. The reported moments are

$$
\sigma_j=\operatorname{sd}(c_{j,t}),\qquad
r_{j,g}=\operatorname{corr}(c_{j,t},c_{g,t}),\qquad
a_j=\operatorname{corr}(c_{j,t},c_{j,t-1}).
$$

The Okun diagnostic is the finite-sample regression

$$
c_{u,t}=\alpha_O+\beta_O c_{g,t}+e_t,
$$

where $c_{g,t}$ is the GDP-growth cycle and $c_{u,t}$ is the unemployment
cycle.

## Model Setup

**Quarterly sample**

| Object | Value | Role |
|---|---:|---|
| $T$ | 200 | Main sample, 50 years of quarters |
| $T_B$ | 5000 | Long simulation used only as a DGP benchmark |
| $\lambda$ | 1600 | HP smoothing parameter for quarterly data |

**Series-level primitives**

| Series | Mean | Std. dev. | Persistence | Economic role |
|---|---:|---:|---:|---|
| GDP growth | 2.5 | 3.0 | 0.30 | Output-growth cycle |
| CPI inflation | 2.0 | 1.5 | 0.70 | Price-pressure cycle |
| Unemployment | 5.5 | 1.5 | 0.85 | Labor-market slack |
| Fed funds | 4.0 | 3.0 | 0.80 | Short-rate policy indicator |

**Innovation correlation matrix $C$**

| | GDP | CPI | Unemployment | Fed funds |
|---|---:|---:|---:|---:|
| GDP | 1.00 | 0.20 | -0.60 | 0.30 |
| CPI | 0.20 | 1.00 | -0.30 | 0.50 |
| Unemployment | -0.60 | -0.30 | 1.00 | -0.20 |
| Fed funds | 0.30 | 0.50 | -0.20 | 1.00 |

## Solution Method

The computation turns a macro panel into the moment vector a modeler might match or inspect. The HP filter solves one sparse linear system for each series. It chooses a trend that stays close to the data while penalizing changes in trend growth. The remaining deviations define the cycles, and the moment table uses ordinary sample statistics on those cycles.

The long simulation has a narrower role. It runs the same data-generating process for many quarters, then applies the same HP filter and moment calculation. That comparison separates sampling variation in a 50-year panel from the co-movement imposed by the DGP.

```text
Algorithm: HP-filtered business-cycle moments
Inputs: quarterly panel y_t, HP parameter lambda, benchmark horizon T_B
Outputs: cycles c_t, moment table M, Okun slope beta_O

1. Simulate the four-variable macro vector y_t from the calibrated DGP.
2. For each series j:
      solve (I + lambda K'K) tau_j = y_j
      set c_j = y_j - tau_j
3. Compute volatility, relative volatility, GDP correlation, and lag-1
   autocorrelation from the cycles c_j.
4. Regress the unemployment cycle on the GDP-growth cycle to estimate beta_O.
5. Repeat steps 1-4 with T_B quarters and use those moments only as a
   long-sample benchmark for the finite 50-year run.
```

Interpretation starts when the cycles are defined. A positive GDP-growth cycle means output growth is above its smooth trend. A positive unemployment cycle means labor-market slack is above trend. The signs have to be read by series before the moments can be used as economic targets.

## Results

The raw panel shows the object a macroeconomist starts from: rates and growth rates in their observed units. A model calibration would rarely match these raw levels directly. GDP growth is noisy, while unemployment and the funds rate move more slowly, so the trend-cycle step affects each series in a different way.

<img src="figures/time-series.png" alt="Quarterly FRED-style macro series before detrending." width="80%">

After detrending, the comparison is in deviations from each series' own smooth path. This convention creates the object in the moment table. A different trend rule could change the size and persistence of measured fluctuations, which is why the data construction belongs in the economic discussion.

<img src="figures/hp-cycles.png" alt="HP-filtered cyclical components for the four macro series." width="80%">

The Okun scatter makes the economic content easiest to see. Output above trend is associated with unemployment below trend. The dashed line is a 5,000-quarter simulation from the same DGP, so it is a numerical benchmark for the finite sample rather than a claim about historical U.S. data. The 50-year sample correlation is -0.423; the long-sample benchmark is -0.450.

<img src="figures/okuns-law.png" alt="Okun relationship with finite-sample and long-sample regression lines." width="80%">

The correlation matrix gives a compact target for calibration or model checking. A structural RBC or New Keynesian model would need mechanisms that generate these signs. A reduced-form VAR would summarize the same object dynamically. Here the matrix checks whether the synthetic panel delivers the relationships built into the calibration.

<img src="figures/cross-correlation.png" alt="Cross-correlation matrix of HP-filtered cyclical components." width="80%">

The table separates the finite 50-year sample from the long-sample benchmark. The benchmark columns come from the same synthetic DGP, so their role is to show sampling variation and HP-filter effects, not to replace actual empirical validation.

**Business-cycle moments from HP-filtered quarterly cycles**

| Variable      |   Volatility (%) |   Rel. volatility |   Corr. with GDP |   Long-sample corr. |   Autocorr. |   Long-sample autocorr. |
|:--------------|-----------------:|------------------:|-----------------:|--------------------:|------------:|------------------------:|
| GDP growth    |            2.896 |             1     |            1     |               1     |       0.282 |                   0.223 |
| CPI inflation |            1.159 |             0.4   |            0.174 |               0.171 |       0.483 |                   0.529 |
| Unemployment  |            0.975 |             0.337 |           -0.423 |              -0.45  |       0.649 |                   0.638 |
| Fed funds     |            2.007 |             0.693 |            0.187 |               0.222 |       0.599 |                   0.622 |

## Takeaway

Business-cycle moments are constructed before they are matched. In this run, the HP-filtered panel recovers the intended signs: GDP growth and unemployment move against each other, with an Okun slope of -0.142 in the 50-year sample, and unemployment is the most persistent cycle. The long-sample benchmark shows the finite-sample point: the measured moments are close to the DGP's implications, but sampling and filtering keep them from matching exactly. Read these moments that way before using them as targets for a DSGE, RBC, or reduced-form forecasting exercise.

## References

- Federal Reserve Bank of St. Louis. FRED, Federal Reserve Economic Data.
- Hodrick, R. and Prescott, E. (1997). "Postwar U.S. Business Cycles: An Empirical Investigation." *Journal of Money, Credit and Banking*, 29(1), 1-16.
- Stock, J. and Watson, M. (1999). "Business Cycle Fluctuations in U.S. Macroeconomic Time Series." *Handbook of Macroeconomics*, Vol. 1A, Ch. 1.
- Okun, A. (1962). "Potential GNP: Its Measurement and Significance." *Proceedings of the Business and Economic Statistics Section*, ASA.
- Phillips, A. W. (1958). "The Relation Between Unemployment and the Rate of Change of Money Wage Rates in the United Kingdom, 1861-1957." *Economica*, 25(100), 283-299.
