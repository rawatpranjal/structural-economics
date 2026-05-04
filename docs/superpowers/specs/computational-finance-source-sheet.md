# Computational Finance Source Sheet

This sheet locks the interpretation for the computational-finance chapter before
implementation. The old notebooks from `rawatpranjal/computational-finance` are
used as topic provenance, not as authoritative empirical references. Two source
notebooks are placeholders, and the Fama-Bliss notebook depends on an unavailable
local CRSP file.

## External Topic Scan: Macro_Finance

Additional source scanned:
`https://github.com/thallesqliduares/Macro_Finance`. The repository describes
itself as notebook exercises drawn from Dixon et al. (2021), *Machine Learning
in Finance: From Theory to Practice*, and Tsay (2005), *Analysis of Financial
Time Series*. Treat it as a topic-discovery source, not as an authority: some
notebooks use runtime `yfinance`/Colab calls, some are formula-only, and the PDF
exercise files are image-like exports rather than easily auditable source text.

Integration decisions:

- **Asset return measurement and annualization** from
  `AFTS/Ch1/AFTS_Ch1_Ex3.ipynb`: integrate into the existing
  `efficient-market-tests` tutorial if that page is expanded. It is prerequisite
  notation for return predictability, not a standalone chapter topic.
- **MA(1), AR(1), and ARMA forecasting mechanics** from
  `AFTS Ch2/AFTS_Ch2_Ex1.ipynb`, `Fin_ML_Ch6_Ex1.ipynb`, and
  `Fin_ML_Ch6_Ex2.ipynb`: integrate only as a compact extension to
  `ar1-rate-forecasting`. Do not create separate MA or ARMA tutorials unless the
  chapter later becomes a financial-time-series chapter.
- **Bayesian rare-event classification in finance** from
  `Fin_ML_Ch2_Ex1.ipynb`: route to the existing `choice/bayesian-learning`
  material or a future general diagnostic tutorial. Do not add it to
  computational finance just because the example uses fraud screening.
- **Runtime equity-data downloads** from the return notebook: do not use in
  active tutorials. Use static or synthetic data.

Genuinely new standalone candidate:

- **Geometric Brownian motion and lognormal price distributions** from
  `AFTS Ch6/AFTS_Ch6_Ex6.ipynb`. This could become one short tutorial on
  log-price dynamics, arithmetic price moments, simulated paths, and confidence
  bands. It is the only Macro_Finance item that is distinct enough to add to the
  compact computational-finance chapter later.

## Bond Yield to Maturity

- Source notebook: `01_Yeild_to_Maturity_for_Bonds.ipynb`.
- Core equation: bond price equals the present value of promised coupon and face
  value cash flows; YTM is the discount rate that sets this present value equal
  to the observed price.
- Tutorial should teach: loans, discount bonds, perpetuities, fixed-payment
  loans, coupon bonds, price-yield monotonicity, and duration-like sensitivity.
- Do not overclaim: YTM is not a realized return unless the bond is held to
  maturity, payments are made as promised, and interim coupons are treated
  consistently.
- Planned outputs: price-yield curve, yield sensitivities, instrument summary.
- References used:
  - OpenStax, "10.2 Bond Valuation," https://openstax.org/books/principles-finance/pages/10-2-bond-valuation
  - CFA Institute, "Fixed-Income Bond Valuation: Prices and Yields," https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/fixed-income-bond-valuation-prices-and-yields

## Treasury Yield Curve

- Source notebook: `02_Yeild_on_Treasuries.ipynb` and bundled
  `data/daily-treasury-rates.csv`.
- Core object: daily Treasury constant-maturity rates across maturities.
- Tutorial should teach: curve levels, slope, maturity panels, and the difference
  between cross-sectional curve shape and time-series movement.
- Do not overclaim: the bundled CSV is a static 1990 teaching snapshot. Treasury
  CMT rates are interpolated par-yield-curve rates derived from market quotes,
  not raw transaction yields.
- Planned outputs: selected yield curves, time series by maturity, spread table.
- References used:
  - U.S. Treasury, "Treasury Yield Curve Methodology," https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology
  - U.S. Treasury, "Daily Treasury Rates," https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve

## Fama-Bliss Forward Regression

- Source notebook: `03_Fama_Bliss_Regression.ipynb`.
- Core equation: regress future yield changes or returns on initial forward-rate
  or yield-spread information.
- Tutorial should teach: expectations-hypothesis intuition, forward rates as
  predictors, and why predictability is an empirical question.
- Do not overclaim: without CRSP bond data or Fama-Bliss zero-coupon yields, this
  is a Fama-Bliss-style teaching exercise using the bundled Treasury curve.
- Planned outputs: fitted regression line, coefficient table, realized versus
  predicted changes.
- References used:
  - Fama and Bliss (1987), "The information in long-maturity forward rates,"
    American Economic Review 77(4), 680-692, https://www.econbiz.de/Record/the-information-in-long-maturity-forward-rates-fama-eugene/10015130928
  - Campbell and Shiller (1991), "Yield Spreads and Interest Rate Movements,"
    Review of Economic Studies 58(3), 495-514, https://doi.org/10.2307/2298008
  - Cochrane and Piazzesi (2005), "Bond Risk Premia," American Economic Review
    95(1), 138-160, https://www.aeaweb.org/articles?id=10.1257/0002828053828581

## AR(1) Rate Forecasting

- Source notebook: `04_Forecasting_AR(1)_Process.ipynb`, which is only a
  placeholder.
- Core equation: `x[t+1] = alpha + rho x[t] + eps[t+1]`.
- Tutorial should teach: persistence, stationarity, one-step forecasts, and a
  random-walk/no-change benchmark.
- Do not overclaim: AR(1) is a small forecasting benchmark, not a complete term
  structure model.
- Planned outputs: observed and fitted series, forecast-error comparison,
  residual diagnostics.
- References used:
  - QuantEcon, "AR(1) Processes," https://intro.quantecon.org/ar1_processes.html

## Efficient-Market Tests

- Source notebook: `05_Tests_of_Efficient_Market_Hypothesis.ipynb`, which is
  only a placeholder.
- Core tests: autocorrelation, variance-ratio diagnostics, and predictive
  regressions for changes or returns.
- Tutorial should teach: weak-form predictability tests and why rejection of a
  random walk is not automatically a money-making strategy.
- Do not overclaim: every EMH test is also a test of the expected-return model
  and measurement assumptions.
- Planned outputs: autocorrelation bars, variance-ratio table, forecast test.
- References used:
  - Fama (1970), "Efficient Capital Markets: A Review of Theory and Empirical
    Work," Journal of Finance 25(2), 383-417, https://doi.org/10.2307/2325486
  - Lo and MacKinlay (1988), "Stock Market Prices Do Not Follow Random Walks,"
    Review of Financial Studies 1, 41-66, https://web.mit.edu/~alo/www/Papers/lo-mackinlay-88.html

## Mean-Variance Frontier

- Source notebook: `06_Optimal_Portfolio_Mean_Variance_Frontier.ipynb`.
- Core equation: portfolio mean is `w' mu` and variance is `w' Sigma w`.
- Tutorial should teach: diversification, covariance, minimum variance,
  tangency portfolios, and the efficient frontier.
- Do not overclaim: the frontier is input-sensitive and should not be presented
  as investment advice.
- Planned outputs: simulated portfolios, analytic frontier, portfolio summary.
- References used:
  - Markowitz (1952), "Portfolio Selection," Journal of Finance 7(1), 77-91,
    https://doi.org/10.2307/2975974
