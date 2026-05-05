#!/usr/bin/env python3
"""Fama-Bliss-style forward-rate regressions on static Treasury data."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


MATURITY_MAP = {
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "30 Yr": 30.0,
}
HORIZON_DAYS = 20


def load_treasury_data() -> pd.DataFrame:
    """Load static Treasury curve data and convert yields to decimals."""
    path = Path(__file__).resolve().parents[1] / "_data" / "daily-treasury-rates.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["1 Yr", *MATURITY_MAP.keys()]:
        df[col] = df[col] / 100.0
    return df


def approximate_forward_rate(short_rate: pd.Series, long_rate: pd.Series, maturity: float) -> pd.Series:
    """Approximate the one-to-n-year forward rate from par-yield data."""
    short_cc = np.log1p(short_rate)
    long_cc = np.log1p(long_rate)
    forward_cc = (maturity * long_cc - short_cc) / (maturity - 1.0)
    return np.expm1(forward_cc)


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Estimate y = alpha + beta x and return alpha, beta, R-squared."""
    design = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.array([alpha, beta])
    ssr = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    return float(alpha), float(beta), float(r2)


def build_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """Construct Fama-Bliss-style predictors and future yield changes."""
    rows = []
    short = df["1 Yr"]
    for col, maturity in MATURITY_MAP.items():
        forward = approximate_forward_rate(short, df[col], maturity)
        predictor = forward - short
        future_change = df[col].shift(-HORIZON_DAYS) - df[col]
        temp = pd.DataFrame(
            {
                "Date": df["Date"],
                "Maturity": col,
                "Forward minus 1Y": predictor,
                "Future yield change": future_change,
            }
        ).dropna()
        rows.append(temp)
    return pd.concat(rows, ignore_index=True)


def coefficient_table(regdata: pd.DataFrame) -> pd.DataFrame:
    """Estimate one regression per maturity and compare with no-change forecasts."""
    rows = []
    for maturity, group in regdata.groupby("Maturity", sort=False):
        x = group["Forward minus 1Y"].to_numpy()
        y = group["Future yield change"].to_numpy()
        alpha, beta, r2 = ols(y, x)
        fitted = alpha + beta * x
        ols_rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
        zero_rmse = float(np.sqrt(np.mean(y ** 2)))
        rmse_gain = 100.0 * (1.0 - ols_rmse / zero_rmse)
        rows.append(
            {
                "Maturity": maturity,
                "Intercept (bp)": f"{10000.0 * alpha:.2f}",
                "Slope": f"{beta:.2f}",
                "R-squared": f"{r2:.3f}",
                "OLS RMSE (bp)": f"{10000.0 * ols_rmse:.2f}",
                "No-change RMSE (bp)": f"{10000.0 * zero_rmse:.2f}",
                "RMSE gain vs zero (%)": f"{rmse_gain:.1f}",
                "Obs.": len(group),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    df = load_treasury_data()
    regdata = build_regression_data(df)
    coefs = coefficient_table(regdata)
    ten_year = regdata[regdata["Maturity"] == "10 Yr"].copy()
    alpha_10, beta_10, r2_10 = ols(
        ten_year["Future yield change"].to_numpy(),
        ten_year["Forward minus 1Y"].to_numpy(),
    )
    fitted_10 = alpha_10 + beta_10 * ten_year["Forward minus 1Y"].to_numpy()
    y_10 = ten_year["Future yield change"].to_numpy()
    ols_rmse_10 = float(np.sqrt(np.mean((y_10 - fitted_10) ** 2)))
    zero_rmse_10 = float(np.sqrt(np.mean(y_10 ** 2)))
    rmse_gain_10 = 100.0 * (1.0 - ols_rmse_10 / zero_rmse_10)

    print("Fama-Bliss-style forward regression")
    print(coefs.to_string(index=False))

    report = ModelReport(
        "Fama-Bliss Forward-Rate Predictability",
        "Forward spreads, future yield changes, and the limits of a static Treasury CMT snapshot.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Forward rates are useful because they turn the yield curve into a forecast-like "
        "object. If long rates mainly average expected future short rates, a steep forward "
        "curve should say something about later rate movements. If time-varying risk premia "
        "are important, the same spread can instead predict bond returns or compensation "
        "for bearing duration risk. The Fama-Bliss regressions sit exactly on that boundary: "
        "they are simple predictive regressions with an economic interpretation that depends "
        "on the maintained term-structure model.\n\n"
        "This page is a teaching analogue, not a replication. The [Treasury yield-curve tutorial](../treasury-yield-curve/) "
        "works with the same offline 1990 Treasury CMT panel and explains the measurement "
        "object. Here we approximate forward-rate spreads from those par-yield nodes and ask "
        "whether they predict twenty-trading-day changes in longer yields. The short horizon "
        "and CMT measurement make the exercise diagnostic; they do not deliver the zero-coupon "
        "bond panel used by Fama and Bliss."
    )

    report.add_equations(
        rf"""
Let $Y_t^1$ be the annual effective one-year CMT yield and $Y_t^n$ be the
annual effective CMT yield at maturity $n$. The code first maps each yield into
a continuously compounded rate,

$$
q_t^m = \log(1+Y_t^m).
$$

For $n>1$, the forward-rate analogue from year 1 to year $n$ is

$$
f_t^{{1,n}} =
\exp\left(\frac{{n q_t^n - q_t^1}}{{n-1}}\right)-1.
$$

The predictor is the forward-minus-short spread

$$
x_t^n = f_t^{{1,n}} - Y_t^1,
$$

and the dependent variable is the future yield change

$$
\Delta_h Y_{{t+h}}^n = Y_{{t+h}}^n - Y_t^n.
$$

The maturity-by-maturity predictive regression is

$$
\Delta_h Y_{{t+h}}^n = \alpha_n + \beta_n x_t^n + \varepsilon_{{t+h}}^n,
$$

with $h={HORIZON_DAYS}$ trading days. Because CMT rates are par-yield curve
nodes, $f_t^{{1,n}}$ is an approximation to the forward-rate object, not an
arbitrage-free zero-coupon forward rate.
"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| Data | Static 1990 Treasury CMT panel |\n"
        f"| Date range | {df['Date'].min().date()} to {df['Date'].max().date()} |\n"
        f"| Usable observations | {len(df) - HORIZON_DAYS} per maturity after the lead |\n"
        f"| Short yield | 1-year CMT rate |\n"
        f"| Long maturities | {', '.join(MATURITY_MAP.keys())} |\n"
        f"| Forecast horizon | {HORIZON_DAYS} trading days |\n"
        f"| Benchmark forecast | No yield change, $\\Delta_h Y_{{t+h}}^n=0$ |\n"
        f"| Data limitation | CMT par yields, not a zero-coupon Fama-Bliss panel |"
    )

    report.add_solution_method(
        "The estimator is deliberately transparent: build the spread implied by the current "
        "curve, line it up with a future yield change, and run OLS separately by maturity. "
        "The important discipline is not the linear algebra. It is keeping the forecast "
        "horizon, maturity, and measurement object fixed when interpreting $\\beta_n$.\n\n"
        "```text\n"
        "Algorithm: Fama-Bliss-style forward-spread regression\n"
        "Input: daily yields Y_t^1 and Y_t^n, maturity set N, horizon h\n"
        "Output: maturity-specific alpha_n, beta_n, R^2, and forecast errors\n"
        "Sort observations by date\n"
        "For each maturity n in N:\n"
        "    convert yields to q_t^m = log(1 + Y_t^m)\n"
        "    compute f_t^{1,n} = exp((n q_t^n - q_t^1) / (n - 1)) - 1\n"
        "    form x_t^n = f_t^{1,n} - Y_t^1\n"
        "    form Delta_h Y_{t+h}^n = Y_{t+h}^n - Y_t^n\n"
        "    drop the last h dates, where the future yield is unavailable\n"
        "    estimate Delta_h Y_{t+h}^n = alpha_n + beta_n x_t^n + epsilon_{t+h}^n by OLS\n"
        "    compare fitted errors with the no-change forecast Delta_h Y_{t+h}^n = 0\n"
        "Return the coefficient table and the ten-year fitted path\n"
        "```\n\n"
        "There is no ground-truth term premium in this dataset. The no-change forecast is "
        "included only as a modest benchmark: it asks whether the forward spread improves on "
        "a random-walk-style prediction for this short sample."
    )

    fig1, ax1 = plt.subplots(figsize=(7.2, 5.2))
    x_bp = 10000.0 * ten_year["Forward minus 1Y"].to_numpy()
    y_bp = 10000.0 * ten_year["Future yield change"].to_numpy()
    ax1.scatter(x_bp, y_bp, alpha=0.75, s=28)
    x_line = np.linspace(x_bp.min(), x_bp.max(), 100)
    y_line = 10000.0 * alpha_10 + beta_10 * x_line
    ax1.plot(x_line, y_line, color="black", linewidth=2.0, label="OLS fit")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, label="No-change forecast")
    ax1.set_xlabel("10Y forward-minus-1Y spread (bp)")
    ax1.set_ylabel(f"Next {HORIZON_DAYS}-day 10Y yield change (bp)")
    ax1.set_title("Forward Spread and Future Yield Changes")
    ax1.legend()
    report.add_results(
        "For the ten-year maturity, wider forward-minus-one-year spreads in this 1990 "
        f"snapshot are associated with lower subsequent ten-year yields: the estimated slope "
        f"is **{beta_10:.2f}** with $R^2={r2_10:.3f}$. The sign should not be turned into a "
        "structural claim. It is a short-horizon relationship in one CMT panel, useful because "
        "it shows how the Fama-Bliss object is constructed and how sensitive interpretation is "
        "to the data object."
    )
    report.add_figure(
        "figures/forward-regression-10y.png",
        "Ten-year forward-spread predictability regression",
        fig1,
        description=(
            "The horizontal benchmark is a zero predicted yield change. The fitted line has "
            "visible slope, but the cloud also makes clear that this is a noisy predictive "
            "relationship rather than a pricing identity."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(7.2, 5.0))
    ax2.plot(ten_year["Date"], 10000.0 * ten_year["Future yield change"], label="Realized")
    ax2.plot(ten_year["Date"], 10000.0 * fitted_10, label="OLS fitted", linewidth=2.0)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="No-change forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Basis points")
    ax2.set_title("Realized and Fitted Yield Changes")
    ax2.legend()
    report.add_results(
        "The fitted ten-year series does better than a flat no-change forecast in this run, "
        f"with RMSE **{10000.0 * ols_rmse_10:.2f} bp** versus "
        f"**{10000.0 * zero_rmse_10:.2f} bp** for the benchmark, a "
        f"**{rmse_gain_10:.1f}%** reduction. That comparison is deliberately modest. It "
        "checks whether the spread has in-sample predictive content; it is not an out-of-sample "
        "trading rule."
    )
    report.add_figure(
        "figures/fitted-vs-realized.png",
        "Realized versus fitted ten-year yield changes",
        fig2,
        description=(
            "The fitted line moves slowly because it is driven by the current curve, while "
            "realized twenty-day yield changes contain high-frequency surprises that the spread "
            "cannot absorb."
        ),
    )

    report.add_table(
        "tables/forward-regression-coefficients.csv",
        "Forward-regression coefficients by maturity",
        coefs,
        description=(
            "Across maturities, the slopes are negative in this panel and the simple RMSE "
            "comparison favors OLS over the no-change benchmark. The pattern is suggestive, "
            "not definitive: all regressions use overlapping short-horizon changes from the "
            "same single-year CMT sample."
        ),
    )

    report.add_takeaway(
        "The forward spread is an economically meaningful summary of the term structure, not "
        "just a plotted difference between two rates. In this snapshot it predicts short-run "
        "yield changes better than a no-change benchmark, but the result inherits the limits "
        "of CMT par yields, overlapping horizons, and a single year of data. The reusable lesson "
        "is how to map a yield curve into a predictive regression while keeping the term-"
        "structure interpretation honest."
    )
    report.add_references(
        [
            "[Fama, E. F., and Bliss, R. R. (1987). The information in long-maturity forward rates. American Economic Review, 77(4), 680-692.](https://www.econbiz.de/Record/the-information-in-long-maturity-forward-rates-fama-eugene/10015130928)",
            "[Campbell, J. Y., and Shiller, R. J. (1991). Yield Spreads and Interest Rate Movements: A Bird's Eye View. Review of Economic Studies, 58(3), 495-514.](https://doi.org/10.2307/2298008)",
            "[Cochrane, J. H., and Piazzesi, M. (2005). Bond Risk Premia. American Economic Review, 95(1), 138-160.](https://www.aeaweb.org/articles?id=10.1257/0002828053828581)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
