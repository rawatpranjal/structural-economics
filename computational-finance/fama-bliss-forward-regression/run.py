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


MATURITY_MAP = {"2 Yr": 2.0, "3 Yr": 3.0, "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "30 Yr": 30.0}
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
    """Estimate one regression per maturity."""
    rows = []
    for maturity, group in regdata.groupby("Maturity", sort=False):
        x = group["Forward minus 1Y"].to_numpy()
        y = group["Future yield change"].to_numpy()
        alpha, beta, r2 = ols(y, x)
        rows.append(
            {
                "Maturity": maturity,
                "Intercept (bp)": f"{10000.0 * alpha:.2f}",
                "Slope": f"{beta:.2f}",
                "R-squared": f"{r2:.3f}",
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

    print("Fama-Bliss-style forward regression")
    print(coefs.to_string(index=False))

    report = ModelReport(
        "Fama-Bliss-Style Forward Regressions",
        "A small term-structure predictability exercise using a static Treasury CMT snapshot.",
    )

    report.add_overview(
        "The expectations hypothesis links long rates and forward rates to expected future "
        "short rates, with risk premia determining how far the link is from a one-for-one "
        "prediction. Fama and Bliss use this idea to ask whether long-maturity forward rates "
        "contain information about future interest rates and bond returns.\n\n"
        "The data here are a static 1990 Treasury CMT snapshot, not the CRSP zero-coupon "
        "bond panel needed for a full Fama-Bliss replication. The exercise approximates "
        "forward rates from observed par-yield maturities and asks whether the "
        "forward-minus-short spread predicts future yield changes over a short horizon."
    )

    report.add_equations(
        rf"""
Let $y_t^1$ be the one-year yield and $y_t^n$ be an $n$-year yield. Using
continuously compounded rates, approximate the forward rate from year 1 to year
$n$ as

$$
f_t^{{1,n}} = \frac{{n y_t^n - y_t^1}}{{n-1}}.
$$

The predictive regression is

$$
y_{{t+h}}^n - y_t^n = \alpha_n + \beta_n (f_t^{{1,n}} - y_t^1) + \epsilon_{{t+h}}^n,
$$

with $h = {HORIZON_DAYS}$ trading days in this static dataset.
"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| Data | Static 1990 Treasury CMT snapshot |\n"
        f"| Short rate | 1-year CMT rate |\n"
        f"| Long maturities | {', '.join(MATURITY_MAP.keys())} |\n"
        f"| Forecast horizon | {HORIZON_DAYS} trading days |\n"
        f"| Data limitation | CMT snapshot, not full Fama-Bliss replication |"
    )

    report.add_solution_method(
        "Percentage yields are converted to decimal rates, forward rates are approximated "
        "from one-year and longer-maturity yields, and separate OLS regressions are estimated "
        "by maturity. Because the data are par-yield CMT rates and cover only one year, the "
        "results should be read as mechanics and diagnostics rather than a published-style "
        "bond-risk-premium estimate."
    )

    fig1, ax1 = plt.subplots(figsize=(7.2, 5.2))
    x_bp = 10000.0 * ten_year["Forward minus 1Y"].to_numpy()
    y_bp = 10000.0 * ten_year["Future yield change"].to_numpy()
    ax1.scatter(x_bp, y_bp, alpha=0.75, s=28)
    x_line = np.linspace(x_bp.min(), x_bp.max(), 100)
    y_line = 10000.0 * alpha_10 + beta_10 * x_line
    ax1.plot(x_line, y_line, color="black", linewidth=2.0)
    ax1.axhline(0.0, color="gray", linewidth=1.0)
    ax1.set_xlabel("10Y forward-minus-1Y spread (bp)")
    ax1.set_ylabel(f"Next {HORIZON_DAYS}-day 10Y yield change (bp)")
    ax1.set_title("Forward Spread and Future Yield Changes")
    report.add_figure(
        "figures/forward-regression-10y.png",
        "Ten-year forward-spread predictability regression",
        fig1,
        description=(
            f"For the ten-year maturity, the fitted slope is {beta_10:.2f} and the R-squared "
            f"is {r2_10:.3f}. The fitted relationship illustrates the regression mechanics; "
            "the short snapshot does not establish a stable term-structure premium."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(7.2, 5.0))
    fitted = alpha_10 + beta_10 * ten_year["Forward minus 1Y"].to_numpy()
    ax2.plot(ten_year["Date"], 10000.0 * ten_year["Future yield change"], label="Realized")
    ax2.plot(ten_year["Date"], 10000.0 * fitted, label="Fitted", linewidth=2.0)
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Basis points")
    ax2.set_title("Realized and Fitted Yield Changes")
    ax2.legend()
    report.add_figure(
        "figures/fitted-vs-realized.png",
        "Realized versus fitted ten-year yield changes",
        fig2,
        description=(
            "A predictive regression can fit broad movement without becoming a trading rule. "
            "Overlapping horizons, short samples, and measurement choices matter."
        ),
    )

    report.add_table(
        "tables/forward-regression-coefficients.csv",
        "Forward-regression coefficients by maturity",
        coefs,
    )

    report.add_takeaway(
        "Forward rates are not just curve decoration: they can be used as predictors in "
        "term-structure regressions. But the interpretation is delicate. With this static "
        "snapshot, the result is a compact predictability exercise with clear data limits, "
        "not a full Fama-Bliss or Cochrane-Piazzesi replication."
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
