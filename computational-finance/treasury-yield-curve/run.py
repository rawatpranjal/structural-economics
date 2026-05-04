#!/usr/bin/env python3
"""Treasury yield-curve snapshots from bundled static data."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


MATURITY_COLUMNS = ["3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
MATURITY_YEARS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])


def load_treasury_data() -> pd.DataFrame:
    """Load the static Treasury-rate snapshot bundled with this chapter."""
    path = Path(__file__).resolve().parents[1] / "_data" / "daily-treasury-rates.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df.sort_values("Date").reset_index(drop=True)


def selected_curve_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return beginning, middle, and end yield curves."""
    target_dates = [df["Date"].min(), pd.Timestamp("1990-06-29"), df["Date"].max()]
    rows = []
    for target in target_dates:
        idx = int((df["Date"] - target).abs().idxmin())
        row = df.loc[idx, ["Date", *MATURITY_COLUMNS]].copy()
        row["Date"] = row["Date"].strftime("%Y-%m-%d")
        rows.append(row)
    return pd.DataFrame(rows)


def spread_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple slope and curvature summaries."""
    work = df.copy()
    work["10Y-3M"] = work["10 Yr"] - work["3 Mo"]
    work["30Y-10Y"] = work["30 Yr"] - work["10 Yr"]
    work["5Y-2Y"] = work["5 Yr"] - work["2 Yr"]
    stats = work[["10Y-3M", "30Y-10Y", "5Y-2Y"]].agg(["mean", "min", "max", "std"]).T
    stats = stats.reset_index().rename(columns={"index": "Spread"})
    for col in ["mean", "min", "max", "std"]:
        stats[col] = stats[col].map(lambda x: f"{x:.2f}")
    return stats.rename(columns={"mean": "Mean", "min": "Min", "max": "Max", "std": "Std. dev."})


def main() -> None:
    setup_style()
    df = load_treasury_data()
    curves = selected_curve_rows(df)
    spreads = spread_summary(df)

    print("Treasury yield curve snapshot")
    print(f"  observations: {len(df)}")
    print(f"  date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    report = ModelReport(
        "Treasury Yield Curve Snapshots",
        "Static Treasury-rate data used to read level, slope, and curve shape.",
    )

    report.add_overview(
        "The source notebook plotted Treasury yields from online tickers and from a local CSV. "
        "This chapter uses only the bundled CSV so the tutorial is reproducible without a "
        "network call.\n\n"
        "The data are a 1990 teaching snapshot of Treasury constant-maturity rates. Treasury "
        "constant-maturity rates should be interpreted as par-yield-curve rates derived from "
        "market quotes and interpolation, not as raw transaction yields for one traded bond."
    )

    report.add_equations(
        r"""
A simple slope measure compares a long maturity with a short maturity:

$$
\text{Slope}_t = y_{10,t} - y_{3m,t}.
$$

A simple long-end spread compares two longer maturities:

$$
\text{LongSpread}_t = y_{30,t} - y_{10,t}.
$$

These are descriptive statistics. They summarize curve shape but do not by
themselves identify risk premia or expected future short rates.
"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| Data source | Bundled source-repo Treasury CSV |\n"
        f"| Date range | {df['Date'].min().date()} to {df['Date'].max().date()} |\n"
        f"| Observations | {len(df)} daily rows |\n"
        f"| Maturities | {', '.join(MATURITY_COLUMNS)} |\n"
        f"| Runtime data calls | None |"
    )

    report.add_solution_method(
        "The script parses the CSV dates, orders the series from January to December 1990, "
        "then treats each row as a cross-sectional curve. It reports selected curve snapshots "
        "and computes simple term spreads over time."
    )

    fig1, ax1 = plt.subplots(figsize=(7.4, 5.2))
    for _, row in curves.iterrows():
        rates = row[MATURITY_COLUMNS].astype(float).to_numpy()
        ax1.plot(MATURITY_YEARS, rates, marker="o", label=row["Date"])
    ax1.set_xscale("log")
    ax1.set_xticks(MATURITY_YEARS)
    ax1.set_xticklabels(MATURITY_COLUMNS, rotation=35)
    ax1.set_xlabel("Maturity")
    ax1.set_ylabel("Yield (%)")
    ax1.set_title("Treasury Curve Snapshots")
    ax1.legend()
    report.add_figure(
        "figures/yield-curve-snapshots.png",
        "Selected Treasury yield curves",
        fig1,
        description=(
            "A yield curve is a cross-section: it compares yields across maturities on the "
            "same date. Here the curve generally slopes upward during 1990, but the level "
            "and slope move over the year."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(7.4, 5.2))
    for col in ["3 Mo", "2 Yr", "10 Yr", "30 Yr"]:
        ax2.plot(df["Date"], df[col], label=col)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Yield (%)")
    ax2.set_title("Treasury Rates Over Time")
    ax2.legend(ncol=2)
    report.add_figure(
        "figures/yields-over-time.png",
        "Selected maturity yields over time",
        fig2,
        description=(
            "Time-series movement is different from curve shape. A maturity can rise or fall "
            "over time even when the cross-sectional curve remains upward sloping."
        ),
    )

    work = df.copy()
    work["10Y-3M"] = work["10 Yr"] - work["3 Mo"]
    work["30Y-10Y"] = work["30 Yr"] - work["10 Yr"]
    fig3, ax3 = plt.subplots(figsize=(7.4, 4.8))
    ax3.plot(work["Date"], work["10Y-3M"], label="10Y minus 3M")
    ax3.plot(work["Date"], work["30Y-10Y"], label="30Y minus 10Y")
    ax3.axhline(0.0, color="black", linewidth=1.0)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Spread (percentage points)")
    ax3.set_title("Yield-Curve Slope Measures")
    ax3.legend()
    report.add_figure(
        "figures/term-spreads.png",
        "Treasury term spreads",
        fig3,
        description=(
            "Spreads summarize curve shape in one number. They are useful descriptors, but "
            "their interpretation depends on expectations, risk premia, and measurement."
        ),
    )

    table_curves = curves.copy()
    for col in MATURITY_COLUMNS:
        table_curves[col] = table_curves[col].map(lambda x: f"{float(x):.2f}")
    report.add_table(
        "tables/selected-curves.csv",
        "Selected curve snapshots",
        table_curves,
    )
    report.add_table(
        "tables/spread-summary.csv",
        "Spread summary statistics",
        spreads,
    )

    report.add_takeaway(
        "The Treasury curve is a compact picture of interest rates by maturity. The key "
        "interpretive discipline is to keep curve description separate from causal claims: "
        "level, slope, and curvature can be plotted directly, while expectations and risk "
        "premia require additional assumptions or regressions."
    )
    report.add_references(
        [
            "[U.S. Treasury. Treasury Yield Curve Methodology.](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology)",
            "[U.S. Treasury. Daily Treasury Rates.](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
