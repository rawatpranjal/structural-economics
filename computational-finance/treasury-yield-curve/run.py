#!/usr/bin/env python3
"""Treasury yield-curve snapshots from static CMT data."""
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
    """Load the static Treasury-rate snapshot."""
    path = Path(__file__).resolve().parents[1] / "_data" / "daily-treasury-rates.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df.sort_values("Date").reset_index(drop=True)


def selected_curve_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return beginning, middle, and end yield curves."""
    targets = [
        ("Start of 1990", df["Date"].min()),
        ("Mid-1990", pd.Timestamp("1990-06-29")),
        ("End of 1990", df["Date"].max()),
    ]
    rows = []
    for label, target in targets:
        idx = int((df["Date"] - target).abs().idxmin())
        source = df.loc[idx, ["Date", *MATURITY_COLUMNS]]
        row = {"Snapshot": label, "Date": source["Date"].strftime("%Y-%m-%d")}
        for col in MATURITY_COLUMNS:
            row[col] = float(source[col])
        rows.append(row)
    return pd.DataFrame(rows)


def curve_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute level, slope, and curvature diagnostics for each date."""
    work = df[["Date", *MATURITY_COLUMNS]].copy()
    work["Level 10Y"] = work["10 Yr"]
    work["Slope 10Y-3M"] = work["10 Yr"] - work["3 Mo"]
    work["Curvature 2x5Y-2Y-10Y"] = 2.0 * work["5 Yr"] - work["2 Yr"] - work["10 Yr"]
    work["Long-end 30Y-10Y"] = work["30 Yr"] - work["10 Yr"]
    return work


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize curve-shape diagnostics over the sample."""
    metric_cols = [
        "Level 10Y",
        "Slope 10Y-3M",
        "Curvature 2x5Y-2Y-10Y",
        "Long-end 30Y-10Y",
    ]
    stats = metrics[metric_cols].agg(["mean", "min", "max", "std"]).T
    stats = stats.reset_index().rename(columns={"index": "Metric"})
    for col in ["mean", "min", "max", "std"]:
        stats[col] = stats[col].map(lambda x: f"{x:.2f}")
    return stats.rename(
        columns={
            "mean": "Mean",
            "min": "Min",
            "max": "Max",
            "std": "Std. dev.",
        }
    )


def main() -> None:
    setup_style()
    df = load_treasury_data()
    curves = selected_curve_rows(df)
    metrics = curve_metrics(df)
    summary = metric_summary(metrics)
    curve_lookup = curves.set_index("Snapshot")

    def snapshot_slope(label: str) -> float:
        row = curve_lookup.loc[label]
        return float(row["10 Yr"] - row["3 Mo"])

    start_slope = snapshot_slope("Start of 1990")
    mid_slope = snapshot_slope("Mid-1990")
    end_slope = snapshot_slope("End of 1990")
    front_end_drop = float(
        curve_lookup.loc["Start of 1990", "3 Mo"] - curve_lookup.loc["End of 1990", "3 Mo"]
    )
    ten_year_change = float(
        curve_lookup.loc["End of 1990", "10 Yr"] - curve_lookup.loc["Start of 1990", "10 Yr"]
    )

    print("Treasury CMT yield curve snapshots")
    print(f"  observations: {len(df)}")
    print(f"  date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    report = ModelReport(
        "Treasury Yield Curves and Term-Structure Shape",
        "CMT curve snapshots, level-slope-curvature summaries, and static interpretation.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The term structure is a price system viewed across maturities. On one date, "
        "short, intermediate, and long Treasury rates summarize how markets price dollars "
        "delivered at different horizons. The first economic question is therefore not "
        "which plot to draw, but what part of the curve is moving: the common level, the "
        "long-minus-short slope, or the middle-maturity belly.\n\n"
        "This tutorial uses an offline 1990 panel of Treasury constant-maturity rates. "
        "The observations are CMT par-yield-curve rates constructed from market quotes and "
        "Treasury interpolation. They are not zero-coupon spot rates, and they are not raw "
        "transaction yields on one traded bond. The previous [bond-pricing tutorial](../bond-yield-to-maturity/) "
        "works with a single promised cash-flow claim; the later [Fama-Bliss-style regression](../fama-bliss-forward-regression/) "
        "uses term-structure spreads for predictability."
    )

    report.add_equations(
        r"""
Let $t=1,\ldots,T$ index dates and let $\tau_j$ denote a maturity in years, with

$$
\tau_j \in \{0.25,0.5,1,2,3,5,7,10,30\}.
$$

The daily CMT curve is the vector

$$
\boldsymbol{y}_t =
(y_t(\tau_1),\ldots,y_t(\tau_J)),
$$

where $y_t(\tau_j)$ is the annualized par yield in percent at maturity $\tau_j$.
The tutorial reduces this cross-section to a few transparent diagnostics:

$$
L_t = y_t(10),
$$

$$
S_t = y_t(10)-y_t(0.25),
$$

and

$$
B_t = 2y_t(5)-y_t(2)-y_t(10).
$$

Here $L_t$ is a level proxy, $S_t$ is the ten-year minus three-month slope, and
$B_t$ is a simple belly measure: positive values put the five-year rate above
the average of the two-year and ten-year rates. These are descriptive
statistics in percentage points. They do not identify expected future short
rates, term premia, or arbitrage-free discount factors without additional
structure.
"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| Data | Static 1990 Treasury CMT panel |\n"
        f"| Date range | {df['Date'].min().date()} to {df['Date'].max().date()} |\n"
        f"| Observations | {len(df)} daily rows |\n"
        f"| Maturities | {', '.join(MATURITY_COLUMNS)} |\n"
        f"| Unit | Annual percentage yield |\n"
        f"| Measurement | Constant-maturity par-yield rates, not zero-coupon spot rates |"
    )

    report.add_solution_method(
        "The computation is deliberately descriptive. Each row is a cross-section of "
        "maturity-specific rates, and the code asks how much of the movement is level, "
        "slope, or curvature. A denser maturity grid would mostly interpolate the "
        "published CMT nodes; it would not create a ground-truth zero-coupon curve unless "
        "we added a separate term-structure model.\n\n"
        "```text\n"
        "Algorithm: CMT curve-shape summaries\n"
        "Input: daily table of dates t and CMT yields y_t(tau_j)\n"
        "Output: selected yield curves, time-series plots, and summary moments\n"
        "Sort the table by date\n"
        "For each date t:\n"
        "    form the maturity vector y_t = (y_t(tau_1), ..., y_t(tau_J))\n"
        "    compute L_t = y_t(10)\n"
        "    compute S_t = y_t(10) - y_t(0.25)\n"
        "    compute B_t = 2 y_t(5) - y_t(2) - y_t(10)\n"
        "Select start, middle, and end-of-sample rows for cross-sectional plots\n"
        "Plot selected maturity rates over calendar time\n"
        "Report means, minima, maxima, and standard deviations for L_t, S_t, and B_t\n"
        "```\n\n"
        "This is the right level of machinery for a snapshot tutorial. Forecasting, "
        "expectations-hypothesis tests, and term-premium decompositions require stronger "
        "objects than the static CMT panel used here."
    )

    fig1, ax1 = plt.subplots(figsize=(7.4, 5.2))
    for _, row in curves.iterrows():
        rates = row[MATURITY_COLUMNS].astype(float).to_numpy()
        ax1.plot(MATURITY_YEARS, rates, marker="o", label=f"{row['Snapshot']} ({row['Date']})")
    ax1.set_xscale("log")
    ax1.set_xticks(MATURITY_YEARS)
    ax1.set_xticklabels(MATURITY_COLUMNS, rotation=35)
    ax1.set_xlabel("Maturity (log scale)")
    ax1.set_ylabel("Yield (%)")
    ax1.set_title("Treasury CMT Curves in 1990")
    ax1.legend()
    report.add_results(
        "The three selected dates make the cross-sectional object concrete. The curve was "
        f"nearly flat at the start of the sample, with a ten-year minus three-month slope "
        f"of {start_slope:.2f} percentage points. It was still only {mid_slope:.2f} "
        "points in late June. By year end the slope had widened to "
        f"{end_slope:.2f} points. That steepening came mainly from the front end: the "
        f"three-month rate fell by {front_end_drop:.2f} points over the year, while the "
        f"ten-year rate changed by {ten_year_change:.2f} points."
    )
    report.add_figure(
        "figures/yield-curve-snapshots.png",
        "Selected Treasury yield curves",
        fig1,
    )

    fig2, ax2 = plt.subplots(figsize=(7.4, 5.2))
    for col in ["3 Mo", "2 Yr", "10 Yr", "30 Yr"]:
        ax2.plot(df["Date"], df[col], label=col)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Yield (%)")
    ax2.set_title("Selected CMT Rates Over Time")
    ax2.legend(ncol=2)
    report.add_results(
        "The time-series view separates calendar movement from curve shape. A maturity "
        "series can fall over the year even when the cross-section remains upward sloping "
        "on most dates. This distinction matters because statements about a yield curve "
        "are always indexed by both date and maturity."
    )
    report.add_figure(
        "figures/yields-over-time.png",
        "Selected maturity yields over time",
        fig2,
    )

    fig3, ax3 = plt.subplots(figsize=(7.4, 4.8))
    ax3.plot(metrics["Date"], metrics["Slope 10Y-3M"], label="10Y minus 3M")
    ax3.plot(metrics["Date"], metrics["Curvature 2x5Y-2Y-10Y"], label="2 x 5Y - 2Y - 10Y")
    ax3.axhline(0.0, color="black", linewidth=1.0)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Percentage points")
    ax3.set_title("Slope and Belly Diagnostics")
    ax3.legend()
    report.add_results(
        "The diagnostic series compress each daily curve into economically interpretable "
        "numbers. The slope carries most of the visible late-year movement. The belly "
        "measure is smaller, so in this sample the main fact is not a dramatic hump in "
        "middle maturities; it is the widening gap between short and long yields."
    )
    report.add_figure(
        "figures/term-spreads.png",
        "Yield-curve slope and belly diagnostics",
        fig3,
    )

    table_curves = curves.copy()
    for col in MATURITY_COLUMNS:
        table_curves[col] = table_curves[col].map(lambda x: f"{float(x):.2f}")
    report.add_table(
        "tables/selected-curves.csv",
        "Selected curve snapshots",
        table_curves,
        description=(
            "The selected rows keep the raw CMT rates visible. They are useful as an audit "
            "trail for the plotted curves and for the slope numbers quoted above."
        ),
    )
    report.add_table(
        "tables/spread-summary.csv",
        "Curve-shape summary statistics",
        summary,
        description=(
            "The summary statistics are descriptive moments of the 1990 panel. They should "
            "be read as facts about this static sample, not as estimates of a structural "
            "term-premium model."
        ),
    )

    report.add_takeaway(
        "A Treasury CMT curve is a maturity cross-section, not a single interest rate. "
        "Level, slope, and curvature are useful because they discipline what we mean by "
        "curve movements before imposing a richer model. The limitation is equally "
        "important: static CMT summaries describe the term structure, but expectations, "
        "risk premia, and arbitrage-free discount factors require additional assumptions "
        "and usually different data objects."
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
