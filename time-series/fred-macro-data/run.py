#!/usr/bin/env python3
"""FRED-style macro data and business-cycle moment measurement.

The tutorial is self-contained: it simulates a small quarterly macro panel with
FRED-like units and moments, applies the HP trend-cycle convention, and reports
the business-cycle moments that structural macro models often try to match.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


SERIES = ["GDP_growth", "CPI_inflation", "Unemployment", "FedFunds"]
SERIES_LABELS = {
    "GDP_growth": "GDP growth",
    "CPI_inflation": "CPI inflation",
    "Unemployment": "Unemployment",
    "FedFunds": "Fed funds",
}
SERIES_TITLES = {
    "GDP_growth": "Real GDP growth (%)",
    "CPI_inflation": "CPI inflation (%)",
    "Unemployment": "Unemployment rate (%)",
    "FedFunds": "Fed funds rate (%)",
}
SERIES_COLORS = {
    "GDP_growth": "#2166ac",
    "CPI_inflation": "#b3202a",
    "Unemployment": "#3f8f4f",
    "FedFunds": "#7b4fa3",
}


def hp_filter(y: np.ndarray, lamb: float = 1600.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the HP cycle and trend for one time series."""
    T = len(y)
    e = np.ones(T)
    diags = np.array([e, -2.0 * e, e])
    offsets = np.array([0, 1, 2])
    K = spdiags(diags, offsets, T - 2, T)
    I = eye(T, format="csc")
    trend = spsolve(I + lamb * K.T @ K, y)
    cycle = y - trend
    return cycle, trend


def generate_synthetic_macro_data(
    T: int = 200,
    seed: int = 42,
    dated: bool = True,
) -> pd.DataFrame:
    """Generate a quarterly macro panel with FRED-like units and moments."""
    rng = np.random.default_rng(seed)

    means = np.array([2.5, 2.0, 5.5, 4.0])
    stds = np.array([3.0, 1.5, 1.5, 3.0])
    persistence = np.array([0.30, 0.70, 0.85, 0.80])

    # Innovation correlations encode the intended macro co-movement.
    corr = np.array(
        [
            [1.0, 0.2, -0.6, 0.3],
            [0.2, 1.0, -0.3, 0.5],
            [-0.6, -0.3, 1.0, -0.2],
            [0.3, 0.5, -0.2, 1.0],
        ]
    )

    innovations = rng.multivariate_normal(np.zeros(4), corr, size=T)
    standardized = np.zeros((T, 4))
    standardized[0] = innovations[0]
    innovation_scale = np.sqrt(1.0 - persistence**2)

    for t in range(1, T):
        standardized[t] = (
            persistence * standardized[t - 1]
            + innovation_scale * innovations[t]
        )

    data = means[None, :] + stds[None, :] * standardized
    data[:, 2] = np.clip(data[:, 2], 1.0, 15.0)
    data[:, 3] = np.clip(data[:, 3], 0.0, 20.0)

    if dated:
        index = pd.date_range("1960-01-01", periods=T, freq="QS")
    else:
        index = pd.RangeIndex(T, name="quarter")

    return pd.DataFrame(data, index=index, columns=SERIES)


def compute_business_cycle_stats(
    df: pd.DataFrame,
    lamb: float = 1600.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """HP-filter the panel and compute business-cycle moments."""
    cycles = {}
    trends = {}
    for col in df.columns:
        cycle, trend = hp_filter(df[col].to_numpy(), lamb=lamb)
        cycles[col] = cycle
        trends[col] = trend

    cycles_df = pd.DataFrame(cycles, index=df.index)
    trends_df = pd.DataFrame(trends, index=df.index)

    vols = cycles_df.std()
    rel_vols = vols / vols["GDP_growth"]
    corrs_with_gdp = cycles_df.corrwith(cycles_df["GDP_growth"])
    autocorrs = cycles_df.apply(lambda s: s.autocorr(lag=1))

    stats = pd.DataFrame(
        {
            "Volatility (%)": vols,
            "Rel. Volatility": rel_vols,
            "Corr. with GDP": corrs_with_gdp,
            "Autocorrelation": autocorrs,
        }
    )
    return cycles_df, trends_df, stats


def okun_line(cycles_df: pd.DataFrame) -> tuple[float, float, float]:
    """Fit unemployment cycles on GDP-growth cycles."""
    x = cycles_df["GDP_growth"].to_numpy()
    y = cycles_df["Unemployment"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), corr


def stats_table(sample_stats: pd.DataFrame, benchmark_stats: pd.DataFrame) -> pd.DataFrame:
    """Build a compact table comparing sample and long-sample moments."""
    rows = []
    for series in SERIES:
        rows.append(
            {
                "Variable": SERIES_LABELS[series],
                "Volatility (%)": sample_stats.loc[series, "Volatility (%)"],
                "Rel. volatility": sample_stats.loc[series, "Rel. Volatility"],
                "Corr. with GDP": sample_stats.loc[series, "Corr. with GDP"],
                "Long-sample corr.": benchmark_stats.loc[series, "Corr. with GDP"],
                "Autocorr.": sample_stats.loc[series, "Autocorrelation"],
                "Long-sample autocorr.": benchmark_stats.loc[series, "Autocorrelation"],
            }
        )

    table = pd.DataFrame(rows)
    numeric_cols = table.columns.drop("Variable")
    table[numeric_cols] = table[numeric_cols].round(3)
    return table


def quarter_label(timestamp: pd.Timestamp) -> str:
    """Format a pandas quarterly timestamp as YYYYQq."""
    quarter = (timestamp.month - 1) // 3 + 1
    return f"{timestamp.year}Q{quarter}"


def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    T = 200
    T_benchmark = 5000
    lamb_hp = 1600.0

    print("Generating a self-contained FRED-style macro panel...")
    df = generate_synthetic_macro_data(T=T, seed=42, dated=True)
    print(f"  Sample: {quarter_label(df.index[0])} to {quarter_label(df.index[-1])}")
    print(f"  Variables: {', '.join(SERIES_LABELS[col] for col in SERIES)}")
    print(f"  Observations: {len(df)}")

    print("\nComputing HP-filtered business-cycle moments...")
    cycles_df, trends_df, sample_stats = compute_business_cycle_stats(df, lamb=lamb_hp)

    print("Computing a long-sample benchmark from the same DGP...")
    benchmark_df = generate_synthetic_macro_data(
        T=T_benchmark,
        seed=2026,
        dated=False,
    )
    benchmark_cycles_df, _, benchmark_stats = compute_business_cycle_stats(
        benchmark_df,
        lamb=lamb_hp,
    )

    sample_okun_slope, sample_okun_intercept, sample_okun_corr = okun_line(cycles_df)
    benchmark_okun_slope, benchmark_okun_intercept, benchmark_okun_corr = okun_line(
        benchmark_cycles_df
    )
    cross_corr = cycles_df.corr()
    display_stats = stats_table(sample_stats, benchmark_stats)

    print(display_stats.to_string(index=False))
    print(
        "\nOkun slope: "
        f"sample = {sample_okun_slope:.3f}, "
        f"long-sample benchmark = {benchmark_okun_slope:.3f}"
    )

    setup_style()
    report = ModelReport(
        "FRED-Style Macro Data and Business-Cycle Moments",
        "HP-filtered macro cycles in a self-contained quarterly panel.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Before a structural macro model is estimated or simulated, the researcher has "
        "to decide what the data object is. Quarterly GDP growth, inflation, "
        "unemployment, and the policy rate are not yet business-cycle facts; they "
        "become facts only after a trend-cycle convention and a set of moments are "
        "chosen.\n\n"
        "This tutorial keeps that measurement step small and explicit. It uses a "
        "synthetic panel with FRED-style units, so the example is reproducible without "
        "an API key or changing data release. The maintained co-movement is familiar: "
        "output and unemployment move in opposite directions, inflation and slack move "
        "against each other, and the funds rate comoves with inflation. The exercise "
        "then asks how much of that structure is visible in a finite 50-year quarterly "
        "sample.\n\n"
        "The tutorial is a bridge between the scalar persistence logic in "
        "[Persistent Shocks](../ar-processes/) and the larger-panel forecasting "
        "problem in [Stock-Watson Factor Forecasting](../stock-watson/)."
    )

    report.add_equations(
        r"""
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
sets the intended contemporaneous macro relationships, while $\rho_j$ controls
how slowly each series adjusts.

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
"""
    )

    report.add_model_setup(
        "**Quarterly sample**\n\n"
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        f"| $T$ | {T} | Main sample, 50 years of quarters |\n"
        f"| $T_B$ | {T_benchmark} | Long simulation used only as a DGP benchmark |\n"
        f"| $\\lambda$ | {lamb_hp:.0f} | HP smoothing parameter for quarterly data |\n\n"
        "**Series-level primitives**\n\n"
        "| Series | Mean | Std. dev. | Persistence | Economic role |\n"
        "|---|---:|---:|---:|---|\n"
        "| GDP growth | 2.5 | 3.0 | 0.30 | Output-growth cycle |\n"
        "| CPI inflation | 2.0 | 1.5 | 0.70 | Price-pressure cycle |\n"
        "| Unemployment | 5.5 | 1.5 | 0.85 | Labor-market slack |\n"
        "| Fed funds | 4.0 | 3.0 | 0.80 | Short-rate policy indicator |\n\n"
        "**Innovation correlation matrix $C$**\n\n"
        "| | GDP | CPI | Unemployment | Fed funds |\n"
        "|---|---:|---:|---:|---:|\n"
        "| GDP | 1.00 | 0.20 | -0.60 | 0.30 |\n"
        "| CPI | 0.20 | 1.00 | -0.30 | 0.50 |\n"
        "| Unemployment | -0.60 | -0.30 | 1.00 | -0.20 |\n"
        "| Fed funds | 0.30 | 0.50 | -0.20 | 1.00 |"
    )

    report.add_solution_method(
        "The computation is a measurement exercise, not a structural estimation "
        "routine. The HP filter supplies the trend-cycle split, and the moment table "
        "summarizes the cyclical panel. The long simulation is not an external truth; "
        "it is the same data-generating process run long enough to show what the "
        "finite sample is trying to recover.\n\n"
        "```text\n"
        "Algorithm: FRED-style business-cycle measurement\n"
        "Inputs: quarterly panel y_t, HP parameter lambda, benchmark horizon T_B\n"
        "Outputs: cycles c_t, moment table M, Okun slope beta_O\n\n"
        "1. Simulate the four-variable macro vector y_t from the calibrated DGP.\n"
        "2. For each series j:\n"
        "      solve (I + lambda K'K) tau_j = y_j\n"
        "      set c_j = y_j - tau_j\n"
        "3. Compute volatility, relative volatility, GDP correlation, and lag-1\n"
        "   autocorrelation from the cycles c_j.\n"
        "4. Regress the unemployment cycle on the GDP-growth cycle to estimate beta_O.\n"
        "5. Repeat steps 1-4 with T_B quarters and use those moments only as a\n"
        "   long-sample benchmark for the finite 50-year run.\n"
        "```\n\n"
        "The key interpretive step is step 2. A positive GDP-growth cycle means output "
        "growth is above its smooth trend; a positive unemployment cycle means labor "
        "market slack is above trend. The signs should therefore be read by variable, "
        "not as a generic good-versus-bad cycle."
    )

    # --- Figure 1: raw macro panel ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col in zip(axes1.flat, SERIES):
        ax.plot(df.index, df[col], color=SERIES_COLORS[col], linewidth=1.2)
        ax.axhline(df[col].mean(), color="0.35", linestyle="--", alpha=0.6, linewidth=0.9)
        ax.set_title(SERIES_TITLES[col])
        ax.set_ylabel("Percent")
    fig1.suptitle("Quarterly macro series before detrending", fontsize=14, y=1.01)
    fig1.tight_layout()

    report.add_results(
        "The raw panel shows the object a macroeconomist would start from: rates and "
        "growth rates in their observed units. GDP growth is noisy, while "
        "unemployment and the funds rate move more slowly. That difference matters "
        "because the HP filter will treat high-frequency movement and slow adjustment "
        "differently."
    )
    report.add_figure(
        "figures/time-series.png",
        "Quarterly FRED-style macro series before detrending.",
        fig1,
    )

    # --- Figure 2: HP-filtered cyclical components ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col in zip(axes2.flat, SERIES):
        ax.plot(cycles_df.index, cycles_df[col], color=SERIES_COLORS[col], linewidth=1.1)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_title(f"{SERIES_TITLES[col]} cycle")
        ax.set_ylabel("Deviation from trend")
    fig2.suptitle("HP-filtered cycles with quarterly smoothing", fontsize=14, y=1.01)
    fig2.tight_layout()

    report.add_results(
        "After detrending, the comparison is in deviations from each series' own "
        "smooth path. This is the measurement convention behind the moment table. It "
        "is useful, but it is not innocuous: trend-cycle choices can change the size "
        "and persistence of measured fluctuations."
    )
    report.add_figure(
        "figures/hp-cycles.png",
        "HP-filtered cyclical components for the four macro series.",
        fig2,
    )

    # --- Figure 3: Okun relationship with long-sample benchmark ---
    fig3, ax3 = plt.subplots(figsize=(7.5, 5.3))
    x = cycles_df["GDP_growth"].to_numpy()
    y = cycles_df["Unemployment"].to_numpy()
    x_pad = 0.05 * (x.max() - x.min())
    x_fit = np.linspace(x.min() - x_pad, x.max() + x_pad, 120)

    ax3.scatter(
        x,
        y,
        alpha=0.55,
        s=24,
        color="#2166ac",
        edgecolors="white",
        linewidth=0.35,
        label="50-year sample",
    )
    ax3.plot(
        x_fit,
        sample_okun_intercept + sample_okun_slope * x_fit,
        color="#b3202a",
        linewidth=2.0,
        label=f"sample slope {sample_okun_slope:.2f}",
    )
    ax3.plot(
        x_fit,
        benchmark_okun_intercept + benchmark_okun_slope * x_fit,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label=f"long-sample slope {benchmark_okun_slope:.2f}",
    )
    ax3.axhline(0.0, color="0.55", linewidth=0.7)
    ax3.axvline(0.0, color="0.55", linewidth=0.7)
    ax3.set_xlabel("GDP-growth cycle")
    ax3.set_ylabel("Unemployment cycle")
    ax3.set_title(
        f"Okun relationship in cycles: corr = {sample_okun_corr:.3f}"
    )
    ax3.legend(frameon=False)
    fig3.tight_layout()

    report.add_results(
        "The Okun scatter is where the economic content is easiest to see. Output "
        "above trend is associated with unemployment below trend. The dashed line is "
        f"a {T_benchmark:,}-quarter simulation from the same DGP, so it is a numerical "
        "benchmark for the finite sample rather than a claim about historical U.S. "
        f"data. The 50-year sample correlation is {sample_okun_corr:.3f}; the "
        f"long-sample benchmark is {benchmark_okun_corr:.3f}."
    )
    report.add_figure(
        "figures/okuns-law.png",
        "Okun relationship with finite-sample and long-sample regression lines.",
        fig3,
    )

    # --- Figure 4: Cross-correlation heatmap ---
    fig4, ax4 = plt.subplots(figsize=(7.2, 6.0))
    im = ax4.imshow(cross_corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    labels = [SERIES_LABELS[col] for col in SERIES]
    ax4.set_xticks(range(len(SERIES)))
    ax4.set_xticklabels(labels, rotation=30, ha="right")
    ax4.set_yticks(range(len(SERIES)))
    ax4.set_yticklabels(labels)

    for i in range(len(SERIES)):
        for j in range(len(SERIES)):
            val = cross_corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax4.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )

    fig4.colorbar(im, ax=ax4, shrink=0.82, label="Correlation")
    ax4.set_title("Cross-correlation of HP-filtered cycles")
    fig4.tight_layout()

    report.add_results(
        "The correlation matrix is a compact target, not a causal model. A structural "
        "RBC or New Keynesian model would have to explain why these variables comove; "
        "a reduced-form VAR would instead summarize the same object dynamically. Here "
        "the matrix is enough to check whether the synthetic panel delivers the signs "
        "built into the calibration."
    )
    report.add_figure(
        "figures/cross-correlation.png",
        "Cross-correlation matrix of HP-filtered cyclical components.",
        fig4,
    )

    report.add_table(
        "tables/business-cycle-stats.csv",
        "Business-cycle moments from HP-filtered quarterly cycles",
        display_stats,
        description=(
            "The table separates the finite 50-year sample from the long-sample "
            "benchmark. The benchmark columns come from the same synthetic DGP, so "
            "their role is to show sampling variation and HP-filter effects, not to "
            "replace actual empirical validation."
        ),
    )

    report.add_takeaway(
        "Business-cycle measurement is already a modeling choice. In this run, the "
        "HP-filtered panel recovers the intended signs: GDP growth and unemployment "
        f"move against each other, with an Okun slope of {sample_okun_slope:.3f} in "
        "the 50-year sample, and unemployment is the most persistent cycle. The "
        "long-sample benchmark makes the finite-sample point explicit: the measured "
        "moments are close to the DGP's implications, but not identical. That is the "
        "right way to read these moments before using them as targets for a DSGE, "
        "RBC, or reduced-form forecasting exercise."
    )

    report.add_references(
        [
            "Federal Reserve Bank of St. Louis. FRED, Federal Reserve Economic Data.",
            "Hodrick, R. and Prescott, E. (1997). \"Postwar U.S. Business Cycles: An Empirical Investigation.\" *Journal of Money, Credit and Banking*, 29(1), 1-16.",
            "Stock, J. and Watson, M. (1999). \"Business Cycle Fluctuations in U.S. Macroeconomic Time Series.\" *Handbook of Macroeconomics*, Vol. 1A, Ch. 1.",
            "Okun, A. (1962). \"Potential GNP: Its Measurement and Significance.\" *Proceedings of the Business and Economic Statistics Section*, ASA.",
            "Phillips, A. W. (1958). \"The Relation Between Unemployment and the Rate of Change of Money Wage Rates in the United Kingdom, 1861-1957.\" *Economica*, 25(100), 283-299.",
        ]
    )

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} table")


if __name__ == "__main__":
    main()
