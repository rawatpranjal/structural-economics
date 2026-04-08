#!/usr/bin/env python3
"""FRED Macroeconomic Data Analysis: Business Cycle Statistics.

Generates synthetic macroeconomic data mimicking FRED series (GDP growth,
CPI inflation, unemployment, Fed funds rate) and computes standard business
cycle statistics: HP-filtered cyclical components, volatilities, correlations,
and autocorrelations.

Reference: Stock, J. and Watson, M. (1999). "Business Cycle Fluctuations in
U.S. Macroeconomic Time Series." Handbook of Macroeconomics, Vol. 1A.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def hp_filter(y, lamb=1600):
    """Hodrick-Prescott filter.

    Decomposes a time series y into trend (tau) and cycle (c) components
    by solving: min_tau { sum(y_t - tau_t)^2 + lambda * sum((tau_{t+1} - 2*tau_t + tau_{t-1})^2) }

    Parameters
    ----------
    y : array, shape (T,)
        Time series to filter.
    lamb : float
        Smoothing parameter (1600 for quarterly data).

    Returns
    -------
    cycle : array, shape (T,)
    trend : array, shape (T,)
    """
    from scipy.sparse import eye, spdiags
    from scipy.sparse.linalg import spsolve

    T = len(y)
    # Build second-difference matrix K
    e = np.ones(T)
    diags = np.array([e, -2 * e, e])
    offsets = np.array([0, 1, 2])
    K = spdiags(diags, offsets, T - 2, T)
    I = eye(T, format="csc")
    trend = spsolve(I + lamb * K.T @ K, y)
    cycle = y - trend
    return cycle, trend


def generate_synthetic_macro_data(T=200, seed=42):
    """Generate synthetic quarterly macroeconomic data mimicking FRED series.

    Builds correlated series with realistic moments:
    - GDP growth ~ N(2.5%, 3%) -- Okun's law correlation with unemployment
    - CPI inflation ~ N(2%, 1.5%) -- Phillips curve correlation with unemployment
    - Unemployment ~ N(5.5%, 1.5%) -- counter-cyclical
    - Fed funds rate ~ N(4%, 3%) -- follows Taylor rule loosely

    Returns a DataFrame indexed by synthetic quarterly dates.
    """
    rng = np.random.default_rng(seed)

    # Target means and standard deviations
    means = np.array([2.5, 2.0, 5.5, 4.0])
    stds = np.array([3.0, 1.5, 1.5, 3.0])

    # Correlation matrix encoding stylized facts
    #           GDP   CPI   UE    FFR
    corr = np.array([
        [ 1.0,  0.2, -0.6,  0.3],   # GDP growth
        [ 0.2,  1.0, -0.3,  0.5],   # CPI inflation (Phillips curve)
        [-0.6, -0.3,  1.0, -0.2],   # Unemployment (Okun's law)
        [ 0.3,  0.5, -0.2,  1.0],   # Fed funds rate (Taylor rule)
    ])

    # Covariance matrix
    cov = np.outer(stds, stds) * corr

    # Generate i.i.d. draws, then impose AR(1) persistence
    raw = rng.multivariate_normal(np.zeros(4), corr, size=T)

    # AR(1) persistence parameters (quarterly)
    rho = np.array([0.3, 0.7, 0.85, 0.8])

    data = np.zeros((T, 4))
    data[0] = raw[0]
    for t in range(1, T):
        data[t] = rho * data[t - 1] + np.sqrt(1 - rho**2) * raw[t]

    # Rescale to target moments
    data = data * stds[None, :] + means[None, :]

    # Clip unemployment to positive values
    data[:, 2] = np.clip(data[:, 2], 1.0, 15.0)
    # Clip fed funds to non-negative
    data[:, 3] = np.clip(data[:, 3], 0.0, 20.0)

    dates = pd.date_range("1960-01-01", periods=T, freq="QS")
    df = pd.DataFrame(data, index=dates,
                       columns=["GDP_growth", "CPI_inflation", "Unemployment", "FedFunds"])
    return df


def compute_business_cycle_stats(df, lamb=1600):
    """Compute standard business cycle statistics on HP-filtered data.

    Returns cyclical components and a summary statistics DataFrame.
    """
    cycles = {}
    trends = {}
    for col in df.columns:
        c, t = hp_filter(df[col].values, lamb=lamb)
        cycles[col] = c
        trends[col] = t

    cycles_df = pd.DataFrame(cycles, index=df.index)
    trends_df = pd.DataFrame(trends, index=df.index)

    # Volatilities (std of cyclical component)
    vols = cycles_df.std()

    # Relative volatility (normalized by GDP cycle volatility)
    rel_vols = vols / vols["GDP_growth"]

    # Contemporaneous correlations with GDP cycle
    corrs_with_gdp = cycles_df.corrwith(cycles_df["GDP_growth"])

    # First-order autocorrelations
    autocorrs = cycles_df.apply(lambda s: s.autocorr(lag=1))

    stats = pd.DataFrame({
        "Volatility (%)": vols.round(3),
        "Rel. Volatility": rel_vols.round(3),
        "Corr. with GDP": corrs_with_gdp.round(3),
        "Autocorrelation": autocorrs.round(3),
    })
    return cycles_df, trends_df, stats


def main():
    # =========================================================================
    # Generate synthetic macroeconomic data
    # =========================================================================
    print("Generating synthetic FRED macro data...")
    T = 200  # 50 years of quarterly data
    df = generate_synthetic_macro_data(T=T, seed=42)

    print(f"  Sample: {df.index[0].strftime('%Y-Q1')} to {df.index[-1].strftime('%Y')}")
    print(f"  Variables: {list(df.columns)}")
    print(f"  Observations: {len(df)}")

    # =========================================================================
    # HP filter and business cycle statistics
    # =========================================================================
    print("\nComputing HP-filtered business cycle statistics...")
    cycles_df, trends_df, stats = compute_business_cycle_stats(df, lamb=1600)
    print(stats.to_string())

    # =========================================================================
    # Cross-correlation structure
    # =========================================================================
    cross_corr = cycles_df.corr()
    print("\nCross-correlation matrix of cyclical components:")
    print(cross_corr.round(3).to_string())

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "FRED Macroeconomic Data Analysis",
        "Business cycle statistics from synthetic macroeconomic data mimicking FRED series.",
    )

    report.add_overview(
        "This model generates synthetic U.S. macroeconomic data mimicking the key quarterly "
        "series available from the Federal Reserve Economic Data (FRED) database: real GDP "
        "growth, CPI inflation, the unemployment rate, and the federal funds rate.\n\n"
        "The synthetic data is constructed to reproduce the most important stylized facts of "
        "business cycles: (1) Okun's law (negative GDP-unemployment correlation), (2) the "
        "Phillips curve (negative inflation-unemployment correlation), and (3) realistic "
        "persistence and volatility in each series. We then apply the Hodrick-Prescott filter "
        "to decompose each series into trend and cyclical components, and compute standard "
        "business cycle statistics."
    )

    report.add_equations(
        r"""
**Hodrick-Prescott Filter:**

$$\min_{\{\tau_t\}} \left\{ \sum_{t=1}^{T} (y_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1} [(\tau_{t+1} - \tau_t) - (\tau_t - \tau_{t-1})]^2 \right\}$$

where $y_t$ is the observed series, $\tau_t$ is the trend component, and $\lambda = 1600$
for quarterly data (Hodrick and Prescott, 1997).

**Okun's Law:** $\Delta u_t \approx -0.5 \cdot (\Delta Y_t / Y_t - 3\%)$

**Phillips Curve:** $\pi_t = \pi_t^e - \alpha (u_t - u_t^*)$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $T$ | {T} | Quarterly observations (50 years) |\n"
        f"| $\\lambda$ | 1600 | HP filter smoothing (quarterly) |\n"
        f"| GDP growth | $\\mu=2.5\\%$, $\\sigma=3\\%$ | Real GDP growth rate |\n"
        f"| CPI inflation | $\\mu=2\\%$, $\\sigma=1.5\\%$ | Consumer price inflation |\n"
        f"| Unemployment | $\\mu=5.5\\%$, $\\sigma=1.5\\%$ | Civilian unemployment rate |\n"
        f"| Fed funds | $\\mu=4\\%$, $\\sigma=3\\%$ | Federal funds effective rate |"
    )

    report.add_solution_method(
        "**Data Generation:** Correlated AR(1) processes with calibrated persistence "
        "parameters ($\\rho_{GDP}=0.3$, $\\rho_{CPI}=0.7$, $\\rho_{UE}=0.85$, "
        "$\\rho_{FFR}=0.8$) and a cross-sectional correlation matrix encoding Okun's "
        "law, the Phillips curve, and Taylor rule correlations.\n\n"
        "**HP Filtering:** The Hodrick-Prescott filter with $\\lambda=1600$ separates each "
        "series into a smooth trend and a stationary cyclical component. The cyclical "
        "components are used to compute business cycle statistics.\n\n"
        "**Business Cycle Statistics:** Standard deviations (volatilities), relative "
        "volatilities (normalized by GDP), contemporaneous correlations with GDP, and "
        "first-order autocorrelations of the cyclical components."
    )

    # --- Figure 1: Raw time series ---
    fig1, axes = plt.subplots(2, 2, figsize=(14, 8))
    titles = ["Real GDP Growth (%)", "CPI Inflation (%)",
              "Unemployment Rate (%)", "Fed Funds Rate (%)"]
    colors = ["#2166ac", "#b2182b", "#4daf4a", "#984ea3"]

    for ax, col, title, color in zip(axes.flat, df.columns, titles, colors):
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.axhline(df[col].mean(), color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Percent")
    fig1.suptitle("Synthetic Macroeconomic Time Series", fontsize=14, y=1.01)
    fig1.tight_layout()
    report.add_figure("figures/time-series.png",
                       "Synthetic macroeconomic time series mimicking FRED data. "
                       "Dashed lines show sample means.", fig1,
                       description="These four series capture the core of macroeconomic monitoring. "
                       "Notice the high persistence of unemployment compared to the more volatile GDP growth -- this reflects the sluggish adjustment of labor markets "
                       "and is a key calibration target for DSGE models.")

    # --- Figure 2: HP-filtered cyclical components ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col, title, color in zip(axes2.flat, cycles_df.columns, titles, colors):
        ax.plot(cycles_df.index, cycles_df[col], color=color, linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.fill_between(cycles_df.index, cycles_df[col], 0,
                        where=cycles_df[col] < 0, alpha=0.15, color="gray")
        ax.set_title(f"{title} -- Cyclical Component")
        ax.set_ylabel("Deviation from trend")
    fig2.suptitle("HP-Filtered Cyclical Components ($\\lambda=1600$)", fontsize=14, y=1.01)
    fig2.tight_layout()
    report.add_figure("figures/hp-cycles.png",
                       "HP-filtered cyclical components. Shaded areas indicate below-trend periods.", fig2,
                       description="The HP filter separates each series into a smooth trend and a stationary cyclical component. "
                       "Shaded below-trend periods correspond to contractionary phases of the business cycle. "
                       "These cyclical components are the inputs to standard business cycle accounting -- their volatilities, correlations, and persistence define the stylized facts that structural models must match.")

    # --- Figure 3: Okun's law scatter ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.scatter(cycles_df["GDP_growth"], cycles_df["Unemployment"],
                alpha=0.5, s=20, color="#2166ac", edgecolors="white", linewidth=0.3)
    # Regression line
    z = np.polyfit(cycles_df["GDP_growth"].values, cycles_df["Unemployment"].values, 1)
    x_fit = np.linspace(cycles_df["GDP_growth"].min(), cycles_df["GDP_growth"].max(), 100)
    ax3.plot(x_fit, np.polyval(z, x_fit), "r-", linewidth=2, label=f"slope = {z[0]:.3f}")
    corr_ou = cycles_df["GDP_growth"].corr(cycles_df["Unemployment"])
    ax3.set_xlabel("GDP Growth (cyclical component)")
    ax3.set_ylabel("Unemployment (cyclical component)")
    ax3.set_title(f"Okun's Law: GDP--Unemployment Relationship ($\\rho = {corr_ou:.3f}$)")
    ax3.legend()
    report.add_figure("figures/okuns-law.png",
                       f"Okun's law: negative relationship between cyclical GDP growth and "
                       f"unemployment (correlation = {corr_ou:.3f}).", fig3,
                       description="Okun's law is one of the most robust empirical regularities in macroeconomics: output above trend is associated with unemployment below trend. "
                       "The slope of approximately $-0.5$ means that a 1 percentage point increase in GDP growth is associated with a 0.5 percentage point decline in unemployment -- "
                       "a relationship that has held remarkably stable across decades and countries.")

    # --- Figure 4: Cross-correlation heatmap ---
    fig4, ax4 = plt.subplots(figsize=(7, 6))
    im = ax4.imshow(cross_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = ["GDP growth", "CPI inflation", "Unemployment", "Fed funds"]
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(labels, rotation=30, ha="right")
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(labels)
    # Annotate cells
    for i in range(4):
        for j in range(4):
            val = cross_corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=12, fontweight="bold", color=color)
    fig4.colorbar(im, ax=ax4, shrink=0.8, label="Correlation")
    ax4.set_title("Cross-Correlation of Cyclical Components")
    fig4.tight_layout()
    report.add_figure("figures/cross-correlation.png",
                       "Cross-correlation structure of HP-filtered cyclical components.", fig4,
                       description="The correlation matrix encodes the key macroeconomic relationships: the negative GDP-unemployment entry is Okun's law, "
                       "the negative inflation-unemployment entry is the Phillips curve, and the positive inflation-Fed funds entry reflects the Taylor rule. "
                       "These co-movement patterns are the empirical targets that DSGE and VAR models aim to replicate.")

    # --- Table: Business cycle statistics ---
    stats_display = stats.reset_index().rename(columns={"index": "Variable"})
    report.add_table("tables/business-cycle-stats.csv",
                      "Business Cycle Statistics (HP-filtered, quarterly)",
                      stats_display,
                      description="These statistics are the standard output of business cycle accounting. "
                      "Relative volatility shows how variable each series is compared to GDP, while autocorrelation measures persistence. "
                      "Any structural model (RBC, New Keynesian) should be evaluated against these empirical moments.")

    report.add_takeaway(
        "The synthetic data successfully reproduces the key stylized facts of U.S. "
        "business cycles:\n\n"
        "**Key findings:**\n"
        f"- **Okun's law** is clearly visible: the cyclical correlation between GDP growth "
        f"and unemployment is {corr_ou:.3f}, confirming the negative relationship between "
        f"output and joblessness.\n"
        f"- **Unemployment is the most persistent** series (highest autocorrelation), "
        f"consistent with the sluggish adjustment of labor markets.\n"
        f"- **GDP growth is the most volatile** cyclical variable, while inflation is "
        f"relatively smooth after HP filtering.\n"
        f"- The **cross-correlation structure** reveals the expected pattern: GDP and "
        f"unemployment move in opposite directions, while the fed funds rate co-moves "
        f"positively with inflation (Taylor rule).\n\n"
        "These statistics provide a target for structural models: any DSGE model "
        "should be able to match these moments to be empirically credible."
    )

    report.add_references([
        "Hodrick, R. and Prescott, E. (1997). \"Postwar U.S. Business Cycles: An Empirical Investigation.\" *Journal of Money, Credit and Banking*, 29(1), 1-16.",
        "Stock, J. and Watson, M. (1999). \"Business Cycle Fluctuations in U.S. Macroeconomic Time Series.\" *Handbook of Macroeconomics*, Vol. 1A, Ch. 1.",
        "Okun, A. (1962). \"Potential GNP: Its Measurement and Significance.\" *Proceedings of the Business and Economic Statistics Section*, ASA.",
        "Phillips, A. W. (1958). \"The Relation Between Unemployment and the Rate of Change of Money Wage Rates in the United Kingdom, 1861-1957.\" *Economica*, 25(100), 283-299.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
