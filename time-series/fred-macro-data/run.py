#!/usr/bin/env python3
"""Business-cycle moments from a FRED-style macro panel.

The tutorial is self-contained: it simulates quarterly GDP growth, inflation,
unemployment, and policy-rate series with FRED-like units, applies the HP
trend-cycle convention, and reports the moments that business-cycle models
often try to match.
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
from lib.plotting import save_figure, save_thumbnail, setup_style


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

    print("Computing a long-sample benchmark from the same process...")
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

    # --- Figure 1: raw macro panel ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col in zip(axes1.flat, SERIES):
        ax.plot(df.index, df[col], color=SERIES_COLORS[col], linewidth=1.2)
        ax.axhline(df[col].mean(), color="0.35", linestyle="--", alpha=0.6, linewidth=0.9)
        ax.set_title(SERIES_TITLES[col])
        ax.set_ylabel("Percent")
    fig1.suptitle("Quarterly macro series before detrending", fontsize=14, y=1.01)
    fig1.tight_layout()
    save_figure(fig1, "figures/time-series.png", dpi=150)

    # --- Figure 2: HP-filtered cyclical components ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col in zip(axes2.flat, SERIES):
        ax.plot(cycles_df.index, cycles_df[col], color=SERIES_COLORS[col], linewidth=1.1)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_title(f"{SERIES_TITLES[col]} cycle")
        ax.set_ylabel("Deviation from trend")
    fig2.suptitle("HP-filtered cycles with quarterly smoothing", fontsize=14, y=1.01)
    fig2.tight_layout()
    save_figure(fig2, "figures/hp-cycles.png", dpi=150)

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
    save_figure(fig3, "figures/okuns-law.png", dpi=150)

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
    save_figure(fig4, "figures/cross-correlation.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    display_stats.to_csv("tables/business-cycle-stats.csv", index=False)

    save_thumbnail("figures/time-series.png", "figures/thumb.png")

    print(f"\nGenerated: 4 figures + 1 table")


if __name__ == "__main__":
    main()
