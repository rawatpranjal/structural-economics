#!/usr/bin/env python3
"""Weak-form efficient-market diagnostics for yield changes."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def load_changes() -> pd.DataFrame:
    """Load ten-year Treasury yield changes from the static dataset."""
    path = Path(__file__).resolve().parents[1] / "_data" / "daily-treasury-rates.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    df["dy"] = df["10 Yr"].diff()
    return df.dropna().reset_index(drop=True)


def autocorrelations(x: np.ndarray, max_lag: int = 20) -> pd.DataFrame:
    """Compute sample autocorrelations by lag."""
    rows = []
    centered = x - x.mean()
    denom = np.sum(centered**2)
    for lag in range(1, max_lag + 1):
        corr = float(np.sum(centered[lag:] * centered[:-lag]) / denom)
        rows.append({"Lag": lag, "Autocorrelation": corr})
    return pd.DataFrame(rows)


def variance_ratio(x: np.ndarray, q: int) -> float:
    """Compute the simple variance ratio for q-period changes."""
    overlapping = np.convolve(x, np.ones(q), mode="valid")
    return float(np.var(overlapping, ddof=1) / (q * np.var(x, ddof=1)))


def predictive_regression(x: np.ndarray) -> tuple[float, float, float]:
    """Estimate dy[t+1] = alpha + beta dy[t]."""
    y = x[1:]
    lag = x[:-1]
    design = np.column_stack([np.ones_like(lag), lag])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.array([alpha, beta])
    ssr = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return float(alpha), float(beta), 1.0 - ssr / sst


def diagnostic_table(changes: np.ndarray) -> pd.DataFrame:
    """Return weak-form diagnostic statistics."""
    alpha, beta, r2 = predictive_regression(changes)
    rows = [
        {"Diagnostic": "Lag-1 predictive slope", "Value": f"{beta:.3f}", "Null benchmark": "0"},
        {"Diagnostic": "Predictive R-squared", "Value": f"{r2:.3f}", "Null benchmark": "0"},
    ]
    for q in [2, 5, 10, 20]:
        rows.append(
            {
                "Diagnostic": f"Variance ratio q={q}",
                "Value": f"{variance_ratio(changes, q):.3f}",
                "Null benchmark": "1",
            }
        )
    rows.append({"Diagnostic": "Mean daily change (bp)", "Value": f"{100.0 * alpha:.2f}", "Null benchmark": "0"})
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    df = load_changes()
    changes = df["dy"].to_numpy()
    acf = autocorrelations(changes)
    diagnostics = diagnostic_table(changes)

    print("Efficient-market diagnostics")
    print(diagnostics.to_string(index=False))

    report = ModelReport(
        "Weak-Form Efficient-Market Diagnostics",
        "Autocorrelation, variance-ratio, and predictive checks for Treasury yield changes.",
    )

    report.add_overview(
        "Weak-form efficiency asks whether information in past prices or returns predicts "
        "future returns. In this interest-rate setting, the return-like object is the daily "
        "change in the ten-year Treasury yield, so the diagnostics ask whether past yield "
        "changes forecast future yield changes.\n\n"
        "The efficient-market interpretation must be modest. Yield changes are not fully "
        "measured holding-period bond returns, and every market-efficiency test is also a "
        "test of the maintained expected-return model, sampling choices, and measurement "
        "assumptions. The data are a static 1990 Treasury CMT snapshot."
    )

    report.add_equations(
        r"""
Define the daily yield change as

$$
\Delta y_t = y_t - y_{t-1}.
$$

A weak-form predictability regression is

$$
\Delta y_{t+1} = \alpha + \beta \Delta y_t + \epsilon_{t+1}.
$$

For independent increments, autocorrelations should be near zero. The variance
ratio compares a multi-period variance with scaled one-period variance:

$$
VR(q) = \frac{\operatorname{Var}(\Delta_q y_t)}
{q \operatorname{Var}(\Delta y_t)}.
$$

The random-walk benchmark has $VR(q) = 1$.
"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| Series | Daily 10-year Treasury yield changes |\n"
        f"| Data | Static 1990 Treasury CMT snapshot |\n"
        f"| Autocorrelation lags | 1 to 20 |\n"
        f"| Variance-ratio horizons | 2, 5, 10, 20 days |\n"
        f"| Interpretation | Weak-form diagnostic, not a trading backtest |"
    )

    report.add_solution_method(
        "Daily yield changes, sample autocorrelations, simple variance ratios, and a one-lag "
        "predictive regression provide complementary diagnostics. The variance-ratio "
        "diagnostic follows the Lo-MacKinlay intuition, but the exercise does not implement "
        "the full heteroskedasticity-robust test statistic."
    )

    fig1, ax1 = plt.subplots(figsize=(7.4, 4.8))
    ax1.bar(acf["Lag"], acf["Autocorrelation"], color="tab:blue")
    ax1.axhline(0.0, color="black", linewidth=1.0)
    band = 2.0 / np.sqrt(len(changes))
    ax1.axhline(band, color="gray", linestyle="--", linewidth=1.0)
    ax1.axhline(-band, color="gray", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_title("Autocorrelation of Yield Changes")
    report.add_figure(
        "figures/autocorrelations.png",
        "Autocorrelations of daily ten-year yield changes",
        fig1,
        description=(
            "Weak-form tests ask whether past changes predict future changes. Sample "
            "autocorrelation is the first diagnostic, but isolated bars in a short sample "
            "should not be overread."
        ),
    )

    q_values = np.array([2, 5, 10, 20])
    vr_values = np.array([variance_ratio(changes, int(q)) for q in q_values])
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.8))
    ax2.plot(q_values, vr_values, marker="o", linewidth=2.0)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Horizon q")
    ax2.set_ylabel("Variance ratio")
    ax2.set_title("Variance-Ratio Diagnostics")
    report.add_figure(
        "figures/variance-ratios.png",
        "Variance ratios for yield changes",
        fig2,
        description=(
            "Under independent increments, q-period variance should scale roughly linearly "
            "with q. Deviations are evidence about serial dependence, not automatically "
            "evidence of exploitable profits."
        ),
    )

    report.add_table(
        "tables/efficient-market-diagnostics.csv",
        "Weak-form diagnostic summary",
        diagnostics,
    )

    report.add_takeaway(
        "Efficient-market tests should be read as disciplined diagnostics. Autocorrelation "
        "or variance-ratio deviations can reject a random-walk benchmark, but interpretation "
        "requires care because the null bundles market efficiency, expected returns, sampling, "
        "and measurement assumptions."
    )
    report.add_references(
        [
            "[Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance, 25(2), 383-417.](https://doi.org/10.2307/2325486)",
            "[Lo, A. W., and MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks. Review of Financial Studies, 1, 41-66.](https://web.mit.edu/~alo/www/Papers/lo-mackinlay-88.html)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
