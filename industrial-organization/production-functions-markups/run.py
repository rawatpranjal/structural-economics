#!/usr/bin/env python3
"""Production function estimation and markup measurement."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


TRUE_BETA = {"Labor": 0.32, "Capital": 0.24, "Materials": 0.44}


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return beta, resid


def simulate_panel(n_firms: int = 320, n_years: int = 6, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    omega = rng.normal(0.0, 0.35, n_firms)
    capital = rng.normal(2.2, 0.45, n_firms)
    for t in range(n_years):
        omega = 0.72 * omega + rng.normal(0.0, 0.22, n_firms)
        capital = 0.82 * capital + 0.25 * omega + rng.normal(0.0, 0.18, n_firms) + 0.45
        labor = 1.05 + 0.10 * omega + 0.18 * capital + rng.normal(0.0, 0.22, n_firms)
        materials = 1.20 + 0.10 * omega + 0.15 * capital + rng.normal(0.0, 0.18, n_firms)
        investment = 0.75 + 0.90 * omega + 0.20 * capital + rng.normal(0.0, 0.05, n_firms)
        eps = rng.normal(0.0, 0.08, n_firms)
        y = (
            TRUE_BETA["Labor"] * labor
            + TRUE_BETA["Capital"] * capital
            + TRUE_BETA["Materials"] * materials
            + omega
            + eps
        )
        true_markup = np.clip(1.18 + 0.34 * (omega - omega.mean()) / omega.std() + rng.normal(0.0, 0.08, n_firms), 0.85, 2.60)
        material_share = np.clip(TRUE_BETA["Materials"] / true_markup + rng.normal(0.0, 0.025, n_firms), 0.16, 0.75)
        for i in range(n_firms):
            rows.append({
                "Firm": i,
                "Year": t,
                "log_output": y[i],
                "log_labor": labor[i],
                "log_capital": capital[i],
                "log_materials": materials[i],
                "investment_proxy": investment[i],
                "productivity": omega[i],
                "material_share": material_share[i],
                "true_markup": true_markup[i],
            })
    return pd.DataFrame(rows)


def estimate_production(df: pd.DataFrame) -> pd.DataFrame:
    y = df["log_output"].to_numpy()
    l = df["log_labor"].to_numpy()
    k = df["log_capital"].to_numpy()
    m = df["log_materials"].to_numpy()
    inv = df["investment_proxy"].to_numpy()

    X_ols = np.column_stack([np.ones(len(df)), l, k, m])
    beta_ols, _ = ols(y, X_ols)

    omega_proxy = (inv - 0.75 - 0.20 * k) / 0.90
    X_proxy = np.column_stack([np.ones(len(df)), l, k, m, omega_proxy])
    beta_proxy, _ = ols(y, X_proxy)

    estimates = pd.DataFrame({
        "Input": ["Labor", "Capital", "Materials"],
        "True elasticity": [TRUE_BETA["Labor"], TRUE_BETA["Capital"], TRUE_BETA["Materials"]],
        "OLS": beta_ols[1:4],
        "Proxy-control": beta_proxy[1:4],
    })
    return estimates


def markup_decomposition(df: pd.DataFrame, theta_m: float) -> pd.DataFrame:
    out = df.copy()
    out["estimated_markup"] = theta_m / out["material_share"]
    out["productivity_bin"] = pd.qcut(out["productivity"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return out


def main() -> None:
    setup_style()
    df = simulate_panel()
    estimates = estimate_production(df)
    theta_m_hat = float(estimates.loc[estimates["Input"] == "Materials", "Proxy-control"].iloc[0])
    markups = markup_decomposition(df, theta_m_hat)

    print("Production functions and markups tutorial")
    print(estimates.to_string(index=False))
    print(f"Mean estimated markup: {markups['estimated_markup'].mean():.3f}")

    report = ModelReport(
        "Production Functions and Markup Measurement",
        "Proxy-control production estimation and markup recovery from variable input cost shares.",
    )

    report.add_overview(
        "Production-function estimation is central to IO because productivity is observed "
        "by firms when they choose inputs but not by the econometrician. If more productive "
        "firms choose more labor or materials, a naive regression of output on inputs "
        "confounds technology with input choice.\n\n"
        "The tutorial simulates a Cobb-Douglas panel, compares OLS with a simple "
        "investment-proxy control regression, and then uses the De Loecker-Warzynski markup formula: "
        "markup equals an output elasticity divided by the revenue share of that variable input."
    )

    report.add_equations(r"""
Cobb-Douglas production:
$$y_{it} = \beta_l l_{it} + \beta_k k_{it} + \beta_m m_{it} + \omega_{it} + \epsilon_{it}$$

Investment responds monotonically to productivity:
$$i_{it} = h(k_{it}, \omega_{it})$$

Proxy-control estimators invert this policy to control for productivity.

Markup from a variable input:
$$\mu_{it} = \frac{\theta^m_{it}}{\alpha^m_{it}}$$

where $\theta^m$ is the output elasticity of materials and $\alpha^m$ is the materials expenditure share in revenue.
""")

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        f"| Firms | {df['Firm'].nunique()} |\n"
        f"| Years | {df['Year'].nunique()} |\n"
        "| Technology | Cobb-Douglas in labor, capital, and materials |\n"
        "| Productivity | Persistent AR(1), observed by firms before input choice |\n"
        "| Proxy variable | Investment, increasing in productivity conditional on capital |\n"
        "| Markup formula | Materials output elasticity divided by materials revenue share |"
    )

    report.add_solution_method(
        "OLS regresses log output on log inputs directly. The proxy-control regression "
        "uses the simulated investment policy to construct a noisy productivity proxy, "
        "which absorbs much of the productivity term that drives simultaneity. The "
        "estimated materials elasticity then enters the markup calculation for every "
        "firm-year observation."
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(estimates))
    width = 0.25
    ax1.bar(x - width, estimates["True elasticity"], width, label="True", color="#54A24B")
    ax1.bar(x, estimates["OLS"], width, label="OLS", color="#E45756")
    ax1.bar(x + width, estimates["Proxy-control"], width, label="Proxy-control", color="#4C78A8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(estimates["Input"])
    ax1.set_ylabel("Output elasticity")
    ax1.set_title("Production Function Coefficients")
    ax1.legend()
    report.add_figure(
        "figures/production-estimates.png",
        "True and estimated output elasticities",
        fig1,
        description="OLS loads part of unobserved productivity onto flexible inputs. "
        "The proxy-control regression moves the input elasticities closer to the data-generating values.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(markups["true_markup"], bins=35, alpha=0.55, label="True", color="#54A24B")
    ax2.hist(markups["estimated_markup"], bins=35, alpha=0.55, label="Estimated", color="#4C78A8")
    ax2.set_xlabel("Markup")
    ax2.set_ylabel("Firm-year count")
    ax2.set_title("Markup Distribution")
    ax2.legend()
    report.add_figure(
        "figures/markup-distribution.png",
        "True and estimated markup distributions",
        fig2,
        description="Markup estimates inherit any error in the production elasticity and any "
        "noise in the expenditure share. The distribution is still informative about dispersion.",
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sample = markups.sample(700, random_state=3)
    ax3.scatter(sample["productivity"], sample["estimated_markup"], alpha=0.35, s=18, color="#4C78A8")
    bins = markups.groupby("productivity_bin", observed=False)[["productivity", "estimated_markup"]].mean()
    ax3.plot(bins["productivity"], bins["estimated_markup"], color="black", marker="o", label="Bin mean")
    ax3.set_xlabel("Productivity")
    ax3.set_ylabel("Estimated markup")
    ax3.set_title("Productivity and Markups")
    ax3.legend()
    report.add_figure(
        "figures/productivity-markups.png",
        "Estimated markups rise with productivity in the simulated panel",
        fig3,
        description="The same production data can be used to study heterogeneity: high-productivity "
        "firms have lower material shares and therefore higher measured markups in this design.",
    )

    table = estimates.copy()
    for col in ["True elasticity", "OLS", "Proxy-control"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    report.add_table("tables/production-estimates.csv", "Production function estimates", table)

    markup_table = markups.groupby("productivity_bin", observed=False).agg(
        mean_productivity=("productivity", "mean"),
        mean_markup=("estimated_markup", "mean"),
        median_markup=("estimated_markup", "median"),
    ).reset_index()
    for col in ["mean_productivity", "mean_markup", "median_markup"]:
        markup_table[col] = markup_table[col].map(lambda x: f"{x:.3f}")
    report.add_table("tables/markup-by-productivity.csv", "Markup moments by productivity quintile", markup_table)

    report.add_takeaway(
        "Production-function estimates are not just technology parameters. Once combined "
        "with expenditure shares, they become markup estimates. That makes simultaneity, "
        "proxy assumptions, and revenue-vs-quantity measurement central to market-power claims."
    )

    report.add_references([
        "Olley, S., and Pakes, A. (1996). The Dynamics of Productivity in the Telecommunications Equipment Industry. *Econometrica*, 64(6), 1263-1297.",
        "Levinsohn, J., and Petrin, A. (2003). Estimating Production Functions Using Inputs to Control for Unobservables. *Review of Economic Studies*, 70(2), 317-341.",
        "De Loecker, J., and Warzynski, F. (2012). Markups and Firm-Level Export Status. *American Economic Review*, 102(6), 2437-2471.",
        "Lectures 10-12 Slides 2023: Production functions, proxy methods, and markups.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
