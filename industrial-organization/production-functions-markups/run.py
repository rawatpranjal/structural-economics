#!/usr/bin/env python3
"""Production-function estimation and markup measurement."""
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


def markup_decomposition(df: pd.DataFrame, theta_m_proxy: float, theta_m_ols: float) -> pd.DataFrame:
    out = df.copy()
    out["proxy_markup"] = theta_m_proxy / out["material_share"]
    out["ols_markup"] = theta_m_ols / out["material_share"]
    out["productivity_bin"] = pd.qcut(out["productivity"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return out


def main() -> None:
    setup_style()
    df = simulate_panel()
    estimates = estimate_production(df)
    theta_m_proxy = float(estimates.loc[estimates["Input"] == "Materials", "Proxy-control"].iloc[0])
    theta_m_ols = float(estimates.loc[estimates["Input"] == "Materials", "OLS"].iloc[0])
    markups = markup_decomposition(df, theta_m_proxy, theta_m_ols)

    print("Production elasticities and firm markups tutorial")
    print(estimates.to_string(index=False))
    print(f"Mean true markup:        {markups['true_markup'].mean():.3f}")
    print(f"Mean OLS-implied markup: {markups['ols_markup'].mean():.3f}")
    print(f"Mean proxy markup:       {markups['proxy_markup'].mean():.3f}")

    report = ModelReport(
        "Production Elasticities and Firm Markups",
        "Recover firm-year markups from a corrected materials elasticity and materials shares.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Plant panels often record output, labor, capital, materials, and input spending. "
        "They usually do not record marginal cost.\n\n"
        "The object is the firm-year markup. This tutorial recovers it from a materials "
        "output elasticity and a materials revenue share.\n\n"
        "Inputs respond to productivity before output is observed. The computation uses "
        "an investment proxy to correct the materials elasticity before forming markups."
    )

    report.add_equations(r"""
Let $i$ index firms and $t$ index years. Output, labor, capital, and materials
are logs. They are denoted by $y_{it}$, $l_{it}$, $k_{it}$, and $m_{it}$. The
plant technology is Cobb-Douglas:

$$y_{it} = \beta_l l_{it}+\beta_k k_{it}+\beta_m m_{it} +\omega_{it}+\varepsilon_{it}.$$

The firm observes productivity $\omega_{it}$ before choosing flexible inputs.
This timing makes $l_{it}$ and $m_{it}$ correlated with $\omega_{it}$. Naive
OLS therefore has a nonzero input-error covariance.

The proxy variable is investment $I_{it}$. In the synthetic data, investment
follows a monotone policy:

$$I_{it}=h(k_{it},\omega_{it})+\nu_{it}, \qquad \frac{\partial h(k,\omega)}{\partial \omega}>0.$$

The control-function estimator uses this monotonicity to form a productivity
control $\tilde \omega_{it}=h^{-1}(k_{it},I_{it})$. It estimates

$$y_{it} = \beta_l l_{it}+\beta_k k_{it}+\beta_m m_{it} +\rho \tilde\omega_{it}+u_{it}.$$

Markup recovery uses materials as the variable input. For Cobb-Douglas
production, the materials elasticity is $\theta^m=\beta_m$. Let

$$\alpha^m_{it} = \frac{\text{materials expenditure}_{it}}{\text{revenue}_{it}}$$

be the materials revenue share. Cost minimization implies the gross markup

$$\mu_{it}=\frac{\theta^m}{\alpha^m_{it}}.$$
""")

    report.add_model_setup(
        "| Object | Value | Role in the exercise |\n"
        "|--------|-------|----------------------|\n"
        f"| Firm-year panel | {df['Firm'].nunique()} firms, {df['Year'].nunique()} years | Lets input choices respond to persistent productivity |\n"
        "| Technology | Cobb-Douglas in labor, capital, materials | Gives known output elasticities for the benchmark |\n"
        "| True elasticities | $\\beta_l=0.32$, $\\beta_k=0.24$, $\\beta_m=0.44$ | Ground truth for the coefficient comparison |\n"
        "| Productivity | Persistent AR(1), observed by firms | Source of simultaneity in flexible inputs |\n"
        "| Proxy variable | Investment, monotone in productivity conditional on capital | Control for the unobserved productivity state |\n"
        "| Markup measure | $\\theta^m / \\alpha^m_{it}$ | Maps the materials elasticity into firm-year markups |"
    )

    report.add_solution_method(
        "The calculation first estimates the materials elasticity while controlling for "
        "productivity. It then divides that elasticity by each firm-year materials share. "
        "The proxy-control regression uses the synthetic investment schedule to form the "
        "productivity control.\n\n"
        "```text\n"
        "Algorithm: proxy-control markup measurement\n"
        "Input: panel {y_it, l_it, k_it, m_it, I_it, alpha^m_it}, proxy policy h, true benchmark mu_it\n"
        "Output: production elasticities and firm-year markup estimates\n"
        "1. Estimate the naive production regression:\n"
        "       y_it = b_l l_it + b_k k_it + b_m m_it + residual_it\n"
        "   and record the OLS materials elasticity b_m^OLS.\n"
        "2. Use monotonic investment to build a productivity control:\n"
        "       omega_tilde_it = h^{-1}(k_it, I_it).\n"
        "3. Re-estimate production with the control included:\n"
        "       y_it = b_l l_it + b_k k_it + b_m m_it + rho omega_tilde_it + u_it.\n"
        "   The controlled b_m is the markup-relevant elasticity theta_hat^m.\n"
        "4. For every firm-year, compute\n"
        "       mu_hat_it = theta_hat^m / alpha^m_it.\n"
        "5. Compare theta_hat^m and mu_hat_it with the simulated truth, and aggregate\n"
        "   markups by productivity quintile to inspect heterogeneity.\n"
        "```\n\n"
        "The inverted proxy control is the numerical step. The final markup calculation "
        "is a firm-year division."
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
    ax1.set_title("Production elasticities: truth and estimators")
    ax1.legend()
    report.add_figure(
        "figures/production-estimates.png",
        "True and estimated output elasticities",
        fig1,
        description="The production-function step drives the markup calculation. OLS overstates "
        "the flexible-input elasticities because high-productivity firms choose more inputs. "
        "The proxy-control estimate corrects for the omitted productivity state. It moves "
        "the materials elasticity close to its true value.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(markups["true_markup"], bins=35, alpha=0.50, label="True", color="#54A24B")
    ax2.hist(markups["ols_markup"], bins=35, alpha=0.42, label="OLS-implied", color="#E45756")
    ax2.hist(markups["proxy_markup"], bins=35, alpha=0.55, label="Proxy-control", color="#4C78A8")
    ax2.set_xlabel("Markup")
    ax2.set_ylabel("Firm-year count")
    ax2.set_title("Markup distribution: true and recovered")
    ax2.legend()
    report.add_figure(
        "figures/markup-distribution.png",
        "True and estimated markup distributions",
        fig2,
        description="The coefficient bias passes through the markup formula. The OLS-implied "
        "distribution sits too far to the right because the materials elasticity is inflated. "
        "The proxy-control markups stay much closer to the truth.",
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sample = markups.sample(700, random_state=3)
    ax3.scatter(sample["productivity"], sample["proxy_markup"], alpha=0.32, s=18, color="#4C78A8", label="Firm-year estimate")
    bins = markups.groupby("productivity_bin", observed=False)[["productivity", "proxy_markup", "true_markup"]].mean()
    ax3.plot(bins["productivity"], bins["proxy_markup"], color="black", marker="o", label="Proxy bin mean")
    ax3.plot(bins["productivity"], bins["true_markup"], color="#54A24B", marker="s", linestyle="--", label="True bin mean")
    ax3.set_xlabel("Productivity")
    ax3.set_ylabel("Markup")
    ax3.set_title("Productivity and markup heterogeneity")
    ax3.legend()
    report.add_figure(
        "figures/productivity-markups.png",
        "Estimated markups rise with productivity in the simulated panel",
        fig3,
        description="The simulated truth lets us check the markup gradient. More productive "
        "firms have lower materials shares in this design. True markups rise with productivity. "
        "The recovered quintile means trace that gradient.",
    )

    table = estimates.copy()
    table["OLS bias"] = table["OLS"] - table["True elasticity"]
    table["Proxy bias"] = table["Proxy-control"] - table["True elasticity"]
    for col in ["True elasticity", "OLS", "Proxy-control", "OLS bias", "Proxy bias"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    report.add_table(
        "tables/production-estimates.csv",
        "Production function estimates",
        table,
        description="The coefficient table is read through the markup formula. Materials is the "
        "main row because $\\theta^m$ is divided by the materials revenue share.",
    )

    markup_table = markups.groupby("productivity_bin", observed=False).agg(
        mean_productivity=("productivity", "mean"),
        true_markup=("true_markup", "mean"),
        ols_markup=("ols_markup", "mean"),
        proxy_markup=("proxy_markup", "mean"),
    ).reset_index()
    markup_table["proxy_bias"] = markup_table["proxy_markup"] - markup_table["true_markup"]
    markup_table = markup_table.rename(columns={
        "productivity_bin": "productivity_quintile",
    })
    for col in ["mean_productivity", "true_markup", "ols_markup", "proxy_markup", "proxy_bias"]:
        markup_table[col] = markup_table[col].map(lambda x: f"{x:.3f}")
    report.add_table(
        "tables/markup-by-productivity.csv",
        "Markup moments by productivity quintile",
        markup_table,
        description="The quintile table makes the ground-truth comparison explicit. OLS-based "
        "markups are too high in every productivity cell. The proxy-control markups keep the "
        "right ordering and a much smaller level error.",
    )

    report.add_takeaway(
        "The markup estimate is only as credible as the production elasticity and the "
        "materials share behind it. In this controlled panel, correcting for productivity "
        "greatly reduces markup error."
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
