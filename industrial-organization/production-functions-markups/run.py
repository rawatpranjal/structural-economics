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

    print("Production functions and markups tutorial")
    print(estimates.to_string(index=False))
    print(f"Mean true markup:        {markups['true_markup'].mean():.3f}")
    print(f"Mean OLS-implied markup: {markups['ols_markup'].mean():.3f}")
    print(f"Mean proxy markup:       {markups['proxy_markup'].mean():.3f}")

    report = ModelReport(
        "Production Functions and Markup Measurement",
        "Productivity, input choice, and markup recovery from variable-input cost shares.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Markup measurement links two objects that are often taught separately: a production "
        "function and the firm's first-order condition for a flexible input. The economic "
        "problem is not just estimating a technology parameter. If productive firms choose "
        "more labor and materials, OLS attributes part of productivity to those inputs, and "
        "that bias flows directly into markup estimates.\n\n"
        "The data are synthetic so the truth is visible. A Cobb-Douglas panel generates "
        "output, flexible input choices, investment, productivity, and markups. The exercise "
        "compares a naive production regression with a transparent investment-proxy control, "
        "then applies the De Loecker-Warzynski markup formula. It complements "
        "[Logit Demand and Markup Recovery](../logit-supply-side/): that tutorial recovers "
        "markups from demand and pricing FOCs, while this one recovers them from production "
        "elasticities and input shares."
    )

    report.add_equations(r"""
Let $i$ index firms and $t$ years. Output, labor, capital, and materials are in
logs and are denoted by $y_{it}$, $l_{it}$, $k_{it}$, and $m_{it}$. The simulated
production function is

$$
y_{it}
= \beta_l l_{it}+\beta_k k_{it}+\beta_m m_{it}
+\omega_{it}+\varepsilon_{it}.
$$

The productivity state $\omega_{it}$ is observed by the firm before flexible
inputs are chosen. That timing makes $l_{it}$ and $m_{it}$ correlated with
$\omega_{it}$, so the population regression error in a naive OLS equation is not
orthogonal to the inputs.

The proxy variable is investment $I_{it}$. In the synthetic data it follows a
monotone policy

$$
I_{it}=h(k_{it},\omega_{it})+\nu_{it},
\qquad \frac{\partial h(k,\omega)}{\partial \omega}>0.
$$

The control-function idea is to use the monotonicity of $h$ to form a control
$\tilde \omega_{it}=h^{-1}(k_{it},I_{it})$ and estimate

$$
y_{it}
= \beta_l l_{it}+\beta_k k_{it}+\beta_m m_{it}
+\rho \tilde\omega_{it}+u_{it}.
$$

The markup formula uses a variable input. For materials, the Cobb-Douglas output
elasticity is $\theta^m=\beta_m$. Let

$$
\alpha^m_{it}
= \frac{\text{materials expenditure}_{it}}{\text{revenue}_{it}}
$$

be the materials revenue share. Cost minimization implies the gross markup

$$
\mu_{it}=\frac{\theta^m}{\alpha^m_{it}}.
$$

In field data, the hard part is justifying the proxy and the variable-input
first-order condition. In this synthetic run, the hard part is stripped down so
the mapping from production-elasticity bias to markup bias is observable.
""")

    report.add_model_setup(
        "| Object | Value | Role in the exercise |\n"
        "|--------|-------|----------------------|\n"
        f"| Firm-year panel | {df['Firm'].nunique()} firms, {df['Year'].nunique()} years | Lets input choices respond to persistent productivity |\n"
        "| Technology | Cobb-Douglas in labor, capital, materials | Gives known output elasticities for the benchmark |\n"
        "| True elasticities | $\\beta_l=0.32$, $\\beta_k=0.24$, $\\beta_m=0.44$ | Ground truth for the coefficient comparison |\n"
        "| Productivity | Persistent AR(1), observed by firms | Source of simultaneity in flexible inputs |\n"
        "| Proxy variable | Investment, monotone in productivity conditional on capital | Control for the unobserved productivity state |\n"
        "| Markup measure | $\\theta^m / \\alpha^m_{it}$ | Converts the materials elasticity into firm-year markups |"
    )

    report.add_solution_method(
        "The computation has two layers. First estimate the production elasticity of the "
        "variable input. Then divide that elasticity by each firm-year materials share. "
        "The proxy-control regression here uses the known synthetic investment schedule "
        "to form a noisy productivity control; a full Olley-Pakes or Levinsohn-Petrin "
        "application would estimate the nuisance policy and productivity law from data.\n\n"
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
        "The markup step is mechanically simple. The identification burden sits in the "
        "elasticity and in the assumption that materials are a flexible input with the "
        "right first-order condition."
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
        description="Why the production-function step matters for IO markups. OLS overstates "
        "the flexible-input elasticities because high-productivity firms choose more inputs. "
        "The proxy-control estimate is not a new economic object; it is a correction for the "
        "omitted productivity state, and in this simulation it moves the materials elasticity "
        "close to its true value.",
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
        description="Pushing the coefficient bias through the markup formula. The OLS-implied "
        "distribution is too far to the right because the materials elasticity is inflated. "
        "The proxy-control markups stay much closer to the truth, though they still inherit "
        "noise from the expenditure share and the proxy.",
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
        description="The simulated truth lets us check the markup gradient, not just the level. "
        "More productive firms have lower materials shares in this design, so true markups "
        "rise with productivity. The recovered quintile means trace that gradient fairly "
        "closely; the scatter is a reminder that firm-level markups are noisy even when the "
        "production elasticity is well estimated.",
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
        "crucial row because $\\theta^m$ is divided by the materials revenue share; an "
        "elasticity bias of this size would become a markup bias almost one-for-one.",
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
        "variable-input share behind it. In this controlled panel, correcting for productivity "
        "substantially reduces the markup error. In real IO work, the analogous scrutiny falls "
        "on the proxy monotonicity, the timing of input choices, and whether revenue data are "
        "being mistaken for physical output."
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
