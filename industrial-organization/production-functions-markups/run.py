#!/usr/bin/env python3
"""Production-function estimation and markup measurement."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


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
    beta_m = TRUE_BETA["Materials"]
    # Unit input and output prices, so the materials revenue share is
    # alpha_m = exp(log_materials) / exp(log_output).
    for t in range(n_years):
        omega = 0.72 * omega + rng.normal(0.0, 0.22, n_firms)
        capital = 0.82 * capital + 0.25 * omega + rng.normal(0.0, 0.18, n_firms) + 0.45
        labor = 1.05 + 0.10 * omega + 0.18 * capital + rng.normal(0.0, 0.22, n_firms)
        investment = 0.75 + 0.90 * omega + 0.20 * capital + rng.normal(0.0, 0.05, n_firms)
        eps = rng.normal(0.0, 0.08, n_firms)

        # The markup is the firm-year primitive. Cost minimisation makes
        # materials demand respond to it: the firm sets materials so the
        # planned materials revenue share equals beta_m / mu. With unit prices,
        # log materials and log expected output satisfy m = E[y] + log(beta_m / mu)
        # up to an optimisation error u. Materials is chosen on expected output,
        # before the output shock eps is realised.
        true_markup = np.clip(
            1.18 + 0.34 * (omega - omega.mean()) / omega.std()
            + rng.normal(0.0, 0.08, n_firms),
            0.85, 2.60,
        )
        log_share_target = np.log(beta_m / true_markup)
        opt_error = rng.normal(0.0, 0.12, n_firms)
        expected_y = (
            TRUE_BETA["Labor"] * labor
            + TRUE_BETA["Capital"] * capital
            + beta_m * (log_share_target + opt_error)
            + omega
        ) / (1.0 - beta_m)
        materials = expected_y + log_share_target + opt_error
        eps_y = (
            TRUE_BETA["Labor"] * labor
            + TRUE_BETA["Capital"] * capital
            + beta_m * materials
            + omega
            + eps
        )
        y = eps_y

        # material_share is read off the panel: materials expenditure over
        # revenue. With unit prices this is exp(log_materials) / exp(log_output).
        material_share = np.exp(materials - y)
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

    # Nonparametric proxy inversion. Investment is monotone in productivity
    # given capital, so the part of investment not explained by a polynomial
    # in capital is monotone in productivity. np.polyfit estimates that
    # polynomial from the data; the residual is the productivity control.
    # No true investment-schedule coefficient is used here.
    capital_trend = np.polyfit(k, inv, deg=2)
    omega_proxy = inv - np.polyval(capital_trend, k)
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
    save_figure(fig1, "figures/production-estimates.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(markups["true_markup"], bins=35, alpha=0.50, label="True", color="#54A24B")
    ax2.hist(markups["ols_markup"], bins=35, alpha=0.42, label="OLS-implied", color="#E45756")
    ax2.hist(markups["proxy_markup"], bins=35, alpha=0.55, label="Proxy-control", color="#4C78A8")
    ax2.set_xlabel("Markup")
    ax2.set_ylabel("Firm-year count")
    ax2.set_title("Markup distribution: true and recovered")
    ax2.legend()
    save_figure(fig2, "figures/markup-distribution.png", dpi=150)

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
    save_figure(fig3, "figures/productivity-markups.png", dpi=150)

    table = estimates.copy()
    table["OLS bias"] = table["OLS"] - table["True elasticity"]
    table["Proxy bias"] = table["Proxy-control"] - table["True elasticity"]
    for col in ["True elasticity", "OLS", "Proxy-control", "OLS bias", "Proxy bias"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    Path("tables/production-estimates.csv").parent.mkdir(parents=True, exist_ok=True)
    table.to_csv("tables/production-estimates.csv", index=False)

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
    Path("tables/markup-by-productivity.csv").parent.mkdir(parents=True, exist_ok=True)
    markup_table.to_csv("tables/markup-by-productivity.csv", index=False)

    save_thumbnail("figures/production-estimates.png", "figures/thumb.png")
    print("Figures and tables written.")


if __name__ == "__main__":
    main()
