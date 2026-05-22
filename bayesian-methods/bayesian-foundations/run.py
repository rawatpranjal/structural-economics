#!/usr/bin/env python3
"""Bayesian foundations: conjugate Beta-Binomial, Gaussian-Gaussian regression, and prior sensitivity.

Three minimal worked examples in one script.

Example A: Beta-Binomial coin flip. Three priors (uniform, conservative,
confident-wrong) are updated against Bernoulli draws from a true probability
p_star. The posterior is computed in closed form after 0, 10, 100, and 1000
observations. The figure shows posterior contraction onto p_star.

Example B: Gaussian-Gaussian conjugate linear regression. The posterior mean
is the precision-weighted average of the OLS estimate and the prior mean.
Posterior bands shrink as sample size grows.

Example C: Prior-sensitivity sweep on the Beta-Binomial. The data are fixed
and the prior strength is swept; the posterior mean interpolates from the
prior anchor to the sample fraction as the prior weakens.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


# =============================================================================
# Example A: Beta-Binomial with three priors and growing sample size
# =============================================================================
TRUE_P = 0.55
PRIORS = {
    "Uniform Beta(1, 1)": (1.0, 1.0),
    "Conservative Beta(20, 20)": (20.0, 20.0),
    "Confident-wrong Beta(40, 10)": (40.0, 10.0),
}
SAMPLE_SIZES = [0, 10, 100, 1000]
SEED_BB = 20260520


def beta_log_pdf(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """Log density of Beta(a, b) on the open unit interval."""
    log_norm = gammaln(a + b) - gammaln(a) - gammaln(b)
    return log_norm + (a - 1.0) * np.log(theta) + (b - 1.0) * np.log(1.0 - theta)


def beta_pdf(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.exp(beta_log_pdf(theta, a, b))


def beta_binomial_panel() -> dict:
    """Run Example A. Return draws and per-prior posterior parameters."""
    rng = np.random.default_rng(SEED_BB)
    draws = rng.binomial(1, TRUE_P, size=max(SAMPLE_SIZES))
    cumsum = np.cumsum(draws)

    results: dict = {}
    for label, (a0, b0) in PRIORS.items():
        rows = []
        for n in SAMPLE_SIZES:
            s = int(cumsum[n - 1]) if n > 0 else 0
            a_post = a0 + s
            b_post = b0 + n - s
            mean = a_post / (a_post + b_post)
            var = (a_post * b_post) / (
                (a_post + b_post) ** 2 * (a_post + b_post + 1)
            )
            rows.append(
                {
                    "Sample size": n,
                    "Successes": s,
                    "Posterior alpha": a_post,
                    "Posterior beta": b_post,
                    "Posterior mean": mean,
                    "Posterior variance": var,
                }
            )
        results[label] = pd.DataFrame(rows)
    return results


# =============================================================================
# Example B: Gaussian-Gaussian conjugate linear regression
# =============================================================================
TRUE_BETA = np.array([1.5, -0.8])
NOISE_SIGMA = 1.0
SAMPLE_GRID_B = [10, 50, 200, 1000]
PRIOR_MEAN_B = np.zeros(2)
PRIOR_PRECISION_B = (1.0 / 5.0**2) * np.eye(2)  # weakly informative
SEED_REG = 20260521


def make_design(n: int, rng: np.random.Generator) -> np.ndarray:
    """Build a regression design with intercept and one standard-normal covariate."""
    x = rng.normal(size=n)
    return np.column_stack([np.ones(n), x])


def conjugate_posterior(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Gaussian-Gaussian posterior precision and mean.

    Posterior precision V^{-1} = V0^{-1} + X'X / sigma^2.
    Posterior mean V (V0^{-1} m0 + X'y / sigma^2).
    """
    data_precision = (X.T @ X) / NOISE_SIGMA**2
    V_inv = PRIOR_PRECISION_B + data_precision
    V = np.linalg.inv(V_inv)
    rhs = PRIOR_PRECISION_B @ PRIOR_MEAN_B + (X.T @ y) / NOISE_SIGMA**2
    mean = V @ rhs
    return mean, V


def regression_panel() -> dict:
    rng = np.random.default_rng(SEED_REG)
    results = {}
    for n in SAMPLE_GRID_B:
        X = make_design(n, rng)
        y = X @ TRUE_BETA + NOISE_SIGMA * rng.normal(size=n)
        ols = np.linalg.solve(X.T @ X, X.T @ y)
        post_mean, post_cov = conjugate_posterior(X, y)
        results[n] = {
            "X": X,
            "y": y,
            "ols": ols,
            "post_mean": post_mean,
            "post_cov": post_cov,
        }
    return results


# =============================================================================
# Example C: Prior-sensitivity sweep (Beta-Binomial)
# =============================================================================
PRIOR_STRENGTHS = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
PRIOR_MEAN_C = 0.3  # anchor far from the data
DATA_N_C = 50
DATA_K_C = 32  # sample fraction 0.64


def prior_sensitivity_panel() -> pd.DataFrame:
    rows = []
    for strength in PRIOR_STRENGTHS:
        a0 = PRIOR_MEAN_C * strength
        b0 = (1.0 - PRIOR_MEAN_C) * strength
        a_post = a0 + DATA_K_C
        b_post = b0 + DATA_N_C - DATA_K_C
        post_mean = a_post / (a_post + b_post)
        lo, hi = beta_dist.ppf([0.025, 0.975], a_post, b_post)
        rows.append(
            {
                "Prior strength": strength,
                "Prior mean": PRIOR_MEAN_C,
                "Posterior mean": post_mean,
                "Lower 2.5%": lo,
                "Upper 97.5%": hi,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_beta_posteriors(panels: dict) -> None:
    theta_grid = np.linspace(1e-3, 1 - 1e-3, 600)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(panels), figsize=(13, 4.4), sharey=True
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(SAMPLE_SIZES)))
    for ax, (label, df) in zip(axes, panels.items()):
        for color, (_, row) in zip(colors, df.iterrows()):
            a_post = row["Posterior alpha"]
            b_post = row["Posterior beta"]
            n = int(row["Sample size"])
            pdf = beta_pdf(theta_grid, a_post, b_post)
            ax.plot(theta_grid, pdf, color=color, linewidth=1.8,
                    label=f"n = {n}")
        ax.axvline(TRUE_P, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"True p = {TRUE_P}")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"$\theta$")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Posterior density")
    fig.suptitle("Beta-Binomial posterior contraction under three priors")
    fig.tight_layout()
    save_figure(fig, "figures/beta-posteriors.png", dpi=150)


def plot_regression_bands(panels: dict) -> None:
    x_plot = np.linspace(-3.0, 3.0, 200)
    design_plot = np.column_stack([np.ones_like(x_plot), x_plot])
    fig, axes = plt.subplots(nrows=1, ncols=len(panels), figsize=(14, 4.4),
                              sharey=True)
    for ax, (n, payload) in zip(axes, panels.items()):
        mean = payload["post_mean"]
        cov = payload["post_cov"]
        y_pred = design_plot @ mean
        # Posterior predictive uncertainty in the regression line itself
        # (mean function only, excluding observation noise).
        line_var = np.einsum("ij,jk,ik->i", design_plot, cov, design_plot)
        line_sd = np.sqrt(line_var)
        ax.fill_between(
            x_plot,
            y_pred - 1.96 * line_sd,
            y_pred + 1.96 * line_sd,
            color="tab:blue",
            alpha=0.25,
            label="95% posterior band",
        )
        ax.plot(x_plot, y_pred, color="tab:blue", linewidth=2,
                label="Posterior mean")
        ax.plot(x_plot, design_plot @ TRUE_BETA, color="crimson",
                linestyle="--", linewidth=1.4, label="Truth")
        ax.scatter(payload["X"][:, 1], payload["y"], s=14, color="tab:gray",
                   alpha=0.5, label="Data")
        ax.set_xlim(-3.0, 3.0)
        ax.set_xlabel("Covariate")
        ax.set_title(f"n = {n}")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Response")
    fig.suptitle("Gaussian-Gaussian conjugate regression: posterior bands shrink with n")
    fig.tight_layout()
    save_figure(fig, "figures/gaussian-regression-bands.png", dpi=150)


def plot_prior_sensitivity(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.fill_between(
        df["Prior strength"],
        df["Lower 2.5%"],
        df["Upper 97.5%"],
        color="tab:blue",
        alpha=0.22,
        label="95% credible band",
    )
    ax.plot(df["Prior strength"], df["Posterior mean"], color="tab:blue",
            linewidth=2, label="Posterior mean")
    ax.axhline(PRIOR_MEAN_C, color="tab:gray", linestyle="--", linewidth=1.2,
               label=f"Prior mean = {PRIOR_MEAN_C}")
    ax.axhline(DATA_K_C / DATA_N_C, color="crimson", linestyle="--",
               linewidth=1.2, label=f"Sample fraction = {DATA_K_C/DATA_N_C:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"Prior strength $\alpha_0 + \beta_0$")
    ax.set_ylabel(r"Posterior mean of $\theta$")
    ax.set_title("Beta-Binomial prior sensitivity (50 trials, 32 successes)")
    ax.legend(loc="lower left")
    save_figure(fig, "figures/prior-sensitivity-fan.png", dpi=150)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    setup_style()
    np.seterr(divide="ignore")

    # Example A
    bb_panels = beta_binomial_panel()
    print("Example A: Beta-Binomial posterior contraction")
    for label, df in bb_panels.items():
        print(f"  {label}")
        for _, row in df.iterrows():
            print(
                f"    n={int(row['Sample size']):>4d}: "
                f"posterior Beta({row['Posterior alpha']:.1f}, "
                f"{row['Posterior beta']:.1f}), "
                f"mean={row['Posterior mean']:.4f}"
            )
    plot_beta_posteriors(bb_panels)

    # Example B
    reg_panels = regression_panel()
    print("\nExample B: Gaussian-Gaussian conjugate regression")
    reg_rows = []
    for n, payload in reg_panels.items():
        post_mean = payload["post_mean"]
        ols = payload["ols"]
        post_sd = np.sqrt(np.diag(payload["post_cov"]))
        print(
            f"  n={n:>4d}: posterior intercept={post_mean[0]:+.4f} "
            f"(sd {post_sd[0]:.4f}), slope={post_mean[1]:+.4f} "
            f"(sd {post_sd[1]:.4f}); OLS=({ols[0]:+.4f}, {ols[1]:+.4f})"
        )
        reg_rows.append(
            {
                "Sample size": n,
                "OLS intercept": ols[0],
                "OLS slope": ols[1],
                "Posterior intercept": post_mean[0],
                "Posterior slope": post_mean[1],
                "Intercept posterior sd": post_sd[0],
                "Slope posterior sd": post_sd[1],
            }
        )
    pd.DataFrame(reg_rows).to_csv("tables/regression-summary.csv", index=False)
    plot_regression_bands(reg_panels)

    # Example C
    sens_df = prior_sensitivity_panel()
    print("\nExample C: Prior-sensitivity sweep")
    for _, row in sens_df.iterrows():
        print(
            f"  strength={row['Prior strength']:>6.1f}: "
            f"posterior mean={row['Posterior mean']:.4f}, "
            f"95% band=[{row['Lower 2.5%']:.4f}, {row['Upper 97.5%']:.4f}]"
        )
    sens_df.to_csv("tables/prior-sensitivity.csv", index=False)
    plot_prior_sensitivity(sens_df)

    save_thumbnail("figures/beta-posteriors.png", "figures/thumb.png")
    print("\nGenerated: figures + tables + thumb")


if __name__ == "__main__":
    main()
