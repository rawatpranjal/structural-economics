#!/usr/bin/env python3
"""Monetary policy SVARs with Minnesota-prior shrinkage."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


VARIABLES = ["Output gap", "Inflation", "Policy rate"]
SHORT_NAMES = ["y", "pi", "i"]
COLORS = {
    "Actual": "black",
    "OLS VAR": "#d95f02",
    "Minnesota BVAR": "#1b9e77",
}


@dataclass(frozen=True)
class MinnesotaHyperparameters:
    """Hyperparameters for a compact Minnesota prior."""

    own_persistence: float = 0.85
    overall_tightness: float = 0.18
    cross_tightness: float = 0.35
    lag_decay: float = 1.4
    intercept_sd: float = 3.0


def companion_from_lags(lag_matrices: list[np.ndarray]) -> np.ndarray:
    """Build the companion matrix for a VAR coefficient list."""
    p = len(lag_matrices)
    k = lag_matrices[0].shape[0]
    companion = np.zeros((k * p, k * p))
    companion[:k, :] = np.hstack(lag_matrices)
    if p > 1:
        companion[k:, :-k] = np.eye(k * (p - 1))
    return companion


def simulate_macro_panel(
    periods: int = 132,
    burn_in: int = 120,
    seed: int = 2027,
) -> dict[str, np.ndarray]:
    """Simulate output, inflation, and the policy rate from a stable VAR."""
    rng = np.random.default_rng(seed)
    k = len(VARIABLES)
    total = periods + burn_in
    y = np.zeros((total, k))

    a1 = np.array(
        [
            [0.67, 0.06, -0.16],
            [0.08, 0.62, -0.05],
            [0.13, 0.24, 0.70],
        ]
    )
    a2 = np.array(
        [
            [0.11, -0.03, -0.08],
            [0.03, 0.16, -0.04],
            [0.02, 0.06, -0.18],
        ]
    )
    impact = np.array(
        [
            [0.48, 0.00, 0.00],
            [0.12, 0.26, 0.00],
            [0.10, 0.11, 0.30],
        ]
    )

    structural_shocks = rng.normal(size=(total, k))
    reduced_shocks = structural_shocks @ impact.T
    for t in range(2, total):
        y[t] = a1 @ y[t - 1] + a2 @ y[t - 2] + reduced_shocks[t]

    sl = slice(burn_in, None)
    return {
        "series": y[sl],
        "structural_shocks": structural_shocks[sl],
        "impact": impact,
        "a1": a1,
        "a2": a2,
    }


def make_var_design(series: np.ndarray, lag_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return design matrix and targets for a VAR with an intercept."""
    rows = []
    targets = []
    for t in range(lag_order, len(series)):
        lagged = [series[t - lag] for lag in range(1, lag_order + 1)]
        rows.append(np.concatenate([[1.0], *lagged]))
        targets.append(series[t])
    return np.asarray(rows), np.asarray(targets)


def residual_covariance(residuals: np.ndarray, dof_adjustment: int = 0) -> np.ndarray:
    """Return a symmetric positive definite residual covariance estimate."""
    denom = max(residuals.shape[0] - dof_adjustment, 1)
    sigma = residuals.T @ residuals / denom
    sigma = 0.5 * (sigma + sigma.T)
    jitter = 1e-8 * np.trace(sigma) / sigma.shape[0]
    return sigma + jitter * np.eye(sigma.shape[0])


def fit_ols_var(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate an unrestricted reduced-form VAR by equation-by-equation OLS."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    sigma_u = residual_covariance(residuals, dof_adjustment=x.shape[1])
    return {"beta": beta, "sigma_u": sigma_u, "residuals": residuals}


def minnesota_prior(
    k: int,
    lag_order: int,
    residual_std: np.ndarray,
    hyper: MinnesotaHyperparameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return prior means and variances for each equation."""
    n_coef = 1 + k * lag_order
    prior_mean = np.zeros((n_coef, k))
    prior_var = np.zeros((n_coef, k))
    prior_var[0, :] = hyper.intercept_sd**2

    for equation in range(k):
        for lag in range(1, lag_order + 1):
            lag_shrink = hyper.overall_tightness / (lag**hyper.lag_decay)
            for variable in range(k):
                idx = 1 + (lag - 1) * k + variable
                scale_ratio = (residual_std[equation] / residual_std[variable]) ** 2
                cross_scale = 1.0 if equation == variable else hyper.cross_tightness**2
                prior_var[idx, equation] = max(
                    (lag_shrink**2) * scale_ratio * cross_scale,
                    1e-10,
                )

        own_first_lag = 1 + equation
        prior_mean[own_first_lag, equation] = hyper.own_persistence

    return prior_mean, prior_var


def fit_minnesota_bvar(
    x: np.ndarray,
    y: np.ndarray,
    lag_order: int,
    hyper: MinnesotaHyperparameters,
) -> dict[str, np.ndarray]:
    """Estimate posterior means and covariances of a Minnesota-prior BVAR."""
    ols = fit_ols_var(x, y)
    residual_std = np.sqrt(np.diag(ols["sigma_u"]))
    residual_std = np.maximum(residual_std, 1e-8)
    prior_mean, prior_var = minnesota_prior(
        y.shape[1],
        lag_order,
        residual_std,
        hyper,
    )

    beta = np.zeros_like(prior_mean)
    posterior_cov = np.zeros((y.shape[1], x.shape[1], x.shape[1]))
    xtx = x.T @ x
    for equation in range(y.shape[1]):
        sigma2 = residual_std[equation] ** 2
        prior_precision = np.diag(1.0 / prior_var[:, equation])
        lhs = xtx / sigma2 + prior_precision
        rhs = x.T @ y[:, equation] / sigma2 + prior_precision @ prior_mean[:, equation]
        cov_i = np.linalg.solve(lhs, np.eye(x.shape[1]))
        cov_i = 0.5 * (cov_i + cov_i.T)
        posterior_cov[equation] = cov_i
        beta[:, equation] = cov_i @ rhs

    residuals = y - x @ beta
    sigma_u = residual_covariance(residuals)
    return {
        "beta": beta,
        "sigma_u": sigma_u,
        "residuals": residuals,
        "prior_mean": prior_mean,
        "prior_var": prior_var,
        "posterior_cov": posterior_cov,
        "residual_std": residual_std,
    }


def beta_to_lag_matrices(beta: np.ndarray, lag_order: int, k: int) -> list[np.ndarray]:
    """Convert a stacked coefficient matrix into VAR lag matrices."""
    lag_matrices = []
    for lag in range(lag_order):
        block = beta[1 + lag * k : 1 + (lag + 1) * k, :]
        lag_matrices.append(block.T)
    return lag_matrices


def stability_radius(beta: np.ndarray, lag_order: int, k: int) -> float:
    """Return the largest absolute companion eigenvalue."""
    lag_matrices = beta_to_lag_matrices(beta, lag_order, k)
    roots = np.linalg.eigvals(companion_from_lags(lag_matrices))
    return float(np.max(np.abs(roots)))


def one_step_forecast(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute one-step-ahead VAR forecasts."""
    return x @ beta


def rmse_by_variable(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Root mean squared forecast error by variable."""
    return np.sqrt(np.mean((actual - forecast) ** 2, axis=0))


def recursive_irf(
    beta: np.ndarray,
    sigma_u: np.ndarray,
    lag_order: int,
    horizon: int,
    shock_index: int,
    target_impact: float,
) -> np.ndarray:
    """Compute recursively identified impulse responses to a scaled shock."""
    k = sigma_u.shape[0]
    lag_matrices = beta_to_lag_matrices(beta, lag_order, k)
    impact = np.linalg.cholesky(sigma_u)
    shock = impact[:, shock_index]
    scale = target_impact / shock[shock_index]
    shock = shock * scale

    responses = np.zeros((horizon + 1, k))
    responses[0] = shock
    for h in range(1, horizon + 1):
        for lag in range(1, lag_order + 1):
            if h - lag >= 0:
                responses[h] += lag_matrices[lag - 1] @ responses[h - lag]
    return responses


def build_prior_heatmap_data(
    prior_var: np.ndarray,
    lag_order: int,
    k: int,
) -> tuple[np.ndarray, list[str]]:
    """Return prior standard deviations and coefficient labels for plotting."""
    prior_sd = np.sqrt(prior_var[1:, :]).T
    labels = []
    for lag in range(1, lag_order + 1):
        labels.extend([f"{name}(t-{lag})" for name in SHORT_NAMES])
    return prior_sd, labels


def lag_coefficient_index(variable: int, lag: int, k: int) -> int:
    """Return the stacked VAR coefficient index for a lagged variable."""
    return 1 + (lag - 1) * k + variable


def selected_coefficient_posteriors(
    bvar: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize posterior means and intervals for key VAR coefficients."""
    selected = [
        ("Output persistence", 0, 0, 1),
        ("Inflation persistence", 1, 1, 1),
        ("Policy-rate persistence", 2, 2, 1),
        ("Policy rate effect on output", 0, 2, 1),
        ("Policy rate effect on inflation", 1, 2, 1),
    ]

    table_rows = []
    plot_rows = []
    for label, equation, variable, lag in selected:
        idx = lag_coefficient_index(variable, lag, len(VARIABLES))
        prior_mean = float(bvar["prior_mean"][idx, equation])
        posterior_mean = float(bvar["beta"][idx, equation])
        posterior_sd = float(
            np.sqrt(max(bvar["posterior_cov"][equation, idx, idx], 0.0))
        )
        lower = posterior_mean - 1.96 * posterior_sd
        upper = posterior_mean + 1.96 * posterior_sd
        regressor = f"{SHORT_NAMES[variable]}(t-{lag})"
        table_rows.append(
            {
                "Coefficient": label,
                "Equation": VARIABLES[equation],
                "Regressor": regressor,
                "Prior mean": format_float(prior_mean),
                "Posterior mean": format_float(posterior_mean),
                "Posterior sd": format_float(posterior_sd),
                "Approx 95% interval": f"[{lower:.3f}, {upper:.3f}]",
            }
        )
        plot_rows.append(
            {
                "Coefficient": label,
                "Prior mean": prior_mean,
                "Posterior mean": posterior_mean,
                "Lower": lower,
                "Upper": upper,
            }
        )

    return pd.DataFrame(table_rows), pd.DataFrame(plot_rows)


def format_float(value: float, digits: int = 3) -> str:
    """Format a float for Markdown tables."""
    return f"{value:.{digits}f}"


def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    periods = 132
    lag_order = 4
    train_obs = 88
    horizon = 20
    target_policy_impact = 0.25
    hyper = MinnesotaHyperparameters()

    data = simulate_macro_panel(periods=periods)
    series = data["series"]
    x_all, y_all = make_var_design(series, lag_order)
    x_train, y_train = x_all[:train_obs], y_all[:train_obs]
    x_test, y_test = x_all[train_obs:], y_all[train_obs:]
    test_quarters = np.arange(lag_order + train_obs, periods)

    ols = fit_ols_var(x_train, y_train)
    bvar = fit_minnesota_bvar(x_train, y_train, lag_order, hyper)

    ols_forecast = one_step_forecast(x_test, ols["beta"])
    bvar_forecast = one_step_forecast(x_test, bvar["beta"])
    ols_rmse = rmse_by_variable(y_test, ols_forecast)
    bvar_rmse = rmse_by_variable(y_test, bvar_forecast)
    overall_ols_rmse = float(np.sqrt(np.mean((y_test - ols_forecast) ** 2)))
    overall_bvar_rmse = float(np.sqrt(np.mean((y_test - bvar_forecast) ** 2)))

    ols_radius = stability_radius(ols["beta"], lag_order, len(VARIABLES))
    bvar_radius = stability_radius(bvar["beta"], lag_order, len(VARIABLES))
    shrinkage_ratio = float(
        np.linalg.norm(bvar["beta"][1:, :]) / np.linalg.norm(ols["beta"][1:, :])
    )

    ols_irf = recursive_irf(
        ols["beta"],
        ols["sigma_u"],
        lag_order,
        horizon,
        shock_index=2,
        target_impact=target_policy_impact,
    )
    bvar_irf = recursive_irf(
        bvar["beta"],
        bvar["sigma_u"],
        lag_order,
        horizon,
        shock_index=2,
        target_impact=target_policy_impact,
    )

    forecast_table = pd.DataFrame(
        [
            {
                "Variable": name,
                "OLS VAR RMSE": format_float(ols_rmse[i]),
                "Minnesota BVAR RMSE": format_float(bvar_rmse[i]),
                "BVAR / OLS": format_float(bvar_rmse[i] / ols_rmse[i]),
            }
            for i, name in enumerate(VARIABLES)
        ]
        + [
            {
                "Variable": "All variables",
                "OLS VAR RMSE": format_float(overall_ols_rmse),
                "Minnesota BVAR RMSE": format_float(overall_bvar_rmse),
                "BVAR / OLS": format_float(overall_bvar_rmse / overall_ols_rmse),
            }
        ]
    )

    hyper_table = pd.DataFrame(
        [
            {
                "Hyperparameter": "Own first-lag mean",
                "Value": format_float(hyper.own_persistence, 2),
                "Role": "Pulls each variable toward persistent own-lag dynamics",
            },
            {
                "Hyperparameter": "Overall tightness",
                "Value": format_float(hyper.overall_tightness, 2),
                "Role": "Controls how strongly coefficients are shrunk toward prior means",
            },
            {
                "Hyperparameter": "Cross-variable tightness",
                "Value": format_float(hyper.cross_tightness, 2),
                "Role": "Shrinks lags of other variables more tightly than own lags",
            },
            {
                "Hyperparameter": "Lag decay",
                "Value": format_float(hyper.lag_decay, 2),
                "Role": "Makes distant lags less important a priori",
            },
            {
                "Hyperparameter": "Intercept prior sd",
                "Value": format_float(hyper.intercept_sd, 1),
                "Role": "Leaves equation intercepts weakly regularized",
            },
        ]
    )

    coefficient_table, coefficient_plot_data = selected_coefficient_posteriors(bvar)

    identification_table = pd.DataFrame(
        [
            {
                "Object": "Reduced-form innovation",
                "Meaning": "Forecast error $u_t$ from the VAR",
                "Assumption": "Estimated covariance can be non-diagonal",
            },
            {
                "Object": "Structural shock",
                "Meaning": "Orthogonal shock $\\varepsilon_t$",
                "Assumption": "Unit variance and no cross-shock correlation",
            },
            {
                "Object": "Recursive ordering",
                "Meaning": "Output gap, inflation, then policy rate",
                "Assumption": "Policy can react within quarter to output and inflation",
            },
            {
                "Object": "Policy shock",
                "Meaning": "Third Cholesky innovation",
                "Assumption": "No impact effect on output or inflation",
            },
            {
                "Object": "Shock scale",
                "Meaning": "Impact rise in policy rate",
                "Assumption": f"Normalized to {target_policy_impact:.2f} rate points",
            },
        ]
    )

    irf_records = []
    for idx, name in enumerate(VARIABLES):
        if idx == 2:
            selected_idx = int(np.argmax(bvar_irf[:, idx]))
            response_type = "Peak"
        else:
            selected_idx = int(np.argmin(bvar_irf[:, idx]))
            response_type = "Trough"
        irf_records.append(
            {
                "Variable": name,
                "Impact": format_float(bvar_irf[0, idx]),
                "After 4 quarters": format_float(bvar_irf[4, idx]),
                "After 8 quarters": format_float(bvar_irf[8, idx]),
                "Selected response": format_float(bvar_irf[selected_idx, idx]),
                "Response type": response_type,
                "Quarter": selected_idx,
            }
        )
    irf_table = pd.DataFrame(irf_records)

    setup_style()
    report = ModelReport(
        "Monetary Policy SVARs with Minnesota Priors",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A quarterly macro VAR tries to summarize the joint dynamics of output, "
        "inflation, and the policy rate. The object of interest is a monetary "
        "policy shock: an unexpected policy tightening after current output and "
        "inflation have been accounted for.\n\n"
        "The problem is that even a small VAR can be noisy in a short macro sample. "
        "Four lags of three variables already give each equation thirteen "
        "coefficients including the intercept. Unrestricted OLS can fit accidental "
        "lag patterns and then produce unstable forecasts or impulse responses.\n\n"
        "The Minnesota prior is ridge-like shrinkage for dynamic systems. It puts "
        "prior mass on persistent own first lags, pulls most other coefficients "
        "toward zero, and tightens the prior for cross-variable and distant-lag "
        "effects. The shrinkage is soft, so the data can still move coefficients "
        "away from the prior when the sample is informative."
    )

    report.add_equations(
        r"""
Let $y_t=(x_t,\pi_t,i_t)'$ collect the output gap, inflation, and the policy
rate. The reduced-form VAR is

$$
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t,
\qquad u_t \sim N(0,\Sigma_u).
$$

Stack the observations equation by equation. Let $X$ contain an intercept and
the $p$ lagged values of $y_t$. For equation $i$,

$$
y_i = X\beta_i + e_i,
\qquad
e_i \sim N(0,\sigma_i^2 I_T).
$$

Conditional on the residual scale $\sigma_i^2$, the Gaussian likelihood is

$$
p(y_i \mid \beta_i,\sigma_i^2)
\propto
\exp\left[-\frac{1}{2\sigma_i^2}(y_i-X\beta_i)'(y_i-X\beta_i)
\right].
$$

The Minnesota prior is Gaussian:

$$
\beta_i \sim N(b_i^0,V_i^0).
$$

For variable $j$ and lag $\ell$, the prior mean is persistent only for the own
first lag:

$$
b_{i,j,\ell}^0 =
\begin{cases}
\rho_0, & i=j \ \mathrm{and}\ \ell=1,\\
0, & \mathrm{otherwise}.
\end{cases}
$$

The prior variance is larger for own lags and smaller for cross lags and distant
lags:

$$
v_{i,j,\ell}
= \left(\frac{\lambda}{\ell^d}\right)^2
\left(\frac{\sigma_i}{\sigma_j}\right)^2
\theta_{ij}^2,
\qquad
\theta_{ij}=1 \ \mathrm{for}\ i=j,\quad
\theta_{ij}=\theta \ \mathrm{for}\ i\ne j.
$$

This tutorial plugs in $\hat\sigma_i^2$ from OLS residuals. Conditional on that
plug-in scale, conjugacy gives a Gaussian posterior:

$$
V_i^{-1}
=
\frac{X'X}{\hat\sigma_i^2} + (V_i^0)^{-1}.
$$

$$
b_i
=
V_i\left(
\frac{X'y_i}{\hat\sigma_i^2} + (V_i^0)^{-1}b_i^0
\right).
$$

Coefficient uncertainty comes from the posterior covariance:

$$
\mathrm{sd}(\beta_{im}\mid y_i)
=
\sqrt{(V_i)_{mm}},
\qquad
\beta_{im}\approx b_{im}\pm 1.96\sqrt{(V_i)_{mm}}.
$$

Recursive SVAR identification factors the BVAR reduced-form covariance as

$$
\Sigma_u = PP',
\qquad
u_t=P\varepsilon_t,\qquad
E[\varepsilon_t\varepsilon_t']=I.
$$

This factorization is not unique. A recursive SVAR chooses the lower-triangular
Cholesky factor $P$ after fixing an ordering. With ordering output gap,
inflation, policy rate,

$$
\begin{bmatrix}
u_{y,t}\\
u_{\pi,t}\\
u_{i,t}
\end{bmatrix}
=
\begin{bmatrix}
p_{11} & 0 & 0\\
p_{21} & p_{22} & 0\\
p_{31} & p_{32} & p_{33}
\end{bmatrix}
\begin{bmatrix}
\varepsilon_{y,t}\\
\varepsilon_{\pi,t}\\
\varepsilon_{i,t}
\end{bmatrix}.
$$

The policy shock is the third structural innovation. It has zero impact effect
on output and inflation because the third column of $P$ is zero in those rows.
It can still affect output and inflation after one or more quarters through the
lag matrices. The policy rate can react on impact to output and inflation shocks
through $p_{31}$ and $p_{32}$.

The plotted shock is scaled to move the policy rate by $\tau=0.25$ on impact:

$$
q
=
\tau \frac{P e_3}{e_3'P e_3}.
$$

Impulse responses then propagate the scaled impact vector through the posterior
mean VAR dynamics:

$$
\psi_0=q,
\qquad
\psi_h=A_1\psi_{h-1}+A_2\psi_{h-2}+\cdots+A_p\psi_{h-p}.
$$
"""
    )

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        f"| Variables | {len(VARIABLES)} | Output gap, inflation, and policy rate |\n"
        f"| Simulated quarters | {periods} | Short macro panel after burn-in |\n"
        f"| VAR lag order $p$ | {lag_order} | Quarterly dynamics with one year of lags |\n"
        f"| Training observations | {train_obs} | Sample used to estimate each VAR |\n"
        f"| Test observations | {len(y_test)} | Held-out quarters for one-step forecasts |\n"
        f"| Coefficients per equation | {x_train.shape[1]} | Intercept plus lagged variables |\n"
        f"| Structural ordering | y, pi, i | Policy shock ordered last |\n"
        f"| Policy shock scale | {target_policy_impact:.2f} | Impact rise in the policy rate |"
    )

    report.add_solution_method(
        "The tutorial estimates the same reduced-form VAR in two ways. OLS treats "
        "all lag coefficients as free. The Minnesota BVAR treats the OLS residual "
        "scales as fixed, then computes the Gaussian posterior for each equation. "
        "That makes this an empirical-Bayes shrinkage estimator: posterior means "
        "and posterior covariance matrices are available without running an MCMC "
        "sampler.\n\n"
        "```text\n"
        "Procedure: Minnesota-prior monetary policy SVAR\n"
        "Inputs: quarterly series y_t, lag order p, prior hyperparameters\n"
        "Output: posterior coefficients, forecasts, and policy-shock impulse responses\n\n"
        "1. Build X from an intercept and p lags of output, inflation, and the rate.\n"
        "2. Fit the unrestricted OLS VAR and estimate residual scales sigma_i.\n"
        "3. Construct the Minnesota prior mean b_i^0 and diagonal covariance V_i^0.\n"
        "4. For each equation i:\n"
        "   precision_i <- X'X / sigma_i^2 + inv(V_i^0)\n"
        "   covariance_i <- inv(precision_i)\n"
        "   mean_i <- covariance_i * (X'y_i / sigma_i^2 + inv(V_i^0) b_i^0)\n"
        "5. Report selected posterior means and 1.96 posterior-sd intervals.\n"
        "6. Estimate the BVAR residual covariance and take its Cholesky factor P.\n"
        "7. Pick the third structural shock because policy is ordered last.\n"
        "8. Scale that shock to raise the policy rate by 25 bp on impact.\n"
        "9. Use posterior mean VAR coefficients to propagate the impulse responses.\n"
        "```"
    )

    quarters = np.arange(periods)
    train_boundary = lag_order + train_obs
    fig1, axes1 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for idx, ax in enumerate(axes1):
        ax.plot(quarters, series[:, idx], color="#2b2b2b", linewidth=1.6)
        ax.axvspan(train_boundary, periods - 1, color="#d9d9d9", alpha=0.35)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylabel(VARIABLES[idx])
        if idx == 0:
            ax.set_title("Simulated quarterly macro panel")
    axes1[-1].set_xlabel("Quarter")
    fig1.tight_layout()
    report.add_figure(
        "figures/simulated-macro-series.png",
        "Simulated output, inflation, and policy-rate series",
        fig1,
        description=(
            "The shaded region is the held-out forecast block. The data are "
            "stationary deviations from a macro steady state, so the policy rate "
            "should be read as a rate gap rather than a literal nominal level."
        ),
    )

    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for idx, ax in enumerate(axes2):
        ax.plot(
            test_quarters,
            y_test[:, idx],
            color=COLORS["Actual"],
            linewidth=1.7,
            label="Actual",
        )
        ax.plot(
            test_quarters,
            ols_forecast[:, idx],
            color=COLORS["OLS VAR"],
            linestyle=":",
            linewidth=1.8,
            label="OLS VAR",
        )
        ax.plot(
            test_quarters,
            bvar_forecast[:, idx],
            color=COLORS["Minnesota BVAR"],
            linewidth=1.8,
            label="Minnesota BVAR",
        )
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylabel(VARIABLES[idx])
        if idx == 0:
            ax.set_title("One-step forecasts on the held-out block")
            ax.legend(loc="upper right", fontsize=8)
    axes2[-1].set_xlabel("Quarter")
    fig2.tight_layout()
    report.add_figure(
        "figures/forecast-comparison.png",
        "OLS VAR and Minnesota BVAR one-step forecasts",
        fig2,
        description=(
            f"The Minnesota BVAR has an overall test RMSE of {overall_bvar_rmse:.3f}, "
            f"compared with {overall_ols_rmse:.3f} for the unrestricted OLS VAR. "
            "The gain comes from accepting small bias in exchange for lower "
            "coefficient variance."
        ),
    )

    horizons = np.arange(horizon + 1)
    fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for idx, ax in enumerate(axes3):
        ax.plot(
            horizons,
            ols_irf[:, idx],
            color=COLORS["OLS VAR"],
            linestyle=":",
            linewidth=2.0,
            label="OLS VAR",
        )
        ax.plot(
            horizons,
            bvar_irf[:, idx],
            color=COLORS["Minnesota BVAR"],
            linewidth=2.0,
            label="Minnesota BVAR",
        )
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_title(VARIABLES[idx])
        ax.set_xlabel("Quarters after shock")
        if idx == 0:
            ax.set_ylabel("Response")
            ax.legend(fontsize=8)
    fig3.suptitle("Responses to a 25 bp policy tightening", y=1.03)
    fig3.tight_layout()
    output_trough = float(np.min(bvar_irf[:, 0]))
    inflation_trough = float(np.min(bvar_irf[:, 1]))
    report.add_figure(
        "figures/policy-shock-irfs.png",
        "Impulse responses to a recursively identified policy-rate shock",
        fig3,
        description=(
            "The policy shock is ordered last, so output and inflation do not "
            "jump on impact. That zero-impact pattern is the recursive identifying "
            "restriction, not a coefficient estimate. The BVAR response is smoother "
            "than the OLS response. "
            f"In this run, output reaches {output_trough:.3f} and inflation reaches "
            f"{inflation_trough:.3f} after the tightening."
        ),
    )

    prior_sd, coef_labels = build_prior_heatmap_data(
        bvar["prior_var"],
        lag_order,
        len(VARIABLES),
    )
    fig4, ax4 = plt.subplots(figsize=(12, 4.5))
    image = ax4.imshow(prior_sd, aspect="auto", cmap="viridis")
    ax4.set_yticks(np.arange(len(VARIABLES)))
    ax4.set_yticklabels(VARIABLES)
    ax4.set_xticks(np.arange(len(coef_labels)))
    ax4.set_xticklabels(coef_labels, rotation=45, ha="right")
    ax4.set_title("Minnesota prior standard deviations")
    colorbar = fig4.colorbar(image, ax=ax4, fraction=0.035, pad=0.02)
    colorbar.set_label("Prior sd")
    fig4.tight_layout()
    report.add_figure(
        "figures/minnesota-prior-heatmap.png",
        "Prior standard deviations by equation and lagged regressor",
        fig4,
        description=(
            "The heatmap shows the prior before seeing the VAR coefficients. Own "
            "first lags have the loosest priors. Cross-variable lags and later "
            "lags are pulled more tightly toward zero."
        ),
    )

    coef_positions = np.arange(len(coefficient_plot_data))
    prior_means = coefficient_plot_data["Prior mean"].to_numpy()
    posterior_means = coefficient_plot_data["Posterior mean"].to_numpy()
    lower_bounds = coefficient_plot_data["Lower"].to_numpy()
    upper_bounds = coefficient_plot_data["Upper"].to_numpy()
    xerr = np.vstack(
        [posterior_means - lower_bounds, upper_bounds - posterior_means]
    )
    fig5, ax5 = plt.subplots(figsize=(10, 4.6))
    ax5.errorbar(
        posterior_means,
        coef_positions,
        xerr=xerr,
        fmt="o",
        color=COLORS["Minnesota BVAR"],
        ecolor=COLORS["Minnesota BVAR"],
        elinewidth=1.5,
        capsize=4,
        label="Posterior mean and approx 95% interval",
    )
    ax5.scatter(
        prior_means,
        coef_positions,
        marker="x",
        s=60,
        color="#7570b3",
        label="Prior mean",
    )
    ax5.axvline(0.0, color="black", linewidth=0.7)
    ax5.set_yticks(coef_positions)
    ax5.set_yticklabels(coefficient_plot_data["Coefficient"])
    ax5.invert_yaxis()
    ax5.set_xlabel("VAR coefficient")
    ax5.set_title("Selected coefficient posteriors")
    ax5.legend(loc="lower right", fontsize=8)
    fig5.tight_layout()
    report.add_figure(
        "figures/coefficient-posteriors.png",
        "Prior means, posterior means, and approximate intervals for selected coefficients",
        fig5,
        description=(
            "The coefficient posterior plot shows where the data move important "
            "VAR slopes relative to the Minnesota prior. The intervals condition "
            "on the OLS plug-in residual scales, so they are empirical-Bayes "
            "coefficient intervals rather than full posterior draws over all "
            "hyperparameters."
        ),
    )

    report.add_table(
        "tables/forecast-rmse.csv",
        "Forecast RMSE comparison",
        forecast_table,
        description=(
            "The RMSE table reports one-step forecast errors on quarters not used "
            "for estimation. Values below one in the ratio column favor the "
            "Minnesota BVAR."
        ),
    )
    report.add_table(
        "tables/prior-hyperparameters.csv",
        "Minnesota prior hyperparameters",
        hyper_table,
        description=(
            "These hyperparameters encode the economic belief that macro variables "
            "are persistent, but that distant and cross-variable lags should need "
            "strong evidence before receiving large coefficients."
        ),
    )
    report.add_table(
        "tables/coefficient-posteriors.csv",
        "Selected coefficient posterior summaries",
        coefficient_table,
        description=(
            "The posterior table reports the same selected slopes as the interval "
            "plot. Own first lags measure persistence, while the policy-rate "
            "slopes in the output and inflation equations summarize the first "
            "dynamic transmission channel."
        ),
    )
    report.add_table(
        "tables/shock-identification.csv",
        "Recursive shock-identification assumptions",
        identification_table,
        description=(
            "The identification table separates estimated VAR objects from the "
            "extra restrictions used to name a residual innovation as a monetary "
            "policy shock."
        ),
    )
    report.add_table(
        "tables/irf-summary.csv",
        "Selected BVAR policy-shock responses",
        irf_table,
        description=(
            "The impulse-response table summarizes the BVAR response to a policy "
            "shock scaled to raise the policy rate by 25 basis points on impact."
        ),
    )

    report.add_takeaway(
        "The Minnesota prior does not replace the VAR with a theory model. It "
        "regularizes the reduced form toward simple own-lag dynamics while still "
        "letting the data estimate monetary-policy transmission. The posterior "
        "coefficient intervals show which slopes are pulled close to the prior "
        "and which remain informed by the sample. The policy-shock responses also "
        "depend on the recursive ordering, so they should be read as conditional "
        "responses under that timing assumption. In a short macro sample, the "
        f"shrinkage lowers the coefficient norm to {shrinkage_ratio:.2f} of the "
        "OLS norm and gives smoother policy-shock responses. The stability radius "
        f"falls from {ols_radius:.2f} under OLS to {bvar_radius:.2f} under the "
        "Minnesota BVAR."
    )

    report.add_references(
        [
            'Doan, Thomas, Robert Litterman, and Christopher Sims (1984). "Forecasting and Conditional Projection Using Realistic Prior Distributions." Econometric Reviews, 3(1), 1-100.',
            'Litterman, Robert B. (1986). "Forecasting with Bayesian Vector Autoregressions: Five Years of Experience." Journal of Business & Economic Statistics, 4(1), 25-38.',
            'Banbura, Marta, Domenico Giannone, and Lucrezia Reichlin (2010). "Large Bayesian Vector Auto Regressions." Journal of Applied Econometrics, 25(1), 71-92.',
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
