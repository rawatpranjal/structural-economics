#!/usr/bin/env python3
"""Policy-rate forecasting with ridge, lasso, and sparse shrinkage."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


CONCEPTS = ["inflation", "labor", "credit", "output", "stress"]
CONCEPT_LABELS = {
    "inflation": "Inflation",
    "labor": "Labor",
    "credit": "Credit",
    "output": "Output",
    "stress": "Financial stress",
}


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized data, means, and standard deviations."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x - mean) / scale, mean, scale


def apply_standardization(
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply training-set standardization to a new block."""
    return (x - mean) / scale


def soft_threshold(value: float, penalty: float) -> float:
    """Soft-thresholding map used by lasso coordinate descent."""
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def fit_least_squares(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit an intercept plus linear coefficients by least squares."""
    x_design = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(x_design, y, rcond=None)[0]
    return {"intercept": float(beta[0]), "coef": beta[1:]}


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> dict[str, np.ndarray | float]:
    """Fit ridge regression with an unpenalized intercept."""
    x_std, x_mean, x_scale = standardize(x)
    y_mean = float(y.mean())
    y_centered = y - y_mean
    n_obs, n_features = x_std.shape
    gram = x_std.T @ x_std
    coef_std = np.linalg.solve(
        gram + n_obs * penalty * np.eye(n_features),
        x_std.T @ y_centered,
    )
    coef = coef_std / x_scale
    intercept = y_mean - float(x_mean @ coef)
    return {"intercept": intercept, "coef": coef}


def fit_lasso(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float,
    max_iter: int = 5_000,
    tol: float = 1e-8,
) -> dict[str, np.ndarray | float]:
    """Fit lasso by cyclic coordinate descent with an unpenalized intercept."""
    x_std, x_mean, x_scale = standardize(x)
    y_mean = float(y.mean())
    y_centered = y - y_mean
    n_obs, n_features = x_std.shape
    coef_std = np.zeros(n_features)
    fitted = np.zeros(n_obs)
    column_norm = (x_std**2).mean(axis=0)

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(n_features):
            old = coef_std[j]
            residual_plus_j = y_centered - fitted + x_std[:, j] * old
            raw = float(x_std[:, j] @ residual_plus_j / n_obs)
            new = soft_threshold(raw, penalty) / column_norm[j]
            if new != old:
                fitted += x_std[:, j] * (new - old)
                coef_std[j] = new
                max_change = max(max_change, abs(new - old))
        if max_change < tol:
            break

    coef = coef_std / x_scale
    intercept = y_mean - float(x_mean @ coef)
    return {"intercept": intercept, "coef": coef}


def predict(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    """Predict from a fitted linear model."""
    return float(model["intercept"]) + x @ np.asarray(model["coef"])


def rmse(actual: np.ndarray, fitted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((actual - fitted) ** 2)))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Sample correlation with a zero-variance guard."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def simulate_policy_panel(
    meetings: int = 260,
    indicators_per_concept: int = 24,
    seed: int = 730,
) -> dict[str, np.ndarray | list[str] | list[str]]:
    """Simulate policy meetings, text-like indicators, and rate changes."""
    rng = np.random.default_rng(seed)
    n_concepts = len(CONCEPTS)
    concept_state = np.zeros((meetings, n_concepts))
    persistence = np.array([0.82, 0.76, 0.70, 0.78, 0.66])
    shock_scale = np.array([0.75, 0.70, 0.85, 0.72, 0.90])

    innovations = rng.normal(size=(meetings, n_concepts)) * shock_scale
    for t in range(1, meetings):
        spillover = 0.08 * concept_state[t - 1].mean()
        concept_state[t] = persistence * concept_state[t - 1] + spillover + innovations[t]

    predictors = []
    predictor_names = []
    predictor_groups = []
    for g, concept in enumerate(CONCEPTS):
        for k in range(indicators_per_concept):
            own_loading = rng.uniform(0.55, 1.25)
            common_loading = rng.normal(0.10, 0.05)
            cross_loading = rng.normal(0.05, 0.04, n_concepts)
            cross_loading[g] = 0.0
            measurement_noise = rng.normal(scale=rng.uniform(0.55, 1.10), size=meetings)
            series = (
                own_loading * concept_state[:, g]
                + common_loading * concept_state.mean(axis=1)
                + concept_state @ cross_loading
                + measurement_noise
            )
            predictors.append(series)
            predictor_names.append(f"{concept}_{k + 1:02d}")
            predictor_groups.append(concept)

    x_raw = np.column_stack(predictors)
    x, _, _ = standardize(x_raw)
    n_features = x.shape[1]

    beta = rng.normal(0.0, 0.008, n_features)
    concept_signs = {
        "inflation": 1.0,
        "labor": 0.75,
        "credit": -0.65,
        "output": 0.70,
        "stress": -0.85,
    }
    for g, concept in enumerate(CONCEPTS):
        sl = slice(g * indicators_per_concept, (g + 1) * indicators_per_concept)
        weak = concept_signs[concept] * rng.uniform(0.006, 0.018, indicators_per_concept)
        beta[sl] += weak
        strong_positions = [g * indicators_per_concept, g * indicators_per_concept + 1]
        beta[strong_positions[0]] += concept_signs[concept] * rng.uniform(0.09, 0.15)
        beta[strong_positions[1]] += concept_signs[concept] * rng.uniform(0.06, 0.11)

    policy_rate = np.zeros(meetings)
    delta_rate = np.zeros(meetings)
    systematic = np.zeros(meetings)
    true_shock = rng.normal(scale=0.20, size=meetings)
    for t in range(1, meetings):
        systematic[t] = -0.10 * policy_rate[t - 1] + float(x[t] @ beta)
        delta_rate[t] = systematic[t] + true_shock[t]
        policy_rate[t] = policy_rate[t - 1] + delta_rate[t]

    lagged_rate = policy_rate[:-1]
    design = np.column_stack([lagged_rate, x[1:]])
    feature_names = ["lagged_policy_rate"] + predictor_names
    feature_groups = ["policy"] + predictor_groups
    beta_design = np.concatenate([[-0.10], beta])
    systematic_target = systematic[1:]

    return {
        "design": design,
        "delta_rate": delta_rate[1:],
        "systematic": systematic_target,
        "shock": true_shock[1:],
        "policy_rate": policy_rate[1:],
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "beta": beta_design,
    }


def validation_path(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    penalties: np.ndarray,
    estimator: str,
) -> pd.DataFrame:
    """Evaluate ridge or lasso on a blocked validation set."""
    records = []
    for penalty in penalties:
        if estimator == "ridge":
            model = fit_ridge(x_train, y_train, float(penalty))
        elif estimator == "lasso":
            model = fit_lasso(x_train, y_train, float(penalty))
        else:
            raise ValueError(f"unknown estimator: {estimator}")
        y_hat = predict(model, x_valid)
        records.append(
            {
                "estimator": estimator,
                "penalty": float(penalty),
                "validation_rmse": rmse(y_valid, y_hat),
                "selected": int(np.sum(np.abs(np.asarray(model["coef"])[1:]) > 1e-6)),
            }
        )
    return pd.DataFrame(records)


def fit_forecast_models(
    data: dict[str, np.ndarray | list[str] | list[str]],
) -> dict[str, object]:
    """Select penalties on validation data and refit models for the test block."""
    x = np.asarray(data["design"])
    y = np.asarray(data["delta_rate"])
    systematic = np.asarray(data["systematic"])
    shock = np.asarray(data["shock"])
    beta = np.asarray(data["beta"])

    train_end = 125
    valid_end = 180
    x_train, y_train = x[:train_end], y[:train_end]
    x_valid, y_valid = x[train_end:valid_end], y[train_end:valid_end]
    x_fit, y_fit = x[:valid_end], y[:valid_end]
    x_test, y_test = x[valid_end:], y[valid_end:]
    systematic_test = systematic[valid_end:]
    shock_test = shock[valid_end:]

    ridge_penalties = np.logspace(-4, 1.0, 32)
    lasso_penalties = np.logspace(-3.3, -0.65, 32)
    ridge_curve = validation_path(
        x_train,
        y_train,
        x_valid,
        y_valid,
        ridge_penalties,
        "ridge",
    )
    lasso_curve = validation_path(
        x_train,
        y_train,
        x_valid,
        y_valid,
        lasso_penalties,
        "lasso",
    )

    ridge_penalty = float(
        ridge_curve.loc[ridge_curve["validation_rmse"].idxmin(), "penalty"]
    )
    lasso_penalty = float(
        lasso_curve.loc[lasso_curve["validation_rmse"].idxmin(), "penalty"]
    )

    lag_model = fit_least_squares(x_fit[:, :1], y_fit)
    ols_model = fit_least_squares(x_fit, y_fit)
    ridge_model = fit_ridge(x_fit, y_fit, ridge_penalty)
    lasso_model = fit_lasso(x_fit, y_fit, lasso_penalty)

    models = {
        "Lag-only": {
            "model": lag_model,
            "x_test": x_test[:, :1],
            "coef": np.concatenate(
                [
                    np.asarray(lag_model["coef"]),
                    np.zeros(x.shape[1] - 1),
                ]
            ),
            "penalty": np.nan,
        },
        "Wide OLS": {
            "model": ols_model,
            "x_test": x_test,
            "coef": np.asarray(ols_model["coef"]),
            "penalty": 0.0,
        },
        "Ridge": {
            "model": ridge_model,
            "x_test": x_test,
            "coef": np.asarray(ridge_model["coef"]),
            "penalty": ridge_penalty,
        },
        "Lasso": {
            "model": lasso_model,
            "x_test": x_test,
            "coef": np.asarray(lasso_model["coef"]),
            "penalty": lasso_penalty,
        },
    }

    predictions = {}
    shocks = {}
    metrics = []
    lag_rmse = None
    for name, entry in models.items():
        y_hat = predict(entry["model"], entry["x_test"])
        predictions[name] = y_hat
        shocks[name] = y_test - y_hat
        test_rmse = rmse(y_test, y_hat)
        if name == "Lag-only":
            lag_rmse = test_rmse
        metrics.append(
            {
                "Model": name,
                "Penalty": entry["penalty"],
                "Test RMSE": test_rmse,
                "Relative RMSE": test_rmse / lag_rmse if lag_rmse else 1.0,
                "Corr. with true systematic policy": correlation(y_hat, systematic_test),
                "Shock correlation": correlation(shocks[name], shock_test),
                "Selected indicators": int(np.sum(np.abs(entry["coef"][1:]) > 1e-6)),
            }
        )

    lasso_coef = np.asarray(models["Lasso"]["coef"])
    ridge_coef = np.asarray(models["Ridge"]["coef"])
    ols_coef = np.asarray(models["Wide OLS"]["coef"])
    true_text_beta = beta[1:]
    lasso_selected = np.abs(lasso_coef[1:]) > 1e-6
    true_nonzero = np.abs(true_text_beta) > 1e-10
    strong_cutoff = np.quantile(np.abs(true_text_beta), 0.90)
    dense_mask = true_nonzero & (np.abs(true_text_beta) < strong_cutoff)
    missed_dense_mass = np.sum(np.abs(true_text_beta[dense_mask & ~lasso_selected]))
    dense_mass = np.sum(np.abs(true_text_beta[dense_mask]))
    selection_summary = pd.DataFrame(
        [
            {
                "Statistic": "True nonzero policy indicators",
                "Value": int(np.sum(true_nonzero)),
            },
            {
                "Statistic": "Lasso-selected policy indicators",
                "Value": int(np.sum(lasso_selected)),
            },
            {
                "Statistic": "False inclusions by lasso",
                "Value": int(np.sum(lasso_selected & ~true_nonzero)),
            },
            {
                "Statistic": "True indicators missed by lasso",
                "Value": int(np.sum(true_nonzero & ~lasso_selected)),
            },
            {
                "Statistic": "Dense-signal share missed by lasso",
                "Value": missed_dense_mass / dense_mass,
            },
            {
                "Statistic": "Ridge coefficient correlation with truth",
                "Value": correlation(ridge_coef[1:], true_text_beta),
            },
            {
                "Statistic": "Lasso coefficient correlation with truth",
                "Value": correlation(lasso_coef[1:], true_text_beta),
            },
        ]
    )

    return {
        "x_test": x_test,
        "y_test": y_test,
        "systematic_test": systematic_test,
        "shock_test": shock_test,
        "predictions": predictions,
        "shocks": shocks,
        "metrics": pd.DataFrame(metrics),
        "selection_summary": selection_summary,
        "ridge_curve": ridge_curve,
        "lasso_curve": lasso_curve,
        "ridge_penalty": ridge_penalty,
        "lasso_penalty": lasso_penalty,
        "coef": {
            "true": beta,
            "ols": ols_coef,
            "ridge": ridge_coef,
            "lasso": lasso_coef,
        },
        "splits": {"train_end": train_end, "valid_end": valid_end, "n_obs": len(y)},
    }


def round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Round metrics for report display."""
    out = df.copy()
    out["Penalty"] = out["Penalty"].map(
        lambda value: "not tuned"
        if pd.isna(value)
        else ("0" if float(value) == 0.0 else f"{float(value):.4f}")
    )
    for col in ["Test RMSE", "Relative RMSE", "Corr. with true systematic policy", "Shock correlation"]:
        out[col] = out[col].astype(float).round(4)
    return out


def round_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Round mixed integer and decimal selection diagnostics."""
    out = df.copy()
    out["Value"] = out["Value"].map(
        lambda value: f"{value:.3f}" if isinstance(value, float) and not value.is_integer() else str(int(value))
    )
    return out


def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    meetings = 260
    indicators_per_concept = 24
    data = simulate_policy_panel(
        meetings=meetings,
        indicators_per_concept=indicators_per_concept,
    )
    results = fit_forecast_models(data)
    feature_names = list(data["feature_names"])
    feature_groups = list(data["feature_groups"])
    n_indicators = len(feature_names) - 1
    split = results["splits"]

    metrics_table = round_metrics(results["metrics"])
    selection_table = round_selection(results["selection_summary"])

    setup_style()
    report = ModelReport(
        "Policy Forecasting with Ridge, Lasso, and Sparsity",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A central bank moves the policy rate after reading a large flow of economic text. "
        "Inflation, labor, credit, output, and financial-stress language all contain signals. "
        "Each individual indicator is noisy.\n\n"
        "The economic object is the policy shock: the part of the rate change not predicted "
        "by the information set available at the meeting. A better forecast changes the "
        "measured shock series.\n\n"
        "The example simulates many correlated policy-concept indicators. A few signals are "
        "strong, but many weak signals also matter. That distinction is useful because a "
        "sparse selected model is not the same statement as a sparse economy."
    )

    report.add_equations(
        r"""
Let $r_t$ be the policy-rate level and let $\Delta r_t$ be the rate change at
meeting $t$. The information set contains a lagged policy rate and a vector
$x_t$ of standardized policy-concept indicators.

$$\Delta r_t = \phi r_{t-1} + x_t'\beta + u_t.$$

Here $\phi$ is the autoregressive coefficient on the lagged policy rate.

The systematic policy component is

$$m_t = \phi r_{t-1} + x_t'\beta,$$

and the policy shock is the residual

$$u_t = \Delta r_t - m_t.$$

The forecast uses a linear rule $f_t=b_0+z_t'b$, where
$z_t=(r_{t-1},x_t')'$. Ridge estimates the coefficients by

$$\hat b_{\mathrm{ridge}} = \arg\min_b \frac{1}{n}\sum_{t=1}^n (\Delta r_t-b_0-z_t'b)^2 +\lambda\sum_{j=1}^p b_j^2.$$

Lasso replaces the quadratic penalty with an absolute-value penalty.

$$\hat b_{\mathrm{lasso}} = \arg\min_b \frac{1}{n}\sum_{t=1}^n (\Delta r_t-b_0-z_t'b)^2 +\lambda\sum_{j=1}^p |b_j|.$$

The tuning parameter $\lambda$ is chosen on a blocked validation sample. Ridge
keeps many small correlated signals. Lasso can set coefficients exactly to
zero, so it produces a compressed selected set.
"""
    )

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        f"| Policy meetings | {meetings} | Synthetic rate-setting observations |\n"
        f"| Policy-concept groups | {len(CONCEPTS)} | Inflation, labor, credit, output, and financial stress |\n"
        f"| Indicators per group | {indicators_per_concept} | Noisy text-like signals per concept |\n"
        f"| Total indicators | {n_indicators} | Wide predictor block used by ridge and lasso |\n"
        f"| Training meetings | {split['train_end']} | First block used to tune penalties |\n"
        f"| Validation meetings | {split['valid_end'] - split['train_end']} | Middle block used to choose $\\lambda$ |\n"
        f"| Test meetings | {split['n_obs'] - split['valid_end']} | Final block used for reported forecast losses |\n"
        f"| True shock sd | 0.20 | Innovation in the policy rule |\n"
        f"| Ridge $\\lambda$ | {results['ridge_penalty']:.4f} | Validation-selected shrinkage |\n"
        f"| Lasso $\\lambda$ | {results['lasso_penalty']:.4f} | Validation-selected sparsity |"
    )

    report.add_solution_method(
        "The forecast exercise uses time blocks rather than random folds. The validation block "
        "comes after the training block, and the test block comes last. This keeps the tuning "
        "exercise close to a real policy-forecasting problem.\n\n"
        "Ridge has a closed-form penalized least-squares solution after centering and scaling "
        "the regressors. Lasso uses cyclic coordinate descent. The intercept is never penalized.\n\n"
        "```text\n"
        "Procedure: policy-shock measurement with shrinkage forecasts\n"
        "Inputs: rate changes Delta r_t, lagged rate r_{t-1}, indicators x_t\n"
        "Output: forecasts f_t and measured shocks e_t = Delta r_t - f_t\n\n"
        "1. Split meetings into training, validation, and test blocks.\n"
        "2. Fit the lag-only benchmark on the training-plus-validation block.\n"
        "3. For each ridge penalty lambda:\n"
        "       fit ridge on the training block and record validation RMSE.\n"
        "4. For each lasso penalty lambda:\n"
        "       fit lasso on the training block and record validation RMSE.\n"
        "5. Refit ridge and lasso on training-plus-validation data using selected lambdas.\n"
        "6. On the test block, compare forecast RMSE and residual-shock correlation.\n"
        "```"
    )

    y_test = results["y_test"]
    predictions = results["predictions"]
    shock_test = results["shock_test"]
    recovered_shocks = results["shocks"]
    test_axis = np.arange(len(y_test))

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 5))
    ax1a.plot(test_axis, y_test, color="black", linewidth=1.6, label="Actual rate change")
    ax1a.plot(test_axis, predictions["Lag-only"], color="#7570b3", linestyle="--", label="Lag-only")
    ax1a.plot(test_axis, predictions["Wide OLS"], color="#d95f02", linestyle=":", label="Wide OLS")
    ax1a.plot(test_axis, predictions["Ridge"], color="#1b9e77", label="Ridge")
    ax1a.plot(test_axis, predictions["Lasso"], color="#e7298a", linestyle="-.", label="Lasso")
    ax1a.axhline(0.0, color="black", linewidth=0.6)
    ax1a.set_xlabel("Test meeting")
    ax1a.set_ylabel("Policy-rate change")
    ax1a.set_title("Forecasts of policy-rate changes")
    ax1a.legend(fontsize=8)

    for name, color, linestyle in [
        ("Lag-only", "#7570b3", "--"),
        ("Wide OLS", "#d95f02", ":"),
        ("Ridge", "#1b9e77", "-"),
        ("Lasso", "#e7298a", "-."),
    ]:
        cse = np.cumsum((y_test - predictions[name]) ** 2)
        ax1b.plot(test_axis, cse, color=color, linestyle=linestyle, linewidth=2.0, label=name)
    ax1b.set_xlabel("Test meeting")
    ax1b.set_ylabel("Cumulative squared error")
    ax1b.set_title("Forecast loss over the test block")
    ax1b.legend(fontsize=8)
    fig1.tight_layout()
    ridge_rmse = float(metrics_table.loc[metrics_table["Model"] == "Ridge", "Test RMSE"].iloc[0])
    lag_rmse = float(metrics_table.loc[metrics_table["Model"] == "Lag-only", "Test RMSE"].iloc[0])
    report.add_figure(
        "figures/forecast-comparison.png",
        "Test-set policy-rate forecasts and cumulative squared errors",
        fig1,
        description=(
            "The forecast plot compares measured policy movements on the held-out meetings. "
            f"Ridge lowers RMSE from {lag_rmse:.3f} for the lag-only benchmark to {ridge_rmse:.3f}. "
            "Wide OLS has more freedom but pays for estimating many noisy coefficients."
        ),
    )

    coef = results["coef"]
    true_beta = coef["true"][1:]
    ridge_beta = coef["ridge"][1:]
    lasso_beta = coef["lasso"][1:]
    top = np.argsort(np.abs(true_beta))[::-1][:30]
    x_pos = np.arange(len(top))
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    width = 0.26
    ax2.bar(x_pos - width, true_beta[top], width=width, color="#4d4d4d", label="True")
    ax2.bar(x_pos, ridge_beta[top], width=width, color="#1b9e77", label="Ridge")
    ax2.bar(x_pos + width, lasso_beta[top], width=width, color="#e7298a", label="Lasso")
    ax2.axhline(0.0, color="black", linewidth=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([feature_names[i + 1].replace("_", " ") for i in top], rotation=60, ha="right")
    ax2.set_ylabel("Coefficient")
    ax2.set_title("Shrinkage on the strongest policy indicators")
    ax2.legend()
    fig2.tight_layout()
    lasso_selected = int(metrics_table.loc[metrics_table["Model"] == "Lasso", "Selected indicators"].iloc[0])
    report.add_figure(
        "figures/coefficient-shrinkage.png",
        "True, ridge, and lasso coefficients for the largest true signals",
        fig2,
        description=(
            "The coefficient plot sorts indicators by the true signal size. Ridge shrinks many "
            "correlated predictors toward zero without selecting a small subset. "
            f"Lasso selects {lasso_selected} indicators, so it compresses the rule more aggressively."
        ),
    )

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))
    ax3a.plot(test_axis, shock_test, color="black", linewidth=1.6, label="True shock")
    ax3a.plot(test_axis, recovered_shocks["Lag-only"], color="#7570b3", linestyle="--", label="Lag-only residual")
    ax3a.plot(test_axis, recovered_shocks["Ridge"], color="#1b9e77", label="Ridge residual")
    ax3a.plot(test_axis, recovered_shocks["Lasso"], color="#e7298a", linestyle="-.", label="Lasso residual")
    ax3a.axhline(0.0, color="black", linewidth=0.6)
    ax3a.set_xlabel("Test meeting")
    ax3a.set_ylabel("Policy shock")
    ax3a.set_title("Recovered residual shocks")
    ax3a.legend(fontsize=8)

    ax3b.scatter(shock_test, recovered_shocks["Lag-only"], color="#7570b3", alpha=0.55, label="Lag-only")
    ax3b.scatter(shock_test, recovered_shocks["Ridge"], color="#1b9e77", alpha=0.65, label="Ridge")
    ax3b.scatter(shock_test, recovered_shocks["Lasso"], color="#e7298a", alpha=0.65, label="Lasso")
    lim = max(np.max(np.abs(shock_test)), np.max(np.abs(recovered_shocks["Lag-only"]))) * 1.05
    ax3b.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.8, linestyle=":")
    ax3b.set_xlim(-lim, lim)
    ax3b.set_ylim(-lim, lim)
    ax3b.set_xlabel("True residual shock")
    ax3b.set_ylabel("Measured residual shock")
    ax3b.set_title("Shock measurement")
    ax3b.legend(fontsize=8)
    fig3.tight_layout()
    ridge_shock_corr = float(metrics_table.loc[metrics_table["Model"] == "Ridge", "Shock correlation"].iloc[0])
    report.add_figure(
        "figures/shock-recovery.png",
        "Recovered policy shocks against the true residual shocks",
        fig3,
        description=(
            "Policy shocks are residuals from the forecast rule. When the systematic component "
            f"is forecast better, the residual series lines up more closely with the true shock. "
            f"The ridge residual-shock correlation is {ridge_shock_corr:.3f}."
        ),
    )

    ridge_curve = results["ridge_curve"]
    lasso_curve = results["lasso_curve"]
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))
    ax4a.plot(ridge_curve["penalty"], ridge_curve["validation_rmse"], marker="o", markersize=3, color="#1b9e77")
    ax4a.axvline(results["ridge_penalty"], color="#1b9e77", linestyle="--", linewidth=1.2)
    ax4a.set_xscale("log")
    ax4a.set_xlabel("Ridge penalty")
    ax4a.set_ylabel("Validation RMSE")
    ax4a.set_title("Ridge validation curve")

    ax4b.plot(lasso_curve["penalty"], lasso_curve["validation_rmse"], marker="o", markersize=3, color="#e7298a")
    ax4b.axvline(results["lasso_penalty"], color="#e7298a", linestyle="--", linewidth=1.2)
    ax4b_twin = ax4b.twinx()
    ax4b_twin.plot(lasso_curve["penalty"], lasso_curve["selected"], color="#4d4d4d", linestyle=":", linewidth=1.5)
    ax4b.set_xscale("log")
    ax4b.set_xlabel("Lasso penalty")
    ax4b.set_ylabel("Validation RMSE")
    ax4b_twin.set_ylabel("Selected indicators")
    ax4b.set_title("Lasso validation and selected-set size")
    fig4.tight_layout()
    report.add_figure(
        "figures/validation-curves.png",
        "Blocked-validation curves over ridge and lasso penalty strengths",
        fig4,
        description=(
            "The validation curves show the tuning tradeoff. Low penalties fit many noisy "
            "coefficients. High penalties can underfit. For lasso, the selected-set size falls "
            "as the penalty rises."
        ),
    )

    report.add_table(
        "tables/forecast_metrics.csv",
        "Forecast and shock-measurement comparison",
        metrics_table,
        description=(
            "The forecast table reports test-block loss and residual-shock recovery. "
            "Relative RMSE divides each model's RMSE by the lag-only benchmark."
        ),
    )

    report.add_table(
        "tables/selection_summary.csv",
        "Coefficient and selection summary",
        selection_table,
        description=(
            "The selection table separates statistical selection from economic sparsity. "
            "The true rule contains many small nonzero indicators, so missed dense signal "
            "matters even when the selected model forecasts well. Note that the "
            "false-inclusion count is always zero by DGP construction: every one of "
            "the 120 indicators has a nonzero true coefficient, so any indicator lasso "
            "selects is true by construction. The zero reflects the DGP, not lasso "
            "precision, and should not be read as a measurement of lasso selectivity."
        ),
    )

    dense_share_value = float(results["selection_summary"].loc[
        selection_table["Statistic"] == "Dense-signal share missed by lasso",
        "Value",
    ].iloc[0])
    report.add_takeaway(
        "Ridge is useful when many weak correlated predictors contain real information. "
        "Lasso is useful when the researcher wants selection and compression. In this run, "
        f"lasso misses about {100.0 * dense_share_value:.1f}% of the weak dense signal while still producing a "
        "compact forecasting rule. Sparsity is therefore a modeling restriction, not an "
        "economic conclusion by itself."
    )

    report.write("README.md")
    print(f"Generated README.md with {len(report._figures)} figures and {len(report._tables)} tables.")


if __name__ == "__main__":
    main()
