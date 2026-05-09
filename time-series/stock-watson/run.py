#!/usr/bin/env python3
"""Macroeconomic forecasting with Stock-Watson diffusion indexes.

Builds a small, self-contained version of the Stock and Watson (2002)
many-predictor forecasting problem. A synthetic macro panel has a known latent
factor, so the report can compare principal-component estimates with the true
economic state and with a benchmark forecast that observes that state directly.

Reference: Stock, J. and Watson, M. (2002). "Forecasting Using Principal
Components from a Large Number of Predictors." JASA, 97(460), 1167-1179.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import eigh

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def generate_factor_panel(N=100, T=200, n_factors=1, seed=42):
    """Generate a synthetic panel with a known factor structure.

    Data generating process:
        X_it = lambda_i * F_t + e_it

    where F_t is an AR(1) common factor, lambda_i are factor loadings
    drawn from N(1, 0.5^2), and e_it ~ N(0, sigma_i^2) are idiosyncratic.

    Parameters
    ----------
    N : int
        Number of series (cross-section).
    T : int
        Number of time periods.
    n_factors : int
        Number of true common factors.
    seed : int
        Random seed.

    Returns
    -------
    X : array, shape (T, N)
        Observed panel data.
    F_true : array, shape (T, n_factors)
        True common factor(s).
    Lambda_true : array, shape (N, n_factors)
        True factor loadings.
    """
    rng = np.random.default_rng(seed)

    # Generate AR(1) factor: F_t = rho * F_{t-1} + eta_t
    rho_f = 0.8
    eta = rng.standard_normal((T, n_factors))
    F_true = np.zeros((T, n_factors))
    F_true[0] = eta[0]
    for t in range(1, T):
        F_true[t] = rho_f * F_true[t - 1] + eta[t]

    # Factor loadings: lambda_i ~ N(1, 0.5^2)
    Lambda_true = 1.0 + 0.5 * rng.standard_normal((N, n_factors))

    # Idiosyncratic errors: e_it ~ N(0, sigma_i^2), sigma_i ~ U(0.5, 1.5)
    sigma_e = 0.5 + rng.random(N)
    E = rng.standard_normal((T, N)) * sigma_e[None, :]

    # Observed data
    X = F_true @ Lambda_true.T + E

    return X, F_true, Lambda_true


def estimate_factors_pca(X, n_factors=1):
    """Estimate common factors via principal components analysis.

    Standardizes the panel, computes the cross-sectional covariance matrix,
    and extracts the top eigenvectors. The estimated factors are projections
    of the standardized panel onto those eigenvectors.

    Parameters
    ----------
    X : array, shape (T, N)
        Panel data (T periods, N series).
    n_factors : int
        Number of factors to extract.

    Returns
    -------
    F_hat : array, shape (T, n_factors)
        Estimated factors.
    Lambda_hat : array, shape (N, n_factors)
        Estimated loadings.
    eigenvalues : array, shape (N,)
        All eigenvalues of the correlation matrix (for scree plot).
    explained_var : array, shape (n_factors,)
        Fraction of variance explained by each factor.
    """
    T, N = X.shape

    # Standardize each series to zero mean and unit variance
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Z = (X - X_mean) / X_std

    # Eigendecomposition of (1/T) * Z'Z
    cov_matrix = Z.T @ Z / T
    eigenvalues, eigenvectors = eigh(cov_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Estimated loadings: top eigenvectors scaled by sqrt(eigenvalue)
    Lambda_hat = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    # Estimated factors: F_hat = Z @ Lambda_hat / N (projection)
    # Standard normalization: F_hat = Z @ V_r (where V_r are top eigenvectors)
    F_hat = Z @ eigenvectors[:, :n_factors]

    # Explained variance
    total_var = eigenvalues.sum()
    explained_var = eigenvalues[:n_factors] / total_var

    return F_hat, Lambda_hat, eigenvalues, explained_var


def factor_augmented_forecast(y, F_hat, p_ar=2, h=1):
    """Factor-augmented autoregressive (FAAR) forecast.

    Model: y_{t+h} = alpha + beta_1*y_t + ... + beta_p*y_{t-p+1} + gamma*F_t + e_t

    Compares:
    1. AR(p): only lagged y
    2. Factor AR(p): lagged y + supplied factor series

    Uses expanding-window out-of-sample evaluation.

    Parameters
    ----------
    y : array, shape (T,)
        Target series to forecast.
    F_hat : array, shape (T, n_factors)
        Estimated factors.
    p_ar : int
        AR lag order.
    h : int
        Forecast horizon.

    Returns
    -------
    results : dict with keys 'rmse_ar', 'rmse_faar', 'y_actual', 'y_ar', 'y_faar', 'eval_start'
    """
    T = len(y)
    n_factors = F_hat.shape[1]

    # Build regressor matrices
    # We need p_ar lags and h-step ahead target
    start = p_ar
    end = T - h

    # Target
    y_target = y[start + h: end + h]
    n_eval = len(y_target)

    # AR regressors
    X_ar = np.ones((end - start, p_ar + 1))
    for lag in range(p_ar):
        X_ar[:, lag + 1] = y[start - lag - 1: end - lag - 1]

    # FAAR regressors: AR + factors
    X_faar = np.ones((end - start, p_ar + 1 + n_factors))
    X_faar[:, :p_ar + 1] = X_ar
    X_faar[:, p_ar + 1:] = F_hat[start:end]

    # Expanding window evaluation (use first 60% for initial training)
    train_frac = 0.6
    n_train_init = int(train_frac * n_eval)

    y_pred_ar = np.zeros(n_eval - n_train_init)
    y_pred_faar = np.zeros(n_eval - n_train_init)
    y_actual = y_target[n_train_init:]

    for t in range(n_train_init, n_eval):
        # Train on 0:t, predict at t
        # AR
        X_tr_ar = X_ar[:t]
        y_tr = y_target[:t]
        beta_ar = np.linalg.lstsq(X_tr_ar, y_tr, rcond=None)[0]
        y_pred_ar[t - n_train_init] = X_ar[t] @ beta_ar

        # FAAR
        X_tr_faar = X_faar[:t]
        beta_faar = np.linalg.lstsq(X_tr_faar, y_tr, rcond=None)[0]
        y_pred_faar[t - n_train_init] = X_faar[t] @ beta_faar

    rmse_ar = np.sqrt(np.mean((y_actual - y_pred_ar) ** 2))
    rmse_faar = np.sqrt(np.mean((y_actual - y_pred_faar) ** 2))

    return {
        "rmse_ar": rmse_ar,
        "rmse_faar": rmse_faar,
        "y_actual": y_actual,
        "y_ar": y_pred_ar,
        "y_faar": y_pred_faar,
        "eval_start": n_train_init,
    }


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    N = 100          # Number of series
    T = 200          # Time periods
    n_factors = 1    # True number of factors
    p_ar = 2         # AR lag order for forecasting
    h = 1            # Forecast horizon

    # =========================================================================
    # Generate synthetic factor model data
    # =========================================================================
    print("Generating synthetic panel data (N=%d, T=%d)..." % (N, T))
    X, F_true, Lambda_true = generate_factor_panel(N=N, T=T, n_factors=n_factors, seed=42)
    print(f"  Panel shape: {X.shape}")
    print(f"  True factor shape: {F_true.shape}")
    print(f"  True loadings shape: {Lambda_true.shape}")

    # =========================================================================
    # Estimate factors via PCA
    # =========================================================================
    print("\nEstimating factors via PCA...")
    n_estimate = 5  # Extract top 5 for scree plot, use 1 for forecasting
    F_hat, _, eigenvalues, explained_var = estimate_factors_pca(X, n_factors=n_estimate)
    print(f"  Top 5 eigenvalues: {eigenvalues[:5].round(2)}")
    print(f"  Variance explained by first factor: {explained_var[0]:.1%}")

    # Use first estimated factor for comparison
    F_hat_1 = F_hat[:, 0:1]

    # Fix sign: align estimated factor with true factor
    corr_sign = np.sign(np.corrcoef(F_true[:, 0], F_hat[:, 0])[0, 1])
    F_hat_aligned = F_hat[:, 0] * corr_sign

    # Correlation between true and estimated factor
    factor_corr = np.corrcoef(F_true[:, 0], F_hat_aligned)[0, 1]
    print(f"  Correlation(F_true, F_hat): {factor_corr:.4f}")

    # =========================================================================
    # Forecasting comparison
    # =========================================================================
    print("\nRunning forecast evaluation...")
    # Use first series as target (representative macro variable)
    y_target = X[:, 0]
    forecast_results = factor_augmented_forecast(y_target, F_hat_1, p_ar=p_ar, h=h)
    true_factor_results = factor_augmented_forecast(y_target, F_true, p_ar=p_ar, h=h)
    print(f"  AR({p_ar}) RMSE:   {forecast_results['rmse_ar']:.4f}")
    print(f"  PCA factor RMSE:   {forecast_results['rmse_faar']:.4f}")
    print(f"  True factor RMSE:  {true_factor_results['rmse_faar']:.4f}")
    improvement = (1 - forecast_results['rmse_faar'] / forecast_results['rmse_ar']) * 100
    true_factor_improvement = (1 - true_factor_results['rmse_faar'] / forecast_results['rmse_ar']) * 100
    print(f"  PCA improvement: {improvement:.1f}%")
    print(f"  True factor improvement: {true_factor_improvement:.1f}%")

    # Standardized series-factor exposures are the scale-free analogue of
    # loadings. This avoids comparing arbitrary PCA normalizations directly.
    Z_panel = (X - X.mean(axis=0)) / X.std(axis=0)
    F_true_std = (F_true[:, 0] - F_true[:, 0].mean()) / F_true[:, 0].std()
    F_hat_std = (F_hat_aligned - F_hat_aligned.mean()) / F_hat_aligned.std()
    true_exposure = Z_panel.T @ F_true_std / T
    estimated_exposure = Z_panel.T @ F_hat_std / T
    exposure_corr = np.corrcoef(true_exposure, estimated_exposure)[0, 1]

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Macro Forecasting with Stock-Watson Diffusion Indexes",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A forecaster wants next month's industrial production. The data are monthly indicators "
        "such as employment and prices. Each series is noisy, but they move together over the "
        "business cycle.\n\n"
        "The object is a common macro factor. It is the shared state behind many observed "
        "indicators.\n\n"
        "The panel has 100 series and 200 months. A forecast regression cannot use every "
        "series directly. PCA estimates the factor, then a small AR forecast adds it to own "
        "lags."
    )

    report.add_equations(
        r"""
Let $X_t=(X_{1t},\ldots,X_{Nt})'$ collect the macro panel at date $t$. The
static factor model writes each indicator as common movement plus series noise:

$$X_{it}=\lambda_i'F_t+e_{it}, \qquad i=1,\ldots,N,\quad t=1,\ldots,T.$$

Here $F_t\in\mathbb{R}^r$ is the common macro factor. The loading
$\lambda_i\in\mathbb{R}^r$ measures exposure. The error $e_{it}$ is
series-specific noise. In this simulated panel, $r=1$ and

$$F_t=\rho_F F_{t-1}+\eta_t,\qquad \eta_t\sim N(0,1), \qquad \lambda_i\sim N(1,0.5^2), \qquad e_{it}\sim N(0,\sigma_{e,i}^2).$$

Each series is standardized before PCA:

$$Z_{it}=\frac{X_{it}-\bar X_i}{s_i}.$$

PCA uses the eigenvectors with the largest eigenvalues of $T^{-1}Z'Z$. The
estimated factor projects each date's standardized panel onto those directions:

$$\hat F_t=(Z_t'v_1,\ldots,Z_t'v_r)'.$$

Factors are identified only up to scale, sign, and rotation. The plots align
signs and compare standardized factors. The forecast regression adds the
estimated factor to own lags of a target series:

$$y_{t+h} =\alpha+\sum_{\ell=1}^{p}\beta_\ell y_{t-\ell+1} +\gamma'\hat F_t+\varepsilon_{t+h}.$$

The AR benchmark sets $\gamma=0$. A true-factor benchmark replaces $\hat F_t$
with the simulated $F_t$.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $N$ | {N} | Number of series (cross-section) |\n"
        f"| $T$ | {T} | Number of time periods |\n"
        f"| $r$ | {n_factors} | True number of factors |\n"
        f"| $\\rho_F$ | 0.8 | Factor AR(1) persistence |\n"
        f"| $\\lambda_i$ | $\\sim N(1, 0.25)$ | Factor loadings |\n"
        f"| $\\sigma_{{e,i}}$ | $\\sim U(0.5, 1.5)$ | Idiosyncratic std. deviations |\n"
        f"| AR lags ($p$) | {p_ar} | Lags in forecasting equation |\n"
        f"| Horizon ($h$) | {h} | Forecast horizon |\n"
        f"| Initial training share | 60% | Expanding-window forecast start |\n"
        f"| Target series | $X_{{1t}}$ | Representative observed macro variable |"
    )

    report.add_solution_method(
        "The computation has two steps. First, PCA estimates one common state from the "
        "standardized panel. Second, expanding-window regressions compare forecasts with and "
        "without that state.\n\n"
        "The wide panel supplies repeated signals about the same business-cycle movement. The "
        "leading component averages through series-specific noise.\n\n"
        "```text\n"
        "Algorithm: Stock-Watson diffusion-index forecast\n"
        "Inputs: panel X_it, target y_t, number of factors r, AR lag order p,\n"
        "        forecast horizon h, initial training share q\n"
        "Outputs: estimated factors Fhat_t, AR RMSE, PCA-factor RMSE, true-factor RMSE\n"
        "\n"
        "1. Standardize each series: Z_it = (X_it - mean_i) / sd_i.\n"
        "2. Form the cross-sectional covariance matrix S = T^{-1} Z'Z.\n"
        "3. Extract the r largest eigenvectors v_1,...,v_r of S.\n"
        "4. Set Fhat_t = (Z_t'v_1,...,Z_t'v_r) for each date t.\n"
        "5. For each expanding-window forecast origin tau:\n"
        "      fit AR(p): y_{t+h} on 1, y_t,...,y_{t-p+1}\n"
        "      fit factor AR(p): add Fhat_t to the same regression\n"
        "      fit true-factor AR(p): replace Fhat_t with the simulated F_t\n"
        "      record each h-step forecast error\n"
        "6. Compare RMSEs and cumulative squared errors over the evaluation window.\n"
        "```\n"
    )

    # --- Figure 1: True factor vs estimated factor ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    t_axis = np.arange(T)
    # Scale estimated factor to match true factor's scale for visual comparison
    scale = F_true[:, 0].std() / F_hat_aligned.std()
    F_hat_scaled = F_hat_aligned * scale
    ax1.plot(t_axis, F_true[:, 0], "b-", linewidth=1.5, label="True factor $F_t$", alpha=0.8)
    ax1.plot(t_axis, F_hat_scaled, "r--", linewidth=1.5, label="Estimated $\\hat{F}_t$ (PCA)", alpha=0.8)
    ax1.set_xlabel("Time period")
    ax1.set_ylabel("Factor value")
    ax1.set_title("Common macro state: true factor and PCA estimate")
    ax1.legend()
    report.add_figure("figures/factor-comparison.png",
                       f"True common factor vs PCA estimate (correlation = {factor_corr:.4f}). "
                       "PCA recovers the latent factor up to a scale normalization.", fig1,
                       description="The first plot checks whether PCA measured the simulated state. "
                       "Sign and scale are arbitrary, so the series are aligned before plotting. "
                       "The estimate tracks the latent AR(1) factor closely. "
                       f"The sample correlation is {factor_corr:.4f}.")

    # --- Figure 2: Scree plot ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    n_show = min(20, len(eigenvalues))
    ax2a.bar(range(1, n_show + 1), eigenvalues[:n_show], color="#2166ac", alpha=0.7)
    ax2a.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Kaiser criterion")
    ax2a.set_xlabel("Component number")
    ax2a.set_ylabel("Eigenvalue")
    ax2a.set_title("Eigenvalue drop")
    ax2a.legend()

    # Cumulative variance explained
    cum_var = np.cumsum(eigenvalues[:n_show]) / eigenvalues.sum()
    ax2b.plot(range(1, n_show + 1), cum_var * 100, "bo-", markersize=4)
    ax2b.axhline(90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="90% threshold")
    ax2b.set_xlabel("Number of components")
    ax2b.set_ylabel("Cumulative variance explained (%)")
    ax2b.set_title("Cumulative panel variance")
    ax2b.legend()
    fig2.tight_layout()
    report.add_figure("figures/scree-plot.png",
                       "Scree plot and cumulative variance explained. The sharp drop after the "
                       "first eigenvalue indicates one dominant factor.", fig2,
                       description="The scree plot checks factor count. "
                       f"PC1 explains {explained_var[0]:.1%} of standardized variance. "
                       "Later components look small in this controlled one-factor panel.")

    # --- Figure 3: Factor loadings ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sort_idx = np.argsort(true_exposure)
    ax3.scatter(range(N), true_exposure[sort_idx], s=25, alpha=0.6,
                color="#2166ac", label="True exposure $corr(X_i,F)$", zorder=3)
    ax3.scatter(range(N), estimated_exposure[sort_idx], s=25, alpha=0.6,
                color="#b2182b", marker="x",
                label="PCA exposure $corr(X_i,\\hat F)$", zorder=3)
    ax3.set_xlabel("Series (sorted by true exposure)")
    ax3.set_ylabel("Standardized exposure")
    ax3.set_title("Which series load on the common state?")
    ax3.legend()
    report.add_figure("figures/factor-loadings.png",
                       "Standardized series-factor exposures sorted by the true exposure.",
                       fig3,
                       description="The exposure plot shows which indicators carry the common state. "
                       "The PCA exposure ranking almost matches the true ranking. "
                       f"The correlation is {exposure_corr:.4f}.")

    # --- Figure 4: Forecast comparison ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    n_oos = len(forecast_results["y_actual"])
    t_oos = np.arange(n_oos)

    ax4a.plot(t_oos, forecast_results["y_actual"], "k-", linewidth=1.2,
              label="Actual", alpha=0.8)
    ax4a.plot(t_oos, forecast_results["y_ar"], "b--", linewidth=1.0,
              label=f"AR({p_ar})", alpha=0.7)
    ax4a.plot(t_oos, forecast_results["y_faar"], "r--", linewidth=1.0,
              label=f"PCA factor AR({p_ar})", alpha=0.7)
    ax4a.plot(t_oos, true_factor_results["y_faar"], color="#4d9221", linestyle=":", linewidth=1.2,
              label=f"True factor AR({p_ar})", alpha=0.8)
    ax4a.set_xlabel("Out-of-sample period")
    ax4a.set_ylabel("Value")
    ax4a.set_title("Forecasts of the target series")
    ax4a.legend()

    # Cumulative squared forecast errors
    cse_ar = np.cumsum((forecast_results["y_actual"] - forecast_results["y_ar"]) ** 2)
    cse_faar = np.cumsum((forecast_results["y_actual"] - forecast_results["y_faar"]) ** 2)
    cse_true_factor = np.cumsum((forecast_results["y_actual"] - true_factor_results["y_faar"]) ** 2)
    ax4b.plot(t_oos, cse_ar, "b-", linewidth=1.5, label=f"AR({p_ar})")
    ax4b.plot(t_oos, cse_faar, "r-", linewidth=1.5, label=f"PCA factor AR({p_ar})")
    ax4b.plot(t_oos, cse_true_factor, color="#4d9221", linestyle=":", linewidth=1.8,
              label=f"True factor AR({p_ar})")
    ax4b.set_xlabel("Out-of-sample period")
    ax4b.set_ylabel("Cumulative squared error")
    ax4b.set_title("Cumulative forecast loss")
    ax4b.legend()
    fig4.tight_layout()
    report.add_figure("figures/forecast-comparison.png",
                       f"Forecast comparison: the PCA factor forecast reduces RMSE by {improvement:.1f}% "
                       f"relative to AR({p_ar}). Right panel shows cumulative squared errors.", fig4,
                       description="The forecast plot compares one-step predictions. "
                       f"AR({p_ar}) uses only the target's own lags. "
                       "The Stock-Watson regression adds the estimated factor. "
                       f"RMSE falls from {forecast_results['rmse_ar']:.3f} to {forecast_results['rmse_faar']:.3f}. "
                       f"The true-factor forecast has RMSE {true_factor_results['rmse_faar']:.3f}.")

    # --- Tables ---
    # Eigenvalue table
    eig_table = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(5)],
        "Eigenvalue": eigenvalues[:5].round(3),
        "Var. Explained (%)": (eigenvalues[:5] / eigenvalues.sum() * 100).round(2),
        "Cumulative (%)": (np.cumsum(eigenvalues[:5]) / eigenvalues.sum() * 100).round(2),
    })
    report.add_table("tables/eigenvalues.csv",
                      "Top five eigenvalues and variance explained", eig_table,
                      description="The eigenvalue table repeats the scree evidence. "
                      "The large first eigenvalue is the simulated common factor. "
                      "The remaining entries mostly reflect series-specific variation.")

    # Forecast comparison table
    forecast_table = pd.DataFrame({
        "Model": [f"AR({p_ar})", f"PCA factor AR({p_ar})", f"True factor AR({p_ar})"],
        "RMSE": [
            forecast_results["rmse_ar"],
            forecast_results["rmse_faar"],
            true_factor_results["rmse_faar"],
        ],
        "Relative RMSE": [
            1.0,
            forecast_results["rmse_faar"] / forecast_results["rmse_ar"],
            true_factor_results["rmse_faar"] / forecast_results["rmse_ar"],
        ],
    }).round(4)
    report.add_table("tables/forecast-comparison.csv",
                      "Out-of-sample forecast comparison", forecast_table,
                      description="The forecast table reports the same loss comparison. "
                      f"The estimated factor and true factor both beat AR({p_ar}). "
                      "The close ordering should not be overinterpreted.")

    report.add_takeaway(
        "Stock-Watson diffusion indexes let a forecaster use many macro indicators without "
        "estimating one coefficient per series. In this run, PCA recovers the common state "
        f"almost exactly. The factor forecast lowers one-step RMSE by {improvement:.1f}% "
        f"relative to AR({p_ar}). The practical lesson is simple: estimate the shared state "
        "first, then forecast with a small regression."
    )

    report.add_references([
        "Stock, J. and Watson, M. (2002). \"Forecasting Using Principal Components from a Large Number of Predictors.\" *Journal of the American Statistical Association*, 97(460), 1167-1179.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
