#!/usr/bin/env python3
"""Stock-Watson Diffusion Index / Factor Model.

Implements the Stock and Watson (2002) approach to forecasting with many
predictors: extract a small number of common factors from a large panel of
macroeconomic series using principal components, then use these factors in
forecasting regressions.

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
from lib.plotting import setup_style, save_figure
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

    Standardizes the panel, computes the covariance matrix, and extracts
    the top eigenvectors. The estimated factors are sqrt(T) * eigenvectors
    of (1/T) X X' (the Bai-Ng normalization).

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
    2. FAAR(p): lagged y + estimated factors

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
    F_hat, Lambda_hat, eigenvalues, explained_var = estimate_factors_pca(X, n_factors=n_estimate)
    print(f"  Top 5 eigenvalues: {eigenvalues[:5].round(2)}")
    print(f"  Variance explained by first factor: {explained_var[0]:.1%}")

    # Use first estimated factor for comparison
    F_hat_1 = F_hat[:, 0:1]

    # Fix sign: align estimated factor with true factor
    corr_sign = np.sign(np.corrcoef(F_true[:, 0], F_hat[:, 0])[0, 1])
    F_hat_aligned = F_hat[:, 0] * corr_sign
    Lambda_hat_aligned = Lambda_hat[:, 0] * corr_sign

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
    print(f"  AR({p_ar}) RMSE:   {forecast_results['rmse_ar']:.4f}")
    print(f"  FAAR({p_ar}) RMSE:  {forecast_results['rmse_faar']:.4f}")
    improvement = (1 - forecast_results['rmse_faar'] / forecast_results['rmse_ar']) * 100
    print(f"  Improvement: {improvement:.1f}%")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Stock-Watson Diffusion Index / Factor Model",
        "Principal component estimation of common factors from a large panel, with "
        "factor-augmented forecasting.",
    )

    report.add_overview(
        "Stock and Watson (2002) showed that a small number of estimated factors, extracted "
        "via principal components from a large panel of macroeconomic time series, can "
        "substantially improve forecasts relative to standard autoregressive models.\n\n"
        "The key insight is that in a data-rich environment with $N$ series and $T$ periods, "
        "one can consistently estimate the latent common factors as $N, T \\to \\infty$, even "
        "though the factor loadings are unknown. This model demonstrates the approach with a "
        "synthetic panel where the true data generating process is known, allowing us to "
        "verify that PCA recovers the true factor."
    )

    report.add_equations(
        r"""
**Static Factor Model:**

$$X_{it} = \lambda_i' F_t + e_{it}, \qquad i = 1, \ldots, N, \quad t = 1, \ldots, T$$

where $F_t$ is a $r \times 1$ vector of common factors, $\lambda_i$ is the $r \times 1$
loading vector for series $i$, and $e_{it}$ is the idiosyncratic component.

**PCA Estimation (Bai and Ng, 2002):**

The estimated factors $\hat{F}$ are $\sqrt{T}$ times the eigenvectors corresponding to the
$r$ largest eigenvalues of the $T \times T$ matrix $(NT)^{-1} X X'$.

**Factor-Augmented Autoregression (FAAR):**

$$y_{t+h} = \alpha + \sum_{j=1}^{p} \beta_j y_{t-j+1} + \gamma' \hat{F}_t + \varepsilon_{t+h}$$
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
        f"| Horizon ($h$) | {h} | Forecast horizon |"
    )

    report.add_solution_method(
        "**Step 1 -- Standardization:** Each series is demeaned and scaled to unit variance.\n\n"
        "**Step 2 -- Eigendecomposition:** Compute the eigenvalues and eigenvectors of the "
        "$N \\times N$ sample covariance matrix $(1/T) Z'Z$ where $Z$ is the standardized "
        "panel. The estimated factors are the projections of the data onto the top "
        "eigenvectors.\n\n"
        "**Step 3 -- Forecasting:** Compare an AR(p) model (using only own lags) with a "
        "factor-augmented AR model (FAAR) that adds the estimated factor as a predictor. "
        "Evaluation uses an expanding-window out-of-sample exercise.\n\n"
        f"**Key result:** The first principal component explains **{explained_var[0]:.1%}** "
        f"of the total variance and has correlation **{factor_corr:.4f}** with the true "
        f"factor. The FAAR model achieves **{improvement:.1f}%** lower RMSE than the pure "
        f"AR({p_ar}) benchmark."
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
    ax1.set_title(f"True vs Estimated Factor (correlation = {factor_corr:.4f})")
    ax1.legend()
    report.add_figure("figures/factor-comparison.png",
                       f"True common factor vs PCA estimate (correlation = {factor_corr:.4f}). "
                       "PCA recovers the latent factor up to a scale normalization.", fig1)

    # --- Figure 2: Scree plot ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    n_show = min(20, len(eigenvalues))
    ax2a.bar(range(1, n_show + 1), eigenvalues[:n_show], color="#2166ac", alpha=0.7)
    ax2a.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Kaiser criterion")
    ax2a.set_xlabel("Component number")
    ax2a.set_ylabel("Eigenvalue")
    ax2a.set_title("Scree Plot")
    ax2a.legend()

    # Cumulative variance explained
    cum_var = np.cumsum(eigenvalues[:n_show]) / eigenvalues.sum()
    ax2b.plot(range(1, n_show + 1), cum_var * 100, "bo-", markersize=4)
    ax2b.axhline(90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="90% threshold")
    ax2b.set_xlabel("Number of components")
    ax2b.set_ylabel("Cumulative variance explained (%)")
    ax2b.set_title("Cumulative Explained Variance")
    ax2b.legend()
    fig2.tight_layout()
    report.add_figure("figures/scree-plot.png",
                       "Scree plot and cumulative variance explained. The sharp drop after the "
                       "first eigenvalue correctly indicates one dominant factor.", fig2)

    # --- Figure 3: Factor loadings ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sort_idx = np.argsort(Lambda_true[:, 0])
    ax3.scatter(range(N), Lambda_true[sort_idx, 0], s=25, alpha=0.6,
                color="#2166ac", label="True $\\lambda_i$", zorder=3)
    ax3.scatter(range(N), Lambda_hat_aligned[sort_idx] * (F_true[:, 0].std() / F_hat_aligned.std()),
                s=25, alpha=0.6, color="#b2182b", marker="x",
                label="Estimated $\\hat{\\lambda}_i$ (scaled)", zorder=3)
    ax3.set_xlabel("Series (sorted by true loading)")
    ax3.set_ylabel("Factor loading")
    ax3.set_title("Factor Loadings: True vs Estimated")
    ax3.legend()
    report.add_figure("figures/factor-loadings.png",
                       "Factor loadings sorted by true value. PCA estimates track the cross-sectional "
                       "pattern of true loadings.", fig3)

    # --- Figure 4: Forecast comparison ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    n_oos = len(forecast_results["y_actual"])
    t_oos = np.arange(n_oos)

    ax4a.plot(t_oos, forecast_results["y_actual"], "k-", linewidth=1.2,
              label="Actual", alpha=0.8)
    ax4a.plot(t_oos, forecast_results["y_ar"], "b--", linewidth=1.0,
              label=f"AR({p_ar})", alpha=0.7)
    ax4a.plot(t_oos, forecast_results["y_faar"], "r--", linewidth=1.0,
              label=f"FAAR({p_ar})", alpha=0.7)
    ax4a.set_xlabel("Out-of-sample period")
    ax4a.set_ylabel("Value")
    ax4a.set_title("Out-of-Sample Forecasts")
    ax4a.legend()

    # Cumulative squared forecast errors
    cse_ar = np.cumsum((forecast_results["y_actual"] - forecast_results["y_ar"]) ** 2)
    cse_faar = np.cumsum((forecast_results["y_actual"] - forecast_results["y_faar"]) ** 2)
    ax4b.plot(t_oos, cse_ar, "b-", linewidth=1.5, label=f"AR({p_ar})")
    ax4b.plot(t_oos, cse_faar, "r-", linewidth=1.5, label=f"FAAR({p_ar})")
    ax4b.set_xlabel("Out-of-sample period")
    ax4b.set_ylabel("Cumulative squared error")
    ax4b.set_title("Cumulative Forecast Error Comparison")
    ax4b.legend()
    fig4.tight_layout()
    report.add_figure("figures/forecast-comparison.png",
                       f"Forecast comparison: FAAR reduces RMSE by {improvement:.1f}% "
                       f"relative to AR({p_ar}). Right panel shows cumulative squared errors.", fig4)

    # --- Tables ---
    # Eigenvalue table
    eig_table = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(5)],
        "Eigenvalue": eigenvalues[:5].round(3),
        "Var. Explained (%)": (eigenvalues[:5] / eigenvalues.sum() * 100).round(2),
        "Cumulative (%)": (np.cumsum(eigenvalues[:5]) / eigenvalues.sum() * 100).round(2),
    })
    report.add_table("tables/eigenvalues.csv",
                      "Top 5 Eigenvalues and Variance Explained", eig_table)

    # Forecast comparison table
    forecast_table = pd.DataFrame({
        "Model": [f"AR({p_ar})", f"FAAR({p_ar})"],
        "RMSE": [forecast_results["rmse_ar"], forecast_results["rmse_faar"]],
        "Relative RMSE": [1.0, forecast_results["rmse_faar"] / forecast_results["rmse_ar"]],
    }).round(4)
    report.add_table("tables/forecast-comparison.csv",
                      "Out-of-Sample Forecast Comparison", forecast_table)

    report.add_takeaway(
        "The Stock-Watson diffusion index approach demonstrates a powerful principle: "
        "when many correlated time series share a common source of variation, principal "
        "components can extract this latent factor even with unknown loadings.\n\n"
        "**Key insights:**\n"
        f"- **Factor recovery:** With N={N} series and T={T} periods, PCA achieves a "
        f"correlation of {factor_corr:.4f} with the true factor. The Bai-Ng (2002) theory "
        f"guarantees consistency as min(N,T) grows.\n"
        f"- **Scree plot diagnostics:** The sharp drop after the first eigenvalue correctly "
        f"identifies the true number of factors (r=1). The first PC explains "
        f"{explained_var[0]:.1%} of total variance.\n"
        f"- **Forecast gains:** Adding the estimated factor to an AR({p_ar}) model reduces "
        f"RMSE by {improvement:.1f}%. This gain comes from the factor capturing "
        f"common movements that predict the target series.\n"
        "- **Practical implication:** Factor models are the workhorse for nowcasting and "
        "short-term macro forecasting at central banks. The approach scales naturally to "
        "hundreds of predictors without running into overfitting problems."
    )

    report.add_references([
        "Stock, J. and Watson, M. (2002). \"Forecasting Using Principal Components from a Large Number of Predictors.\" *Journal of the American Statistical Association*, 97(460), 1167-1179.",
        "Bai, J. and Ng, S. (2002). \"Determining the Number of Factors in Approximate Factor Models.\" *Econometrica*, 70(1), 191-221.",
        "Stock, J. and Watson, M. (2006). \"Forecasting with Many Predictors.\" *Handbook of Economic Forecasting*, Vol. 1, Ch. 10.",
        "Bai, J. (2003). \"Inferential Theory for Factor Models of Large Dimensions.\" *Econometrica*, 71(1), 135-171.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
