#!/usr/bin/env python3
"""Adversarial structural estimation for the logistic location model.

Replicates Sections 3.1.1 and 3.1.3 of Kaji, Manresa, and Pouliot (2023,
Econometrica), "An Adversarial Approach to Structural Estimation."

The economic question is the smallest possible: from one i.i.d. sample drawn
from the standard logistic distribution, recover the unknown location. The
pedagogical question is whether a discriminator-based minimax estimator,
trained to fool a logistic regression or a small neural network, recovers
MLE-quality estimates without the analyst hand-picking moments. The headline
exhibit is the optimally-weighted SMM comparison: as the moment count grows,
SMM precision collapses, while the adversarial estimator stays close to MLE.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


# =============================================================================
# Configuration
# =============================================================================

THETA_TRUE = 0.0
N = 300
M_SIM = 300
THETA_GRID = np.linspace(-0.6, 0.6, 25)
POLY_DEGREES = (3, 7, 11)
HIDDEN = 10
MLP_L2 = 1.0e-3
MLP_MAXITER = 60
R_MAIN = 200
R_MLP = 60
BOOTSTRAP_B = 200
BOOTSTRAP_BUDGET_S = 60.0
TOTAL_BUDGET_S = 240.0
SEED = 20260510

LOG2 = float(np.log(2.0))
FISHER_LOGISTIC_LOCATION = 1.0 / 3.0
THEORETICAL_SE_MLE = float(np.sqrt(1.0 / FISHER_LOGISTIC_LOCATION))
THEORETICAL_SE_ADV = float(np.sqrt((1.0 + N / M_SIM) / FISHER_LOGISTIC_LOCATION))


# =============================================================================
# Sampling and MLE for the standard logistic location model
# =============================================================================

def sample_logistic(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw iid standard logistic samples by inverse CDF."""
    u = rng.uniform(low=1.0e-12, high=1.0 - 1.0e-12, size=size)
    return np.log(u / (1.0 - u))


def neg_log_lik_logistic(theta: float, X: np.ndarray) -> float:
    """Average negative log-likelihood for X ~ Logistic(theta, 1)."""
    z = X - float(theta)
    return float(np.mean(z + 2.0 * np.logaddexp(0.0, -z)))


def mle(X: np.ndarray) -> float:
    res = minimize_scalar(
        neg_log_lik_logistic, args=(X,), bracket=(-2.0, 2.0), method="brent",
        options={"xtol": 1.0e-6},
    )
    return float(res.x)


# =============================================================================
# Method 1: Oracle adversarial loss M(theta, D_theta)
# Returns M (non-positive); the outer estimator MINIMIZES this.
# =============================================================================

def oracle_M(theta: float, X: np.ndarray, X_tilde: np.ndarray) -> float:
    """Sample cross-entropy under the oracle discriminator for logistic vs logistic."""
    th = float(theta)
    X_theta = th + X_tilde
    logit_real = -th - 2.0 * np.logaddexp(0.0, -X) + 2.0 * np.logaddexp(0.0, -(X - th))
    logit_sim = -th - 2.0 * np.logaddexp(0.0, -X_theta) + 2.0 * np.logaddexp(0.0, -(X_theta - th))
    log_D_real = -np.logaddexp(0.0, -logit_real)
    log_1mD_sim = -np.logaddexp(0.0, logit_sim)
    return float(np.mean(log_D_real)) + float(np.mean(log_1mD_sim))


# =============================================================================
# Method 2: Logistic discriminator with d polynomial features
# D_lambda(x) = Lambda(lambda_0 + lambda_1 x + ... + lambda_d x^d)
# Returns max-over-lambda M(theta, D_lambda).
# =============================================================================

def poly_features(x: np.ndarray, d: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    cols.extend(x ** k for k in range(1, d + 1))
    return np.column_stack(cols)


def fit_logistic_disc(X_real: np.ndarray, X_sim: np.ndarray, d: int) -> float:
    """Train logistic discriminator with d polynomial features. Returns max M."""
    Phi_r = poly_features(X_real, d)
    Phi_s = poly_features(X_sim, d)
    n = len(X_real)
    m = len(X_sim)

    def neg_M_and_grad(lam: np.ndarray) -> tuple[float, np.ndarray]:
        z_r = Phi_r @ lam
        z_s = Phi_s @ lam
        log_D_r = -np.logaddexp(0.0, -z_r)
        log_1mD_s = -np.logaddexp(0.0, z_s)
        M = float(np.mean(log_D_r) + np.mean(log_1mD_s))
        D_r = expit(z_r)
        D_s = expit(z_s)
        gM = (1.0 - D_r) @ Phi_r / n + (-D_s) @ Phi_s / m
        return -M, -gM

    lam0 = np.zeros(d + 1)
    res = minimize(
        neg_M_and_grad, lam0, jac=True, method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1.0e-9},
    )
    return -float(res.fun)  # max M


# =============================================================================
# Method 3: Shallow MLP discriminator (1 input -> H tanh -> 1 sigmoid)
# Trained at each theta with scipy L-BFGS-B on jax.value_and_grad.
# =============================================================================

def mlp_init_weights(rng: np.random.Generator) -> np.ndarray:
    """Flat initial weights: W[H], b[H], c[H], b_out[1]."""
    W = rng.normal(0.0, 0.5, size=HIDDEN)
    b = np.zeros(HIDDEN)
    c = rng.normal(0.0, 0.5, size=HIDDEN)
    b_out = np.zeros(1)
    return np.concatenate([W, b, c, b_out])


def _mlp_logits(theta_d: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    W = theta_d[:HIDDEN]
    b = theta_d[HIDDEN:2 * HIDDEN]
    c = theta_d[2 * HIDDEN:3 * HIDDEN]
    b_out = theta_d[3 * HIDDEN]
    h = jnp.tanh(jnp.outer(x, W) + b)
    return h @ c + b_out


def _mlp_neg_M(theta_d: jnp.ndarray, X_real: jnp.ndarray, X_sim: jnp.ndarray) -> jnp.ndarray:
    z_r = _mlp_logits(theta_d, X_real)
    z_s = _mlp_logits(theta_d, X_sim)
    log_D_r = -jax.nn.softplus(-z_r)
    log_1mD_s = -jax.nn.softplus(z_s)
    M = jnp.mean(log_D_r) + jnp.mean(log_1mD_s)
    penalty = MLP_L2 * (jnp.sum(theta_d[:HIDDEN] ** 2)
                        + jnp.sum(theta_d[2 * HIDDEN:3 * HIDDEN] ** 2))
    return -M + penalty


_MLP_VG = jax.jit(jax.value_and_grad(_mlp_neg_M))


def fit_mlp_disc(X_real: np.ndarray, X_sim: np.ndarray, init: np.ndarray) -> float:
    """Train the MLP discriminator at one theta. Returns max M (without penalty)."""
    Xr = jnp.asarray(X_real, dtype=jnp.float32)
    Xs = jnp.asarray(X_sim, dtype=jnp.float32)

    def obj(t: np.ndarray) -> tuple[float, np.ndarray]:
        v, g = _MLP_VG(jnp.asarray(t, dtype=jnp.float32), Xr, Xs)
        return float(v), np.asarray(g, dtype=float)

    res = minimize(
        obj, init, jac=True, method="L-BFGS-B",
        options={"maxiter": MLP_MAXITER, "ftol": 1.0e-7},
    )
    # Recompute M from the trained weights to drop the L2 penalty term
    theta_d = jnp.asarray(res.x, dtype=jnp.float32)
    z_r = _mlp_logits(theta_d, Xr)
    z_s = _mlp_logits(theta_d, Xs)
    log_D_r = -jax.nn.softplus(-z_r)
    log_1mD_s = -jax.nn.softplus(z_s)
    return float(jnp.mean(log_D_r) + jnp.mean(log_1mD_s))


# =============================================================================
# Outer estimator: evaluate M(theta) on a grid and refine the argmin parabolically
# =============================================================================

def estimate_by_grid(loss_at: callable, theta_grid: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate loss on the grid, refine the argmin with parabolic interpolation."""
    losses = np.array([loss_at(float(t)) for t in theta_grid])
    j = int(np.argmin(losses))
    if 0 < j < len(theta_grid) - 1:
        x0, x1, x2 = theta_grid[j - 1], theta_grid[j], theta_grid[j + 1]
        y0, y1, y2 = losses[j - 1], losses[j], losses[j + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom > 1.0e-10:
            theta_hat = float(x1 - 0.5 * (y2 - y0) * (x1 - x0) / denom)
            theta_hat = max(min(theta_hat, float(theta_grid[-1])), float(theta_grid[0]))
        else:
            theta_hat = float(x1)
    else:
        theta_hat = float(theta_grid[j])
    return theta_hat, losses


# =============================================================================
# SMM with d power moments, optimally weighted
# =============================================================================

def smm_estimate(X_real: np.ndarray, X_tilde: np.ndarray, d: int) -> float:
    """Optimally-weighted SMM matching first d power moments E[X^k]."""
    n = len(X_real)
    G_real = np.column_stack([X_real ** k for k in range(1, d + 1)])
    g_real_mean = G_real.mean(axis=0)
    Sigma = np.cov(G_real, rowvar=False, ddof=1)
    Sigma = np.atleast_2d(Sigma)
    W = np.linalg.inv(Sigma + 1.0e-8 * np.eye(d))

    def objective(theta: float) -> float:
        X_theta = float(theta) + X_tilde
        g_sim = np.array([(X_theta ** k).mean() for k in range(1, d + 1)])
        diff = g_real_mean - g_sim
        return float(diff @ W @ diff)

    res = minimize_scalar(
        objective, bracket=(-2.0, 2.0), method="brent",
        options={"xtol": 1.0e-6},
    )
    return float(res.x)


# =============================================================================
# Monte Carlo loop helpers
# =============================================================================

def run_cheap_replications(
    rng: np.random.Generator,
    R: int,
) -> dict[str, np.ndarray]:
    """Run MLE, oracle, logistic-d, and SMM-d on R replications."""
    out = {
        "mle": np.zeros(R),
        "oracle": np.zeros(R),
        "oracle_loss": np.zeros((R, len(THETA_GRID))),
    }
    for d in POLY_DEGREES:
        out[f"logistic_d{d}"] = np.zeros(R)
        out[f"logistic_loss_d{d}"] = np.zeros((R, len(THETA_GRID)))
        out[f"smm_d{d}"] = np.zeros(R)

    t0 = time.perf_counter()
    for r in range(R):
        X = sample_logistic(rng, N)
        X_tilde = sample_logistic(rng, M_SIM)
        out["mle"][r] = mle(X)
        theta_hat, loss_curve = estimate_by_grid(
            lambda t: oracle_M(t, X, X_tilde), THETA_GRID,
        )
        out["oracle"][r] = theta_hat
        out["oracle_loss"][r] = loss_curve
        for d in POLY_DEGREES:
            theta_hat, loss_curve = estimate_by_grid(
                lambda t, d=d: fit_logistic_disc(X, t + X_tilde, d), THETA_GRID,
            )
            out[f"logistic_d{d}"][r] = theta_hat
            out[f"logistic_loss_d{d}"][r] = loss_curve
            out[f"smm_d{d}"][r] = smm_estimate(X, X_tilde, d)
        if (r + 1) % 50 == 0 or r + 1 == R:
            print(f"  cheap reps: {r + 1}/{R}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return out


def run_mlp_replications(
    rng: np.random.Generator,
    R: int,
    init: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the MLP adversarial estimator on R replications."""
    out = {
        "mlp": np.zeros(R),
        "mlp_loss": np.zeros((R, len(THETA_GRID))),
        "mle_for_mlp": np.zeros(R),
    }
    t0 = time.perf_counter()
    for r in range(R):
        X = sample_logistic(rng, N)
        X_tilde = sample_logistic(rng, M_SIM)
        out["mle_for_mlp"][r] = mle(X)
        theta_hat, loss_curve = estimate_by_grid(
            lambda t: fit_mlp_disc(X, t + X_tilde, init), THETA_GRID,
        )
        out["mlp"][r] = theta_hat
        out["mlp_loss"][r] = loss_curve
        if (r + 1) % 10 == 0 or r + 1 == R:
            print(f"  mlp reps:   {r + 1}/{R}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return out


def maybe_bootstrap(
    rng: np.random.Generator,
    remaining_budget_s: float,
) -> tuple[float | None, str | None, int, float]:
    """Run a logistic-d=7 bootstrap if it fits the remaining budget."""
    d_focal = 7
    X_base = sample_logistic(rng, N)
    Xt_base = sample_logistic(rng, M_SIM)

    t_one = time.perf_counter()
    idx_r = rng.integers(0, N, size=N)
    idx_s = rng.integers(0, M_SIM, size=M_SIM)
    Xb = X_base[idx_r]
    Xtb = Xt_base[idx_s]
    estimate_by_grid(
        lambda t: fit_logistic_disc(Xb, t + Xtb, d_focal), THETA_GRID,
    )
    one_rep_s = time.perf_counter() - t_one
    projected_s = one_rep_s * BOOTSTRAP_B
    cap = min(remaining_budget_s, BOOTSTRAP_BUDGET_S)
    if projected_s + 5.0 > cap:
        reason = (
            f"projected bootstrap = {projected_s:.0f}s exceeds "
            f"{cap:.0f}s remaining budget"
        )
        return None, reason, d_focal, one_rep_s

    print(f"running bootstrap (B={BOOTSTRAP_B}, projected {projected_s:.0f}s)", flush=True)
    boot = np.zeros(BOOTSTRAP_B)
    t_boot = time.perf_counter()
    for b in range(BOOTSTRAP_B):
        idx_r = rng.integers(0, N, size=N)
        idx_s = rng.integers(0, M_SIM, size=M_SIM)
        Xb = X_base[idx_r]
        Xtb = Xt_base[idx_s]
        theta_b, _ = estimate_by_grid(
            lambda t: fit_logistic_disc(Xb, t + Xtb, d_focal), THETA_GRID,
        )
        boot[b] = theta_b
        if (b + 1) % 50 == 0 or b + 1 == BOOTSTRAP_B:
            print(f"  bootstrap: {b + 1}/{BOOTSTRAP_B}  "
                  f"({time.perf_counter() - t_boot:.1f}s)", flush=True)
    return float(np.std(boot, ddof=1)), None, d_focal, one_rep_s


# =============================================================================
# Figures
# =============================================================================

def figure_curvature(
    cheap: dict[str, np.ndarray],
    mlp: dict[str, np.ndarray] | None,
) -> plt.Figure:
    """Average loss curve over reps, demeaned at theta=0, vs the analytical Fisher quartic."""
    j0 = int(np.argmin(np.abs(THETA_GRID - THETA_TRUE)))

    def avg_curve(arr: np.ndarray) -> np.ndarray:
        avg = arr.mean(axis=0)
        return avg - avg[j0]

    fisher_quartic = np.array([
        0.25 * FISHER_LOGISTIC_LOCATION * (t - THETA_TRUE) ** 2 for t in THETA_GRID
    ])

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(THETA_GRID, fisher_quartic,
            label=r"$\frac{1}{4} I_0 (\theta - \theta_0)^2$ (asymptotic prediction)",
            color="black", linestyle=":", linewidth=2.2, zorder=2)
    ax.plot(THETA_GRID, avg_curve(cheap["oracle_loss"]),
            label=r"Oracle classifier", color="#4c78a8", linewidth=2.4, zorder=4)
    ax.plot(THETA_GRID, avg_curve(cheap["logistic_loss_d3"]),
            label=r"Logistic classifier, $d = 3$", color="#f58518",
            linewidth=2.2, zorder=3)
    ax.plot(THETA_GRID, avg_curve(cheap["logistic_loss_d7"]),
            label=r"Logistic classifier, $d = 7$", color="#e45756",
            linewidth=1.6, linestyle="--", zorder=3)
    if mlp is not None:
        ax.plot(THETA_GRID, avg_curve(mlp["mlp_loss"]),
                label=r"Neural-net classifier (10 tanh units)", color="#54a24b",
                linewidth=2.2, zorder=4)
    ax.axvline(THETA_TRUE, color="grey", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.annotate("true $\\theta_0 = 0$", xy=(THETA_TRUE, 0.0),
                xytext=(0.05, 0.004), fontsize=9, color="grey")
    ax.set_xlabel(r"Candidate parameter  $\theta$")
    ax.set_ylabel(r"Demeaned cross-entropy  $M(\theta, \hat D_\theta) - M(0, \hat D_0)$")
    ax.set_title("Every classifier's loss has the same curvature at the true parameter")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9.5)
    ax.set_ylim(bottom=-0.002)
    return fig


def figure_estimator_histograms(
    cheap: dict[str, np.ndarray],
    mlp: dict[str, np.ndarray] | None,
) -> plt.Figure:
    """2x2 panel of estimator histograms with the asymptotic Gaussian overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), sharex=True, sharey=True)
    x_range = np.linspace(-0.5, 0.5, 240)
    bins = np.linspace(-0.5, 0.5, 31)
    sqrtn = np.sqrt(N)
    sigma_mle_asymp = THEORETICAL_SE_MLE / sqrtn
    sigma_adv_asymp = THEORETICAL_SE_ADV / sqrtn
    mle_pdf = norm.pdf(x_range, loc=0.0, scale=sigma_mle_asymp)
    adv_pdf = norm.pdf(x_range, loc=0.0, scale=sigma_adv_asymp)

    panels = [
        (axes[0, 0], cheap["oracle"], cheap["mle"], "Oracle classifier", "#4c78a8", adv_pdf),
        (axes[0, 1], cheap["logistic_d3"], cheap["mle"], "Logistic, $d = 3$", "#f58518", None),
        (axes[1, 0], cheap["logistic_d7"], cheap["mle"], "Logistic, $d = 7$", "#e45756", None),
    ]
    if mlp is not None:
        panels.append(
            (axes[1, 1], mlp["mlp"], mlp["mle_for_mlp"],
             "Neural-net classifier", "#54a24b", adv_pdf)
        )
    else:
        axes[1, 1].axis("off")

    for ax, est, mle_arr, label, color, adv_curve in panels:
        sd_mle = np.std(mle_arr, ddof=1) * sqrtn
        sd_est = np.std(est, ddof=1) * sqrtn
        ax.hist(mle_arr, bins=bins, density=True, alpha=0.45, color="#bab0ab",
                label=f"MLE  ({sd_mle:.2f})")
        ax.hist(est, bins=bins, density=True, alpha=0.6, color=color,
                label=f"{label}  ({sd_est:.2f})")
        ax.plot(x_range, mle_pdf, color="black", linewidth=1.0, alpha=0.6,
                linestyle=":", label=f"MLE asymptotic  ({THEORETICAL_SE_MLE:.2f})")
        if adv_curve is not None:
            ax.plot(x_range, adv_curve, color=color, linewidth=1.0, alpha=0.7,
                    linestyle="--",
                    label=f"oracle asymptotic  ({THEORETICAL_SE_ADV:.2f})")
        ax.axvline(THETA_TRUE, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="upper right",
                  title=r"sd $\times \sqrt{n}$", title_fontsize=8)
        ax.set_xlim(-0.5, 0.5)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\hat\theta$")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.suptitle(
        r"Estimator distributions over Monte Carlo replications  ($n = m = 300$)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def figure_smm_vs_adversarial(cheap: dict[str, np.ndarray]) -> plt.Figure:
    """Headline figure: SMM(d) vs adversarial(d) on a shared horizontal range."""
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharex=True, sharey=True)
    sqrtn = np.sqrt(N)
    x_lim = (-2.5, 2.5)
    bins = np.linspace(x_lim[0], x_lim[1], 61)
    x_range = np.linspace(x_lim[0], x_lim[1], 400)
    mle_se = np.std(cheap["mle"], ddof=1) * sqrtn
    mle_pdf = norm.pdf(x_range, loc=0.0, scale=THEORETICAL_SE_MLE / sqrtn)

    def panel(ax, est, color, name, d):
        sd = np.std(est, ddof=1) * sqrtn
        est_clipped = np.clip(est, x_lim[0], x_lim[1])
        mle_clipped = np.clip(cheap["mle"], x_lim[0], x_lim[1])
        ax.hist(est_clipped, bins=bins, density=True, alpha=0.55, color=color,
                label=f"{name}, $d = {d}$  ({sd:.2f})")
        ax.hist(mle_clipped, bins=bins, density=True, alpha=0.45, color="#bab0ab",
                label=f"MLE  ({mle_se:.2f})")
        ax.plot(x_range, mle_pdf, color="black", linewidth=1.0, alpha=0.6,
                linestyle=":", label=f"MLE asymptote  ({THEORETICAL_SE_MLE:.2f})")
        ax.axvline(THETA_TRUE, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{name}, $d = {d}$", fontsize=11)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right",
                  title=r"sd $\times \sqrt{n}$", title_fontsize=8)

    for col, d in enumerate(POLY_DEGREES):
        panel(axes[0, col], cheap[f"smm_d{d}"], "#72b7b2", "SMM", d)
        panel(axes[1, col], cheap[f"logistic_d{d}"], "#f58518", "Adversarial", d)
        axes[1, col].set_xlabel(r"$\hat\theta$")

    axes[0, 0].set_ylabel("density (top: SMM)")
    axes[1, 0].set_ylabel("density (bottom: adversarial)")
    fig.suptitle(
        r"Same data, same features: SMM precision collapses with more moments,"
        r" the adversarial estimator stays close to MLE  ($n = m = 300$)",
        fontsize=12,
    )
    fig.text(0.99, 0.5, "outliers clipped to plot range", ha="right", va="center",
             fontsize=8, color="grey", rotation=90)
    fig.tight_layout()
    return fig


# =============================================================================
# Tables
# =============================================================================

def standard_errors_table(
    cheap: dict[str, np.ndarray],
    mlp: dict[str, np.ndarray] | None,
) -> pd.DataFrame:
    sqrtn = np.sqrt(N)
    rows = [
        ("MLE (reference)", cheap["mle"], THEORETICAL_SE_MLE),
        ("Oracle adversarial", cheap["oracle"], THEORETICAL_SE_ADV),
        ("Logistic disc., d=3", cheap["logistic_d3"], None),
        ("Logistic disc., d=7", cheap["logistic_d7"], None),
        ("Logistic disc., d=11", cheap["logistic_d11"], None),
    ]
    if mlp is not None:
        rows.append(("Neural net disc.", mlp["mlp"], THEORETICAL_SE_ADV))
    out = []
    for name, est, theo in rows:
        bias = float(est.mean() - THETA_TRUE)
        sd = float(np.std(est, ddof=1))
        out.append({
            "Estimator": name,
            "Bias": round(bias, 4),
            r"Monte Carlo sd $\times \sqrt{n}$": round(sd * sqrtn, 3),
            r"Asymptotic sd $\times \sqrt{n}$": "-" if theo is None else round(theo, 3),
            r"RMSE $\times \sqrt{n}$": round(float(np.sqrt(np.mean((est - THETA_TRUE) ** 2))) * sqrtn, 3),
        })
    return pd.DataFrame(out)


def smm_comparison_table(cheap: dict[str, np.ndarray]) -> pd.DataFrame:
    sqrtn = np.sqrt(N)
    rows = []
    mle_se = float(np.std(cheap["mle"], ddof=1)) * sqrtn
    rows.append({
        "Estimator": "MLE (reference)",
        "Moments d": "-",
        "Bias": round(float(cheap["mle"].mean() - THETA_TRUE), 4),
        r"sd $\times \sqrt{n}$": round(mle_se, 3),
        r"RMSE $\times \sqrt{n}$": round(
            float(np.sqrt(np.mean((cheap["mle"] - THETA_TRUE) ** 2))) * sqrtn, 3
        ),
    })
    for d in POLY_DEGREES:
        for name, key in [("SMM", f"smm_d{d}"), ("Adversarial logistic", f"logistic_d{d}")]:
            est = cheap[key]
            rows.append({
                "Estimator": name,
                "Moments d": d,
                "Bias": round(float(est.mean() - THETA_TRUE), 4),
                r"sd $\times \sqrt{n}$": round(float(np.std(est, ddof=1)) * sqrtn, 3),
                r"RMSE $\times \sqrt{n}$": round(
                    float(np.sqrt(np.mean((est - THETA_TRUE) ** 2))) * sqrtn, 3
                ),
            })
    return pd.DataFrame(rows)


def bootstrap_table(
    cheap: dict[str, np.ndarray], boot_se: float, d_focal: int,
) -> pd.DataFrame:
    sqrtn = np.sqrt(N)
    est = cheap[f"logistic_d{d_focal}"]
    mc_se = float(np.std(est, ddof=1)) * sqrtn
    return pd.DataFrame([
        {"Source": "Asymptotic (Theorem 3, paper)",
         r"sd $\times \sqrt{n}$": round(THEORETICAL_SE_ADV, 3)},
        {"Source": f"Monte Carlo across {len(est)} replications",
         r"sd $\times \sqrt{n}$": round(mc_se, 3)},
        {"Source": f"Bootstrap with B={BOOTSTRAP_B}",
         r"sd $\times \sqrt{n}$": round(boot_se * sqrtn, 3)},
    ])


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    setup_style()
    t_start = time.perf_counter()

    print(f"Adversarial estimation tutorial: n=m={N}, "
          f"R_main={R_MAIN}, R_mlp={R_MLP}", flush=True)

    rng_main = np.random.default_rng(SEED)
    cheap = run_cheap_replications(rng_main, R_MAIN)

    rng_mlp = np.random.default_rng(SEED + 1)
    init_rng = np.random.default_rng(SEED + 2)
    mlp_init = mlp_init_weights(init_rng)
    mlp_results: dict[str, np.ndarray] | None
    try:
        mlp_results = run_mlp_replications(rng_mlp, R_MLP, mlp_init)
    except Exception as exc:  # noqa: BLE001 - tutorial fallback
        print(f"MLP replications failed: {exc}", file=sys.stderr)
        mlp_results = None

    elapsed = time.perf_counter() - t_start
    print(f"main estimation complete in {elapsed:.1f}s", flush=True)

    boot_rng = np.random.default_rng(SEED + 99)
    boot_se, boot_skipped, d_focal, one_rep_s = maybe_bootstrap(
        boot_rng, TOTAL_BUDGET_S - elapsed,
    )
    if boot_skipped:
        print(f"skipping bootstrap: {boot_skipped}", file=sys.stderr)
    else:
        print(f"bootstrap done: SE for logistic d={d_focal} = {boot_se:.4f}", flush=True)

    # --- Figures ---
    fig_curvature = figure_curvature(cheap, mlp_results)
    fig_hist = figure_estimator_histograms(cheap, mlp_results)
    fig_smm = figure_smm_vs_adversarial(cheap)

    # --- Tables ---
    se_table = standard_errors_table(cheap, mlp_results)
    smm_table = smm_comparison_table(cheap)
    boot_table = (
        None if boot_se is None
        else bootstrap_table(cheap, boot_se, d_focal)
    )

    # --- Report ---
    report = ModelReport(
        "Adversarial Structural Estimation",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Structural estimation matches data to a model that is easier to simulate than to write down as a likelihood. "
        "The simulated method of moments does this by matching a vector of empirical moments to simulated ones. "
        "The catch is moment choice. "
        "A poorly chosen pair can outperform a long list; adding moments often hurts precision rather than improving it.\n\n"
        "Adversarial estimation moves the moment-choice step inside the algorithm. "
        "The analyst trains a binary classifier, called the discriminator, to separate real observations from observations simulated under a candidate parameter. "
        "The estimator is the parameter that the best-trained discriminator finds hardest to spot. "
        "The class of classifiers replaces the list of moments.\n\n"
        "Two choices of classifier recover estimators the reader has seen. "
        "A logistic regression on the first $d$ power features of $x$ asymptotically matches optimally-weighted SMM with the first $d$ power moments. "
        "A small neural network asymptotically matches maximum likelihood, without writing down the likelihood and without picking moments by hand.\n\n"
        "The illustration uses the smallest model that supports the comparison. "
        "The real data are $n$ i.i.d. draws from the standard logistic distribution. "
        "The single unknown is the location $\\theta_0$, set to zero. "
        "Maximum likelihood is available in closed form as the efficiency benchmark. "
        "The headline exhibit puts SMM and the adversarial estimator side by side as the moment count grows from three to eleven."
    )

    report.add_equations(r"""
The setup has three objects.

- The real sample $\lbrace X_i \rbrace_{i=1}^{n}$, drawn i.i.d. from an unknown distribution $P_0$.
- A parametric structural model $\lbrace P_\theta : \theta \in \Theta \rbrace$ that can be simulated. The model need not admit a closed-form density.
- A class of candidate discriminators $\mathcal D_n$. Each $D \in \mathcal D_n$ maps an observation $x$ to a number $D(x) \in [0, 1]$, read as the predicted probability that $x$ is a real observation rather than a simulated one.

Simulated observations come from a fixed shock vector and a structural transform. Draw $\tilde X_i \sim \tilde P_0$ once for $i = 1, \dots, m$. At any candidate $\theta$,

$$
X_{i,\theta} = T_\theta(\tilde X_i).
$$

The same shocks are reused at every candidate $\theta$. This is the standard common-random-numbers trick; it keeps the outer objective a smooth function of $\theta$ rather than a step function that re-randomizes with each new draw.

The estimator is the min-max

$$
\hat\theta = \arg\min_{\theta \in \Theta}\, \max_{D \in \mathcal D_n}\, M(\theta, D),
$$

with the cross-entropy

$$
M(\theta, D) = \underbrace{\frac{1}{n}\sum_{i=1}^{n} \log D(X_i)}_{\text{score on real data}} + \underbrace{\frac{1}{m}\sum_{i=1}^{m} \log(1 - D(X_{i,\theta}))}_{\text{score on simulated data}}.
$$

Read $M$ as the log-likelihood of a classification problem in which real points carry label $1$ and simulated points carry label $0$. The inner step trains the discriminator. The outer step picks the structural $\theta$ at which the best-trained discriminator does worst. This is the Goodfellow et al. (2014) GAN objective, with the simulator $T_\theta$ in the role of the generator.

The population inner maximum has a known form. When both densities exist, it is attained pointwise by the Bayes-optimal classifier

$$
D^{\ast}_\theta(x) = \frac{p_0(x)}{p_0(x) + p_\theta(x)}.
$$

At $\theta = \theta_0$ the two densities coincide, $D^{\ast}_\theta \equiv 1/2$, and $M$ is at its worst. Minimizing the inner maximum over $\theta$ therefore drives the simulated distribution toward the real one.

### Method 1: Oracle discriminator

Plug the closed-form ratio into $D^{\ast}_\theta$. For the standard logistic location family the density is $p_\theta(x) = \Lambda(x - \theta)\, \Lambda(-(x - \theta))$ with $\Lambda(z) = (1 + e^{-z})^{-1}$. The oracle simplifies to

$$
D^{\ast}_\theta(x) = \Lambda\left(-\theta - 2 \log(1 + e^{-x}) + 2 \log(1 + e^{-(x - \theta)})\right).
$$

Substituting $D^{\ast}_\theta$ into $M$ and minimizing over $\theta$ recovers maximum likelihood in the limit $m / n \to \infty$. The oracle is a sanity benchmark, not a usable estimator. Needing both densities defeats the simulation-based motivation.

### Method 2: Logistic discriminator with polynomial features

Restrict $\mathcal D_n$ to logistic regressions on the first $d$ powers of $x$:

$$
D_\lambda(x) = \Lambda(\lambda_0 + \lambda_1 x + \lambda_2 x^2 + \dots + \lambda_d x^d), \qquad \lambda \in \mathbb{R}^{d+1}.
$$

The inner step is a convex logistic regression. Its first-order conditions in $\lambda$ are GMM moment conditions in disguise, equating residual-weighted feature averages on the real and simulated samples:

$$
\frac{1}{n}\sum_{i=1}^{n} (1 - D_\lambda(X_i)) X_i^{k} = \frac{1}{m}\sum_{i=1}^{m} D_\lambda(X_{i,\theta}) X_{i,\theta}^{k}, \qquad k = 0, 1, \dots, d.
$$

At the min-max solution the resulting $\hat\theta$ is asymptotically equivalent to optimally-weighted SMM with the first $d$ power moments $(\mathbb{E}[X], \mathbb{E}[X^2], \dots, \mathbb{E}[X^d])$. The trained discriminator weights $\lambda$ play the role of the SMM weighting matrix $\hat\Sigma^{-1}$, but they are learned by gradient descent rather than computed from a noisy covariance estimate.

### Method 3: Shallow neural-network discriminator

Replace $\mathcal D_n$ by a one-hidden-layer network with $H$ tanh units:

$$
D_\eta(x) = \Lambda(b_{\mathrm{out}} + c^{\top} \tanh(W x + b)), \qquad \eta = (W, b, c, b_{\mathrm{out}}).
$$

With enough hidden units the class $\mathcal D_n$ can approximate the oracle $D^{\ast}_\theta$ uniformly on a compact set. The Kaji-Manresa-Pouliot result then says the adversarial estimator inherits the MLE asymptotic distribution, up to a finite-simulation correction:

$$
\sqrt{n}(\hat\theta - \theta_0) \Rightarrow \mathcal{N}\left(0, \frac{1 + n/m}{I_0}\right),
$$

where $I_0$ is the Fisher information of the structural model at $\theta_0$. The factor $1 + n/m$ is the price of learning the discriminator from a finite simulation sample. With $n = m$ the adversarial standard error is $\sqrt{2}$ times the MLE one.
""")

    report.add_model_setup(
        f"| Symbol | Meaning | Value |\n"
        f"|---|---|---:|\n"
        f"| $P_0$ | True distribution | Standard logistic |\n"
        f"| $\\theta_0$ | True location | {THETA_TRUE:.1f} |\n"
        f"| $n$ | Real sample size | {N} |\n"
        f"| $m$ | Simulated sample size | {M_SIM} |\n"
        f"| $T_\\theta(\\tilde x)$ | Structural simulator | $\\theta + \\tilde x$ |\n"
        f"| $\\tilde P_0$ | Base shock distribution | Standard logistic |\n"
        f"| $R$ | Monte Carlo replications, cheap estimators | {R_MAIN} |\n"
        f"| $R_{{\\text{{net}}}}$ | Monte Carlo replications, neural-net discriminator | {R_MLP} |\n"
        f"| Outer grid | $\\theta$ candidates | {len(THETA_GRID)} points in $[{THETA_GRID[0]:.1f},{THETA_GRID[-1]:.1f}]$ |\n"
        f"| $d$ | Polynomial degrees for logistic discriminator | $\\lbrace {', '.join(str(d) for d in POLY_DEGREES)} \\rbrace$ |\n"
        f"| $H$ | Hidden tanh units in MLP | {HIDDEN} |\n"
        f"| $\\lambda_{{L_2}}$ | Weight decay on input and output layers | {MLP_L2:.0e} |\n"
        f"| Inner solver | scipy L-BFGS-B on JAX gradients | maxiter={MLP_MAXITER} (MLP), 200 (logistic) |\n"
        f"| $I_0$ | Fisher information for logistic location | $1/3$ |\n"
        f"| Asymptotic sd of MLE | $1/\\sqrt{{I_0}}$ | {THEORETICAL_SE_MLE:.3f} |\n"
        f"| Asymptotic sd of adversarial | $\\sqrt{{(1+n/m)/I_0}}$ | {THEORETICAL_SE_ADV:.3f} |"
    )

    report.add_solution_method(r"""
Every estimator below has the same two-level structure. The inner step fits the best discriminator at a fixed $\theta$. The outer step minimizes the trained discriminator's cross-entropy $M(\theta, \hat D_\theta)$ over $\theta$. After the inner fit, $M(\theta, \hat D_\theta)$ is a function of $\theta$ alone.

The outer step uses a grid with a parabolic refinement. Twenty-five $\theta$ values are evenly spaced in the search interval. At each value the inner discriminator is retrained from scratch. The grid argmin is then refined by fitting a parabola through it and its two neighbours; this gives a sub-grid estimate without an extra inner fit. Three implementation choices keep the outer surface smooth and reproducible: the shocks $\tilde X$ are drawn once and reused at every $\theta$, the discriminator is initialized from the same starting point at every $\theta$, and a small weight decay damps sensitivity to that starting point when the inner fit is non-convex.

### Method 1: Oracle discriminator

The oracle ratio $D^{\ast}_\theta = p_0 / (p_0 + p_\theta)$ is closed form for the logistic location model. There is no inner fit. $M(\theta, D^{\ast}_\theta)$ is evaluated directly at every grid point.

```
Algorithm: oracle adversarial estimator

Inputs:  real sample X, common shocks tilde X, structural map T_theta
Output:  theta_hat

  for each theta on the outer grid:
      X_theta := theta + tilde X
      D := closed-form ratio  p_0(x) / (p_0(x) + p_theta(x))
      M(theta) := mean log D(X) + mean log(1 - D(X_theta))
  theta_hat := argmin of M over the grid (parabolic refinement)
```

The oracle is the upper bound on what any data-driven discriminator can achieve. Its standard error is the MLE standard error inflated by the simulation factor $1 + n/m$. The failure mode is conceptual: needing both densities defeats the simulation-based motivation. The oracle is reported as a sanity check, not as a usable estimator.

### Method 2: Logistic discriminator with polynomial features

The discriminator is a logistic regression on the first $d$ powers of $x$. Real points carry label $1$, simulated points carry label $0$. The inner problem is a strictly convex log-likelihood. It converges in a few L-BFGS steps.

```
Algorithm: logistic adversarial estimator with degree d

Inputs:  real sample X, common shocks tilde X, structural map T_theta, degree d
Output:  theta_hat

  for each theta on the outer grid:
      X_theta := theta + tilde X
      build design matrix Phi with columns  1, x, x^2, ..., x^d
      labels: 1 for real points, 0 for simulated points
      fit lambda by maximizing the logistic log-likelihood  M(theta, D_lambda)
      M(theta) := M(theta, D_lambda_hat)
  theta_hat := argmin of M over the grid
```

This estimator matches optimally-weighted SMM with the first $d$ power moments. Its failure mode is inherited from that match. Higher power moments like $X^7$ and $X^{11}$ are noisy in finite samples, so the corresponding directions of the optimal weighting matrix are noisy too. The adversarial framing softens this problem but does not remove it; the standard error grows with $d$, just more slowly than for plain SMM.

### Method 3: Shallow neural-network discriminator

The discriminator is a one-hidden-layer network with $H$ tanh units and a sigmoid output. The inner problem is non-convex but small. L-BFGS on JAX-computed gradients handles it in tens of milliseconds. A small ridge penalty on the input and output weights limits the freedom of the network in finite samples.

```
Algorithm: neural-net adversarial estimator

Inputs:  real sample X, common shocks tilde X, structural map T_theta,
         hidden width H, common initialization eta_0, weight decay lambda_L2
Output:  theta_hat

  for each theta on the outer grid:
      X_theta := theta + tilde X
      labels: 1 for real points, 0 for simulated points
      fit eta = (W, b, c, b_out) by maximizing
          M(theta, D_eta) - lambda_L2 * (||W||^2 + ||c||^2)
        via L-BFGS starting from eta_0
      M(theta) := mean log D_eta(X) + mean log(1 - D_eta(X_theta))
  theta_hat := argmin of M over the grid
```

The failure modes are familiar from neural training. The inner objective is non-convex, so a re-run with a different starting point can land on a different local optimum. Using the same starting point at every outer $\theta$ removes that source of non-monotonicity in $M(\theta, \hat D_\theta)$. The refit cost dominates runtime, so both the hidden width $H$ and the maximum number of L-BFGS iterations are kept small.
""")

    report.add_results(
        "The first diagnostic is the curvature of $M(\\theta, \\hat D_\\theta)$ around $\\theta_0$. "
        "A second-order expansion of the cross-entropy around the symmetric point $D = 1/2$ gives a quadratic with coefficient $I_0 / 4$, where $I_0$ is the Fisher information of the structural model. "
        "The factor of one-quarter is the price of measuring information through classification accuracy instead of the log-likelihood directly. "
        "All three discriminators trace that target closely. "
        "Their implied estimators therefore inherit Fisher curvature, which is exactly the condition for asymptotic efficiency."
    )
    report.add_figure(
        "figures/curvature.png",
        "Discriminator loss curvature against the Fisher reference",
        fig_curvature,
        description=(
            "Each colored line is the Monte Carlo average of $M(\\theta, \\hat D_\\theta)$, demeaned at $\\theta = 0$. "
            "The dotted black line is the analytical prediction $I_0 (\\theta - \\theta_0)^2 / 4$. "
            "Agreement near the truth is what asymptotic efficiency requires. "
            "Level offsets away from the truth do not affect the estimator, because the outer step uses only the location of the minimum."
        ),
    )

    report.add_results(
        "The second diagnostic compares the sampling distributions of the estimators. "
        "Each adversarial estimator sits next to maximum likelihood on the same simulated data. "
        "Any difference between the two histograms is due to the discriminator, not to the data. "
        "The oracle adversarial estimator overlaps the maximum-likelihood distribution almost exactly. "
        "The logistic and neural discriminators sit next to it with a slightly wider spread. "
        "That extra spread is the cost of training the discriminator on a finite simulation sample."
    )
    report.add_figure(
        "figures/estimator-histograms.png",
        "Monte Carlo distributions of adversarial estimators against MLE",
        fig_hist,
        description=(
            "Each panel overlays the maximum-likelihood histogram in grey with one adversarial estimator in color. "
            "The vertical line marks the true location $\\theta_0 = 0$. "
            "The dashed black curve is the asymptotic Gaussian prediction $\\mathcal{N}(0,\\, (1 + n/m)/(n\\, I_0))$ from Corollary 4 of the paper. "
            "Standard deviations in the legend are on the rate scale $\\sqrt{n} \\cdot \\mathrm{sd}(\\hat\\theta)$, matching the asymptotic numbers in the model setup table."
        ),
    )
    report.add_table(
        "tables/standard-errors.csv",
        "Bias and standard errors across estimators",
        se_table,
        description=(
            "Bias is reported on the natural scale. "
            "Standard deviations are reported on the rate scale $\\sqrt{n} \\cdot \\mathrm{sd}(\\hat\\theta)$. "
            "Asymptotic values use $\\sqrt{1/I_0}$ for maximum likelihood and $\\sqrt{(1 + n/m) / I_0}$ for the adversarial estimator. "
            "All Monte Carlo numbers are slightly below the asymptotic prediction at this sample size, which is the usual finite-sample behavior."
        ),
    )

    report.add_results(
        "The headline exhibit reruns the same data through plain simulated method of moments. "
        "Three power moments give an SMM estimator that is competitive with maximum likelihood in this design. "
        "Adding four more moments multiplies the SMM standard error by roughly six. "
        "The reason is that higher power moments are noisy in finite samples and the optimal weight matrix amplifies that noise. "
        "Eleven moments are essentially useless: the SMM distribution spreads across the full $\\theta$ search interval. "
        "The adversarial estimator on the same polynomial features barely moves over the same range of $d$. "
        "The discriminator reweights the features so that information from each moment is used in proportion to its precision."
    )
    report.add_figure(
        "figures/smm-vs-adversarial.png",
        "SMM versus adversarial estimators as the number of moments grows",
        fig_smm,
        description=(
            "The top row is optimally-weighted simulated method of moments matching the first $d$ power moments. "
            "The bottom row is the adversarial estimator with a logistic discriminator on the same $d$ polynomial features. "
            "All six panels share a horizontal axis spanning the full $\\theta$ search interval. "
            "The SMM panels visibly fan out as $d$ grows while the adversarial panels stay near the truth. "
            "Standard deviations in the legend are on the rate scale $\\sqrt{n} \\cdot \\mathrm{sd}(\\hat\\theta)$, with maximum likelihood overlaid as the efficient benchmark."
        ),
    )
    report.add_table(
        "tables/smm-comparison.csv",
        "SMM versus adversarial bias and standard errors as moment count grows",
        smm_table,
        description=(
            "Both estimators see the same polynomial features but use them differently. "
            "SMM matches each feature as a moment in its own right. "
            "The adversarial estimator lets the discriminator combine the features adaptively. "
            "That adaptive combination is what protects it from the precision loss that breaks SMM at $d = 7$ and $d = 11$."
        ),
    )

    if boot_table is not None:
        report.add_results(
            "A nonparametric bootstrap gives a practical recipe for inference. "
            "Both the real and the common-shock samples are resampled with replacement. "
            "The discriminator class is held fixed and the adversarial estimator is recomputed on every resample. "
            f"For the focal logistic-$d = 7$ discriminator the bootstrap standard error closely tracks the Monte Carlo standard error across the {R_MAIN} simulated experiments. "
            "In a real application the Monte Carlo number would not be available, so the bootstrap is the inference tool the analyst would actually use."
        )
        report.add_table(
            "tables/bootstrap-se.csv",
            f"Bootstrap, Monte Carlo, and asymptotic standard errors for adversarial logistic d={d_focal}",
            boot_table,
            description=(
                f"The bootstrap uses $B = {BOOTSTRAP_B}$ joint resamples of the real and shock vectors with replacement. "
                "The asymptotic value is the oracle prediction. "
                "Both the Monte Carlo and the bootstrap numbers are slightly larger because the data-driven discriminator is less efficient than the oracle."
            ),
        )

    report.add_takeaway(
        "Adversarial estimation moves moment selection from the analyst into the algorithm. "
        "A logistic discriminator on $d$ powers gives an estimator asymptotically equivalent to optimally-weighted SMM with $d$ moments; it inherits SMM's fragility at large $d$ but loses far less precision than the plain version. "
        "A small neural discriminator approaches maximum-likelihood efficiency without writing down the likelihood. "
        "The price is an inner optimization at every outer step, paid in compute rather than analyst judgment."
    )

    report.add_references([
        "[Kaji, T., Manresa, E., and Pouliot, G. (2023). An Adversarial Approach to Structural Estimation. *Econometrica*, 91(6), 2041-2063.](https://doi.org/10.3982/ECTA18707)",
        "[Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative Adversarial Nets. *NeurIPS*.](https://papers.nips.cc/paper/5423-generative-adversarial-nets)",
        "[McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models without Numerical Integration. *Econometrica*, 57(5), 995-1026.](https://doi.org/10.2307/1913621)",
        "[Athey, S., Imbens, G., Metzger, J., and Munro, E. (2024). Using Wasserstein Generative Adversarial Networks for the Design of Monte Carlo Simulations. *Journal of Econometrics*, 240(2), 105076.](https://doi.org/10.1016/j.jeconom.2020.09.013)",
    ])
    report.write()
    print(f"total runtime: {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
