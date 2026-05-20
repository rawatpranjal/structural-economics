#!/usr/bin/env python3
"""Heterogeneous probability distortion via finite-mixture EM (Bruhin-Fehr-Duda-Epper 2010).

Subjects evaluate binary lotteries and report certainty equivalents.
Cumulative prospect theory rationalises the data with three primitives
per subject: a CRRA value-function exponent, a probability-weighting
slope, and a probability-weighting elevation. The population is
heterogeneous: a minority of subjects behave as expected utility
maximisers (linear weighting, near-linear value), while the majority
exhibit substantial probability distortion of varying strength.

Bruhin, Fehr-Duda, and Epper (2010) propose recovering this latent
heterogeneity by a finite-mixture model fitted by the EM algorithm.
The mixture endogenously classifies subjects into types and reports
type-specific parameters together with mixing proportions. The headline
empirical finding across three samples (Zurich 2003, Zurich 2006, Beijing
2005) is a robust C = 3 split: about 20 percent EUT types, 50 percent
mild-distortion CPT types, and 30 percent strong-distortion CPT types.

Three estimators are compared on simulated data faithful to the BFDE
specification:

- Method 1: single-type CPT MLE with no mixture.
- Method 2: finite-mixture EM with C = 2.
- Method 3: finite-mixture EM with C = 3 (the BFDE headline).

Reference: Bruhin, A., Fehr-Duda, H., & Epper, T. (2010). Risk and
Rationality: Uncovering Heterogeneity in Probability Distortion.
Econometrica 78(4), 1375-1412.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


# ---------------------------------------------------------------------------
# CPT primitives: Goldstein-Einhorn weighting, sign-dependent CRRA value
# with loss aversion lambda for mixed and pure-loss lotteries
# ---------------------------------------------------------------------------
def weight(p: np.ndarray, gamma: float, delta: float) -> np.ndarray:
    """Goldstein-Einhorn / Lattimore-Baker-Witte two-parameter weighting."""
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    pg = p ** gamma
    return delta * pg / (delta * pg + (1.0 - p) ** gamma)


def value(x: np.ndarray, alpha: float, lam: float) -> np.ndarray:
    """Sign-dependent CRRA value with loss aversion (Tversky-Kahneman 1992).

    v(x) = x^alpha for x >= 0, v(x) = -lam * (-x)^alpha for x < 0.
    """
    pos = np.maximum(x, 0.0) ** alpha
    neg = -lam * np.maximum(-x, 0.0) ** alpha
    return pos + neg


def value_inv(u: np.ndarray, alpha: float, lam: float) -> np.ndarray:
    """Inverse of v under the same sign-dependent CRRA form."""
    pos = np.maximum(u, 0.0) ** (1.0 / alpha)
    neg_arg = np.maximum(-u, 0.0) / lam
    neg = -(neg_arg ** (1.0 / alpha))
    return pos + neg


def predicted_ce(x1: np.ndarray, x2: np.ndarray, p: np.ndarray,
                 alpha: float, lam: float,
                 gamma: float, delta: float) -> np.ndarray:
    """Closed-form CPT certainty equivalent.

    The lottery (x1, p; x2) places probability p on x1 and 1 - p on x2,
    with x1 > x2 by convention so x1 is the better outcome under any
    sign configuration (gain-gain, mixed, loss-loss). The single
    Goldstein-Einhorn weight w(p) attaches to the better outcome and
    1 - w(p) to the worse one.
    """
    w_p = weight(p, gamma, delta)
    util = value(x1, alpha, lam) * w_p + value(x2, alpha, lam) * (1.0 - w_p)
    return value_inv(util, alpha, lam)


# ---------------------------------------------------------------------------
# Heteroskedastic likelihood
# ---------------------------------------------------------------------------
def lottery_log_density(ce_obs: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                        p: np.ndarray, alpha: float, lam: float,
                        gamma: float, delta: float,
                        xi: float) -> np.ndarray:
    """Per-lottery log density under N(predicted_ce, (xi * range)^2)."""
    sigma = xi * (x1 - x2)
    sigma = np.maximum(sigma, 1e-9)
    pred = predicted_ce(x1, x2, p, alpha, lam, gamma, delta)
    return norm.logpdf(ce_obs, loc=pred, scale=sigma)


def subject_log_density(ce_obs: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                        p: np.ndarray, theta: np.ndarray, xi: float) -> float:
    """Sum of log densities across one subject's lotteries."""
    alpha, lam, gamma, delta = theta
    return float(np.sum(lottery_log_density(ce_obs, x1, x2, p,
                                            alpha, lam, gamma, delta, xi)))


def fit_xi_for_subject(ce_obs: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                       p: np.ndarray, theta: np.ndarray) -> float:
    """Closed-form ML estimate of xi given theta (residual standardisation)."""
    alpha, lam, gamma, delta = theta
    pred = predicted_ce(x1, x2, p, alpha, lam, gamma, delta)
    resid = ce_obs - pred
    range_ = x1 - x2
    range_ = np.maximum(range_, 1e-9)
    standard = resid / range_
    return float(np.sqrt(np.mean(standard ** 2)) + 1e-6)


# ---------------------------------------------------------------------------
# Method 1: single-type CPT MLE
# ---------------------------------------------------------------------------
def fit_single_type(df: pd.DataFrame, theta0: np.ndarray) -> tuple:
    """Single (alpha, lam, gamma, delta) for the whole sample, xi profiled per subject."""
    subjects = df["subject"].unique()

    def neg_ll(theta):
        if np.any(theta <= 0):
            return 1e10
        ll = 0.0
        for s in subjects:
            sub = df[df["subject"] == s]
            xi_hat = fit_xi_for_subject(
                sub["ce"].to_numpy(), sub["x1"].to_numpy(),
                sub["x2"].to_numpy(), sub["p"].to_numpy(), theta,
            )
            ll += subject_log_density(
                sub["ce"].to_numpy(), sub["x1"].to_numpy(),
                sub["x2"].to_numpy(), sub["p"].to_numpy(), theta, xi_hat,
            )
        return -ll

    res = minimize(neg_ll, theta0, method="L-BFGS-B",
                   bounds=[(0.05, 2.0), (1.0, 5.0), (0.05, 2.0), (0.05, 5.0)])
    return res.x, -res.fun


# ---------------------------------------------------------------------------
# Methods 2 and 3: finite-mixture EM
# ---------------------------------------------------------------------------
def neg_weighted_ll_for_type(theta: np.ndarray, weights_subj: np.ndarray,
                             subj_indices: list, df: pd.DataFrame) -> float:
    """Weighted negative log-likelihood for one mixture component."""
    if np.any(theta <= 0):
        return 1e10
    total = 0.0
    for s_idx, s in enumerate(subj_indices):
        sub = df[df["subject"] == s]
        xi_hat = fit_xi_for_subject(
            sub["ce"].to_numpy(), sub["x1"].to_numpy(),
            sub["x2"].to_numpy(), sub["p"].to_numpy(), theta,
        )
        ll = subject_log_density(
            sub["ce"].to_numpy(), sub["x1"].to_numpy(),
            sub["x2"].to_numpy(), sub["p"].to_numpy(), theta, xi_hat,
        )
        total += weights_subj[s_idx] * ll
    return -total


def fit_mixture_em(df: pd.DataFrame, C: int, theta_init: np.ndarray,
                   pi_init: np.ndarray, max_iter: int = 80,
                   tol: float = 1e-4) -> dict:
    """EM for finite-mixture CPT.

    Parameters
    ----------
    df : DataFrame with columns subject, x1, x2, p, ce.
    C : number of mixture components.
    theta_init : (C, 4) initial type parameters (alpha, lam, gamma, delta).
    pi_init : (C,) initial mixing proportions.
    max_iter, tol : EM stopping rule on log-likelihood improvement.

    Returns
    -------
    dict with keys: theta, pi, posteriors (N, C), xi (N,), log_likelihood,
    iterations.
    """
    subjects = df["subject"].unique()
    N = len(subjects)
    theta = theta_init.copy().astype(float)
    pi = pi_init.copy().astype(float)
    pi = pi / pi.sum()
    log_lik_prev = -np.inf

    # Profile xi for each subject under the current per-type params (initialise
    # with type-1 params so xi is not absurd).
    xi = np.zeros(N)
    for s_idx, s in enumerate(subjects):
        sub = df[df["subject"] == s]
        xi[s_idx] = fit_xi_for_subject(
            sub["ce"].to_numpy(), sub["x1"].to_numpy(),
            sub["x2"].to_numpy(), sub["p"].to_numpy(), theta[0],
        )

    for it in range(max_iter):
        # E-step: compute log-likelihood per (subject, type) and posteriors
        log_dens = np.zeros((N, C))
        for s_idx, s in enumerate(subjects):
            sub = df[df["subject"] == s]
            for c in range(C):
                log_dens[s_idx, c] = subject_log_density(
                    sub["ce"].to_numpy(), sub["x1"].to_numpy(),
                    sub["x2"].to_numpy(), sub["p"].to_numpy(),
                    theta[c], xi[s_idx],
                )
        log_pi = np.log(np.maximum(pi, 1e-12))
        log_post_unnorm = log_dens + log_pi[None, :]
        max_log = log_post_unnorm.max(axis=1, keepdims=True)
        log_norm = max_log + np.log(np.sum(
            np.exp(log_post_unnorm - max_log), axis=1, keepdims=True,
        ))
        posteriors = np.exp(log_post_unnorm - log_norm)
        log_lik = float(log_norm.sum())
        # M-step: update mixing proportions
        pi = posteriors.mean(axis=0)
        # M-step: update theta_c by weighted ML for each type
        for c in range(C):
            weights_subj = posteriors[:, c]
            res = minimize(
                neg_weighted_ll_for_type, theta[c],
                args=(weights_subj, list(subjects), df),
                method="L-BFGS-B",
                bounds=[(0.05, 2.0), (1.0, 5.0), (0.05, 2.0), (0.05, 5.0)],
                options={"maxiter": 30},
            )
            theta[c] = res.x
        # M-step: update xi for each subject given the type mixture
        for s_idx, s in enumerate(subjects):
            sub = df[df["subject"] == s]
            # Use the maximum-posterior type to set xi; this is an approximation
            # of the proper weighted update but stays close to the BFDE
            # implementation, which profiles xi per subject.
            best_c = int(np.argmax(posteriors[s_idx]))
            xi[s_idx] = fit_xi_for_subject(
                sub["ce"].to_numpy(), sub["x1"].to_numpy(),
                sub["x2"].to_numpy(), sub["p"].to_numpy(), theta[best_c],
            )
        # Convergence check
        if abs(log_lik - log_lik_prev) < tol:
            break
        log_lik_prev = log_lik

    # Reorder types by gamma so output is reproducible (label-switching fix)
    order = np.argsort(theta[:, 2])[::-1]  # higher gamma = less distortion = type 0
    theta = theta[order]
    pi = pi[order]
    posteriors = posteriors[:, order]
    return {
        "theta": theta, "pi": pi, "posteriors": posteriors,
        "xi": xi, "log_likelihood": log_lik, "iterations": it + 1,
        "n_params": C * 4 + (C - 1) + N,
    }


def bic(log_lik: float, n_obs: int, n_params: int) -> float:
    """Bayesian Information Criterion: -2 ln L + p ln N."""
    return -2.0 * log_lik + n_params * np.log(n_obs)


def normalised_entropy(posteriors: np.ndarray) -> float:
    """Normalised entropy criterion (Celeux-Soromenho 1996); 0 = perfect classification."""
    p = np.maximum(posteriors, 1e-12)
    entropy = -np.sum(p * np.log(p))
    N, C = posteriors.shape
    return float(entropy / (N * np.log(C))) if C > 1 else 0.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def build_lotteries() -> pd.DataFrame:
    """Three-domain lottery design extending BFDE Zurich-2003 (Table II).

    Gain lotteries are pulled from the BFDE design. Pure-loss lotteries
    mirror the gain cells with sign flipped. Mixed lotteries pair a
    positive gain x1 with a negative loss x2 and identify loss aversion
    lambda by the cross-domain payoff trade-off.
    """
    rows = []
    # Gain lotteries (BFDE Zurich-2003 subset)
    for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        for x1, x2 in [(20, 0), (40, 10), (50, 20), (50, 0), (150, 50)]:
            rows.append({"x1": float(x1), "x2": float(x2), "p": float(p),
                          "domain": "gain"})
    # Pure-loss lotteries: same magnitudes, sign-flipped, x1 > x2
    for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        for x1, x2 in [(0, -20), (-10, -40), (-20, -50), (0, -50), (-50, -150)]:
            rows.append({"x1": float(x1), "x2": float(x2), "p": float(p),
                          "domain": "loss"})
    # Mixed lotteries: gain x1 > 0, loss x2 < 0 (these identify lambda)
    for p in [0.25, 0.50, 0.75]:
        for x1, x2 in [(40, -20), (60, -30), (50, -50), (80, -40), (100, -50)]:
            rows.append({"x1": float(x1), "x2": float(x2), "p": float(p),
                          "domain": "mixed"})
    return pd.DataFrame(rows)


def simulate_subjects(lotteries: pd.DataFrame, n_subjects: int,
                      types: np.ndarray, type_pi: np.ndarray,
                      rng: np.random.Generator) -> tuple:
    """Simulate certainty equivalents from the finite-mixture DGP."""
    type_assignments = rng.choice(len(type_pi), size=n_subjects, p=type_pi)
    xi_subjects = rng.uniform(0.05, 0.20, size=n_subjects)
    rows = []
    for s in range(n_subjects):
        c = type_assignments[s]
        alpha, lam, gamma, delta = types[c]
        for _, lot in lotteries.iterrows():
            x1, x2, p = lot["x1"], lot["x2"], lot["p"]
            pred = predicted_ce(np.array([x1]), np.array([x2]),
                                np.array([p]), alpha, lam, gamma, delta)[0]
            sigma = xi_subjects[s] * (x1 - x2)
            ce_obs = pred + rng.normal(0, sigma)
            ce_obs = np.clip(ce_obs, x2, x1)
            rows.append({
                "subject": s, "type_true": c, "domain": lot["domain"],
                "x1": x1, "x2": x2, "p": p, "ce": ce_obs,
            })
    return pd.DataFrame(rows), type_assignments, xi_subjects


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(20260509)

    # True DGP: three-type mixture extending the BFDE classification with
    # type-specific loss aversion. Each type has parameters (alpha, lam,
    # gamma, delta) where lam = 1 means no loss aversion.
    types_true = np.array([
        [0.95, 1.00, 1.00, 1.00],   # Type 1: EUT (no loss aversion, linear weighting)
        [0.85, 1.50, 0.65, 0.85],   # Type 2: mild CPT, moderate loss aversion
        [0.70, 2.50, 0.40, 0.95],   # Type 3: strong CPT, strong loss aversion (close to TK92's lam = 2.25)
    ])
    pi_true = np.array([0.20, 0.50, 0.30])
    n_subjects = 200
    lotteries = build_lotteries()

    df, type_assignments, xi_true = simulate_subjects(
        lotteries, n_subjects, types_true, pi_true, rng,
    )
    n_obs = len(df)

    # Method 1: single-type MLE
    theta0 = np.array([0.85, 1.50, 0.70, 0.95])
    theta_single, ll_single = fit_single_type(df, theta0)
    n_params_single = 4 + n_subjects   # global theta + per-subject xi
    bic_single = bic(ll_single, n_obs, n_params_single)

    # Method 2: C = 2 mixture
    theta_init_c2 = np.array([
        [0.95, 1.00, 1.00, 1.00],
        [0.75, 2.00, 0.55, 0.90],
    ])
    pi_init_c2 = np.array([0.5, 0.5])
    fit_c2 = fit_mixture_em(df, 2, theta_init_c2, pi_init_c2)
    bic_c2 = bic(fit_c2["log_likelihood"], n_obs, fit_c2["n_params"])
    nec_c2 = normalised_entropy(fit_c2["posteriors"])

    # Method 3: C = 3 mixture
    theta_init_c3 = np.array([
        [0.95, 1.00, 1.00, 1.00],
        [0.85, 1.50, 0.65, 0.85],
        [0.70, 2.50, 0.40, 0.95],
    ])
    pi_init_c3 = np.array([0.25, 0.45, 0.30])
    fit_c3 = fit_mixture_em(df, 3, theta_init_c3, pi_init_c3)
    bic_c3 = bic(fit_c3["log_likelihood"], n_obs, fit_c3["n_params"])
    nec_c3 = normalised_entropy(fit_c3["posteriors"])

    # Method 4: C = 4 (over-fit check)
    theta_init_c4 = np.array([
        [0.95, 1.00, 1.00, 1.00],
        [0.85, 1.50, 0.65, 0.85],
        [0.70, 2.50, 0.40, 0.95],
        [0.80, 1.80, 0.50, 0.80],
    ])
    pi_init_c4 = np.array([0.20, 0.30, 0.30, 0.20])
    fit_c4 = fit_mixture_em(df, 4, theta_init_c4, pi_init_c4)
    bic_c4 = bic(fit_c4["log_likelihood"], n_obs, fit_c4["n_params"])
    nec_c4 = normalised_entropy(fit_c4["posteriors"])

    # Classification posterior summary
    max_post_c3 = fit_c3["posteriors"].max(axis=1)
    type_assignments_hat = fit_c3["posteriors"].argmax(axis=1)

    # Relative risk premia for the descriptive figure (replicates BFDE Figure 2).
    # Restrict to gain-only lotteries: relative risk premium (EV - CE) / EV is a
    # gain-domain diagnostic. Loss and mixed lotteries have EV <= 0, so the ratio
    # is not comparable across domains and pooling them would contaminate the
    # median the figure reports.
    df_gain = df[df["domain"] == "gain"].copy()
    df_gain["ev"] = df_gain["p"] * df_gain["x1"] + (1.0 - df_gain["p"]) * df_gain["x2"]
    df_gain["rrp"] = (df_gain["ev"] - df_gain["ce"]) / df_gain["ev"]
    rrp_by_p = df_gain.groupby("p")["rrp"].median()

    # =====================================================================
    # Figures
    # =====================================================================
    setup_style()

    # ------------------------------------------------------------------
    # Figure 1: probability weighting curves for each recovered type
    # ------------------------------------------------------------------
    p_grid = np.linspace(0.001, 0.999, 200)
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5))
    type_colors = ["tab:green", "tab:orange", "tab:red"]
    type_labels = ["EUT", "Mild CPT", "Strong CPT"]
    # Panel A: probability weighting
    ax1a = axes1[0]
    ax1a.plot(p_grid, p_grid, "k--", linewidth=1, alpha=0.5, label="Linear $w(p) = p$")
    for c in range(3):
        w_true = weight(p_grid, types_true[c, 2], types_true[c, 3])
        ax1a.plot(p_grid, w_true, color=type_colors[c], linestyle=":",
                  linewidth=1.0, alpha=0.5)
        w_hat = weight(p_grid, fit_c3["theta"][c, 2], fit_c3["theta"][c, 3])
        ax1a.plot(p_grid, w_hat, color=type_colors[c], linewidth=2,
                  label=fr"{type_labels[c]}: $\gamma = {fit_c3['theta'][c, 2]:.2f}$, $\delta = {fit_c3['theta'][c, 3]:.2f}$")
    ax1a.set_xlabel(r"Objective probability $p$")
    ax1a.set_ylabel(r"Decision weight $w(p)$")
    ax1a.set_title("Probability weighting by type")
    ax1a.legend(loc="lower right", fontsize=9)
    ax1a.set_aspect("equal")
    ax1a.set_xlim(0, 1)
    ax1a.set_ylim(0, 1)
    # Panel B: value function with loss aversion
    ax1b = axes1[1]
    x_grid = np.linspace(-50, 50, 400)
    ax1b.plot(x_grid, x_grid, "k--", linewidth=1, alpha=0.5, label="Linear $v(x) = x$")
    for c in range(3):
        v_hat = value(x_grid, fit_c3["theta"][c, 0], fit_c3["theta"][c, 1])
        ax1b.plot(x_grid, v_hat, color=type_colors[c], linewidth=2,
                  label=fr"{type_labels[c]}: $\alpha = {fit_c3['theta'][c, 0]:.2f}$, $\lambda = {fit_c3['theta'][c, 1]:.2f}$")
    ax1b.axhline(0, color="black", linewidth=0.5)
    ax1b.axvline(0, color="black", linewidth=0.5)
    ax1b.set_xlabel(r"Outcome $x$")
    ax1b.set_ylabel(r"Value $v(x)$")
    ax1b.set_title("Value function by type (loss aversion is the gain-loss asymmetry)")
    ax1b.legend(loc="upper left", fontsize=9)
    fig1.tight_layout()
    save_figure(fig1, "figures/weighting-and-value-functions.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 2: classification posterior histogram
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(max_post_c3, bins=20, color="tab:blue", edgecolor="black",
             alpha=0.8)
    ax2.axvline(0.95, color="tab:red", linestyle="--", linewidth=1.5,
                label="0.95 sharp-classification threshold")
    ax2.set_xlabel(r"Maximum posterior $\max_c \tau_{ic}$")
    ax2.set_ylabel("Number of subjects")
    ax2.set_title("Classification posterior under the three-type mixture")
    ax2.legend(loc="upper left", fontsize=9)
    save_figure(fig2, "figures/classification-posterior.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3: model selection by BIC
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    cs = [1, 2, 3, 4]
    bics = [bic_single, bic_c2, bic_c3, bic_c4]
    bars = ax3.bar(cs, bics, color="tab:blue", edgecolor="black", alpha=0.85)
    best_c = cs[int(np.argmin(bics))]
    bars[best_c - 1].set_color("tab:red")
    ax3.set_xticks(cs)
    ax3.set_xlabel("Number of mixture components $C$")
    ax3.set_ylabel("Bayesian information criterion (lower is better)")
    ax3.set_title("Model selection across mixture sizes")
    for c, b in zip(cs, bics):
        ax3.text(c, b, f"{b:.0f}", ha="center", va="bottom", fontsize=9)
    save_figure(fig3, "figures/model-selection-bic.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 4: relative risk premia by p (replicate BFDE Figure 2)
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    rrp_for_plot = rrp_by_p.reindex([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    bars = ax4.bar(range(len(rrp_for_plot)), rrp_for_plot.values,
                   color=["tab:red" if v > 0 else "tab:blue"
                          for v in rrp_for_plot.values],
                   edgecolor="black", alpha=0.85)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xticks(range(len(rrp_for_plot)))
    ax4.set_xticklabels([f"{p:.2f}" for p in rrp_for_plot.index])
    ax4.set_xlabel(r"Lottery probability $p$")
    ax4.set_ylabel(r"Median relative risk premium $(EV - CE) / EV$")
    ax4.set_title("Median relative risk premia in the gain domain")
    save_figure(fig4, "figures/relative-risk-premia.png", dpi=150)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    type_table = pd.DataFrame({
        "Type": ["EUT", "Mild CPT", "Strong CPT"],
        "True alpha": [f"{types_true[c, 0]:.3f}" for c in range(3)],
        "Estimated alpha": [f"{fit_c3['theta'][c, 0]:.3f}" for c in range(3)],
        "True lambda": [f"{types_true[c, 1]:.3f}" for c in range(3)],
        "Estimated lambda": [f"{fit_c3['theta'][c, 1]:.3f}" for c in range(3)],
        "True gamma": [f"{types_true[c, 2]:.3f}" for c in range(3)],
        "Estimated gamma": [f"{fit_c3['theta'][c, 2]:.3f}" for c in range(3)],
        "True delta": [f"{types_true[c, 3]:.3f}" for c in range(3)],
        "Estimated delta": [f"{fit_c3['theta'][c, 3]:.3f}" for c in range(3)],
        "True share": [f"{pi_true[c]:.2f}" for c in range(3)],
        "Estimated share": [f"{fit_c3['pi'][c]:.2f}" for c in range(3)],
    })
    Path("tables/type-parameters.csv").parent.mkdir(parents=True, exist_ok=True)
    type_table.to_csv("tables/type-parameters.csv", index=False)

    selection_table = pd.DataFrame({
        "Mixture size": ["C = 1 (single type)", "C = 2", "C = 3 (BFDE)", "C = 4"],
        "Log-likelihood": [
            f"{ll_single:.1f}", f"{fit_c2['log_likelihood']:.1f}",
            f"{fit_c3['log_likelihood']:.1f}", f"{fit_c4['log_likelihood']:.1f}",
        ],
        "Number of parameters": [
            n_params_single, fit_c2["n_params"], fit_c3["n_params"], fit_c4["n_params"],
        ],
        "BIC": [f"{bic_single:.0f}", f"{bic_c2:.0f}",
                f"{bic_c3:.0f}", f"{bic_c4:.0f}"],
        "Normalised entropy": ["n/a", f"{nec_c2:.4f}", f"{nec_c3:.4f}", f"{nec_c4:.4f}"],
    })
    Path("tables/model-selection.csv").parent.mkdir(parents=True, exist_ok=True)
    selection_table.to_csv("tables/model-selection.csv", index=False)

    save_thumbnail("figures/weighting-and-value-functions.png", "figures/thumb.png")
    print(f"\nDone: 4 figures, 2 tables")


if __name__ == "__main__":
    main()
