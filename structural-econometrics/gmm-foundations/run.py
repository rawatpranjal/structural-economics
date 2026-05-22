#!/usr/bin/env python3
"""GMM Foundations: moment conditions, identification, and optimal weighting.

A scalar location model x_i = theta + eps_i with skewed-mixture errors. The
estimator targets the location parameter theta from a vector of moment
conditions. The script compares moment sets of size one, two, and five under
identity weighting and Hansen's two-step optimal weighting, then plots the
Monte Carlo sampling distributions, the variance-versus-moment-count
efficiency curve, and the J-statistic histogram under correctly specified and
misspecified data-generating processes.

Figures:
  figures/sampling-distribution.png : sampling distribution of theta_hat per
    moment set and weighting choice.
  figures/efficiency-gain.png       : Monte Carlo variance against moment
    count for identity and optimal weighting.
  figures/j-statistic.png           : J-statistic histogram for the
    five-moment model under correct and misspecified data.
  figures/thumb.png                 : thumbnail derived from efficiency-gain.

Outputs are deterministic given the seeds set below.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


# ----------------------------------------------------------------------------
# Data-generating process: skewed-mixture errors with zero mean.
# ----------------------------------------------------------------------------

# The error is a two-component Gaussian mixture with a heavy right tail. The
# means and weights are chosen so that the mixture has zero mean by
# construction; the third central moment is strictly positive, so a moment
# vector that includes the third central moment carries identifying
# information beyond what the first two moments provide.

MIX_WEIGHT = 0.85
MIX_MEAN_A = -0.30
MIX_MEAN_B = 1.70   # chosen so 0.85 * -0.30 + 0.15 * 1.70 = 0
MIX_SD_A = 0.70
MIX_SD_B = 1.20

THETA_TRUE = 1.0


def sample_errors(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw n errors from the skewed two-component Gaussian mixture."""
    mask = rng.uniform(size=n) < MIX_WEIGHT
    eps_a = rng.normal(MIX_MEAN_A, MIX_SD_A, size=n)
    eps_b = rng.normal(MIX_MEAN_B, MIX_SD_B, size=n)
    return np.where(mask, eps_a, eps_b)


def population_error_moments() -> tuple[float, float, float]:
    """Population mean, variance, and third central moment of the mixture."""
    w = MIX_WEIGHT
    mu_a, mu_b = MIX_MEAN_A, MIX_MEAN_B
    sa2, sb2 = MIX_SD_A ** 2, MIX_SD_B ** 2
    mean = w * mu_a + (1 - w) * mu_b
    second_raw = w * (sa2 + mu_a ** 2) + (1 - w) * (sb2 + mu_b ** 2)
    variance = second_raw - mean ** 2
    third_raw = (
        w * (mu_a ** 3 + 3 * mu_a * sa2)
        + (1 - w) * (mu_b ** 3 + 3 * mu_b * sb2)
    )
    third_central = third_raw - 3 * mean * second_raw + 2 * mean ** 3
    return mean, variance, third_central


ERR_MEAN, ERR_VAR, ERR_THIRD = population_error_moments()

# Population quantiles of the error (computed once on a fine reference sample
# so the same five-moment specification can target population quantiles of
# the error at known levels).
_QUANTILE_RNG = np.random.default_rng(20260522)
_QUANTILE_REF = sample_errors(_QUANTILE_RNG, 4_000_000)
ERR_Q05 = float(np.quantile(_QUANTILE_REF, 0.05))
ERR_Q95 = float(np.quantile(_QUANTILE_REF, 0.95))


# ----------------------------------------------------------------------------
# Moment functions.
# ----------------------------------------------------------------------------

# Each moment function returns an (n, k) array of moment contributions for the
# location model x_i = theta + eps_i. Stacking moments by row lets us compute
# both the average gbar and the sample covariance Omega-hat directly.


def moments_mean(x: np.ndarray, theta: float) -> np.ndarray:
    """Single moment: residual = x - theta."""
    return (x - theta).reshape(-1, 1) - ERR_MEAN


def moments_mean_var(x: np.ndarray, theta: float) -> np.ndarray:
    """Two moments: residual and centred squared residual minus variance."""
    r = x - theta
    return np.column_stack([r - ERR_MEAN, r ** 2 - ERR_VAR - ERR_MEAN ** 2])


def moments_five(x: np.ndarray, theta: float) -> np.ndarray:
    """Five moments: mean, variance, third moment, 0.05 and 0.95 quantiles.

    The quantile moment uses the standard check-function identity
    E[1{eps <= q_alpha} - alpha] = 0 evaluated at the population quantile of
    the error. Subtracting q_alpha from the residual before checking the sign
    gives a finite-sample moment that is mean-zero at theta_true.
    """
    r = x - theta
    second_raw_target = ERR_VAR + ERR_MEAN ** 2
    third_raw_target = (
        ERR_THIRD + 3 * ERR_MEAN * second_raw_target - 2 * ERR_MEAN ** 3
    )
    return np.column_stack([
        r - ERR_MEAN,
        r ** 2 - second_raw_target,
        r ** 3 - third_raw_target,
        (r <= ERR_Q05).astype(float) - 0.05,
        (r <= ERR_Q95).astype(float) - 0.95,
    ])


MOMENT_SETS: dict[str, tuple[Callable[[np.ndarray, float], np.ndarray], int]] = {
    "1 moment": (moments_mean, 1),
    "2 moments": (moments_mean_var, 2),
    "5 moments": (moments_five, 5),
}


# ----------------------------------------------------------------------------
# GMM estimation.
# ----------------------------------------------------------------------------


def gmm_objective(theta: float, x: np.ndarray, moment_fn, W: np.ndarray) -> float:
    """Q(theta) = gbar(theta)' W gbar(theta)."""
    gbar = moment_fn(x, theta).mean(axis=0)
    return float(gbar @ W @ gbar)


def estimate_gmm(
    x: np.ndarray,
    moment_fn,
    k: int,
    two_step: bool,
    start: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Return (theta_hat, W_used). Two-step refreshes W from the identity step."""
    W = np.eye(k)
    res = minimize(
        gmm_objective,
        x0=np.array([start]),
        args=(x, moment_fn, W),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 400},
    )
    theta_hat = float(res.x[0])
    if not two_step or k == 1:
        return theta_hat, W
    g_at_first = moment_fn(x, theta_hat)
    Omega_hat = np.cov(g_at_first, rowvar=False, bias=False)
    if k == 1:
        Omega_hat = np.atleast_2d(Omega_hat)
    # Ridge-stabilised inverse.
    W_opt = np.linalg.inv(Omega_hat + 1e-10 * np.eye(k))
    res2 = minimize(
        gmm_objective,
        x0=np.array([theta_hat]),
        args=(x, moment_fn, W_opt),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 400},
    )
    return float(res2.x[0]), W_opt


def j_statistic(x: np.ndarray, theta_hat: float, moment_fn, W: np.ndarray) -> float:
    """J = n * Q(theta_hat) under optimal weighting."""
    n = x.shape[0]
    gbar = moment_fn(x, theta_hat).mean(axis=0)
    return float(n * gbar @ W @ gbar)


# ----------------------------------------------------------------------------
# Monte Carlo experiment.
# ----------------------------------------------------------------------------


def monte_carlo(
    n_samples: int,
    sample_size: int,
    seed: int,
    misspecify_shift: float = 0.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Run M Monte Carlo replications across moment sets and weightings.

    Returns nested dict: results[label][weighting] -> theta_hat array.
    Additionally tracks the J-statistic for the five-moment case.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, np.ndarray]] = {
        label: {"identity": np.empty(n_samples), "optimal": np.empty(n_samples)}
        for label in MOMENT_SETS
    }
    j_optimal = np.empty(n_samples)

    for m in range(n_samples):
        eps = sample_errors(rng, sample_size)
        if misspecify_shift != 0.0:
            # Inject a model-misspecification: add a state-dependent shift
            # that makes the higher moments inconsistent with the assumed
            # mixture under the null theta = theta_true.
            eps = eps + misspecify_shift * (eps ** 2 - ERR_VAR - ERR_MEAN ** 2)
        x = THETA_TRUE + eps

        for label, (fn, k) in MOMENT_SETS.items():
            theta_id, _ = estimate_gmm(x, fn, k, two_step=False)
            theta_opt, W_opt = estimate_gmm(x, fn, k, two_step=True)
            out[label]["identity"][m] = theta_id
            out[label]["optimal"][m] = theta_opt
            if label == "5 moments":
                j_optimal[m] = j_statistic(x, theta_opt, fn, W_opt)

    out["J statistic"] = {"correct": j_optimal}
    return out


# ----------------------------------------------------------------------------
# Plotting.
# ----------------------------------------------------------------------------


def plot_sampling_distribution(results: dict, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    labels = ["1 moment", "2 moments", "5 moments"]
    for ax, label in zip(axes, labels):
        for weighting, colour in (("identity", "#1f77b4"), ("optimal", "#d62728")):
            theta_hats = results[label][weighting]
            ax.hist(
                theta_hats,
                bins=40,
                alpha=0.55,
                color=colour,
                label=f"{weighting} W",
                density=True,
            )
        ax.axvline(THETA_TRUE, color="black", linestyle="--", linewidth=1.2, label="true theta")
        ax.set_title(label)
        ax.set_xlabel(r"$\hat\theta$")
    axes[0].set_ylabel("density")
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("Sampling distribution of the GMM location estimator", y=1.02)
    out = fig_dir / "sampling-distribution.png"
    save_figure(fig, out)
    return out


def plot_efficiency_gain(results: dict, fig_dir: Path) -> Path:
    labels = ["1 moment", "2 moments", "5 moments"]
    counts = [1, 2, 5]
    var_identity = [np.var(results[label]["identity"], ddof=1) for label in labels]
    var_optimal = [np.var(results[label]["optimal"], ddof=1) for label in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(counts, var_identity, "o-", color="#1f77b4", label="identity W")
    ax.plot(counts, var_optimal, "s-", color="#d62728", label="optimal W (two-step)")
    ax.set_xticks(counts)
    ax.set_xlabel("Number of moments")
    ax.set_ylabel(r"Monte Carlo variance of $\hat\theta$")
    ax.set_title("Efficiency gain from optimal weighting")
    ax.legend(frameon=False)
    out = fig_dir / "efficiency-gain.png"
    save_figure(fig, out)
    return out


def plot_j_statistic(j_correct: np.ndarray, j_misspec: np.ndarray, df: int, fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(j_correct.max(), j_misspec.max(), 30) * 1.05, 60)
    ax.hist(j_correct, bins=bins, alpha=0.55, color="#1f77b4", density=True, label="correctly specified")
    ax.hist(j_misspec, bins=bins, alpha=0.55, color="#d62728", density=True, label="misspecified")
    x_ref = np.linspace(1e-3, bins[-1], 400)
    ax.plot(x_ref, chi2.pdf(x_ref, df=df), color="black", linestyle="--", linewidth=1.2,
            label=f"chi-square (df={df})")
    ax.set_xlabel("J statistic")
    ax.set_ylabel("density")
    ax.set_title("J-statistic distribution: correct vs misspecified")
    ax.legend(frameon=False)
    out = fig_dir / "j-statistic.png"
    save_figure(fig, out)
    return out


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------


def main() -> None:
    setup_style()
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 400
    sample_size = 500

    print(f"Sampling-distribution Monte Carlo: M={n_samples}, n={sample_size}.")
    results = monte_carlo(n_samples=n_samples, sample_size=sample_size, seed=20260522)

    print("Misspecified-DGP Monte Carlo for J statistic.")
    results_mis = monte_carlo(
        n_samples=n_samples,
        sample_size=sample_size,
        seed=99000001,
        misspecify_shift=0.20,
    )

    sampling_path = plot_sampling_distribution(results, fig_dir)
    efficiency_path = plot_efficiency_gain(results, fig_dir)
    j_path = plot_j_statistic(
        results["J statistic"]["correct"],
        results_mis["J statistic"]["correct"],
        df=5 - 1,
        fig_dir=fig_dir,
    )

    save_thumbnail(efficiency_path, fig_dir / "thumb.png")

    print(f"Saved {sampling_path}")
    print(f"Saved {efficiency_path}")
    print(f"Saved {j_path}")
    print(f"Saved {fig_dir / 'thumb.png'}")

    # Compact summary numbers for the README.
    for label in ["1 moment", "2 moments", "5 moments"]:
        v_id = np.var(results[label]["identity"], ddof=1)
        v_op = np.var(results[label]["optimal"], ddof=1)
        m_id = np.mean(results[label]["identity"])
        m_op = np.mean(results[label]["optimal"])
        print(
            f"{label:10s}  identity: mean={m_id:.4f} var={v_id:.5f}  "
            f"optimal: mean={m_op:.4f} var={v_op:.5f}"
        )

    j_c = results["J statistic"]["correct"]
    j_m = results_mis["J statistic"]["correct"]
    print(f"J statistic correct      : mean={j_c.mean():.2f} median={np.median(j_c):.2f}")
    print(f"J statistic misspecified : mean={j_m.mean():.2f} median={np.median(j_m):.2f}")


if __name__ == "__main__":
    main()
