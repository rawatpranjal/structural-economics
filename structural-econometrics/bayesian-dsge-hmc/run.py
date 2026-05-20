#!/usr/bin/env python3
"""Bayesian estimation of a small New Keynesian DSGE by HMC/NUTS.

The pipeline is end-to-end JAX-differentiable: structural parameters theta
go through Klein QZ (custom implicit-IFT JVP in lib/perturbation_jax.py),
into a linear Gaussian state-space form, through a Kalman filter (standard
autodiff in lib/kalman_jax.py), to a scalar log marginal likelihood.
BlackJAX NUTS samples from the resulting posterior. A random-walk
Metropolis baseline is run on the same posterior for an effective-sample-
size comparison.

Data is simulated from the model at a known theta_0 so the posterior can be
audited against ground truth. The economic content (IS, Phillips, Taylor
rule, two AR(1) shocks) is identical to dsge/nkdsge but extended to both
shocks active simultaneously; the only addition here is the estimation
machinery.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.kalman_jax import kalman_loglik
from lib.perturbation import solve_klein
from lib.perturbation_jax import solve_klein_jax
from lib.plotting import save_figure, save_thumbnail, setup_style

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Calibration: the "true" theta used to simulate the observed series.
# =============================================================================
PARAM_NAMES = ["sigma", "phi_pi", "phi_y", "kappa",
               "sigma_v", "rho_v", "sigma_d", "rho_d"]
PARAM_LABELS = [r"$\sigma$", r"$\phi_\pi$", r"$\phi_y$", r"$\kappa$",
                r"$\sigma_v$", r"$\rho_v$", r"$\sigma_d$", r"$\rho_d$"]
THETA0 = np.array([1.0, 1.5, 0.125, 0.3, 0.01, 0.5, 0.01, 0.8])
BETA_FIXED = 0.99
MEAS_STD = 1e-3        # tiny observation noise so the 3-obs / 2-shock Kalman is well-conditioned
T_N = 200              # length of the simulated series
SEED_DATA = 23
NUM_WARMUP = 1000
NUM_SAMPLES = 2000
NUM_CHAINS = 4

# Prior hyperparameters on the constrained (economically interpretable) space.
# Gamma(shape, scale): mean = shape * scale, std = sqrt(shape) * scale.
# Beta(a, b):          mean = a / (a + b).
PRIOR_HYP = {
    "sigma":   ("gamma", 2.0, 0.5),     # mean 1.0
    "phi_pi":  ("gamma", 4.0, 0.375),   # mean 1.5
    "phi_y":   ("gamma", 2.0, 0.0625),  # mean 0.125
    "kappa":   ("beta",  3.0, 7.0),     # mean 0.3
    "sigma_v": ("gamma", 2.0, 0.005),   # mean 0.01
    "rho_v":   ("beta",  2.0, 2.0),     # mean 0.5
    "sigma_d": ("gamma", 2.0, 0.005),   # mean 0.01
    "rho_d":   ("beta",  2.0, 2.0),     # mean 0.5
}
# Transform: gamma priors use exp (positive), beta priors use sigmoid (0,1).
TRANSFORM = {
    "sigma": "exp", "phi_pi": "exp", "phi_y": "exp", "kappa": "sigmoid",
    "sigma_v": "exp", "rho_v": "sigmoid", "sigma_d": "exp", "rho_d": "sigmoid",
}


# =============================================================================
# Model: 2-shock NK pencil, state s = (v, d, y, pi), n_predetermined = 2.
# =============================================================================
def build_AB(sigma, phi_pi, phi_y, kappa, rho_v, rho_d, use_jax=False):
    xp = jnp if use_jax else np
    A = xp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0 / sigma],
        [0.0, 0.0, 0.0, BETA_FIXED],
    ])
    B = xp.array([
        [rho_v, 0.0, 0.0, 0.0],
        [0.0, rho_d, 0.0, 0.0],
        [1.0 / sigma, -1.0, 1.0 + phi_y / sigma, phi_pi / sigma],
        [0.0, 0.0, -kappa, 1.0],
    ])
    return A, B


def state_space_jax(theta):
    sigma, phi_pi, phi_y, kappa, s_v, rho_v, s_d, rho_d = theta
    A, B = build_AB(sigma, phi_pi, phi_y, kappa, rho_v, rho_d, use_jax=True)
    sol = solve_klein_jax(A, B, n_predetermined=2)
    F = sol.F
    P = sol.P
    T_mat = F
    R_mat = jnp.array([[s_v, 0.0], [0.0, s_d]])
    Q_mat = jnp.eye(2)
    H_mat = jnp.stack([
        P[0],
        P[1],
        jnp.array([phi_pi * P[1, 0] + phi_y * P[0, 0] + 1.0,
                   phi_pi * P[1, 1] + phi_y * P[0, 1]]),
    ])
    S_mat = (MEAS_STD ** 2) * jnp.eye(3)
    return T_mat, R_mat, Q_mat, H_mat, S_mat


def state_space_numpy(theta):
    sigma, phi_pi, phi_y, kappa, s_v, rho_v, s_d, rho_d = theta
    A, B = build_AB(sigma, phi_pi, phi_y, kappa, rho_v, rho_d, use_jax=False)
    sol = solve_klein(A, B, n_predetermined=2)
    F = sol.F
    P = sol.P
    R_mat = np.diag([s_v, s_d])
    H_mat = np.vstack([
        P[0],
        P[1],
        np.array([phi_pi * P[1, 0] + phi_y * P[0, 0] + 1.0,
                  phi_pi * P[1, 1] + phi_y * P[0, 1]]),
    ])
    return F, R_mat, np.eye(2), H_mat, (MEAS_STD ** 2) * np.eye(3)


# =============================================================================
# Simulation: generate Y_{1:T} at theta_0 with a fixed seed.
# =============================================================================
def simulate(theta, T_n, seed):
    rng = np.random.default_rng(seed)
    T_mat, R_mat, Q_mat, H_mat, S_mat = state_space_numpy(theta)
    ds = T_mat.shape[0]
    do = H_mat.shape[0]
    x = np.zeros(ds)
    Y = np.zeros((T_n, do))
    chol_state = np.linalg.cholesky(R_mat @ Q_mat @ R_mat.T + 1e-18 * np.eye(ds))
    chol_obs = np.linalg.cholesky(S_mat)
    for t in range(T_n):
        x = T_mat @ x + chol_state @ rng.standard_normal(ds)
        Y[t] = H_mat @ x + chol_obs @ rng.standard_normal(do)
    return Y


# =============================================================================
# Priors and transforms.
# =============================================================================
def transform_z_to_theta(z):
    """Unconstrained z -> economically interpretable theta plus log-Jacobian."""
    pieces = []
    log_jac = 0.0
    for i, name in enumerate(PARAM_NAMES):
        zi = z[i]
        if TRANSFORM[name] == "exp":
            theta_i = jnp.exp(zi)
            log_jac = log_jac + zi
        else:
            theta_i = jax.nn.sigmoid(zi)
            log_jac = log_jac + jnp.log(theta_i) + jnp.log1p(-theta_i)
        pieces.append(theta_i)
    theta = jnp.stack(pieces)
    return theta, log_jac


def log_prior(theta):
    lp = 0.0
    for i, name in enumerate(PARAM_NAMES):
        kind, a, b = PRIOR_HYP[name]
        if kind == "gamma":
            lp = lp + jstats.gamma.logpdf(theta[i], a, scale=b)
        else:
            lp = lp + jstats.beta.logpdf(theta[i], a, b)
    return lp


def make_logdensity(Y_jax):
    """Return log_density(z) on the unconstrained latent space."""
    def log_density(z):
        theta, log_jac = transform_z_to_theta(z)
        lp = log_prior(theta)
        ll = kalman_loglik(*state_space_jax(theta), Y_jax)
        return lp + ll + log_jac
    return log_density


# =============================================================================
# Posterior IRFs.
# =============================================================================
def impulse_responses(theta, shock_kind, horizon=20):
    """Return IRFs of (y, pi, i) to a unit shock in v or d.

    shock_kind = 'v' for monetary, 'd' for demand.
    """
    sigma, phi_pi, phi_y, kappa, s_v, rho_v, s_d, rho_d = [float(t) for t in theta]
    A, B = build_AB(sigma, phi_pi, phi_y, kappa, rho_v, rho_d, use_jax=False)
    sol = solve_klein(A, B, n_predetermined=2)
    F = sol.F
    P = sol.P
    x = np.array([1.0, 0.0]) if shock_kind == "v" else np.array([0.0, 1.0])
    irf_y = np.zeros(horizon)
    irf_pi = np.zeros(horizon)
    irf_i = np.zeros(horizon)
    for t in range(horizon):
        y_t = P[0] @ x
        pi_t = P[1] @ x
        v_t, d_t = x[0], x[1]
        i_t = phi_pi * pi_t + phi_y * y_t + v_t
        irf_y[t] = y_t
        irf_pi[t] = pi_t
        irf_i[t] = i_t
        x = F @ x
    return irf_y, irf_pi, irf_i


# =============================================================================
# Effective sample size (one chain, one parameter): Geyer's initial monotone.
# =============================================================================
def ess_one(chain):
    n = len(chain)
    x = chain - chain.mean()
    # Autocovariances via FFT.
    m = 2
    while m < 2 * n:
        m *= 2
    fx = np.fft.fft(x, n=m)
    acov = np.fft.ifft(fx * np.conj(fx)).real[:n] / (n - np.arange(n))
    rho = acov / acov[0]
    # Geyer initial monotone: sum (rho[2k] + rho[2k+1]) while > 0 and non-increasing.
    s = 1.0
    prev_pair = np.inf
    for k in range(1, n // 2):
        pair = rho[2 * k - 1] + rho[2 * k]
        if pair <= 0 or pair >= prev_pair:
            break
        s += 2.0 * pair
        prev_pair = pair
    return float(n / s) if s > 0 else float(n)


def r_hat(chains):
    """Gelman-Rubin R-hat across multiple chains of equal length."""
    m, n = chains.shape
    means = chains.mean(axis=1)
    B = n * means.var(ddof=1)
    W = chains.var(axis=1, ddof=1).mean()
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


# =============================================================================
# Sampling.
# =============================================================================
def run_nuts(log_density, init_z, key, num_warmup, num_samples, num_chains):
    warmup = blackjax.window_adaptation(blackjax.nuts, log_density,
                                        target_acceptance_rate=0.8)
    keys = jax.random.split(key, num_chains)

    def run_chain(rng_key, init):
        warmup_key, sample_key = jax.random.split(rng_key)
        (state, params), _ = warmup.run(warmup_key, init, num_steps=num_warmup)
        kernel = blackjax.nuts(log_density, **params).step

        def step(carry, k):
            state = carry
            state, info = kernel(k, state)
            return state, (state.position, info.acceptance_rate)

        sample_keys = jax.random.split(sample_key, num_samples)
        _, (positions, accept) = jax.lax.scan(step, state, sample_keys)
        return positions, accept

    positions = []
    accept_rates = []
    for c in range(num_chains):
        pos, accept = jax.jit(run_chain)(keys[c], init_z + 0.01 * jax.random.normal(keys[c], init_z.shape))
        positions.append(np.asarray(pos))
        accept_rates.append(float(accept.mean()))
    return np.stack(positions), np.array(accept_rates)


def run_rwmh(log_density, init_z, key, num_samples, step_size):
    """Single-chain random-walk Metropolis using BlackJAX additive_step_random_walk."""
    cov_diag = step_size ** 2 * jnp.ones_like(init_z)
    random_walk = blackjax.additive_step_random_walk.normal_random_walk(log_density, sigma=jnp.sqrt(cov_diag))
    state = random_walk.init(init_z)
    kernel = jax.jit(random_walk.step)

    def step(carry, k):
        state = carry
        state, info = kernel(k, state)
        return state, (state.position, info.is_accepted)

    keys = jax.random.split(key, num_samples)
    _, (positions, accepts) = jax.lax.scan(step, state, keys)
    return np.asarray(positions), float(np.asarray(accepts).mean())


# =============================================================================
# Main.
# =============================================================================
def main():
    setup_style()
    tutorial_dir = Path(__file__).resolve().parent
    figs_dir = tutorial_dir / "figures"
    tables_dir = tutorial_dir / "tables"
    figs_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    print("Simulating observations at theta_0 ...")
    Y = simulate(THETA0, T_N, SEED_DATA)
    Y_jax = jnp.asarray(Y)
    print(f"  T={T_N}, observables=(y, pi, i), meas_std={MEAS_STD}")

    log_density = make_logdensity(Y_jax)
    init_z = jnp.zeros(len(PARAM_NAMES))
    # Sanity-check the log-posterior is finite at z=0 (theta = (1, 1, 1, 0.5, 1, 0.5, 1, 0.5)).
    ld0 = float(log_density(init_z))
    print(f"  log_posterior(z=0) = {ld0:.3f}")

    print("\nRunning NUTS ...")
    t0 = time.time()
    nuts_pos, nuts_accept = run_nuts(log_density, init_z, jax.random.PRNGKey(0),
                                     NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS)
    t_nuts = time.time() - t0
    print(f"  shape {nuts_pos.shape}, mean accept {nuts_accept.mean():.3f}, wall {t_nuts:.1f}s")

    # Transform NUTS draws back to theta space for diagnostics.
    nuts_theta = np.zeros_like(nuts_pos)
    for c in range(nuts_pos.shape[0]):
        for s in range(nuts_pos.shape[1]):
            th, _ = transform_z_to_theta(jnp.asarray(nuts_pos[c, s]))
            nuts_theta[c, s] = np.asarray(th)

    print("\nRunning RW-MH baseline (single chain, fixed step) ...")
    t0 = time.time()
    rwmh_pos, rwmh_accept = run_rwmh(log_density, init_z,
                                     jax.random.PRNGKey(7),
                                     NUM_SAMPLES * NUM_CHAINS, step_size=0.1)
    t_rwmh = time.time() - t0
    print(f"  shape {rwmh_pos.shape}, accept {rwmh_accept:.3f}, wall {t_rwmh:.1f}s")
    rwmh_theta = np.zeros_like(rwmh_pos)
    for s in range(rwmh_pos.shape[0]):
        th, _ = transform_z_to_theta(jnp.asarray(rwmh_pos[s]))
        rwmh_theta[s] = np.asarray(th)

    # ============================================================ Diagnostics
    print("\nComputing diagnostics ...")
    n_par = len(PARAM_NAMES)
    nuts_flat = nuts_theta.reshape(-1, n_par)  # pool chains
    summary = pd.DataFrame({
        "parameter":  PARAM_NAMES,
        "true":       THETA0,
        "post mean":  nuts_flat.mean(axis=0),
        "post median": np.median(nuts_flat, axis=0),
        "5 percent":  np.quantile(nuts_flat, 0.05, axis=0),
        "95 percent": np.quantile(nuts_flat, 0.95, axis=0),
        "R hat":      [r_hat(nuts_theta[:, :, j]) for j in range(n_par)],
        "NUTS ESS":   [sum(ess_one(nuts_theta[c, :, j]) for c in range(NUM_CHAINS))
                       for j in range(n_par)],
        "RW MH ESS":  [ess_one(rwmh_theta[:, j]) for j in range(n_par)],
    })
    summary["ESS ratio"] = summary["NUTS ESS"] / summary["RW MH ESS"]
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ============================================================ Figures
    # Figure 1: simulated observations.
    fig1, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    obs_names = ["Output gap $y_t$", "Inflation $\\pi_t$", "Policy rate $i_t$"]
    for ax, series, name in zip(axes, Y.T, obs_names):
        ax.plot(series * 100, color="#2c7bb6", linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(name)
        ax.set_ylabel("percent")
    axes[-1].set_xlabel("quarter")
    fig1.suptitle(f"Simulated observations (T = {T_N}, true theta)")
    fig1.tight_layout()
    fig1_path = figs_dir / "observations.png"

    # Figure 2: posterior densities with prior overlay and truth marker.
    fig2, axes = plt.subplots(2, 4, figsize=(14, 7))
    for j, ax in enumerate(axes.flat):
        post = nuts_flat[:, j]
        ax.hist(post, bins=40, density=True, color="#abd9e9", alpha=0.7,
                edgecolor="#2c7bb6", linewidth=0.5, label="posterior")
        ax.axvline(THETA0[j], color="#d7191c", linewidth=2.0, label="truth")
        kind, a, b = PRIOR_HYP[PARAM_NAMES[j]]
        x_grid = np.linspace(max(post.min() - 0.01, 1e-6),
                             post.max() + 0.01, 200)
        if kind == "gamma":
            prior_pdf = jstats.gamma.pdf(x_grid, a, scale=b)
        else:
            prior_pdf = jstats.beta.pdf(x_grid, a, b)
        ax.plot(x_grid, np.asarray(prior_pdf), color="black",
                linewidth=1.2, linestyle="--", label="prior")
        ax.set_title(PARAM_LABELS[j])
    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig2.suptitle("Posterior densities (NUTS) with priors and ground truth")
    fig2.tight_layout()
    fig2_path = figs_dir / "posterior-densities.png"

    # Figure 3: posterior IRFs with 90% bands, both shocks.
    horizon = 20
    irf_subsample = nuts_flat[np.random.default_rng(0)
                              .integers(0, nuts_flat.shape[0], 200)]
    bands = {"v": [], "d": []}
    for shock_kind in ("v", "d"):
        irf_y = np.zeros((len(irf_subsample), horizon))
        irf_pi = np.zeros_like(irf_y)
        irf_i = np.zeros_like(irf_y)
        for s, theta in enumerate(irf_subsample):
            iy, ipi, ii = impulse_responses(theta, shock_kind, horizon)
            irf_y[s], irf_pi[s], irf_i[s] = iy, ipi, ii
        bands[shock_kind] = (irf_y, irf_pi, irf_i)

    fig3, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, shock_kind in enumerate(("v", "d")):
        title_shock = "monetary $v_t$" if shock_kind == "v" else "demand $d_t$"
        iy, ipi, ii = bands[shock_kind]
        for col, (data, label) in enumerate(zip(
                (iy, ipi, ii),
                (r"$y_t$", r"$\pi_t$", r"$i_t$"))):
            ax = axes[row, col]
            lo = np.quantile(data, 0.05, axis=0)
            hi = np.quantile(data, 0.95, axis=0)
            med = np.median(data, axis=0)
            ax.fill_between(np.arange(horizon), lo, hi,
                            color="#abd9e9", alpha=0.5, label="90 percent band")
            ax.plot(med, color="#2c7bb6", linewidth=2.0, label="posterior median")
            true_iy, true_ipi, true_ii = impulse_responses(THETA0, shock_kind, horizon)
            true_path = (true_iy, true_ipi, true_ii)[col]
            ax.plot(true_path, color="#d7191c", linewidth=2.0, linestyle="--",
                    label="truth")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"{label} to {title_shock}")
            ax.set_xlabel("quarter")
    axes[0, 0].legend(loc="best", fontsize=9)
    fig3.suptitle("Posterior impulse responses with 90 percent bands")
    fig3.tight_layout()
    fig3_path = figs_dir / "posterior-irfs.png"

    # Figure 4: ESS comparison (per parameter, NUTS vs RW-MH).
    fig4, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    pos = np.arange(n_par)
    ax.bar(pos - width / 2, summary["NUTS ESS"], width, color="#2c7bb6",
           label=f"NUTS (4 chains x {NUM_SAMPLES} draws)")
    ax.bar(pos + width / 2, summary["RW MH ESS"], width, color="#d7191c",
           label=f"RW-MH ({NUM_SAMPLES * NUM_CHAINS} draws)")
    ax.set_xticks(pos)
    ax.set_xticklabels(PARAM_LABELS)
    ax.set_yscale("log")
    ax.set_ylabel("effective sample size (log)")
    ax.set_title("ESS per parameter: NUTS vs. random-walk Metropolis")
    ax.legend()
    fig4.tight_layout()
    fig4_path = figs_dir / "ess-comparison.png"

    # ============================================================ Tables
    summary_table = summary.copy()
    summary_path = tables_dir / "posterior_summary.csv"

    # ============================================================ Save figures
    save_figure(fig1, str(fig1_path), dpi=150)
    save_figure(fig2, str(fig2_path), dpi=150)
    save_figure(fig3, str(fig3_path), dpi=150)
    save_figure(fig4, str(fig4_path), dpi=150)

    # ============================================================ Save tables
    summary_table.to_csv(str(summary_path), index=False)

    # ============================================================ Thumbnail
    save_thumbnail(str(fig1_path), str(figs_dir / "thumb.png"))
    print(f"\nWrote figures and tables to {tutorial_dir}.")


if __name__ == "__main__":
    main()
