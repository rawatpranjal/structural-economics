#!/usr/bin/env python3
"""Hamiltonian Monte Carlo on a banana-shaped posterior.

Random-walk Metropolis-Hastings struggles on strongly correlated targets.
Each proposal is a local Gaussian step, so the chain spends most of its
time rejecting moves into low-density regions or accepting tiny moves that
leave the autocorrelation high. Hamiltonian Monte Carlo replaces the random
walk with a deterministic trajectory simulated through Hamiltonian dynamics.
The trajectory follows the gradient of the log posterior, traces along the
posterior's ridge geometry, and proposes a far-away state that the
Metropolis correction almost always accepts.

The target here is a banana-shaped posterior, the textbook stress test for
samplers on curved likelihood ridges. Hamiltonian dynamics is simulated
with leapfrog integration; the symplectic structure preserves the
Hamiltonian to second order in the step size, so a long trajectory still
returns a high-probability proposal.

References:
- Duane, S., Kennedy, A. D., Pendleton, B. J., and Roweth, D. (1987). Hybrid Monte Carlo.
- Neal, R. M. (2011). MCMC Using Hamiltonian Dynamics. In Brooks et al., Handbook of Markov Chain Monte Carlo.
- Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn Sampler. JMLR.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


# =============================================================================
# Banana posterior: x ~ N(0, sigma_x^2), y | x ~ N(alpha (x^2 - sigma_x^2), sigma_y^2)
# =============================================================================
SIGMA_X = 2.0
SIGMA_Y = 1.0
ALPHA = 0.5

VAR_X = SIGMA_X ** 2
VAR_Y_MARGINAL = (ALPHA ** 2) * 2.0 * (SIGMA_X ** 4) + SIGMA_Y ** 2  # analytical
MEAN_X = 0.0
MEAN_Y = 0.0  # because E[x^2 - sigma_x^2] = 0 for x ~ N(0, sigma_x^2)


def log_target(q: np.ndarray) -> float:
    """Log density of the banana target (unnormalized)."""
    q = np.atleast_1d(q).astype(float)
    x, y = q[0], q[1]
    return float(
        -0.5 * (x ** 2) / VAR_X
        - 0.5 * ((y - ALPHA * (x ** 2 - VAR_X)) ** 2) / (SIGMA_Y ** 2)
    )


def log_target_grid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorized log target on a meshgrid."""
    return -0.5 * (X ** 2) / VAR_X - 0.5 * ((Y - ALPHA * (X ** 2 - VAR_X)) ** 2) / (SIGMA_Y ** 2)


def grad_neg_log_target(q: np.ndarray) -> np.ndarray:
    """Gradient of U(q) = -log pi(q) for HMC."""
    x, y = float(q[0]), float(q[1])
    resid = y - ALPHA * (x ** 2 - VAR_X)
    dU_dx = x / VAR_X - 2.0 * ALPHA * x * resid / (SIGMA_Y ** 2)
    dU_dy = resid / (SIGMA_Y ** 2)
    return np.array([dU_dx, dU_dy])


# =============================================================================
# Hamiltonian Monte Carlo
# =============================================================================
def leapfrog_trajectory(q0: np.ndarray, p0: np.ndarray, step_size: float,
                        n_leapfrog: int) -> tuple[np.ndarray, np.ndarray]:
    """Run n_leapfrog leapfrog steps and return the full position and momentum traces.

    If the trajectory diverges (e.g., from too large a step size), remaining
    entries are filled with NaN so plotting routines can skip them cleanly.
    """
    q_path = np.full((n_leapfrog + 1, 2), np.nan)
    p_path = np.full((n_leapfrog + 1, 2), np.nan)
    q = q0.copy()
    p = p0.copy()
    q_path[0] = q
    p_path[0] = p
    p = p - 0.5 * step_size * grad_neg_log_target(q)
    for i in range(n_leapfrog):
        q = q + step_size * p
        if not np.all(np.isfinite(q)) or np.any(np.abs(q) > 1e4):
            break
        if i < n_leapfrog - 1:
            p = p - step_size * grad_neg_log_target(q)
            if not np.all(np.isfinite(p)):
                break
        q_path[i + 1] = q
        p_path[i + 1] = p
    if np.all(np.isfinite(q)):
        p = p - 0.5 * step_size * grad_neg_log_target(q)
        if np.all(np.isfinite(p)):
            p_path[-1] = p
    return q_path, p_path


def hmc_sample(n_draws: int, step_size: float, n_leapfrog: int, seed: int,
               start: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Run an HMC chain on the banana target.

    Returns the chain, acceptance rate, and total gradient evaluations.
    Divergent trajectories (where positions blow up at large step sizes) are
    rejected without crashing.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty((n_draws, 2), dtype=float)
    draws[0] = np.asarray(start, dtype=float)
    accepted = 0
    grad_calls = 0
    for t in range(1, n_draws):
        q_current = draws[t - 1].copy()
        p_current = rng.normal(size=2)
        q = q_current.copy()
        p = p_current.copy()
        diverged = False
        try:
            p = p - 0.5 * step_size * grad_neg_log_target(q)
            grad_calls += 1
            for i in range(n_leapfrog):
                q = q + step_size * p
                if not np.all(np.isfinite(q)) or np.any(np.abs(q) > 1e6):
                    diverged = True
                    break
                if i < n_leapfrog - 1:
                    p = p - step_size * grad_neg_log_target(q)
                    grad_calls += 1
                    if not np.all(np.isfinite(p)):
                        diverged = True
                        break
            if not diverged:
                p = p - 0.5 * step_size * grad_neg_log_target(q)
                grad_calls += 1
                if not np.all(np.isfinite(p)):
                    diverged = True
        except (OverflowError, FloatingPointError):
            diverged = True

        if diverged:
            draws[t] = q_current
            continue

        H_current = -log_target(q_current) + 0.5 * float(np.sum(p_current ** 2))
        H_proposed = -log_target(q) + 0.5 * float(np.sum(p ** 2))
        if not np.isfinite(H_proposed):
            draws[t] = q_current
            continue
        if np.log(rng.uniform()) < (H_current - H_proposed):
            draws[t] = q
            accepted += 1
        else:
            draws[t] = q_current
    return draws, accepted / (n_draws - 1), grad_calls


# =============================================================================
# Random-walk Metropolis-Hastings on the same banana target (for comparison)
# =============================================================================
def random_walk_mh(n_draws: int, proposal_step: float, seed: int,
                   start: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Run RW-MH on the banana target. Returns chain, acceptance, and target evaluations."""
    rng = np.random.default_rng(seed)
    draws = np.empty((n_draws, 2), dtype=float)
    draws[0] = np.asarray(start, dtype=float)
    accepted = 0
    current_logp = log_target(draws[0])
    target_calls = 1
    for t in range(1, n_draws):
        proposal = draws[t - 1] + proposal_step * rng.normal(size=2)
        proposal_logp = log_target(proposal)
        target_calls += 1
        if np.log(rng.uniform()) < (proposal_logp - current_logp):
            draws[t] = proposal
            current_logp = proposal_logp
            accepted += 1
        else:
            draws[t] = draws[t - 1]
    return draws, accepted / (n_draws - 1), target_calls


# =============================================================================
# Diagnostics
# =============================================================================
def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(series, dtype=float) - float(np.mean(series))
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.ones(max_lag + 1)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return acf


def effective_sample_size(series: np.ndarray, max_lag: int = 300) -> float:
    acf = autocorrelation(series, min(max_lag, len(series) - 2))
    positive = []
    for v in acf[1:]:
        if v <= 0.0:
            break
        positive.append(v)
    tau = 1.0 + 2.0 * float(np.sum(positive))
    return float(len(series) / max(tau, 1.0))


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    setup_style()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    n_draws_hmc = 4000
    n_draws_mh = 40_000  # more MH draws so per-draw comparison is fair on grad/target calls
    burn_hmc = 500
    burn_mh = 2000
    start = np.array([3.0, 4.0])

    step_size = 0.18
    n_leapfrog = 25
    proposal_step_mh = 0.6

    # -------------------------------------------------------------------------
    # Run HMC
    # -------------------------------------------------------------------------
    hmc_draws, hmc_acceptance, hmc_grads = hmc_sample(
        n_draws=n_draws_hmc, step_size=step_size, n_leapfrog=n_leapfrog,
        seed=1, start=start,
    )
    hmc_kept = hmc_draws[burn_hmc:]

    # -------------------------------------------------------------------------
    # Run RW-MH for comparison
    # -------------------------------------------------------------------------
    mh_draws, mh_acceptance, mh_targets = random_walk_mh(
        n_draws=n_draws_mh, proposal_step=proposal_step_mh, seed=2, start=start,
    )
    mh_kept = mh_draws[burn_mh:]

    # -------------------------------------------------------------------------
    # Step-size sweep for HMC
    # -------------------------------------------------------------------------
    step_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    sweep_rows = []
    for step in step_grid:
        d, a, _ = hmc_sample(n_draws=1500, step_size=step, n_leapfrog=n_leapfrog,
                             seed=42, start=start)
        d_kept = d[300:]
        sweep_rows.append({
            "Leapfrog step size": step,
            "Acceptance rate": f"{a:.3f}",
            "Mean error x": f"{abs(float(np.mean(d_kept[:, 0])) - MEAN_X):.3f}",
            "Mean error y": f"{abs(float(np.mean(d_kept[:, 1])) - MEAN_Y):.3f}",
            "ESS x": f"{effective_sample_size(d_kept[:, 0]):.0f}",
            "ESS y": f"{effective_sample_size(d_kept[:, 1]):.0f}",
        })
    sweep_df = pd.DataFrame(sweep_rows)

    # -------------------------------------------------------------------------
    # Energy diagnostic on a single leapfrog trajectory
    # -------------------------------------------------------------------------
    rng_demo = np.random.default_rng(0)
    p0_demo = rng_demo.normal(size=2)
    q0_demo = np.array([3.5, 0.0])
    q_path, p_path = leapfrog_trajectory(q0_demo, p0_demo, step_size=step_size, n_leapfrog=n_leapfrog)
    H_path = np.array([
        -log_target(q_path[i]) + 0.5 * float(np.sum(p_path[i] ** 2))
        for i in range(len(q_path))
    ])

    # Same trajectory at a larger step size to show energy drift
    q_path_big, p_path_big = leapfrog_trajectory(q0_demo, p0_demo, step_size=0.45, n_leapfrog=n_leapfrog)
    H_path_big = np.full(len(q_path_big), np.nan)
    for i in range(len(q_path_big)):
        if np.all(np.isfinite(q_path_big[i])) and np.all(np.isfinite(p_path_big[i])):
            H_path_big[i] = -log_target(q_path_big[i]) + 0.5 * float(np.sum(p_path_big[i] ** 2))

    # -------------------------------------------------------------------------
    # Method comparison summary
    # -------------------------------------------------------------------------
    method_rows = []
    for name, kept, acc, calls, eval_label in [
        ("Random-walk MH", mh_kept, mh_acceptance, mh_targets, "Target evaluations"),
        ("Hamiltonian Monte Carlo", hmc_kept, hmc_acceptance, hmc_grads, "Gradient evaluations"),
    ]:
        method_rows.append({
            "Method": name,
            "Draws": len(kept),
            "Acceptance rate": f"{acc:.3f}",
            "Mean error x": f"{abs(float(np.mean(kept[:, 0])) - MEAN_X):.3f}",
            "Mean error y": f"{abs(float(np.mean(kept[:, 1])) - MEAN_Y):.3f}",
            "ESS x": f"{effective_sample_size(kept[:, 0]):.0f}",
            "ESS y": f"{effective_sample_size(kept[:, 1]):.0f}",
            eval_label: calls,
        })
    method_df = pd.DataFrame(method_rows).fillna("")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("Hamiltonian Monte Carlo on the banana target")
    print(f"  step size {step_size}, leapfrog steps {n_leapfrog}")
    print(f"  acceptance {hmc_acceptance:.3f}, gradient evaluations {hmc_grads:,}")
    print(f"  ESS x = {effective_sample_size(hmc_kept[:, 0]):.0f}, "
          f"ESS y = {effective_sample_size(hmc_kept[:, 1]):.0f}")
    print()
    print("Random-walk MH on the banana target")
    print(f"  proposal step {proposal_step_mh}")
    print(f"  acceptance {mh_acceptance:.3f}, target evaluations {mh_targets:,}")
    print(f"  ESS x = {effective_sample_size(mh_kept[:, 0]):.0f}, "
          f"ESS y = {effective_sample_size(mh_kept[:, 1]):.0f}")

    # -------------------------------------------------------------------------
    # Figure 1: banana density with HMC and RW-MH overlaid
    # -------------------------------------------------------------------------
    x_grid = np.linspace(-5, 5, 220)
    y_grid = np.linspace(-2, 9, 220)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.exp(log_target_grid(X, Y))

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, draws, label, color in [
        (axes1[0], mh_kept[::8], "Random-walk MH", "tab:orange"),
        (axes1[1], hmc_kept, "Hamiltonian Monte Carlo", "tab:purple"),
    ]:
        ax.contour(X, Y, Z, levels=14, cmap="viridis", alpha=0.85)
        ax.plot(draws[:, 0], draws[:, 1], "-", color=color, alpha=0.35, linewidth=0.6)
        ax.scatter(draws[::20, 0], draws[::20, 1], s=6, color=color, alpha=0.6)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 9)
        ax.set_title(f"{label} ({len(draws):,} kept draws)")
    axes1[0].set_ylabel(r"$\theta_2$")
    fig1.suptitle("Posterior coverage on the banana target", y=1.01)
    fig1.tight_layout()
    save_figure(fig1, "figures/posterior-coverage.png", dpi=150)

    # -------------------------------------------------------------------------
    # Figure 2: leapfrog trajectory + energy conservation
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.0))
    ax2a = axes2[0]
    ax2a.contour(X, Y, Z, levels=14, cmap="viridis", alpha=0.7)
    ax2a.plot(q_path[:, 0], q_path[:, 1], "o-", color="tab:purple", markersize=4,
              linewidth=1.4, label=fr"Leapfrog trajectory, $\varepsilon = {step_size:.2f}$")
    ax2a.plot(q_path[0, 0], q_path[0, 1], "s", color="tab:green", markersize=10, label="Start")
    ax2a.plot(q_path[-1, 0], q_path[-1, 1], "*", color="tab:red", markersize=14, label="Proposed end")
    ax2a.plot(q_path_big[:, 0], q_path_big[:, 1], "o-", color="tab:orange", markersize=3,
              linewidth=1.0, alpha=0.7, label=fr"Same trajectory, $\varepsilon = 0.45$ (drifts)")
    ax2a.set_xlim(-5, 5)
    ax2a.set_ylim(-2, 9)
    ax2a.set_xlabel(r"$\theta_1$")
    ax2a.set_ylabel(r"$\theta_2$")
    ax2a.set_title("Single leapfrog trajectory on the banana")
    ax2a.legend(loc="upper left", fontsize=8)

    ax2b = axes2[1]
    ax2b.plot(np.arange(len(H_path)), H_path - H_path[0], "o-", color="tab:purple",
              linewidth=1.5, markersize=4,
              label=fr"$\varepsilon = {step_size:.2f}$ (acceptable drift)")
    H_diff_big = H_path_big - H_path_big[0]
    finite_big = np.isfinite(H_diff_big)
    ax2b.plot(np.arange(len(H_diff_big))[finite_big], H_diff_big[finite_big], "o-",
              color="tab:orange", linewidth=1.5, markersize=4,
              label=fr"$\varepsilon = 0.45$ (rejection-rate drift)")
    ax2b.axhline(0.0, color="tab:gray", linestyle=":", linewidth=1.0)
    ax2b.set_yscale("symlog", linthresh=1.0)
    ax2b.set_xlabel("Leapfrog step")
    ax2b.set_ylabel(r"$H_t - H_0$ (symlog scale)")
    ax2b.set_title("Hamiltonian conservation along the leapfrog trajectory")
    ax2b.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    save_figure(fig2, "figures/leapfrog-trajectory.png", dpi=150)

    # -------------------------------------------------------------------------
    # Figure 3: efficiency comparison (autocorrelation)
    # -------------------------------------------------------------------------
    max_lag = 200
    acf_hmc_x = autocorrelation(hmc_kept[:, 0], max_lag)
    acf_hmc_y = autocorrelation(hmc_kept[:, 1], max_lag)
    acf_mh_x = autocorrelation(mh_kept[:, 0], max_lag)
    acf_mh_y = autocorrelation(mh_kept[:, 1], max_lag)

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, acf_h, acf_m, dim in [
        (axes3[0], acf_hmc_x, acf_mh_x, r"\theta_1"),
        (axes3[1], acf_hmc_y, acf_mh_y, r"\theta_2"),
    ]:
        ax.plot(np.arange(len(acf_m)), acf_m, color="tab:orange", linewidth=1.5,
                label="Random-walk MH")
        ax.plot(np.arange(len(acf_h)), acf_h, color="tab:purple", linewidth=1.5,
                label="Hamiltonian Monte Carlo")
        ax.axhline(0.0, color="tab:gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Lag")
        ax.set_title(fr"Autocorrelation of ${dim}$ draws")
        ax.legend(loc="upper right", fontsize=9)
    axes3[0].set_ylabel("Autocorrelation")
    fig3.tight_layout()
    ess_hmc_x = effective_sample_size(hmc_kept[:, 0])
    ess_mh_x = effective_sample_size(mh_kept[:, 0])

    # First lag at which each ACF drops below 0.05. argmax returns 0 when the
    # series never crosses the threshold within max_lag; map that to max_lag so
    # the committed artifact reports "still correlated past max_lag" honestly.
    def _first_lag_below(acf: np.ndarray, threshold: float = 0.05) -> int:
        below = acf < threshold
        return int(np.argmax(below)) if below.any() else max_lag

    lag_hmc_x = _first_lag_below(acf_hmc_x)
    acf_summary = pd.DataFrame(
        {
            "series": ["hmc_x", "hmc_y", "mh_x", "mh_y"],
            "lag_below_0.05": [
                lag_hmc_x,
                _first_lag_below(acf_hmc_y),
                _first_lag_below(acf_mh_x),
                _first_lag_below(acf_mh_y),
            ],
        }
    )
    acf_csv = Path(__file__).resolve().parent / "tables" / "acf-summary.csv"
    acf_csv.parent.mkdir(parents=True, exist_ok=True)
    acf_summary.to_csv(acf_csv, index=False)
    save_figure(fig3, "figures/autocorrelation-comparison.png", dpi=150)

    # -------------------------------------------------------------------------
    # Tables
    # -------------------------------------------------------------------------
    Path("tables").mkdir(parents=True, exist_ok=True)
    method_df.to_csv("tables/method-comparison.csv", index=False)
    sweep_df.to_csv("tables/stepsize-sweep.csv", index=False)

    print(f"\nGenerated: figures + tables")

    save_thumbnail("figures/posterior-coverage.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
