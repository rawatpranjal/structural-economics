#!/usr/bin/env python3
"""Gauss-Hermite Quadrature: Nodes, Weights, and AR(1) Conditional Expectations.

Demonstrates Gauss-Hermite quadrature for integrating against a Gaussian weight.
Three test integrands are evaluated at node counts N = 3, 5, 10, 30:
  1. Polynomial: f(z) = z^6 - 3*z^4 + 2  (exact at finite N)
  2. Smooth non-polynomial: f(z) = exp(-z^2 / 4)
  3. Non-smooth: f(z) = |z|

Each is compared against Simpson's rule and Monte Carlo with matched effort.
An AR(1) conditional expectation example shows GH vs MC accuracy across a z grid.

Constants:
  SEED = 0
  AR1_RHO = 0.9
  AR1_SIGMA = 0.5
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_hermite
from scipy.integrate import simpson

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED: int = 0
AR1_RHO: float = 0.9
AR1_SIGMA: float = 0.5
N_GRID: list[int] = [3, 5, 10, 30]

# AR(1) z grid
AR1_Z_GRID: np.ndarray = np.linspace(-2.0, 2.0, 9)
AR1_MC_DRAWS: int = 10_000

# Simpson truncation interval
SIMPSON_LO: float = -6.0
SIMPSON_HI: float = 6.0


# ---------------------------------------------------------------------------
# Test integrands and exact values
#
# All three integrands are for E_phi[f(z)] where phi = N(0, 1).
# Gauss-Hermite (physicists' form) integrates against exp(-x^2).
# Change of variables: z = sqrt(2) * x, dz = sqrt(2) dx, so
#   E_phi[f(z)] = integral f(z) phi(z) dz
#              = integral f(sqrt(2)*x) * (1/sqrt(pi)) * exp(-x^2) dx
#              = (1/sqrt(pi)) * sum_n w_n * f(sqrt(2) * xi_n).
# ---------------------------------------------------------------------------

def f_poly(z: np.ndarray) -> np.ndarray:
    """Polynomial: f(z) = z^6 - 3*z^4 + 2."""
    return z**6 - 3.0 * z**4 + 2.0


def f_smooth(z: np.ndarray) -> np.ndarray:
    """Smooth non-polynomial: f(z) = exp(-z^2 / 4)."""
    return np.exp(-z**2 / 4.0)


def f_nonsmooth(z: np.ndarray) -> np.ndarray:
    """Non-smooth: f(z) = |z|."""
    return np.abs(z)


# Gaussian moments: E[z^k] for z ~ N(0,1).
#   E[z^4] = 3, E[z^6] = 15.
# So E[z^6 - 3*z^4 + 2] = 15 - 9 + 2 = 8.
EXACT_POLY: float = 8.0

# E_phi[exp(-z^2/4)] = integral (1/sqrt(2*pi)) * exp(-z^2/2) * exp(-z^2/4) dz
#   = integral (1/sqrt(2*pi)) * exp(-z^2 * 3/4) dz
#   = (1/sqrt(2*pi)) * sqrt(2*pi / (3/2))
#   = sqrt(1 / (3/2)) = sqrt(2/3) ... let me redo:
# exp(-z^2/2) * exp(-z^2/4) = exp(-3*z^2/4).
# Integral (1/sqrt(2*pi)) * exp(-3*z^2/4) dz
#   = (1/sqrt(2*pi)) * sqrt(2*pi / (3/2))  [since int exp(-a*z^2)dz = sqrt(pi/a)]
#   ... int exp(-3*z^2/4) dz = sqrt(pi / (3/4)) = sqrt(4*pi/3) = 2*sqrt(pi/3).
# So = (1/sqrt(2*pi)) * 2*sqrt(pi/3) = 2 / sqrt(2*3) = 2/sqrt(6) = sqrt(2/3).
# sqrt(2/3) = sqrt(2)/sqrt(3) ~ 0.8165.
EXACT_SMOOTH: float = float(np.sqrt(2.0 / 3.0))

# E_phi[|z|] = 2 * integral_0^inf z * (1/sqrt(2*pi)) * exp(-z^2/2) dz
#   = 2 / sqrt(2*pi) * [- exp(-z^2/2)]_0^inf = 2/sqrt(2*pi) = sqrt(2/pi).
EXACT_NONSMOOTH: float = float(np.sqrt(2.0 / np.pi))

INTEGRANDS: list[tuple[str, object, float]] = [
    ("polynomial",   f_poly,      EXACT_POLY),
    ("smooth-exp",   f_smooth,    EXACT_SMOOTH),
    ("nonsmooth-abs", f_nonsmooth, EXACT_NONSMOOTH),
]


# ---------------------------------------------------------------------------
# Gauss-Hermite approximation
# ---------------------------------------------------------------------------

def gh_expectation(f: object, N: int) -> float:
    """Approximate E_phi[f(z)] for z ~ N(0,1) using N-point Gauss-Hermite.

    The physicists' Hermite weight is exp(-x^2), so we change variables
    z = sqrt(2) * x and apply the (1/sqrt(pi)) prefactor.

    Args:
        f: Callable R -> R applied element-wise.
        N: Number of quadrature nodes.

    Returns:
        Scalar approximation to E_phi[f(z)].
    """
    xi, w = roots_hermite(N)           # nodes and weights for exp(-x^2) weight
    z_nodes = np.sqrt(2.0) * xi        # map to N(0,1) standard deviation scale
    return float(np.dot(w, f(z_nodes)) / np.sqrt(np.pi))


# ---------------------------------------------------------------------------
# Simpson's rule comparator
#
# We integrate f(z) * phi(z) on [-6, 6] with 2*N+1 equally spaced points,
# giving the same number of function evaluations as GH with N points.
# ---------------------------------------------------------------------------

def simpson_expectation(f: object, N: int) -> float:
    """Approximate E_phi[f(z)] via Simpson's rule on [-6, 6] with 2*N+1 nodes.

    Args:
        f: Callable R -> R applied element-wise.
        N: Determines node count: 2*N+1 points (matched to GH effort).

    Returns:
        Scalar approximation to E_phi[f(z)].
    """
    n_pts = 2 * N + 1
    z = np.linspace(SIMPSON_LO, SIMPSON_HI, n_pts)
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    integrand_vals = f(z) * phi_z
    return float(simpson(integrand_vals, x=z))


# ---------------------------------------------------------------------------
# Monte Carlo comparator
# ---------------------------------------------------------------------------

def mc_expectation(f: object, N: int, seed: int) -> float:
    """Approximate E_phi[f(z)] via Monte Carlo with N draws from N(0,1).

    Args:
        f: Callable R -> R applied element-wise.
        N: Number of Monte Carlo draws.
        seed: RNG seed for reproducibility.

    Returns:
        Scalar approximation to E_phi[f(z)].
    """
    rng = np.random.default_rng(seed)
    z_draws = rng.standard_normal(N)
    return float(np.mean(f(z_draws)))


# ---------------------------------------------------------------------------
# AR(1) conditional expectation
#
# z' = rho * z + sigma_eps * eta,  eta ~ N(0, 1).
# E[exp(z') | z] = exp(rho * z + sigma_eps^2 / 2)  (moment generating function).
# GH approximation: (1/sqrt(pi)) * sum_n w_n * exp(rho*z + sigma_eps*sqrt(2)*xi_n).
# ---------------------------------------------------------------------------

def gh_ar1_conditional(
    z_grid: np.ndarray,
    rho: float,
    sigma_eps: float,
    f: object,
    N: int,
) -> np.ndarray:
    """AR(1) conditional expectation E[f(z')|z] at each z in z_grid via GH.

    Args:
        z_grid: Array of conditioning values z.
        rho: AR(1) persistence.
        sigma_eps: Innovation standard deviation.
        f: Callable applied to z' = rho*z + sigma_eps*sqrt(2)*xi.
        N: Number of GH nodes.

    Returns:
        Array of length len(z_grid) with E[f(z')|z] for each z.
    """
    xi, w = roots_hermite(N)
    # z_prime[i, j] = rho * z_grid[i] + sigma_eps * sqrt(2) * xi[j]
    z_prime = rho * z_grid[:, np.newaxis] + sigma_eps * np.sqrt(2.0) * xi[np.newaxis, :]
    fvals = f(z_prime)                             # (len(z_grid), N)
    return np.dot(fvals, w) / np.sqrt(np.pi)      # (len(z_grid),)


def mc_ar1_conditional(
    z_grid: np.ndarray,
    rho: float,
    sigma_eps: float,
    f: object,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """AR(1) conditional expectation E[f(z')|z] at each z in z_grid via MC.

    Args:
        z_grid: Array of conditioning values z.
        rho: AR(1) persistence.
        sigma_eps: Innovation standard deviation.
        f: Callable applied to z'.
        n_draws: Number of Monte Carlo draws per z.
        seed: RNG seed.

    Returns:
        Array of length len(z_grid) with MC estimate of E[f(z')|z].
    """
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((len(z_grid), n_draws))   # (len(z_grid), n_draws)
    z_prime = rho * z_grid[:, np.newaxis] + sigma_eps * eta
    return np.mean(f(z_prime), axis=1)


def exact_ar1_exp(z_grid: np.ndarray, rho: float, sigma_eps: float) -> np.ndarray:
    """Closed-form E[exp(z') | z] for AR(1) z' = rho*z + sigma_eps*eta.

    Uses the moment generating function of a Gaussian:
      E[exp(rho*z + sigma_eps*eta)] = exp(rho*z + sigma_eps^2 / 2).
    """
    return np.exp(rho * z_grid + 0.5 * sigma_eps**2)


def f_exp(z: np.ndarray) -> np.ndarray:
    """Target function for AR(1) example: f(z') = exp(z')."""
    return np.exp(z)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

INTEGRAND_COLORS = {
    "polynomial":    "#4878CF",
    "smooth-exp":    "#D65F5F",
    "nonsmooth-abs": "#6ACC65",
}
INTEGRAND_LABELS = {
    "polynomial":    "Polynomial ($z^6-3z^4+2$)",
    "smooth-exp":    "Smooth ($e^{-z^2/4}$)",
    "nonsmooth-abs": "Non-smooth ($|z|$)",
}
METHOD_MARKERS = {"gh": "o-", "simpson": "s--", "mc": "^:"}
METHOD_LABELS  = {"gh": "Gauss-Hermite", "simpson": "Simpson", "mc": "Monte Carlo"}
METHOD_ALPHAS  = {"gh": 1.0, "simpson": 0.8, "mc": 0.6}


def plot_error_vs_nodes(
    errors: dict[tuple[str, str], list[float]],
    path: str,
) -> None:
    """Log-log plot of absolute error vs N for each integrand and method.

    Args:
        errors: Dict keyed by (integrand_name, method) with list of errors per N.
        path: Output file path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    integrand_names = ["polynomial", "smooth-exp", "nonsmooth-abs"]
    titles = [
        "Polynomial: $z^6 - 3z^4 + 2$",
        "Smooth: $\\exp(-z^2/4)$",
        "Non-smooth: $|z|$",
    ]

    n_arr = np.array(N_GRID, dtype=float)

    for ax, name, title in zip(axes, integrand_names, titles):
        for method in ("gh", "simpson", "mc"):
            key = (name, method)
            if key not in errors:
                continue
            err_arr = np.array(errors[key])
            # Replace zero errors (exact) with small sentinel for log scale.
            err_arr = np.where(err_arr == 0.0, 1e-16, err_arr)
            color = INTEGRAND_COLORS[name]
            lw = 2.0 if method == "gh" else 1.5
            alpha = METHOD_ALPHAS[method]
            ax.loglog(
                n_arr, err_arr,
                METHOD_MARKERS[method],
                color=color,
                lw=lw,
                alpha=alpha,
                label=METHOD_LABELS[method],
                markersize=6,
            )

        ax.set_xlabel("Nodes N")
        ax.set_ylabel("|Error|")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xticks(N_GRID)
        ax.set_xticklabels([str(n) for n in N_GRID])
        ax.set_ylim(bottom=1e-17)

    fig.suptitle("Quadrature Error vs Node Count", fontsize=13)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


def plot_gh_nodes_weights(N_vis: int, path: str) -> None:
    """Scatter of (xi_n, w_n) at N_vis nodes with the N(0,1) density overlaid.

    Nodes are shown in the z = sqrt(2)*xi scale so they align with the N(0,1)
    density curve.

    Args:
        N_vis: Number of GH nodes to visualize.
        path: Output file path.
    """
    import matplotlib.pyplot as plt

    xi, w = roots_hermite(N_vis)
    z_nodes = np.sqrt(2.0) * xi

    # Normalize weights for overlay: GH weights w_n sum to sqrt(pi),
    # so the weight in the N(0,1) representation is w_n / sqrt(pi).
    w_norm = w / np.sqrt(np.pi)

    z_curve = np.linspace(-4.0, 4.0, 400)
    phi_curve = np.exp(-0.5 * z_curve**2) / np.sqrt(2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_curve, phi_curve, color="#444444", lw=2, label="N(0,1) density")
    ax.scatter(
        z_nodes, w_norm,
        s=80, color="#4878CF", zorder=5,
        label=f"GH nodes (N={N_vis}), weight / sqrt(pi)",
    )
    for xi_i, wi in zip(z_nodes, w_norm):
        ax.vlines(xi_i, 0, wi, color="#4878CF", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("z = sqrt(2) * xi (standard-normal scale)")
    ax.set_ylabel("Density / normalized weight")
    ax.set_title(f"Gauss-Hermite Nodes and Weights (N = {N_vis})")
    ax.legend(frameon=False)
    save_figure(fig, path, dpi=150)


def plot_ar1_conditional_error(
    z_grid: np.ndarray,
    gh_est: np.ndarray,
    mc_est: np.ndarray,
    exact: np.ndarray,
    path: str,
) -> None:
    """Relative error of GH and MC for E[exp(z')|z] across z grid.

    Args:
        z_grid: Conditioning values.
        gh_est: GH estimates.
        mc_est: MC estimates.
        exact: Closed-form values.
        path: Output file path.
    """
    import matplotlib.pyplot as plt

    gh_rel_err = np.abs(gh_est - exact) / np.abs(exact)
    mc_rel_err = np.abs(mc_est - exact) / np.abs(exact)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(z_grid, exact,  color="#444444", lw=2, label="Exact")
    axes[0].plot(z_grid, gh_est, "o--", color="#4878CF", lw=1.5, label="GH (N=10)")
    axes[0].plot(z_grid, mc_est, "s:", color="#D65F5F", lw=1.5, label=f"MC ({AR1_MC_DRAWS:,} draws)")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("E[exp(z') | z]")
    axes[0].set_title("AR(1) Conditional Expectation: E[exp(z') | z]")
    axes[0].legend(frameon=False)

    axes[1].semilogy(z_grid, gh_rel_err, "o-", color="#4878CF", lw=2, label="GH (N=10)")
    axes[1].semilogy(z_grid, mc_rel_err, "s--", color="#D65F5F", lw=1.5,
                     label=f"MC ({AR1_MC_DRAWS:,} draws)")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Relative Error in E[exp(z') | z]")
    axes[1].legend(frameon=False)

    fig.suptitle(
        f"AR(1): rho={AR1_RHO}, sigma_eps={AR1_SIGMA}",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute GH quadrature, compare methods, produce figures and tables."""
    setup_style()
    t0 = time.perf_counter()

    Path("figures").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Compute errors for each integrand x method x N
    # -----------------------------------------------------------------------
    errors: dict[tuple[str, str], list[float]] = {
        (name, method): []
        for name, _, _ in INTEGRANDS
        for method in ("gh", "simpson", "mc")
    }
    results_table: list[tuple] = []

    for name, f, exact in INTEGRANDS:
        for N in N_GRID:
            gh_val    = gh_expectation(f, N)
            simp_val  = simpson_expectation(f, N)
            mc_val    = mc_expectation(f, N, seed=SEED)

            gh_err   = abs(gh_val   - exact)
            simp_err = abs(simp_val - exact)
            mc_err   = abs(mc_val   - exact)

            errors[(name, "gh")].append(gh_err)
            errors[(name, "simpson")].append(simp_err)
            errors[(name, "mc")].append(mc_err)

            results_table.append(
                (name, N, gh_val, gh_err, simp_val, simp_err, mc_val, mc_err)
            )

    # -----------------------------------------------------------------------
    # 2. Print integrand comparison table
    # -----------------------------------------------------------------------
    header_fmt = (
        f"{'Integrand':<18} {'N':>4}  "
        f"{'GH value':>12} {'GH err':>12}  "
        f"{'Simpson':>12} {'Simp err':>12}  "
        f"{'MC':>12} {'MC err':>12}"
    )
    print()
    print("Integrand quadrature comparison")
    print("=" * len(header_fmt))
    print(header_fmt)
    print("-" * len(header_fmt))
    last_name = None
    for row in results_table:
        name, N, gh_val, gh_err, simp_val, simp_err, mc_val, mc_err = row
        if last_name is not None and name != last_name:
            print()
        last_name = name
        print(
            f"{name:<18} {N:>4}  "
            f"{gh_val:>12.6f} {gh_err:>12.2e}  "
            f"{simp_val:>12.6f} {simp_err:>12.2e}  "
            f"{mc_val:>12.6f} {mc_err:>12.2e}"
        )
    print()

    # -----------------------------------------------------------------------
    # 3. AR(1) conditional expectation
    # -----------------------------------------------------------------------
    z_grid = AR1_Z_GRID
    gh_est  = gh_ar1_conditional(z_grid, AR1_RHO, AR1_SIGMA, f_exp, N=10)
    mc_est  = mc_ar1_conditional(z_grid, AR1_RHO, AR1_SIGMA, f_exp,
                                 n_draws=AR1_MC_DRAWS, seed=SEED)
    exact   = exact_ar1_exp(z_grid, AR1_RHO, AR1_SIGMA)

    gh_rel  = np.abs(gh_est - exact) / np.abs(exact)
    mc_rel  = np.abs(mc_est - exact) / np.abs(exact)

    ar1_header = (
        f"{'z':>6}  {'GH est':>12}  {'MC est':>12}  {'Exact':>12}  "
        f"{'GH rel err':>12}  {'MC rel err':>12}"
    )
    print("AR(1) conditional expectation: E[exp(z') | z]")
    print(f"rho = {AR1_RHO}, sigma_eps = {AR1_SIGMA}, GH N=10, MC draws={AR1_MC_DRAWS:,}")
    print("=" * len(ar1_header))
    print(ar1_header)
    print("-" * len(ar1_header))
    for zi, ghi, mci, exci, ghr, mcr in zip(z_grid, gh_est, mc_est, exact, gh_rel, mc_rel):
        print(
            f"{zi:>6.2f}  {ghi:>12.6f}  {mci:>12.6f}  {exci:>12.6f}  "
            f"{ghr:>12.2e}  {mcr:>12.2e}"
        )
    print()

    # -----------------------------------------------------------------------
    # 4. Figures
    # -----------------------------------------------------------------------
    plot_error_vs_nodes(errors, path="figures/error-vs-nodes.png")
    plot_gh_nodes_weights(N_vis=10, path="figures/gh-nodes-weights.png")
    plot_ar1_conditional_error(
        z_grid, gh_est, mc_est, exact,
        path="figures/ar1-conditional-error.png",
    )
    save_thumbnail("figures/error-vs-nodes.png", "figures/thumb.png")

    elapsed = time.perf_counter() - t0
    n_lines = len(Path(__file__).read_text().splitlines())
    print(f"Done in {elapsed:.2f}s. Lines: {n_lines}. Figures written to figures/")


if __name__ == "__main__":
    main()
