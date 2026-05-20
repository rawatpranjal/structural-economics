#!/usr/bin/env python3
"""Capital overhang from irreversible investment in a stochastic RBC model.

The representative household chooses next-period capital, but investment cannot
be negative. The friction matters most after capital has already been installed:
in low-productivity states the unconstrained model wants to run capital down
faster than depreciation, while the irreversible model must carry a capital
overhang.

References: Abel and Eberly (1996), Bertola and Caballero (1994), and
Cao, Luo, and Nie (2023).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def utility(c: np.ndarray, sigma: float) -> np.ndarray:
    """CRRA utility with a large penalty for infeasible consumption."""
    safe_c = np.maximum(c, 1e-12)
    u = safe_c ** (1.0 - sigma) / (1.0 - sigma)
    return np.where(c > 1e-12, u, -1e12)


def tauchen_grid(rho: float, sigma_e: float, n_z: int, m: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Discretize log productivity with Tauchen's method."""
    sigma_z = sigma_e / np.sqrt(1.0 - rho ** 2)
    z_log = np.linspace(-m * sigma_z, m * sigma_z, n_z)
    z_grid = np.exp(z_log)
    step = z_log[1] - z_log[0]

    trans_z = np.zeros((n_z, n_z))
    for i in range(n_z):
        for j in range(n_z):
            left = (z_log[j] - rho * z_log[i] - step / 2.0) / sigma_e
            right = (z_log[j] - rho * z_log[i] + step / 2.0) / sigma_e
            if j == 0:
                trans_z[i, j] = norm.cdf(right)
            elif j == n_z - 1:
                trans_z[i, j] = 1.0 - norm.cdf(left)
            else:
                trans_z[i, j] = norm.cdf(right) - norm.cdf(left)
    return z_grid, trans_z


def solve_rbc(
    *,
    constrained: bool,
    beta: float,
    alpha: float,
    sigma: float,
    delta: float,
    rho: float,
    sigma_e: float,
    n_k: int,
    n_z: int,
    tol: float = 1e-6,
    max_iter: int = 500,
    howard_steps: int = 25,
    verbose: bool = True,
) -> dict[str, np.ndarray | float | int | bool]:
    """Solve the RBC model, optionally imposing nonnegative investment."""
    k_ss = (alpha / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    y_ss = k_ss ** alpha
    c_ss = y_ss - delta * k_ss
    i_ss = delta * k_ss

    k_min = 0.55 * k_ss
    k_max = 1.60 * k_ss
    k_grid = np.linspace(k_min, k_max, n_k)
    z_grid, trans_z = tauchen_grid(rho, sigma_e, n_z)

    resources = np.zeros((n_z, n_k))
    u_mats = np.zeros((n_z, n_k, n_k))
    for iz, z in enumerate(z_grid):
        resources[iz] = z * k_grid ** alpha + (1.0 - delta) * k_grid
        c_mat = resources[iz, :, None] - k_grid[None, :]
        u_mats[iz] = utility(c_mat, sigma)

    lower_bound = (1.0 - delta) * k_grid
    feasible_mask = k_grid[None, :] >= lower_bound[:, None] - 1e-12
    boundary_allowed = (lower_bound >= k_min) & (lower_bound <= k_max)

    v = np.zeros((n_z, n_k))
    for iz in range(n_z):
        c_guess = np.maximum(resources[iz] - k_ss, 1e-8)
        v[iz] = utility(c_guess, sigma) / (1.0 - beta)

    policy_k = np.zeros((n_z, n_k))
    policy_c = np.zeros((n_z, n_k))
    binding = np.zeros((n_z, n_k), dtype=bool)
    neg_large = -1e12

    label = "irreversible" if constrained else "standard"
    if verbose:
        print(f"  Solving {label} model on {n_k} x {n_z} grid...")

    for iteration in range(1, max_iter + 1):
        v_new = np.zeros_like(v)

        for iz in range(n_z):
            ev_grid = trans_z[iz] @ v
            val_mat = u_mats[iz] + beta * ev_grid[None, :]
            if constrained:
                val_mat = np.where(feasible_mask, val_mat, neg_large)

            best_idx = np.argmax(val_mat, axis=1)
            best_val = val_mat[np.arange(n_k), best_idx]
            best_k = k_grid[best_idx]
            is_binding = np.zeros(n_k, dtype=bool)

            ev_boundary = np.zeros(n_k)
            for jz in range(n_z):
                ev_boundary += trans_z[iz, jz] * np.interp(lower_bound, k_grid, v[jz])
            boundary_c = resources[iz] - lower_bound
            boundary_val = utility(boundary_c, sigma) + beta * ev_boundary
            boundary_val = np.where(boundary_allowed, boundary_val, neg_large)
            use_boundary = boundary_val >= best_val - 1e-10
            best_val = np.where(use_boundary, boundary_val, best_val)
            best_k = np.where(use_boundary, lower_bound, best_k)
            if constrained:
                is_binding = use_boundary

            v_new[iz] = best_val
            policy_k[iz] = best_k
            policy_c[iz] = resources[iz] - best_k
            binding[iz] = is_binding

        error = float(np.max(np.abs(v_new - v)))
        v = v_new.copy()

        for _ in range(howard_steps):
            v_howard = np.zeros_like(v)
            for iz in range(n_z):
                ev_policy = np.zeros(n_k)
                for jz in range(n_z):
                    ev_policy += trans_z[iz, jz] * np.interp(policy_k[iz], k_grid, v[jz])
                v_howard[iz] = utility(policy_c[iz], sigma) + beta * ev_policy
            v = v_howard

        if verbose and iteration % 10 == 0:
            print(f"    {label} VFI iter {iteration:3d}, error = {error:.2e}")
        if error < tol:
            if verbose:
                print(f"    {label} converged in {iteration} iters (error = {error:.2e})")
            break

    return {
        "constrained": constrained,
        "V": v,
        "policy_k": policy_k,
        "policy_c": policy_c,
        "binding": binding,
        "K_grid": k_grid,
        "z_grid": z_grid,
        "trans_z": trans_z,
        "Kss": k_ss,
        "Yss": y_ss,
        "Css": c_ss,
        "Iss": i_ss,
        "K_min": k_min,
        "K_max": k_max,
        "iterations": iteration,
        "error": error,
        "beta": beta,
        "alpha": alpha,
        "sigma": sigma,
        "delta": delta,
        "rho": rho,
        "sigma_e": sigma_e,
    }


def simulate(
    sol: dict[str, np.ndarray | float | int | bool],
    z_idx: np.ndarray,
    *,
    k0: float,
) -> dict[str, np.ndarray]:
    """Simulate policies on a fixed productivity-state path."""
    k_grid = sol["K_grid"]
    z_grid = sol["z_grid"]
    alpha = float(sol["alpha"])
    delta = float(sol["delta"])
    constrained = bool(sol["constrained"])

    interp_k = RegularGridInterpolator(
        (z_grid, k_grid), sol["policy_k"], method="linear",
        bounds_error=False, fill_value=None
    )
    interp_c = RegularGridInterpolator(
        (z_grid, k_grid), sol["policy_c"], method="linear",
        bounds_error=False, fill_value=None
    )
    interp_bind = None
    if constrained:
        interp_bind = RegularGridInterpolator(
            (z_grid, k_grid), sol["binding"].astype(float), method="linear",
            bounds_error=False, fill_value=0.0
        )

    t_sim = len(z_idx)
    z_sim = z_grid[z_idx]
    k = np.zeros(t_sim)
    c = np.zeros(t_sim)
    y = np.zeros(t_sim)
    inv = np.zeros(t_sim)
    binding = np.zeros(t_sim, dtype=bool)
    k[0] = k0

    for t in range(t_sim):
        point = np.array([[z_sim[t], k[t]]])
        kp = interp_k(point).item()
        c[t] = interp_c(point).item()
        y[t] = z_sim[t] * k[t] ** alpha

        lower = (1.0 - delta) * k[t]
        if constrained and interp_bind is not None:
            binding_score = interp_bind(point).item()
            if binding_score >= 0.5 or kp <= lower + 1e-5:
                kp = lower
                binding[t] = True

        inv[t] = kp - lower
        if constrained:
            inv[t] = max(inv[t], 0.0)
        if t < t_sim - 1:
            k[t + 1] = np.clip(kp, float(sol["K_min"]), float(sol["K_max"]))

    return {"K": k, "C": c, "Y": y, "I": inv, "z": z_sim, "binding": binding}


def draw_markov_path(trans_z: np.ndarray, t_sim: int, seed: int = 42) -> np.ndarray:
    """Draw a productivity-state path from the Markov transition matrix."""
    rng = np.random.default_rng(seed)
    n_z = trans_z.shape[0]
    z_idx = np.zeros(t_sim, dtype=int)
    z_idx[0] = n_z // 2
    cdf = np.cumsum(trans_z, axis=1)
    for t in range(t_sim - 1):
        z_idx[t + 1] = min(np.searchsorted(cdf[z_idx[t]], rng.uniform()), n_z - 1)
    return z_idx


def overhang_path(n_z: int, t_sim: int = 90) -> np.ndarray:
    """Construct a low-productivity episode after the economy starts with high capital."""
    mid = n_z // 2
    low = 0
    path = np.full(t_sim, mid, dtype=int)
    path[8:32] = low
    path[32:48] = 1
    return path


def main() -> None:
    beta = 0.99
    alpha = 0.36
    sigma = 2.0
    delta = 0.025
    rho = 0.90
    sigma_e = 0.05
    n_k = 72
    n_z = 7

    print("Solving RBC models with and without the investment floor...")
    sol_irr = solve_rbc(
        constrained=True, beta=beta, alpha=alpha, sigma=sigma, delta=delta,
        rho=rho, sigma_e=sigma_e, n_k=n_k, n_z=n_z, verbose=True
    )
    sol_std = solve_rbc(
        constrained=False, beta=beta, alpha=alpha, sigma=sigma, delta=delta,
        rho=rho, sigma_e=sigma_e, n_k=n_k, n_z=n_z, verbose=True
    )
    k_grid = sol_irr["K_grid"]
    z_grid = sol_irr["z_grid"]
    k_ss = float(sol_irr["Kss"])
    y_ss = float(sol_irr["Yss"])

    inv_irr = sol_irr["policy_k"] - (1.0 - delta) * k_grid[None, :]
    inv_std = sol_std["policy_k"] - (1.0 - delta) * k_grid[None, :]
    inv_irr = np.maximum(inv_irr, 0.0)

    stress_idx = overhang_path(n_z)
    k0_stress = 1.25 * k_ss
    stress_irr = simulate(sol_irr, stress_idx, k0=k0_stress)
    stress_std = simulate(sol_std, stress_idx, k0=k0_stress)

    stationary_idx = draw_markov_path(sol_irr["trans_z"], 6000, seed=42)
    stat_irr = simulate(sol_irr, stationary_idx, k0=k_ss)
    stat_std = simulate(sol_std, stationary_idx, k0=k_ss)
    burn = 1000
    binding_share_states = float(sol_irr["binding"].mean())
    binding_share_stress = float(stress_irr["binding"].mean())
    binding_share_stationary = float(stat_irr["binding"][burn:].mean())

    def stationary_stats(sim: dict[str, np.ndarray], label: str) -> dict[str, str]:
        y = sim["Y"][burn:]
        c = sim["C"][burn:]
        inv = sim["I"][burn:]
        k = sim["K"][burn:]
        log_y = np.log(y)
        log_c = np.log(c)
        return {
            "Model": label,
            "mean K": f"{k.mean():.3f}",
            "std(Y) %": f"{100.0 * np.std(log_y - log_y.mean()):.3f}",
            "std(C)/std(Y)": f"{np.std(log_c - log_c.mean()) / np.std(log_y - log_y.mean()):.3f}",
            "mean I/Y": f"{np.mean(inv / y):.3f}",
            "I=0 frequency": f"{100.0 * sim['binding'][burn:].mean():.2f}%",
        }

    setup_style()

    # Figure 1: policy functions.
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_z))
    selected_z = [0, n_z // 2]
    for iz in selected_z:
        label = f"$z={z_grid[iz]:.3f}$"
        ax1a.plot(k_grid, inv_irr[iz], "-", color=colors[iz], linewidth=2.1, label=f"Irrev {label}")
        ax1a.plot(k_grid, inv_std[iz], "--", color=colors[iz], linewidth=1.3, label=f"Std {label}")
    ax1a.axhline(0, color="black", linewidth=1.0)
    ax1a.axvline(k_ss, color="0.35", linewidth=1.0, linestyle=":", label="$K_{ss}$")
    ax1a.set_xlabel("Capital $K$")
    ax1a.set_ylabel("Investment $I=K'-(1-\\delta)K$")
    ax1a.set_title("Investment Policy")
    ax1a.legend(fontsize=7, ncol=2)

    for iz in selected_z:
        label = f"$z={z_grid[iz]:.3f}$"
        ax1b.plot(k_grid, sol_irr["policy_c"][iz], "-", color=colors[iz], linewidth=2.1, label=f"Irrev {label}")
        ax1b.plot(k_grid, sol_std["policy_c"][iz], "--", color=colors[iz], linewidth=1.3, label=f"Std {label}")
    ax1b.axvline(k_ss, color="0.35", linewidth=1.0, linestyle=":")
    ax1b.set_xlabel("Capital $K$")
    ax1b.set_ylabel("Consumption $c$")
    ax1b.set_title("Consumption Policy")
    ax1b.legend(fontsize=7, ncol=2)
    fig1.tight_layout()
    save_figure(fig1, "figures/policy-functions.png", dpi=150)

    # Figure 2: binding region.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    k_mesh, z_mesh = np.meshgrid(k_grid, z_grid)
    binding_float = sol_irr["binding"].astype(float)
    ax2.contourf(
        k_mesh, z_mesh, binding_float, levels=[-0.5, 0.5, 1.5],
        colors=["#dbe9f6", "#f3c7b3"], alpha=0.85
    )
    ax2.contour(k_mesh, z_mesh, binding_float, levels=[0.5], colors=["#7f2704"], linewidths=2)
    ax2.axvline(k_ss, color="0.25", linewidth=1.0, linestyle=":")
    ax2.set_xlabel("Capital $K$")
    ax2.set_ylabel("TFP $z$")
    ax2.set_title("States Where the Policy Chooses $I=0$")
    from matplotlib.patches import Patch
    ax2.legend(
        handles=[
            Patch(facecolor="#dbe9f6", edgecolor="k", label="Interior investment"),
            Patch(facecolor="#f3c7b3", edgecolor="k", label="$I=0$ boundary"),
        ],
        loc="upper right",
    )
    fig2.tight_layout()
    save_figure(fig2, "figures/binding-region.png", dpi=150)

    # Figure 3: overhang experiment.
    fig3, axes = plt.subplots(2, 2, figsize=(13, 8))
    t = np.arange(len(stress_idx))
    for ax in axes.flat:
        ax.axvspan(8, 32, color="#eeeeee", alpha=0.8)
        ax.set_xlabel("Period")

    axes[0, 0].plot(t, stress_irr["z"], color="#1f77b4", linewidth=2)
    axes[0, 0].set_title("Productivity Path")
    axes[0, 0].set_ylabel("$z_t$")

    axes[0, 1].plot(t, stress_std["K"] / k_ss, "--", color="#b2182b", label="Standard")
    axes[0, 1].plot(t, stress_irr["K"] / k_ss, "-", color="#2166ac", label="Irreversible")
    axes[0, 1].fill_between(
        t, stress_std["K"] / k_ss, stress_irr["K"] / k_ss,
        where=stress_irr["K"] >= stress_std["K"], color="#92c5de", alpha=0.35
    )
    axes[0, 1].axhline(1.0, color="0.35", linestyle=":", linewidth=1)
    axes[0, 1].set_title("Capital Overhang")
    axes[0, 1].set_ylabel("$K_t/K_{ss}$")
    axes[0, 1].legend()

    axes[1, 0].plot(t, stress_std["I"] / y_ss, "--", color="#b2182b", label="Standard")
    axes[1, 0].plot(t, stress_irr["I"] / y_ss, "-", color="#2166ac", label="Irreversible")
    axes[1, 0].fill_between(
        t, -0.05, 0.25, where=stress_irr["binding"],
        color="#f4a582", alpha=0.35, label="$I=0$"
    )
    axes[1, 0].axhline(0.0, color="0.25", linewidth=1)
    axes[1, 0].set_title("Investment")
    axes[1, 0].set_ylabel("$I_t/Y_{ss}$")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(t, stress_std["C"] / stress_std["Y"], "--", color="#b2182b", label="Standard")
    axes[1, 1].plot(t, stress_irr["C"] / stress_irr["Y"], "-", color="#2166ac", label="Irreversible")
    axes[1, 1].set_title("Consumption Share")
    axes[1, 1].set_ylabel("$C_t/Y_t$")
    axes[1, 1].legend()
    fig3.tight_layout()
    save_figure(fig3, "figures/overhang-experiment.png", dpi=150)

    # Figure 4: value loss.
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    v_loss = np.maximum(sol_std["V"] - sol_irr["V"], 0.0)
    cf = ax4.contourf(k_mesh, z_mesh, v_loss, levels=22, cmap="YlOrRd")
    plt.colorbar(cf, ax=ax4, label="$V^{std}(K,z)-V^{irr}(K,z)$")
    ax4.contour(k_mesh, z_mesh, binding_float, levels=[0.5], colors="black", linewidths=1.8)
    ax4.axvline(k_ss, color="0.25", linestyle=":", linewidth=1)
    ax4.set_xlabel("Capital $K$")
    ax4.set_ylabel("TFP $z$")
    ax4.set_title("Value Loss from the Investment Floor")
    fig4.tight_layout()
    save_figure(fig4, "figures/value-difference.png", dpi=150)

    stationary_table = pd.DataFrame([
        stationary_stats(stat_irr, "Irreversible"),
        stationary_stats(stat_std, "Standard RBC"),
    ])
    Path("tables").mkdir(parents=True, exist_ok=True)
    stationary_table.to_csv("tables/stationary-moments.csv", index=False)

    print(f"\nBinding: {100.0 * binding_share_states:.1f}% grid states, "
          f"{100.0 * binding_share_stress:.1f}% stress path, "
          f"{100.0 * binding_share_stationary:.2f}% stationary")

    save_thumbnail("figures/policy-functions.png", "figures/thumb.png")
    print("Done: figures/ and tables/ regenerated.")


if __name__ == "__main__":
    main()
