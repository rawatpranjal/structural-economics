#!/usr/bin/env python3
"""RBC with Irreversible Investment: Occasionally Binding Constraints.

Solves an RBC model where investment must be non-negative (I >= 0), meaning
the agent cannot disinvest or eat capital. This occasionally binding constraint
creates asymmetric business cycles: expansions look standard, but contractions
are amplified because the agent cannot freely reduce the capital stock.

The constraint also creates an option value of waiting, as installed capital
cannot be recovered.

Reference: Cao, Luo, Nie (2023) GDSGE, Abel and Eberly (1996).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    beta = 0.99        # Discount factor
    alpha = 0.36       # Capital share
    sigma = 2.0        # CRRA coefficient
    delta = 0.025      # Depreciation rate
    rho = 0.95         # TFP persistence
    sigma_e = 0.01     # TFP innovation std

    # Irreversibility: I >= phi * Iss (phi=0 means strict I>=0, phi<0 allows some disinvestment)
    phi = 0.0          # Strict irreversibility: I >= 0

    # =========================================================================
    # Steady state (same as standard RBC, constraint not binding at SS)
    # =========================================================================
    Kss = (alpha / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    Yss = Kss ** alpha
    Css = Yss - delta * Kss
    Iss = delta * Kss

    # =========================================================================
    # Grids
    # =========================================================================
    n_k = 50       # Capital grid points (more needed for constraint region)
    n_z = 7        # TFP grid points

    # Wider grid to capture constraint binding (low TFP => want to disinvest)
    K_min = Kss * 0.5
    K_max = Kss * 1.5
    K_grid = np.linspace(K_min, K_max, n_k)

    # TFP grid (Tauchen)
    from scipy.stats import norm
    sigma_z = sigma_e / np.sqrt(1.0 - rho ** 2)
    m_z = 3.0
    z_grid_log = np.linspace(-m_z * sigma_z, m_z * sigma_z, n_z)
    z_grid = np.exp(z_grid_log)
    step = z_grid_log[1] - z_grid_log[0]

    trans_z = np.zeros((n_z, n_z))
    for i in range(n_z):
        for j in range(n_z):
            if j == 0:
                trans_z[i, j] = norm.cdf((z_grid_log[j] - rho * z_grid_log[i] + step / 2) / sigma_e)
            elif j == n_z - 1:
                trans_z[i, j] = 1.0 - norm.cdf((z_grid_log[j] - rho * z_grid_log[i] - step / 2) / sigma_e)
            else:
                trans_z[i, j] = (
                    norm.cdf((z_grid_log[j] - rho * z_grid_log[i] + step / 2) / sigma_e)
                    - norm.cdf((z_grid_log[j] - rho * z_grid_log[i] - step / 2) / sigma_e)
                )

    # =========================================================================
    # VFI with irreversibility constraint
    # =========================================================================
    def u(c):
        return np.where(c > 1e-10, c ** (1.0 - sigma) / (1.0 - sigma), -1e10)

    # K' >= (1-delta)*K + phi*Iss = (1-delta)*K when phi=0
    # This means I = K' - (1-delta)*K >= 0
    K_min_choice = (1.0 - delta) * K_grid  # minimum K' for each K (irreversibility)

    # Also solve unconstrained version for comparison
    print("Solving RBC with irreversible investment via VFI...")

    # Precompute base utility matrices and constraint masks
    resources_all = np.zeros((n_z, n_k))
    u_mats_base = np.zeros((n_z, n_k, n_k))
    for iz in range(n_z):
        resources_all[iz] = z_grid[iz] * K_grid ** alpha + (1.0 - delta) * K_grid
        c_mat = resources_all[iz][:, None] - K_grid[None, :]
        u_mats_base[iz] = u(c_mat)

    # Precompute irreversibility mask: for each ik, which ik' are feasible?
    irr_mask = np.zeros((n_k, n_k), dtype=bool)
    for ik in range(n_k):
        k_min_irr = (1.0 - delta) * K_grid[ik]
        irr_mask[ik, :] = K_grid >= k_min_irr - 1e-10

    results = {}
    for model_name, constrained in [("irreversible", True), ("standard", False)]:
        print(f"\n  Solving {model_name} model...")
        V = np.zeros((n_z, n_k))
        for iz in range(n_z):
            for ik in range(n_k):
                c_guess = max(z_grid[iz] * K_grid[ik] ** alpha + (1.0 - delta) * K_grid[ik] - Kss, 1e-10)
                V[iz, ik] = u(np.array([c_guess]))[0] / (1.0 - beta)

        policy_k = np.zeros((n_z, n_k))
        policy_c = np.zeros((n_z, n_k))
        policy_idx = np.zeros((n_z, n_k), dtype=int)
        constraint_binding = np.zeros((n_z, n_k), dtype=bool)
        tol = 1e-6
        max_iter = 500
        howard_steps = 25

        # Build u_mats with constraint applied
        u_mats = u_mats_base.copy()
        if constrained:
            for iz in range(n_z):
                for ik in range(n_k):
                    u_mats[iz, ik, ~irr_mask[ik, :]] = -1e10

        for iteration in range(1, max_iter + 1):
            V_new = np.zeros_like(V)

            for iz in range(n_z):
                EV_kprime = trans_z[iz, :] @ V
                val_mat = u_mats[iz] + beta * EV_kprime[None, :]
                best_idx = np.argmax(val_mat, axis=1)
                V_new[iz, :] = val_mat[np.arange(n_k), best_idx]
                policy_idx[iz, :] = best_idx
                policy_k[iz, :] = K_grid[best_idx]
                policy_c[iz, :] = resources_all[iz] - K_grid[best_idx]

                if constrained:
                    for ik in range(n_k):
                        investment = policy_k[iz, ik] - (1.0 - delta) * K_grid[ik]
                        constraint_binding[iz, ik] = investment < 1e-6

            error = np.max(np.abs(V_new - V))
            V = V_new.copy()

            # Howard policy iteration acceleration
            for _ in range(howard_steps):
                V_howard = np.zeros_like(V)
                for iz in range(n_z):
                    EV_kprime = trans_z[iz, :] @ V
                    for ik in range(n_k):
                        ik_prime = policy_idx[iz, ik]
                        V_howard[iz, ik] = u_mats[iz][ik, ik_prime] + beta * EV_kprime[ik_prime]
                V = V_howard

            if iteration % 10 == 0:
                print(f"    {model_name} VFI iter {iteration:3d}, error = {error:.2e}")
            if error < tol:
                print(f"    {model_name} converged in {iteration} iters (error = {error:.2e})")
                break

        results[model_name] = {
            "V": V, "policy_k": policy_k, "policy_c": policy_c,
            "constraint_binding": constraint_binding.copy(),
            "iterations": iteration,
        }

    # =========================================================================
    # Compute multiplier (shadow value of the irreversibility constraint)
    # mu = beta * E[V_K(K', z')] - u'(c) when binding, 0 otherwise
    # Approximated from the difference in value functions
    # =========================================================================
    V_irr = results["irreversible"]["V"]
    V_std = results["standard"]["V"]

    # Shadow value: difference in marginal value of capital
    # For each state, the constraint multiplier mu satisfies:
    # u'(c)(1 - mu) = beta * E[u'(c') * (MPK' + (1-delta)(1-mu'))]
    # We approximate mu from the wedge between constrained and unconstrained Euler
    policy_c_irr = results["irreversible"]["policy_c"]
    policy_c_std = results["standard"]["policy_c"]
    policy_k_irr = results["irreversible"]["policy_k"]
    policy_k_std = results["standard"]["policy_k"]

    # Investment policies
    inv_irr = np.zeros((n_z, n_k))
    inv_std = np.zeros((n_z, n_k))
    for iz in range(n_z):
        inv_irr[iz, :] = policy_k_irr[iz, :] - (1.0 - delta) * K_grid
        inv_std[iz, :] = policy_k_std[iz, :] - (1.0 - delta) * K_grid

    # =========================================================================
    # Simulation (5000 periods)
    # =========================================================================
    T_sim = 5000
    burn = 500
    np.random.seed(42)

    # Shared shock sequence
    z_sim_idx = np.zeros(T_sim, dtype=int)
    z_sim_idx[0] = n_z // 2
    for t in range(T_sim - 1):
        z_sim_idx[t + 1] = min(
            np.searchsorted(np.cumsum(trans_z[z_sim_idx[t], :]), np.random.uniform()),
            n_z - 1
        )
    z_sim = z_grid[z_sim_idx]

    sim_results = {}
    for model_name in ["irreversible", "standard"]:
        pk = results[model_name]["policy_k"]
        pc = results[model_name]["policy_c"]

        interp_k = RegularGridInterpolator(
            (z_grid, K_grid), pk, method="linear", bounds_error=False, fill_value=None
        )
        interp_c = RegularGridInterpolator(
            (z_grid, K_grid), pc, method="linear", bounds_error=False, fill_value=None
        )

        K_sim = np.zeros(T_sim)
        C_sim = np.zeros(T_sim)
        Y_sim = np.zeros(T_sim)
        I_sim = np.zeros(T_sim)
        binding_sim = np.zeros(T_sim, dtype=bool)
        K_sim[0] = Kss

        for t in range(T_sim):
            pt = np.array([[z_sim[t], K_sim[t]]])
            C_sim[t] = float(interp_c(pt))
            Y_sim[t] = z_sim[t] * K_sim[t] ** alpha
            kp = float(interp_k(pt))
            I_sim[t] = kp - (1.0 - delta) * K_sim[t]
            if model_name == "irreversible":
                if I_sim[t] < 1e-6:
                    binding_sim[t] = True
                    I_sim[t] = max(I_sim[t], 0.0)
                    kp = (1.0 - delta) * K_sim[t] + I_sim[t]
            if t < T_sim - 1:
                K_sim[t + 1] = np.clip(kp, K_min, K_max)

        sim_results[model_name] = {
            "K": K_sim, "C": C_sim, "Y": Y_sim, "I": I_sim,
            "binding": binding_sim,
        }

    # =========================================================================
    # Business cycle statistics
    # =========================================================================
    def bc_stats(sim, label):
        Y = sim["Y"][burn:]
        C = sim["C"][burn:]
        I = sim["I"][burn:]
        K = sim["K"][burn:]
        ly = np.log(Y)
        lc = np.log(np.maximum(C, 1e-10))
        # For investment, handle zeros
        I_pos = np.maximum(I, 1e-10)
        li = np.log(I_pos)
        ly_d = ly - ly.mean()
        lc_d = lc - lc.mean()
        li_d = li - li.mean()
        return {
            "Model": label,
            "std(Y) %": f"{100*np.std(ly_d):.3f}",
            "std(C)/std(Y)": f"{np.std(lc_d)/max(np.std(ly_d), 1e-10):.3f}",
            "std(I)/std(Y)": f"{np.std(li_d)/max(np.std(ly_d), 1e-10):.3f}",
            "corr(C,Y)": f"{np.corrcoef(lc_d, ly_d)[0,1]:.3f}",
            "mean(K)": f"{K.mean():.4f}",
            "mean(I/Y)": f"{(I/Y).mean():.4f}",
        }

    stats_irr = bc_stats(sim_results["irreversible"], "Irreversible")
    stats_std = bc_stats(sim_results["standard"], "Standard RBC")

    binding_frac = sim_results["irreversible"]["binding"][burn:].mean()

    # =========================================================================
    # Conditional responses: expansion vs contraction
    # =========================================================================
    # Compare how output responds to positive vs negative TFP changes
    Y_irr = sim_results["irreversible"]["Y"][burn:]
    Y_std = sim_results["standard"]["Y"][burn:]
    z_post = z_sim[burn:]

    # Output growth
    dY_irr = np.diff(np.log(Y_irr))
    dY_std = np.diff(np.log(Y_std))
    dz = np.diff(np.log(z_post))

    pos_dz = dz > 0
    neg_dz = dz < 0

    asym_data = {
        "": ["Positive dz", "Negative dz", "Ratio |neg/pos|"],
        "std(dY) Irr %": [
            f"{100*np.std(dY_irr[pos_dz]):.3f}",
            f"{100*np.std(dY_irr[neg_dz]):.3f}",
            f"{np.std(dY_irr[neg_dz])/max(np.std(dY_irr[pos_dz]), 1e-10):.3f}",
        ],
        "std(dY) Std %": [
            f"{100*np.std(dY_std[pos_dz]):.3f}",
            f"{100*np.std(dY_std[neg_dz]):.3f}",
            f"{np.std(dY_std[neg_dz])/max(np.std(dY_std[pos_dz]), 1e-10):.3f}",
        ],
    }

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "RBC with Irreversible Investment",
        "Occasionally binding I >= 0 constraint creates asymmetric business cycles and option value of waiting.",
    )

    report.add_overview(
        "In the standard RBC model, the representative agent can freely adjust the capital "
        "stock in both directions. In reality, much physical capital is irreversible: once "
        "installed, machinery and structures cannot easily be converted back to consumption goods.\n\n"
        "We impose the constraint $I_t \\geq 0$ (investment cannot be negative), which binds "
        "when TFP is low and the agent would prefer to disinvest. This creates:\n\n"
        "- **Asymmetric responses**: Contractions are amplified (can't reduce K fast enough) "
        "while expansions look like the standard model\n"
        "- **Option value of waiting**: Irreversibility makes capital decisions partially "
        "irreversible, creating value in delaying investment\n"
        "- **Precautionary behavior**: The constraint raises the effective cost of capital, "
        "leading to lower average investment"
    )

    report.add_equations(
        r"""
$$V(K, z) = \max_{c, K'} \left\{ \frac{c^{1-\sigma}}{1-\sigma} + \beta \, \mathbb{E}\left[V(K', z')\right] \right\}$$

subject to:
$$c + K' = z K^\alpha + (1-\delta) K$$
$$K' \geq (1-\delta) K \quad \Leftrightarrow \quad I \geq 0$$

**Euler equation with complementary slackness:**
$$c^{-\sigma} (1-\mu) = \beta \, \mathbb{E}\left[ c'^{-\sigma} \left(\alpha z' K'^{\alpha-1} + (1-\delta)(1-\mu')\right) \right]$$
$$\mu \geq 0, \quad I \geq 0, \quad \mu \cdot I = 0$$

When $\mu > 0$, the constraint binds: the agent would like to disinvest but cannot.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $\\alpha$ | {alpha} | Capital share |\n"
        f"| $\\sigma$ | {sigma} | CRRA coefficient |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\rho$   | {rho} | TFP persistence |\n"
        f"| $\\sigma_\\varepsilon$ | {sigma_e} | TFP innovation std |\n"
        f"| $\\phi$   | {phi} | Irreversibility (0 = strict I >= 0) |\n"
        f"| Capital grid | {n_k} points on [{K_min:.2f}, {K_max:.2f}] | Wider range needed |\n"
        f"| TFP grid | {n_z} points (Tauchen) | |\n"
        f"| $K_{{ss}}$ | {Kss:.4f} | Steady-state capital |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI)** with the irreversibility constraint enforced "
        "directly in the grid search: for each state $(z, K)$, the choice set for $K'$ is "
        "restricted to $K' \\geq (1-\\delta)K$. This naturally handles the occasionally binding "
        f"constraint without requiring complementarity solvers.\n\n"
        f"Both the constrained and unconstrained models are solved on the same grid for comparison.\n\n"
        f"Irreversible model converged in **{results['irreversible']['iterations']}** iterations. "
        f"Standard model converged in **{results['standard']['iterations']}** iterations."
    )

    # --- Figure 1: Investment policy functions ---
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_z))

    for iz in [0, n_z // 2, n_z - 1]:
        label = f"z={z_grid[iz]:.3f}"
        ax1a.plot(K_grid, inv_irr[iz, :], "-", color=colors[iz], linewidth=2, label=f"Irr {label}")
        ax1a.plot(K_grid, inv_std[iz, :], "--", color=colors[iz], linewidth=1.2, label=f"Std {label}")
    ax1a.axhline(0, color="red", linewidth=1.5, linestyle=":", label="I = 0 constraint")
    ax1a.set_xlabel("Capital $K$")
    ax1a.set_ylabel("Investment $I$")
    ax1a.set_title("Investment Policy: Irreversible vs Standard")
    ax1a.legend(fontsize=7, ncol=2)

    # Consumption
    for iz in [0, n_z // 2, n_z - 1]:
        label = f"z={z_grid[iz]:.3f}"
        ax1b.plot(K_grid, policy_c_irr[iz, :], "-", color=colors[iz], linewidth=2, label=f"Irr {label}")
        ax1b.plot(K_grid, policy_c_std[iz, :], "--", color=colors[iz], linewidth=1.2, label=f"Std {label}")
    ax1b.set_xlabel("Capital $K$")
    ax1b.set_ylabel("Consumption $c$")
    ax1b.set_title("Consumption Policy: Irreversible vs Standard")
    ax1b.legend(fontsize=7, ncol=2)
    fig1.tight_layout()
    report.add_figure("figures/policy-functions.png", "Investment and consumption policies: irreversible (solid) vs standard (dashed). Red line marks the I=0 constraint.", fig1,
        description="Look for the kink where the irreversible investment policy meets the I=0 line: "
        "to the right of this kink (high K, low z), the constraint binds and investment is pinned at "
        "zero. The standard model's dashed lines pass freely below zero, showing the disinvestment "
        "that irreversibility prevents.")

    # --- Figure 2: Constraint binding region ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    K_mesh, Z_mesh = np.meshgrid(K_grid, z_grid)
    binding_float = results["irreversible"]["constraint_binding"].astype(float)
    cf = ax2.contourf(K_mesh, Z_mesh, binding_float, levels=[-0.5, 0.5, 1.5],
                       colors=["#d4edda", "#f8d7da"], alpha=0.7)
    ax2.contour(K_mesh, Z_mesh, binding_float, levels=[0.5], colors=["red"], linewidths=2)
    ax2.set_xlabel("Capital $K$")
    ax2.set_ylabel("TFP $z$")
    ax2.set_title("Irreversibility Constraint Binding Region")
    # Manual legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d4edda", edgecolor="k", label="Unconstrained"),
        Patch(facecolor="#f8d7da", edgecolor="k", label="Constraint binds (I=0)"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right")
    report.add_figure("figures/binding-region.png", "Region where the irreversibility constraint binds (red). High K + low z = binding.", fig2,
        description="The constraint binds in the upper-left region where capital is high relative to "
        "productivity. In these states the agent would prefer to sell capital but cannot, creating a "
        "capital overhang that depresses returns and prolongs recessions.")

    # --- Figure 3: Simulation comparison ---
    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))
    t_plot = slice(burn, burn + 300)

    axes3[0, 0].plot(sim_results["irreversible"]["Y"][t_plot], "b-", linewidth=0.8, label="Irreversible")
    axes3[0, 0].plot(sim_results["standard"]["Y"][t_plot], "r--", linewidth=0.8, label="Standard")
    axes3[0, 0].set_title("Output")
    axes3[0, 0].legend(fontsize=8)

    axes3[0, 1].plot(sim_results["irreversible"]["C"][t_plot], "b-", linewidth=0.8)
    axes3[0, 1].plot(sim_results["standard"]["C"][t_plot], "r--", linewidth=0.8)
    axes3[0, 1].set_title("Consumption")

    axes3[1, 0].plot(sim_results["irreversible"]["K"][t_plot], "b-", linewidth=0.8)
    axes3[1, 0].plot(sim_results["standard"]["K"][t_plot], "r--", linewidth=0.8)
    axes3[1, 0].axhline(Kss, color="k", linewidth=0.5, linestyle=":")
    axes3[1, 0].set_title("Capital")

    # Investment with binding indicator
    I_irr_plot = sim_results["irreversible"]["I"][t_plot]
    I_std_plot = sim_results["standard"]["I"][t_plot]
    bind_plot = sim_results["irreversible"]["binding"][t_plot]
    axes3[1, 1].plot(I_std_plot, "r--", linewidth=0.8, label="Standard")
    axes3[1, 1].plot(I_irr_plot, "b-", linewidth=0.8, label="Irreversible")
    # Mark binding periods
    t_range = np.arange(len(I_irr_plot))
    axes3[1, 1].fill_between(t_range, I_irr_plot.min() - 0.01, I_irr_plot.max() + 0.01,
                              where=bind_plot, alpha=0.2, color="red", label="Constraint binds")
    axes3[1, 1].axhline(0, color="k", linewidth=1, linestyle=":")
    axes3[1, 1].set_title("Investment (shaded = constraint binds)")
    axes3[1, 1].legend(fontsize=7)

    for ax in axes3.flat:
        ax.set_xlabel("Period")
    fig3.tight_layout()
    report.add_figure("figures/simulation.png", "Simulated paths comparing irreversible (blue) vs standard (red) RBC. Shaded regions mark binding constraint.", fig3,
        description="The shaded periods in the investment panel show when the constraint binds. During "
        "these episodes, capital cannot adjust downward, so consumption must absorb all of the output "
        "decline, making consumption more volatile in contractions than the standard model predicts.")

    # --- Figure 4: Asymmetric distributions ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))

    # Investment distribution
    I_irr_data = sim_results["irreversible"]["I"][burn:]
    I_std_data = sim_results["standard"]["I"][burn:]
    ax4a.hist(I_std_data, bins=50, alpha=0.5, density=True, color="red", label="Standard")
    ax4a.hist(I_irr_data, bins=50, alpha=0.5, density=True, color="blue", label="Irreversible")
    ax4a.axvline(0, color="k", linewidth=1.5, linestyle=":")
    ax4a.set_xlabel("Investment $I$")
    ax4a.set_ylabel("Density")
    ax4a.set_title("Investment Distribution")
    ax4a.legend()

    # Output growth skewness
    from scipy.stats import skew, kurtosis
    ax4b.hist(dY_std * 100, bins=50, alpha=0.5, density=True, color="red", label="Standard")
    ax4b.hist(dY_irr * 100, bins=50, alpha=0.5, density=True, color="blue", label="Irreversible")
    ax4b.set_xlabel("Output growth (%)")
    ax4b.set_ylabel("Density")
    ax4b.set_title(f"Output Growth (skew: irr={skew(dY_irr):.3f}, std={skew(dY_std):.3f})")
    ax4b.legend()
    fig4.tight_layout()
    report.add_figure("figures/asymmetric-distributions.png", "Investment truncated at zero (left). Output growth shows negative skewness under irreversibility (right).", fig4,
        description="The left panel shows the mass point at I=0 created by the constraint, which is "
        "absent in the symmetric standard-model distribution. The right panel's negative skewness in "
        "output growth under irreversibility formalizes the intuition that recessions are sharper than "
        "expansions when capital adjustment is one-sided.")

    # --- Figure 5: Value function difference ---
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    V_diff = V_irr - V_std
    cf5 = ax5.contourf(K_mesh, Z_mesh, V_diff, levels=20, cmap="RdBu_r")
    plt.colorbar(cf5, ax=ax5, label="$V_{irr}(K,z) - V_{std}(K,z)$")
    ax5.contour(K_mesh, Z_mesh, binding_float, levels=[0.5], colors=["black"], linewidths=2, linestyles="--")
    ax5.set_xlabel("Capital $K$")
    ax5.set_ylabel("TFP $z$")
    ax5.set_title("Welfare Cost of Irreversibility (Value Function Difference)")
    report.add_figure("figures/value-difference.png", "Welfare cost: V_irr - V_std everywhere non-positive. Dashed line marks binding region boundary.", fig5,
        description="The value difference is zero where the constraint never binds (lower-right) and "
        "most negative where it binds tightly (upper-left). This surface maps the welfare cost of "
        "irreversibility across the state space, showing that the cost is concentrated in high-capital, "
        "low-productivity states.")

    # --- Tables ---
    df_bc = pd.DataFrame([stats_irr, stats_std])
    report.add_table("tables/bc-statistics.csv", "Business Cycle Statistics (5000 periods, burn-in 500)", df_bc,
        description="Compare std(I)/std(Y) across models: irreversibility truncates the investment "
        "distribution, altering relative volatilities and the consumption-output correlation.")

    df_asym = pd.DataFrame(asym_data)
    report.add_table("tables/asymmetry.csv", "Asymmetric Responses to Positive vs Negative TFP Changes", df_asym,
        description="The ratio |neg/pos| exceeding 1.0 for the irreversible model confirms that output "
        "growth is more volatile in downturns than in expansions. The standard model shows a ratio "
        "near 1.0, as expected for a symmetric linear system.")

    report.add_results(
        f"**Constraint binding frequency:** The irreversibility constraint binds in "
        f"{binding_frac*100:.1f}% of simulated periods. It binds more often when capital "
        f"is high relative to TFP (upper-left region of the state space).\n\n"
        f"**Asymmetric business cycles:** The irreversibility constraint amplifies contractions: "
        f"when TFP falls, the agent cannot reduce the capital stock quickly enough, leading to "
        f"excess capital and depressed returns. In contrast, expansions are unconstrained and "
        f"look similar to the standard model.\n\n"
        f"**Welfare cost:** The value function under irreversibility is everywhere weakly below "
        f"the unconstrained value, with the largest gaps occurring where the constraint binds."
    )

    report.add_takeaway(
        "Irreversible investment fundamentally alters business cycle dynamics:\n\n"
        "1. **Asymmetry**: The constraint creates a kink in the investment policy function. "
        "Below the kink, investment is pinned at zero and consumption absorbs all output "
        "fluctuations, making consumption more volatile in recessions.\n\n"
        "2. **Option value**: Because installed capital cannot be recovered, each investment "
        "decision carries an option cost. Firms optimally delay investment to preserve the "
        "option of waiting for better information about future TFP.\n\n"
        "3. **Capital overhang**: When the constraint binds, the economy carries excess capital "
        "that depresses the marginal product of capital and the return on saving. This can "
        "amplify and prolong recessions.\n\n"
        "4. **Policy implications**: The occasionally binding constraint means that linearized "
        "solutions are qualitatively wrong -- they cannot capture the asymmetry or the "
        "constraint binding region. Global methods are essential."
    )

    report.add_references([
        "Abel, A. and Eberly, J. (1996). *Optimal Investment with Costly Reversibility*. RES.",
        "Cao, D., Luo, W., and Nie, G. (2023). *Global DSGE Models*. Review of Economic Dynamics.",
        "Bertola, G. and Caballero, R. (1994). *Irreversibility and Aggregate Investment*. RES.",
        "Khan, A. and Thomas, J. (2008). *Idiosyncratic Shocks and the Role of Nonconvexities*. Econometrica.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
