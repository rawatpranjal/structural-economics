#!/usr/bin/env python3
"""Real Business Cycle Model with Endogenous Labor Supply.

Solves the RBC model (Kydland and Prescott, 1982) using value function iteration
with grid search over capital and labor. Aggregate TFP follows a 2-state Markov
chain. Simulates business cycle statistics and compares to stylized facts.

Reference: Kydland, F. and Prescott, E. (1982). "Time to Build and Aggregate Fluctuations."
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def hp_filter(y, lam=1600):
    """Hodrick-Prescott filter. Returns trend and cycle components.

    Uses the standard matrix formulation:
        min_tau  sum(y_t - tau_t)^2 + lambda * sum((tau_{t+1} - tau_t) - (tau_t - tau_{t-1}))^2

    Args:
        y: Time series (1-d array).
        lam: Smoothing parameter (1600 for quarterly data).

    Returns:
        trend: Trend component.
        cycle: Cyclical component (y - trend).
    """
    T = len(y)
    # Build the second-difference matrix D (T-2 x T)
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    I = np.eye(T)
    trend = np.linalg.solve(I + lam * D.T @ D, y)
    cycle = y - trend
    return trend, cycle


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    beta = 0.99        # Discount factor
    delta = 0.0233     # Depreciation rate
    alpha = 1.0 / 3.0  # Capital share
    phi = 1.74         # Weight on leisure in utility

    # TFP states and transition matrix
    z_vals = np.array([0.95, 1.05])   # Low, High productivity
    n_z = len(z_vals)
    P = np.array([[0.95, 0.05],
                  [0.05, 0.95]])      # Transition probabilities

    # Grids
    n_k = 50   # Capital grid points
    n_l = 50   # Labor grid points
    k_min, k_max = 9.0, 12.0
    l_min, l_max = 0.2, 0.6

    k_grid = np.linspace(k_min, k_max, n_k)
    l_grid = np.linspace(l_min, l_max, n_l)

    # VFI settings
    tol = 1e-5
    max_iter = 2000

    # =========================================================================
    # Steady state (deterministic, z=1)
    # =========================================================================
    # From FOC: r = alpha * (k/l)^(alpha-1) = 1/beta - 1 + delta
    # => k/l = ((1/alpha) * (1/beta + delta - 1))^(1/(alpha-1))
    rental_rate_ss = 1.0 / beta - 1.0 + delta
    k_l_ratio = (rental_rate_ss / alpha) ** (1.0 / (alpha - 1.0))
    wage_per_labor_ss = (1.0 - alpha) * k_l_ratio ** alpha
    net_output_per_labor_ss = k_l_ratio ** alpha - delta * k_l_ratio
    l_ss = wage_per_labor_ss / (wage_per_labor_ss + phi * net_output_per_labor_ss)
    k_ss = k_l_ratio * l_ss
    y_ss = k_ss ** alpha * l_ss ** (1.0 - alpha)
    c_ss = y_ss - delta * k_ss
    i_ss = delta * k_ss
    print(f"Steady-state k/l ratio: {k_l_ratio:.4f}")
    print(f"Deterministic steady state: k={k_ss:.4f}, l={l_ss:.4f}, c={c_ss:.4f}")

    # =========================================================================
    # Precompute return matrix for all (k, z, k', l) combinations
    # =========================================================================
    # Production: y = z * k^alpha * l^(1-alpha)
    # Budget: c = y + (1-delta)*k - k'
    # Utility: u(c, l) = log(c) + phi*log(1-l)

    print("\nPrecomputing return matrix...")

    # Vectorized production: shape (n_k, n_z, n_l)
    # k_grid[:, None, None] broadcasts with z_vals[None, :, None] and l_grid[None, None, :]
    production = (z_vals[None, :, None]
                  * k_grid[:, None, None] ** alpha
                  * l_grid[None, None, :] ** (1.0 - alpha))

    # Resources available: y + (1-delta)*k, shape (n_k, n_z, n_l)
    resources = production + (1.0 - delta) * k_grid[:, None, None]

    # Consumption: resources - k', shape (n_k, n_z, n_l, n_k) where last dim is k'
    consumption = resources[:, :, :, None] - k_grid[None, None, None, :]

    # Flow utility: log(c) + phi*log(1-l), with -inf for infeasible c <= 0
    log_leisure = np.log(1.0 - l_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        flow_utility = np.where(
            consumption > 0,
            np.log(np.maximum(consumption, 1e-300)) + phi * log_leisure[None, None, :, None],
            -np.inf,
        )
    # flow_utility shape: (n_k, n_z, n_l, n_k) = (k, z, l, k')

    print("\nStarting Value Function Iteration...")

    # Value function and policy arrays
    # Initialize V with a reasonable guess based on steady-state consumption
    V = np.zeros((n_k, n_z))
    # Better initial guess: assume agent consumes output and stays at same k
    for iz in range(n_z):
        for ik in range(n_k):
            l_guess = 0.33
            y_guess = z_vals[iz] * k_grid[ik] ** alpha * l_guess ** (1.0 - alpha)
            c_guess = max(y_guess - delta * k_grid[ik], 0.01)
            V[ik, iz] = (np.log(c_guess) + phi * np.log(1.0 - l_guess)) / (1.0 - beta)

    policy_k = np.zeros((n_k, n_z), dtype=int)   # Index into k_grid for k'
    policy_l = np.zeros((n_k, n_z), dtype=int)   # Index into l_grid for l

    # =========================================================================
    # Value Function Iteration (vectorized)
    # =========================================================================
    for iteration in range(1, max_iter + 1):
        # Expected continuation value: E[V(k',z')|z] = P @ V.T
        # V shape: (n_k, n_z), P shape: (n_z, n_z)
        # EV[k', z] = sum_z' P[z, z'] * V[k', z']
        EV = V @ P.T  # shape (n_k, n_z)

        # Total value: flow_utility(k, z, l, k') + beta * EV(k', z)
        # flow_utility: (n_k, n_z, n_l, n_k), EV: (n_k, n_z)
        # We need EV[k', z] broadcast: (1, n_z, 1, n_k) -- but k' indexes the last dim
        # EV has shape (n_k_prime, n_z), we need it as (1, n_z, 1, n_k_prime)
        total_value = flow_utility + beta * EV.T[None, :, None, :]
        # EV.T shape: (n_z, n_k), broadcast as (1, n_z, 1, n_k)

        # Maximize over (l, k') for each (k, z)
        # Reshape to (n_k, n_z, n_l * n_k) then argmax
        total_flat = total_value.reshape(n_k, n_z, n_l * n_k)
        best_flat = np.argmax(total_flat, axis=2)

        # Extract policy indices
        policy_l = best_flat // n_k
        policy_k = best_flat % n_k

        # Extract new value function
        V_new = np.max(total_flat, axis=2)

        error = np.max(np.abs(V_new - V))
        V = V_new

        if iteration % 50 == 0:
            print(f"  VFI iteration {iteration:3d}, error = {error:.2e}")

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    if error >= tol:
        print(f"  WARNING: VFI did not converge after {max_iter} iterations (error = {error:.2e})")

    info = {"iterations": iteration, "converged": error < tol, "error": float(error)}

    # Extract policy functions in levels
    k_prime_policy = k_grid[policy_k]   # (n_k, n_z)
    l_policy = l_grid[policy_l]         # (n_k, n_z)

    # =========================================================================
    # Simulate economy for 5000 periods
    # =========================================================================
    T_sim = 5000
    T_burn = 500  # Burn-in periods
    T_total = T_sim + T_burn

    np.random.seed(42)

    # Simulate TFP path
    z_indices = np.zeros(T_total, dtype=int)
    z_indices[0] = 1  # Start in high state
    for t in range(1, T_total):
        if np.random.rand() < P[z_indices[t - 1], z_indices[t - 1]]:
            z_indices[t] = z_indices[t - 1]
        else:
            z_indices[t] = 1 - z_indices[t - 1]

    # Simulate capital, labor, output, consumption, investment
    k_sim = np.zeros(T_total)
    l_sim = np.zeros(T_total)
    y_sim = np.zeros(T_total)
    c_sim = np.zeros(T_total)
    i_sim = np.zeros(T_total)

    # Start near steady state
    k_sim[0] = k_grid[n_k // 2]

    for t in range(T_total):
        iz = z_indices[t]
        # Find nearest grid point for current capital
        ik = np.argmin(np.abs(k_grid - k_sim[t]))

        # Read off policies
        l_sim[t] = l_policy[ik, iz]
        k_next = k_prime_policy[ik, iz]

        # Compute output, consumption, investment
        y_sim[t] = z_vals[iz] * k_sim[t] ** alpha * l_sim[t] ** (1.0 - alpha)
        i_sim[t] = k_next - (1.0 - delta) * k_sim[t]
        c_sim[t] = y_sim[t] - i_sim[t]

        if t < T_total - 1:
            k_sim[t + 1] = k_next

    # Discard burn-in
    k_sim = k_sim[T_burn:]
    l_sim = l_sim[T_burn:]
    y_sim = y_sim[T_burn:]
    c_sim = c_sim[T_burn:]
    i_sim = i_sim[T_burn:]
    z_sim = z_vals[z_indices[T_burn:]]

    # =========================================================================
    # HP filter and business cycle statistics
    # =========================================================================
    print("\nComputing business cycle statistics...")

    # Take logs then HP filter
    log_y = np.log(y_sim)
    log_c = np.log(c_sim)
    log_i = np.log(i_sim)
    log_k = np.log(k_sim)
    log_l = np.log(l_sim)

    _, y_cycle = hp_filter(log_y)
    _, c_cycle = hp_filter(log_c)
    _, i_cycle = hp_filter(log_i)
    _, k_cycle = hp_filter(log_k)
    _, l_cycle = hp_filter(log_l)

    # Percent deviations (multiply by 100)
    y_cycle *= 100
    c_cycle *= 100
    i_cycle *= 100
    k_cycle *= 100
    l_cycle *= 100

    # Standard deviations
    std_y = np.std(y_cycle)
    std_c = np.std(c_cycle)
    std_i = np.std(i_cycle)
    std_k = np.std(k_cycle)
    std_l = np.std(l_cycle)

    # Correlations with output
    corr_cy = np.corrcoef(c_cycle, y_cycle)[0, 1]
    corr_iy = np.corrcoef(i_cycle, y_cycle)[0, 1]
    corr_ky = np.corrcoef(k_cycle, y_cycle)[0, 1]
    corr_ly = np.corrcoef(l_cycle, y_cycle)[0, 1]

    # Relative standard deviations
    rel_c = std_c / std_y
    rel_i = std_i / std_y
    rel_k = std_k / std_y
    rel_l = std_l / std_y

    # First-order autocorrelations
    ac_y = np.corrcoef(y_cycle[1:], y_cycle[:-1])[0, 1]
    ac_c = np.corrcoef(c_cycle[1:], c_cycle[:-1])[0, 1]
    ac_i = np.corrcoef(i_cycle[1:], i_cycle[:-1])[0, 1]
    ac_k = np.corrcoef(k_cycle[1:], k_cycle[:-1])[0, 1]
    ac_l = np.corrcoef(l_cycle[1:], l_cycle[:-1])[0, 1]

    print(f"  std(Y) = {std_y:.2f}%")
    print(f"  std(C) = {std_c:.2f}%, corr(C,Y) = {corr_cy:.2f}")
    print(f"  std(I) = {std_i:.2f}%, corr(I,Y) = {corr_iy:.2f}")
    print(f"  std(K) = {std_k:.2f}%, corr(K,Y) = {corr_ky:.2f}")
    print(f"  std(L) = {std_l:.2f}%, corr(L,Y) = {corr_ly:.2f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "RBC Capital, Labor, and Business-Cycle Moments",
        "Persistent productivity shocks move output on impact, while investment and "
        "labor choices determine how the cycle propagates.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "This tutorial puts the standard real-business-cycle mechanism on a finite "
        "state space. A representative household owns the capital stock, supplies labor, "
        "and chooses investment after observing aggregate productivity. Good technology "
        "states raise the return to working and accumulating capital; bad states make "
        "investment the main margin that absorbs the shock.\n\n"
        "The example is intentionally global and nonlinear. Unlike the later "
        "[Dynare RBC](../../dynare/rbc/) tutorial, it does not log-linearize around the "
        "steady state. It keeps productivity to two Markov states so the Bellman logic, "
        "policy functions, and simulated business-cycle moments fit in one pass."
    )

    report.add_equations(
        r"""
Let $k_t$ be capital at the start of period $t$, $z_t$ aggregate TFP,
$l_t \in (0,1)$ labor, $c_t>0$ consumption, and $k_{t+1}$ next-period
capital. Output is

$$y_t = z_t k_t^\alpha l_t^{1-\alpha}, \qquad \alpha \in (0,1).$$

Capital evolves through the resource constraint

$$c_t + k_{t+1} = z_t k_t^\alpha l_t^{1-\alpha}
    + (1-\delta)k_t.$$

The household has period utility

$$u(c_t,l_t)=\log c_t + \phi \log(1-l_t), \qquad \phi>0,$$

and maximizes $\mathbb{E}_0\sum_{t=0}^{\infty}\beta^t u(c_t,l_t)$.
With $z \in \{z_L,z_H\}=\{0.95,1.05\}$ and
$P_{ij}=\Pr(z_{t+1}=z_j\mid z_t=z_i)$,

$$P = \begin{pmatrix} 0.95 & 0.05 \\ 0.05 & 0.95 \end{pmatrix}$$

is the productivity transition matrix.

The recursive problem is

$$V(k,z_i)=\max_{k',l}
\bigl[
\log c+\phi\log(1-l)+
\beta \sum_j P_{ij}V(k',z_j)
\bigr],$$

where $c=z_i k^\alpha l^{1-\alpha}+(1-\delta)k-k'$ and infeasible choices
with $c \leq 0$ are discarded. The capital policy is $g_k(k,z)$ and the labor
policy is $g_l(k,z)$.

The deterministic $z=1$ steady state is useful as a benchmark. Its capital-labor
ratio satisfies

$$\frac{k}{l} =
\left(\frac{1/\beta-1+\delta}{\alpha}\right)^{1/(\alpha-1)},$$

and labor is pinned down by

$$\frac{\phi}{1-l} =
\frac{(1-\alpha)(k/l)^\alpha}{l\left((k/l)^\alpha-\delta(k/l)\right)}.$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\alpha$ | {alpha:.4f} | Capital share |\n"
        f"| $\\phi$   | {phi} | Weight on leisure |\n"
        f"| $z$       | {{0.95, 1.05}} | Low and high TFP states |\n"
        f"| $P_{{ii}}$ | {P[0, 0]:.2f} | Probability that TFP remains in the same state |\n"
        f"| $k_{{ss}}$ | {k_ss:.4f} | Deterministic steady-state capital at $z=1$ |\n"
        f"| $l_{{ss}}$ | {l_ss:.4f} | Deterministic steady-state labor |\n"
        f"| $c_{{ss}}$ | {c_ss:.4f} | Deterministic steady-state consumption |\n"
        f"| $i_{{ss}}$ | {i_ss:.4f} | Deterministic steady-state investment |\n"
        f"| Capital grid | [{k_min}, {k_max}], {n_k} points | State and $k'$ choice grid |\n"
        f"| Labor grid | [{l_min}, {l_max}], {n_l} points | Candidate values for $l$ |\n"
        f"| Tolerance | {tol:.0e} | Sup-norm convergence criterion |\n"
        f"| Simulation periods | {T_sim} | Kept after {T_burn} burn-in periods |"
    )

    report.add_solution_method(
        "The solution approximates $V(k,z)$ on the capital grid and treats labor as a "
        "static choice nested inside the Bellman update. For every current state, the "
        "solver evaluates all feasible pairs $(l,k')$. The continuation value is the "
        "Markov expectation over tomorrow's productivity state.\n\n"
        "```text\n"
        "Algorithm: global VFI for the two-state RBC model\n"
        "Input: capital grid K, labor grid L, TFP states Z, transition matrix P,\n"
        "       primitives beta, delta, alpha, phi, tolerance epsilon\n"
        "Output: value V(k,z), capital policy g_k(k,z), labor policy g_l(k,z)\n"
        "Precompute u(k,z,l,k') for every feasible c = z k^alpha l^(1-alpha) + (1-delta)k - k'\n"
        "Initialize V_0(k,z) from a steady-state-like consumption rule\n"
        "repeat for n = 0, 1, 2, ...:\n"
        "    for each productivity state z_i:\n"
        "        EV_n(k',z_i) = sum_j P_ij V_n(k',z_j)\n"
        "    for each state (k,z_i):\n"
        "        choose (l,k') maximizing u(k,z_i,l,k') + beta * EV_n(k',z_i)\n"
        "        record V_{n+1}(k,z_i), g_l(k,z_i), and g_k(k,z_i)\n"
        "    error = max_{k,z} |V_{n+1}(k,z) - V_n(k,z)|\n"
        "until error < epsilon\n"
        "Simulate z_t from P, apply the policies, HP-filter log series, and compute moments\n"
        "```\n\n"
        "There is no closed-form stochastic policy here, so the deterministic steady state "
        "serves only as a benchmark location. The economic diagnostics are the policy "
        "functions and the simulated second moments. The VFI loop converged in "
        f"**{info['iterations']} iterations** with sup-norm error **{info['error']:.2e}**."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, V[:, 0], "b-", linewidth=2, label=f"$z = {z_vals[0]:.2f}$ (low)")
    ax1.plot(k_grid, V[:, 1], "r-", linewidth=2, label=f"$z = {z_vals[1]:.2f}$ (high)")
    ax1.axvline(k_ss, color="k", linestyle=":", linewidth=1.0, alpha=0.7, label="$k_{ss}$ at $z=1$")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k, z)$")
    ax1.set_title("Value Function")
    ax1.legend()
    report.add_figure("figures/value-function.png", "Value function by capital and TFP state", fig1,
        description="The value function is increasing in capital and higher in the good productivity state. "
        "The vertical line is the exact deterministic steady state at $z=1$; the stochastic economy does not stay there, "
        "but it is a useful reference point for reading the state space.")

    # --- Figure 2: Capital Policy Function ---
    fig2, ax2 = plt.subplots()
    ax2.plot(k_grid, k_prime_policy[:, 0], "b-", linewidth=2, label=f"$z = {z_vals[0]:.2f}$ (low)")
    ax2.plot(k_grid, k_prime_policy[:, 1], "r-", linewidth=2, label=f"$z = {z_vals[1]:.2f}$ (high)")
    ax2.plot(k_grid, k_grid, "k:", linewidth=0.8, alpha=0.5, label="45-degree line")
    ax2.axvline(k_ss, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7, label="$k_{ss}$ at $z=1$")
    ax2.set_xlabel("Capital $k$")
    ax2.set_ylabel("Next-period capital $k'$")
    ax2.set_title("Capital Policy Function")
    ax2.legend()
    report.add_figure("figures/capital-policy.png", "Capital policy by TFP state", fig2,
        description="The capital policy is the main intertemporal object. Where $g_k(k,z)$ lies above the 45-degree line, "
        "gross investment more than offsets depreciation and capital rises. The high-TFP policy is above the low-TFP policy "
        "because productivity raises the return to carrying capital into the next period.")

    # --- Figure 3: Simulated Output Path ---
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    T_plot = 200  # Show first 200 periods of simulation
    periods = np.arange(T_plot)

    axes3[0].plot(periods, y_sim[:T_plot], "b-", linewidth=1, label="Output $Y$")
    axes3[0].plot(periods, c_sim[:T_plot], "r-", linewidth=1, alpha=0.8, label="Consumption $C$")
    axes3[0].set_ylabel("Level")
    axes3[0].set_title("Simulated RBC Economy (first 200 periods)")
    axes3[0].legend()

    axes3[1].plot(periods, z_sim[:T_plot], "g-", linewidth=1)
    axes3[1].set_xlabel("Period")
    axes3[1].set_ylabel("TFP $z_t$")
    axes3[1].set_title("Productivity Shocks")
    fig3.tight_layout()
    report.add_figure("figures/simulation.png", "Simulated output, consumption, and TFP paths", fig3,
        description="A simulated path makes the mechanism less abstract. Output jumps when TFP switches, while consumption moves less "
        "because the household smooths marginal utility. The gap between output and consumption is investment, so investment takes "
        "much of the adjustment when productivity changes.")

    # --- Figure 4: Business Cycle Comovements ---
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 8))
    T_cyc = 200

    axes4[0, 0].plot(np.arange(T_cyc), y_cycle[:T_cyc], "b-", linewidth=1, label="Output")
    axes4[0, 0].plot(np.arange(T_cyc), c_cycle[:T_cyc], "r-", linewidth=1, alpha=0.8, label="Consumption")
    axes4[0, 0].set_title("Output and Consumption Cycles")
    axes4[0, 0].set_ylabel("% deviation from trend")
    axes4[0, 0].legend()

    axes4[0, 1].plot(np.arange(T_cyc), y_cycle[:T_cyc], "b-", linewidth=1, label="Output")
    axes4[0, 1].plot(np.arange(T_cyc), i_cycle[:T_cyc], "g-", linewidth=1, alpha=0.8, label="Investment")
    axes4[0, 1].set_title("Output and Investment Cycles")
    axes4[0, 1].set_ylabel("% deviation from trend")
    axes4[0, 1].legend()

    axes4[1, 0].plot(np.arange(T_cyc), y_cycle[:T_cyc], "b-", linewidth=1, label="Output")
    axes4[1, 0].plot(np.arange(T_cyc), l_cycle[:T_cyc], "m-", linewidth=1, alpha=0.8, label="Labor")
    axes4[1, 0].set_title("Output and Labor Cycles")
    axes4[1, 0].set_xlabel("Period")
    axes4[1, 0].set_ylabel("% deviation from trend")
    axes4[1, 0].legend()

    axes4[1, 1].plot(np.arange(T_cyc), y_cycle[:T_cyc], "b-", linewidth=1, label="Output")
    axes4[1, 1].plot(np.arange(T_cyc), k_cycle[:T_cyc], "c-", linewidth=1, alpha=0.8, label="Capital")
    axes4[1, 1].set_title("Output and Capital Cycles")
    axes4[1, 1].set_xlabel("Period")
    axes4[1, 1].set_ylabel("% deviation from trend")
    axes4[1, 1].legend()

    fig4.tight_layout()
    report.add_figure("figures/comovements.png", "HP-filtered cyclical comovements", fig4,
        description="The HP-filtered series show the second-moment logic usually used to summarize RBC models. Consumption is smoother "
        "than output, investment is much more volatile, and labor is procyclical. Capital is persistent because it is a stock, "
        "so it comoves less tightly with the contemporaneous output cycle.")

    # --- Table: Business Cycle Statistics ---
    bc_data = {
        "Variable": ["Output (Y)", "Consumption (C)", "Investment (I)", "Capital (K)", "Labor (L)"],
        "Std Dev (%)": [f"{std_y:.2f}", f"{std_c:.2f}", f"{std_i:.2f}", f"{std_k:.2f}", f"{std_l:.2f}"],
        "Relative Std": [f"{1.00:.2f}", f"{rel_c:.2f}", f"{rel_i:.2f}", f"{rel_k:.2f}", f"{rel_l:.2f}"],
        "Corr with Y": [f"{1.00:.2f}", f"{corr_cy:.2f}", f"{corr_iy:.2f}", f"{corr_ky:.2f}", f"{corr_ly:.2f}"],
        "Autocorr(1)": [f"{ac_y:.2f}", f"{ac_c:.2f}", f"{ac_i:.2f}", f"{ac_k:.2f}", f"{ac_l:.2f}"],
    }
    df = pd.DataFrame(bc_data)
    report.add_table(
        "tables/business-cycle-stats.csv",
        "Business Cycle Statistics (HP-filtered, simulated 5000 periods)",
        df,
        description="The table reports model moments, not empirical estimates. They are the standard RBC diagnostic: relative volatility "
        "says which margins absorb shocks, correlations say which variables are procyclical, and autocorrelations summarize propagation.",
    )

    report.add_takeaway(
        "In this finite-state RBC economy, the technology shock is the impulse, but the "
        "capital and labor policies decide the propagation. Investment is the volatile "
        f"margin, with relative standard deviation **{rel_i:.2f}**, while consumption is "
        f"smoother at **{rel_c:.2f}**. Labor is strongly procyclical "
        f"($\\mathrm{{corr}}(L,Y)$ = **{corr_ly:.2f}**), and capital is highly persistent "
        f"(autocorr = **{ac_k:.2f}**) because it moves only through accumulated investment. "
        "The useful lesson is not that this small calibration is a complete quantitative "
        "business-cycle model; it is that a Bellman equation over aggregate capital and TFP "
        "can produce the canonical RBC moment comparisons without leaving the nonlinear policy functions."
    )

    report.add_references([
        "Kydland, F. and Prescott, E. (1982). \"Time to Build and Aggregate Fluctuations.\" *Econometrica*, 50(6), 1345-1370.",
        "Cooley, T. and Prescott, E. (1995). \"Economic Growth and Business Cycles.\" In Cooley (ed.), *Frontiers of Business Cycle Research*, Princeton University Press.",
        "King, R., Plosser, C., and Rebelo, S. (1988). \"Production, Growth and Business Cycles: I. The Basic Neoclassical Model.\" *Journal of Monetary Economics*, 21(2-3), 195-232.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 12.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
