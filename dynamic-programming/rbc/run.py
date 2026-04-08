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
from lib.plotting import setup_style, save_figure
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
    k_l_ratio = ((1.0 / alpha) * (1.0 / beta + delta - 1.0)) ** (1.0 / (alpha - 1.0))
    # Steady state from intratemporal condition: phi/(1-l) = w/c, w = (1-alpha)*(k/l)^alpha
    k_ss_approx = k_l_ratio * 0.33  # l ~ 1/3 in steady state
    print(f"Steady-state k/l ratio: {k_l_ratio:.4f}")
    print(f"Approximate steady-state k: {k_ss_approx:.4f}")

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
        "Real Business Cycle Model",
        "Aggregate TFP shocks drive business cycles through optimal responses of "
        "consumption, investment, and labor supply (Kydland and Prescott, 1982).",
    )

    report.add_overview(
        "The Real Business Cycle (RBC) model is the foundational framework of modern "
        "macroeconomics. A representative agent chooses consumption, labor supply, and "
        "investment to maximize expected discounted utility. Aggregate productivity "
        "follows a stochastic process (here a 2-state Markov chain), and the economy's "
        "response to these shocks generates business cycle fluctuations.\n\n"
        "This implementation solves the full model with endogenous labor supply using "
        "value function iteration with grid search over both the capital and labor "
        "choice variables."
    )

    report.add_equations(
        r"""
$$V(k, z) = \max_{k', l} \left\{ \ln(c) + \phi \ln(1-l) + \beta \, \mathbb{E}\left[ V(k', z') \,|\, z \right] \right\}$$

subject to the budget constraint:

$$c = z \, k^\alpha \, l^{1-\alpha} + (1-\delta) k - k'$$

where $k$ is capital, $z$ is TFP, $l$ is labor, $c$ is consumption, $k'$ is next-period capital, $\alpha$ is the capital share, $\delta$ is depreciation, and $\phi$ is the weight on leisure.

**TFP process:** $z \in \{0.95, 1.05\}$ with transition matrix $P_{ij} = \Pr(z'=z_j \mid z=z_i)$:

$$P = \begin{pmatrix} 0.95 & 0.05 \\ 0.05 & 0.95 \end{pmatrix}$$

**Steady state (deterministic, $z=1$):** $k/l = \left[\frac{1}{\alpha}\left(\frac{1}{\beta} + \delta - 1\right)\right]^{1/(\alpha-1)}$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\alpha$ | {alpha:.4f} | Capital share |\n"
        f"| $\\phi$   | {phi} | Weight on leisure |\n"
        f"| $z$       | {{0.95, 1.05}} | TFP states |\n"
        f"| $K$ grid  | [{k_min}, {k_max}], {n_k} pts | Capital grid |\n"
        f"| $L$ grid  | [{l_min}, {l_max}], {n_l} pts | Labor grid |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI) with grid search:** For each state $(k, z)$, "
        "we search over all combinations of next-period capital $k'$ and current labor $l$ "
        "to find the maximizing pair. The continuation value $\\mathbb{E}[V(k',z') | z]$ "
        "is computed using the Markov transition matrix.\n\n"
        "Starting from $V_0 = 0$, iterate:\n\n"
        "$$V_{n+1}(k, z) = \\max_{k', l} \\left\\{ \\ln(c) + \\phi \\ln(1-l) + "
        "\\beta \\sum_{z'} P(z'|z) V_n(k', z') \\right\\}$$\n\n"
        f"until $\\|V_{{n+1}} - V_n\\|_\\infty < {tol}$.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e})."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, V[:, 0], "b-", linewidth=2, label=f"$z = {z_vals[0]:.2f}$ (low)")
    ax1.plot(k_grid, V[:, 1], "r-", linewidth=2, label=f"$z = {z_vals[1]:.2f}$ (high)")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k, z)$")
    ax1.set_title("Value Function")
    ax1.legend()
    report.add_figure("figures/value-function.png", "Value function V(k,z) for both TFP states", fig1,
        description="The gap between the high- and low-TFP value functions measures the welfare cost of being in a recession. "
        "Both curves are concave, reflecting the agent's desire to smooth consumption across states and over time.")

    # --- Figure 2: Capital Policy Function ---
    fig2, ax2 = plt.subplots()
    ax2.plot(k_grid, k_prime_policy[:, 0], "b-", linewidth=2, label=f"$z = {z_vals[0]:.2f}$ (low)")
    ax2.plot(k_grid, k_prime_policy[:, 1], "r-", linewidth=2, label=f"$z = {z_vals[1]:.2f}$ (high)")
    ax2.plot(k_grid, k_grid, "k:", linewidth=0.8, alpha=0.5, label="45-degree line")
    ax2.set_xlabel("Capital $k$")
    ax2.set_ylabel("Next-period capital $k'$")
    ax2.set_title("Capital Policy Function")
    ax2.legend()
    report.add_figure("figures/capital-policy.png", "Capital policy k'(k,z): investment is higher in the high-TFP state", fig2,
        description="In the high-TFP state, the agent invests more because the marginal product of capital is higher. "
        "The crossing of the 45-degree line identifies the state-dependent steady states; the economy oscillates between them as TFP switches.")

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
        description="Output responds immediately to TFP shocks but consumption is visibly smoother, reflecting the agent's optimal smoothing behavior. "
        "The gap between output and consumption is investment, which absorbs most of the volatility.")

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
    report.add_figure("figures/comovements.png", "Business cycle comovements: cyclical components from HP filter", fig4,
        description="These panels reproduce the core stylized facts that motivated the RBC literature: consumption is less volatile than output, "
        "investment is more volatile, and labor is procyclical. Capital moves sluggishly because it is a stock that adjusts only through the flow of investment.")

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
        description="These statistics are the standard diagnostic for RBC models. The relative standard deviations and correlations "
        "should be compared against U.S. quarterly data: consumption smoother than output, investment 2-3x more volatile, and all variables procyclical.",
    )

    report.add_takeaway(
        "The RBC model replicates several key stylized facts of business cycles:\n\n"
        "**Key insights:**\n"
        f"- **Investment is more volatile than output** (relative std = {rel_i:.2f}), while "
        f"**consumption is smoother** (relative std = {rel_c:.2f}). This reflects consumption "
        "smoothing: agents use investment as a buffer against shocks.\n"
        f"- **Labor and output are strongly procyclical** (corr = {corr_ly:.2f}). When "
        "productivity is high, higher wages induce agents to work more.\n"
        f"- **Capital is a slow-moving state variable** (autocorr = {ac_k:.2f}) that generates "
        "persistence in output even from i.i.d.-like shocks to TFP.\n"
        "- All variables are procyclical, consistent with a supply-driven theory of "
        "business cycles where technology shocks are the main impulse.\n"
        "- The model's main limitation: it struggles to match the observed volatility of hours "
        "worked relative to output, a well-known challenge for the basic RBC framework."
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
