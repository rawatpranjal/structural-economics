#!/usr/bin/env python3
"""Neoclassical Optimal Growth (Ramsey-Cass-Koopmans): deterministic case.

Solves the infinite-horizon planner problem with Cobb-Douglas production, log
utility, and full depreciation. With this configuration the value and policy
functions admit a closed form, so VFI can be audited pointwise against the
exact Ramsey solution.

Reference: Stokey, Lucas, and Prescott (1989), Ch. 2 & 4.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def main() -> None:
    # =========================================================================
    # Calibration
    # =========================================================================
    alpha = 0.3       # Capital share in production
    A = 18.5          # Total factor productivity
    beta = 0.9        # Discount factor
    n_grid = 500      # Grid points for capital
    n_kprime = 500    # Inner grid for k'
    tol = 1e-6

    # Closed-form steady state under log + Cobb-Douglas + full depreciation
    kss = (alpha * beta * A) ** (1 / (1 - alpha))
    css = A * kss ** alpha - kss

    k_min = 0.01
    k_max = kss * 2.5

    # =========================================================================
    # Grid and primitives
    # =========================================================================
    k_grid_np = np.linspace(k_min, k_max, n_grid)

    def f_np(k):
        return A * k ** alpha

    def u_np(c):
        return np.log(np.maximum(c, 1e-15))

    # Closed form (log + Cobb-Douglas + full depreciation):
    #   g(k)   = alpha*beta*A*k^alpha          (saving rate = alpha*beta)
    #   c*(k)  = (1 - alpha*beta)*A*k^alpha
    #   V(k)   = E + B log k,    B = alpha/(1 - alpha*beta)
    B_const = alpha / (1 - alpha * beta)
    E_const = (1 / (1 - beta)) * (
        np.log(A * (1 - alpha * beta))
        + beta * alpha * np.log(A * alpha * beta) / (1 - alpha * beta)
    )

    def analytical_v(k):
        return E_const + B_const * np.log(np.maximum(k, 1e-15))

    def analytical_policy(k):
        return alpha * beta * A * np.maximum(k, 1e-15) ** alpha

    def v_interp(kprime, v_np):
        return np.interp(kprime, k_grid_np, v_np)

    # =========================================================================
    # Value Function Iteration
    # =========================================================================
    # For each capital state k, search over feasible next-period capital
    # k' in [k_min, min(A k^alpha, k_max)] and pick the maximizer of
    # log(A k^alpha - k') + beta * V(k'), where V is interpolated linearly
    # off the state grid.
    v = u_np(f_np(k_grid_np))  # initial guess: consume all output today

    for iteration in range(1, 1001):
        v_new = np.zeros(n_grid)
        policy_kprime = np.zeros(n_grid)

        for ik in range(n_grid):
            k = k_grid_np[ik]
            output = f_np(k)
            kp_max = min(output * 0.9999, k_max)
            kp_grid = np.linspace(k_min, kp_max, n_kprime)
            consumption = output - kp_grid
            values = u_np(consumption) + beta * v_interp(kp_grid, v)
            best = np.argmax(values)
            v_new[ik] = values[best]
            policy_kprime[ik] = kp_grid[best]

        error = np.max(np.abs(v_new - v))
        if iteration % 10 == 0:
            print(f"  VFI iteration {iteration:3d}, error = {error:.2e}")
        v = v_new
        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    v_star = jnp.array(v)
    k_grid = jnp.array(k_grid_np)
    policy_kprime_jnp = jnp.array(policy_kprime)
    consumption_policy = A * k_grid ** alpha - policy_kprime_jnp

    info = {"iterations": iteration, "converged": error < tol, "error": error}

    # =========================================================================
    # Closed-form benchmark on the same grid
    # =========================================================================
    v_analytical = jnp.array(analytical_v(k_grid_np))
    policy_kprime_analytical = jnp.array(analytical_policy(k_grid_np))
    consumption_analytical = A * k_grid ** alpha - policy_kprime_analytical

    # =========================================================================
    # Forward-iterate the policy from a low-capital initial condition
    # =========================================================================
    T_sim = 50
    k0 = kss * 0.1
    capital_path = np.zeros(T_sim)
    capital_path[0] = k0
    for t in range(T_sim - 1):
        kp = np.interp(capital_path[t], k_grid_np, policy_kprime)
        capital_path[t + 1] = kp
    output_path = A * capital_path ** alpha
    consumption_path = output_path - np.concatenate([capital_path[1:], [np.nan]])

    capital_path_exact = np.zeros(T_sim)
    capital_path_exact[0] = k0
    for t in range(T_sim - 1):
        capital_path_exact[t + 1] = analytical_policy(capital_path_exact[t])
    output_path_exact = A * capital_path_exact ** alpha
    consumption_path_exact = output_path_exact - np.concatenate(
        [capital_path_exact[1:], [np.nan]]
    )

    capital_path = jnp.array(capital_path)
    output_path = jnp.array(output_path)
    consumption_path = jnp.array(consumption_path)
    capital_path_exact = jnp.array(capital_path_exact)
    consumption_path_exact = jnp.array(consumption_path_exact)

    print(f"\n  Steady-state capital (closed form): k_ss = {kss:.4f}")
    print(f"  Final capital in simulation:        k_T  = {float(capital_path[-1]):.4f}")
    print(f"  Optimal saving rate:                s    = alpha*beta = {alpha*beta:.2f}")

    valid_start = max(1, n_grid // 10)
    value_error = np.asarray(v_star - v_analytical)
    policy_error = np.asarray(policy_kprime_jnp - policy_kprime_analytical)
    consumption_error = np.asarray(consumption_policy - consumption_analytical)
    max_value_error = float(np.max(np.abs(value_error[valid_start:])))
    max_policy_error = float(np.max(np.abs(policy_error[valid_start:])))
    max_consumption_error = float(np.max(np.abs(consumption_error[valid_start:])))
    max_path_error = float(np.max(np.abs(np.asarray(capital_path - capital_path_exact))))

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Optimal Growth by Value Function Iteration",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A planner allocates output between consumption today and capital tomorrow. "
        "Capital produces future output, so saving has a return that falls with $k$. "
        "The economy settles where impatience balances the marginal product of capital.\n\n"
        "The object is the policy rule $g(k)$ for next-period capital. "
        "Given $g(k)$, consumption is $c^{*}(k)=A k^{\\alpha}-g(k)$, "
        "where $A>0$ is total factor productivity and $\\alpha\\in(0,1)$ is the capital share.\n\n"
        "The log Cobb-Douglas case has the closed-form saving rate $\\alpha\\beta$, "
        "where $\\beta\\in(0,1)$ is the discount factor. "
        "Value function iteration solves the Bellman equation on a grid. "
        "Here the closed form audits the computed value and policy point by point."
    )

    report.add_equations(
        r"""
Capital $k_t$ produces output $y_t = A k_t^{\alpha}$ with $A>0$ and
$\alpha\in(0,1)$. Capital fully depreciates each period, so the resource
constraint is

$$c_t + k_{t+1} = A k_t^{\alpha},
\qquad c_t > 0, k_{t+1} \ge 0.$$

The planner maximizes discounted log utility,

$$\sum_{t=0}^{\infty} \beta^{t} \log c_t,
\qquad \beta \in (0,1),$$

with state $k$ summarizing the entire future. The Bellman equation is

$$V(k) = \max_{0 < k' < A k^{\alpha}}
\{\, \log(A k^{\alpha}-k') + \beta\, V(k') \,\}.$$

Let $g(k)$ denote the optimal $k'$ and $c^{*}(k) = A k^{\alpha} - g(k)$ the
implied consumption. The first-order and envelope conditions deliver the
Euler equation

$$u'(c_t) = \beta\, f'(k_{t+1})\, u'(c_{t+1}),
\qquad f'(k) = \alpha A k^{\alpha-1}.$$

For log utility and Cobb-Douglas production, conjecture $g(k) = s A k^{\alpha}$
with constant saving rate $s$. Substituting into the Euler equation gives
$s = \alpha\beta$, so

$$g(k) = \alpha\beta\, A\, k^{\alpha},
\qquad
c^{*}(k) = (1-\alpha\beta)\, A\, k^{\alpha}.$$

The value function is affine in $\log k$,

$$V(k) = E + B\, \log k,
\qquad
B = \frac{\alpha}{1-\alpha\beta},$$

with intercept

$$E = \frac{1}{1-\beta}\left[\,\log(A(1-\alpha\beta))
+ \frac{\beta\alpha}{1-\alpha\beta}\,\log(A\,\alpha\beta)\,\right].$$

The steady state solves $k = g(k)$, equivalently $\beta f'(k_{ss}) = 1$:

$$k_{ss} = (\alpha\beta A)^{1/(1-\alpha)},
\qquad c_{ss} = A k_{ss}^{\alpha} - k_{ss}.$$
"""
    )

    report.add_model_setup(
        "| Symbol | Value | Role |\n"
        "|--------|-------|------|\n"
        f"| $\\alpha$ | {alpha} | Capital share in $A k^{{\\alpha}}$ |\n"
        f"| $A$ | {A} | Total factor productivity |\n"
        f"| $\\beta$ | {beta} | Discount factor; pins down impatience and the saving rate |\n"
        f"| $k_{{ss}}$ | {kss:.4f} | Closed-form steady-state capital $(\\alpha\\beta A)^{{1/(1-\\alpha)}}$ |\n"
        f"| $c_{{ss}}$ | {css:.4f} | Steady-state consumption $A k_{{ss}}^{{\\alpha}} - k_{{ss}}$ |\n"
        f"| $k$ domain | $[{k_min},\\, {k_max:.2f}]$ | Capital range; upper bound is $2.5\\,k_{{ss}}$ |\n"
        f"| $N_k$ | {n_grid} | Uniform state grid for $k$ |\n"
        f"| $N_{{k'}}$ | {n_kprime} | Inner choice grid for $k'$ at each Bellman update |\n"
        f"| Tolerance $\\varepsilon$ | {tol:.0e} | Sup-norm convergence threshold |\n"
        f"| $T_{{sim}}$ | {T_sim} | Simulation horizon |\n"
        f"| $k_0$ | $0.1\\, k_{{ss}}\\approx{k0:.4f}$ | Initial capital for the transition path |"
    )

    report.add_solution_method(
        "Define the Bellman operator on bounded continuous functions of capital,\n\n"
        r"$$(TV)(k) = \max_{0 < k' < A k^{\alpha}}"
        r"\{\, \log(A k^{\alpha} - k') + \beta\, V(k') \,\}."
        "$$\n\n"
        "VFI starts from an initial value on the capital grid. "
        "At each $k_i$, the code searches over feasible $k'$ values. "
        "The feasible range is $k' \\in [k_{min},\\, A k_i^{\\alpha})$ where $k_{min}=0.01$ is the lower bound on next-period capital. "
        "It chooses the $k'$ with the highest current utility plus interpolated continuation value. "
        "The loop stops when the sup-norm change in $V$ is below $\\varepsilon$.\n\n"
        "```text\n"
        "Algorithm: Optimal-growth VFI with continuous k'\n"
        "Input : capital grid {k_i}_{i=1..N_k}, choice grid size N_{k'},\n"
        "        k_min (lower bound on k'; = 0.01), primitives (A, alpha, beta),\n"
        "        utility u(c) = log c, tolerance epsilon\n"
        "Output: value V*(k_i), capital policy g(k_i)\n"
        "  initialise V_0(k_i) = u(A k_i^alpha)             # eat-everything guess\n"
        "  for n = 0, 1, 2, ... :\n"
        "      for each state k_i :\n"
        "          y_i    <- A * k_i^alpha\n"
        "          kp_max <- min(0.9999 * y_i, k_max)        # 0.9999 keeps c > 0 at the top node\n"
        "          kp     <- N_{k'} points uniform on [k_min, kp_max]\n"
        "          c      <- y_i - kp                         # period consumption\n"
        "          V_cont <- interp(V_n, kp)                  # off-grid continuation\n"
        "          obj    <- log(c) + beta * V_cont\n"
        "          V_{n+1}(k_i) <- max(obj)\n"
        "          g(k_i)       <- argmax(obj)\n"
        "      err <- max_i | V_{n+1}(k_i) - V_n(k_i) |\n"
        "      stop when err < epsilon\n"
        "```\n\n"
        f"The iteration converges in **{info['iterations']} "
        f"steps** with sup-norm residual **{info['error']:.2e}**. "
        "The closed-form rule is computed only after VFI finishes."
    )

    # ------------------------------------------------------------------
    # Figure 1: value function vs closed form
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, v_star, color="tab:blue", linewidth=2, label="Numerical (VFI)")
    ax1.plot(k_grid, v_analytical, color="tab:red", linestyle="--", linewidth=1.5, label="Closed form")
    ax1.axvline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                label=f"$k_{{ss}} = {kss:.2f}$")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k)$")
    ax1.set_title("Value Function vs Closed Form")
    ax1.legend()
    report.add_results(
        "The value function rises and bends because capital has diminishing returns. "
        "The numerical curve matches $E+B\\log k$ except near the lowest grid points. "
        f"Outside the bottom decile, the largest value gap is **{max_value_error:.2e}**."
    )
    report.add_figure(
        "figures/value-function.png",
        "Numerical value function plotted against the closed-form $E + B\\log k$",
        fig1,
    )

    # ------------------------------------------------------------------
    # Figure 2: capital policy vs closed form
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.plot(k_grid, policy_kprime_jnp, color="tab:blue", linewidth=2, label="Numerical $g(k)$")
    ax2.plot(k_grid, policy_kprime_analytical, color="tab:red", linestyle="--", linewidth=1.5,
             label=r"Closed form $\alpha\beta A k^{\alpha}$")
    ax2.plot(k_grid, k_grid, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
             label="$45^{\\circ}$ line")
    ax2.axvline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                label=f"$k_{{ss}}={kss:.2f}$")
    ax2.set_xlabel("Capital $k$")
    ax2.set_ylabel("Next-period capital $k'$")
    ax2.set_title("Capital Policy")
    ax2.legend()
    report.add_results(
        "The policy crosses the $45^{\\circ}$ line at $k_{ss}$. "
        "Below $k_{ss}$, the planner accumulates capital. "
        "Above $k_{ss}$, the planner runs capital down. "
        f"The log case saves the constant share $\\alpha\\beta = {alpha*beta:.2f}$ of output. "
        f"The largest policy gap outside the bottom decile is **{max_policy_error:.2e}**."
    )
    report.add_figure(
        "figures/policy-function.png",
        "Capital policy $g(k)$ versus the closed-form rule $\\alpha\\beta A k^{\\alpha}$",
        fig2,
    )

    # ------------------------------------------------------------------
    # Figure 3: transition paths
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    periods = jnp.arange(T_sim)

    ax3a.plot(periods, capital_path, "o-", color="tab:blue", markersize=3, linewidth=1.5,
              label="Numerical")
    ax3a.plot(periods, capital_path_exact, color="tab:red", linestyle="--", linewidth=1.5,
              label="Closed form")
    ax3a.axhline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                 label=f"$k_{{ss}}={kss:.2f}$")
    ax3a.set_xlabel("Period $t$")
    ax3a.set_ylabel("Capital $k_t$")
    ax3a.set_title("Capital transition")
    ax3a.legend()

    ax3b.plot(periods[:-1], consumption_path[:-1], "o-", color="tab:blue", markersize=3,
              linewidth=1.5, label="Numerical")
    ax3b.plot(periods[:-1], consumption_path_exact[:-1], color="tab:red", linestyle="--",
              linewidth=1.5, label="Closed form")
    ax3b.axhline(css, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                 label=f"$c_{{ss}}={css:.2f}$")
    ax3b.set_xlabel("Period $t$")
    ax3b.set_ylabel("Consumption $c_t$")
    ax3b.set_title("Consumption transition")
    ax3b.legend()
    fig3.tight_layout()
    report.add_results(
        "Starting from $0.1\\,k_{ss}$, capital rises toward the steady state. "
        "It rises fastest when capital is scarce. "
        "Consumption also rises because the saving share is constant. "
        f"The maximum capital-path error is **{max_path_error:.2e}**."
    )
    report.add_figure(
        "figures/simulation.png",
        f"Capital and consumption transitions starting from $k_0={k0:.4f}$",
        fig3,
    )

    # ------------------------------------------------------------------
    # Pointwise audit table
    # ------------------------------------------------------------------
    sample_idx = np.linspace(valid_start, n_grid - 1, 8, dtype=int)
    table_data = {
        "k": [f"{float(k_grid[i]):.3f}" for i in sample_idx],
        "V numerical": [f"{float(v_star[i]):.4f}" for i in sample_idx],
        "V closed form": [f"{float(v_analytical[i]):.4f}" for i in sample_idx],
        "V error": [f"{float(value_error[i]):.2e}" for i in sample_idx],
        "k' numerical": [f"{float(policy_kprime_jnp[i]):.4f}" for i in sample_idx],
        "k' closed form": [f"{float(policy_kprime_analytical[i]):.4f}" for i in sample_idx],
        "k' error": [f"{float(policy_error[i]):.2e}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_results(
        "The table checks eight representative capital states. "
        "Value errors are tiny at each selected state. "
        "Policy errors are larger because $k'$ is chosen on a finite grid."
    )
    report.add_table(
        "tables/comparison.csv",
        "Numerical vs closed-form solution at selected capital states",
        df,
    )

    # ------------------------------------------------------------------
    # Committed audit artifacts so the "max ... outside bottom decile"
    # and convergence claims can be verified without re-running.
    # ------------------------------------------------------------------
    full_errors = pd.DataFrame(
        {
            "k": k_grid_np,
            "V error": value_error,
            "k' error": policy_error,
            "c error": consumption_error,
        }
    )
    full_errors.to_csv(
        Path(__file__).resolve().parent / "tables" / "full-errors.csv", index=False
    )
    convergence_log = pd.DataFrame(
        {
            "metric": [
                "vfi_iterations",
                "vfi_sup_norm_error",
                "max_value_error_above_bottom_decile",
                "max_policy_error_above_bottom_decile",
                "max_capital_path_error",
            ],
            "value": [
                info["iterations"],
                float(info["error"]),
                max_value_error,
                max_policy_error,
                max_path_error,
            ],
        }
    )
    convergence_log.to_csv(
        Path(__file__).resolve().parent / "tables" / "convergence-log.csv", index=False
    )

    report.add_takeaway(
        "The one-capital growth problem makes saving productive. "
        f"In the log Cobb-Douglas case, the exact policy saves $\\alpha\\beta = {alpha*beta:.2f}$ of output. "
        "VFI recovers that rule to grid accuracy. "
        "The example shows how to audit a Bellman solver when an exact benchmark exists."
    )

    report.add_references([
        "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 2 & 4.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.",
        "Ramsey, F. P. (1928). A Mathematical Theory of Saving. *Economic Journal*, 38(152), 543-559.",
        "Cass, D. (1965). Optimum Growth in an Aggregative Model of Capital Accumulation. *Review of Economic Studies*, 32(3), 233-240.",
        "Koopmans, T. C. (1965). On the Concept of Optimal Economic Growth. In *The Econometric Approach to Development Planning*. North-Holland.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
