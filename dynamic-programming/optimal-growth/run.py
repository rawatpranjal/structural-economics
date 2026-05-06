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
        "Productive capital, the Ramsey transition, and a closed-form audit for VFI.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "This is the discrete-time Ramsey-Cass-Koopmans planner: one good, one factor "
        "(capital), and a representative agent who chooses consumption to maximize "
        "discounted utility. Compared with [cake eating](../cake-eating/), the only new "
        "ingredient is that the state is *productive*. Saving a unit of output as capital "
        "delivers $\\alpha A k^{\\alpha-1}$ extra units of output tomorrow rather than "
        "the gross return of one that disciplines the cake problem. That single change "
        "introduces diminishing returns, an interior steady state, and the trade-off "
        "between the impatience rate $1/\\beta - 1$ and the marginal product of capital.\n\n"
        "With log utility, Cobb-Douglas production, and full depreciation, the planner's "
        "problem has a closed form: the optimal saving rate is the constant "
        "$\\alpha\\beta$, the value function is affine in $\\log k$, and the transition "
        "to the Ramsey steady state $k_{ss}=(\\alpha\\beta A)^{1/(1-\\alpha)}$ is "
        "monotone. This is the only one-sector growth calibration where every numerical "
        "object has an exact analytical twin, which is what makes it the natural audit "
        "for a generic Bellman solver before risk, partial depreciation, labor, or "
        "equilibrium prices break the closed form. The same recursion reappears, with "
        "different state spaces, in the [RBC tutorial](../rbc/) once productivity shocks "
        "are added and in [Aiyagari](../aiyagari/) once the planner is replaced by a "
        "continuum of constrained households facing market-determined factor prices."
    )

    report.add_equations(
        r"""
Capital $k_t$ produces output $y_t = A k_t^{\alpha}$ with $A>0$ and
$\alpha\in(0,1)$. Capital fully depreciates each period, so the resource
constraint is

$$c_t + k_{t+1} \;=\; A k_t^{\alpha},
\qquad c_t > 0,\; k_{t+1} \ge 0.$$

The planner maximizes discounted log utility,

$$\sum_{t=0}^{\infty} \beta^{t} \log c_t,
\qquad \beta \in (0,1),$$

with state $k$ summarizing the entire future. The Bellman equation is

$$V(k) \;=\; \max_{0 < k' < A k^{\alpha}}
\Bigl\{\, \log\bigl(A k^{\alpha}-k'\bigr) + \beta\, V(k') \,\Bigr\}.$$

Let $g(k)$ denote the optimal $k'$ and $c^{*}(k) = A k^{\alpha} - g(k)$ the
implied consumption. Differentiating inside the max and applying the envelope
theorem $V'(k) = u'(c^{*}(k))\, f'(k)$ delivers the **Euler equation**

$$u'(c_t) \;=\; \beta\, f'(k_{t+1})\, u'(c_{t+1}),
\qquad f'(k) = \alpha A k^{\alpha-1}.$$

The shadow value of capital today equals discounted shadow value tomorrow
*scaled by the gross return on capital*. The cake-eating Euler equation is the
$f'(k)\equiv 1$ special case.

For log utility and Cobb-Douglas production, conjecture $g(k) = s A k^{\alpha}$
with constant saving rate $s$. Substituting into the Euler equation gives
$s = \alpha\beta$, so

$$g(k) \;=\; \alpha\beta\, A\, k^{\alpha},
\qquad
c^{*}(k) \;=\; (1-\alpha\beta)\, A\, k^{\alpha}.$$

The value function is affine in $\log k$,

$$V(k) \;=\; E + B\, \log k,
\qquad
B \;=\; \frac{\alpha}{1-\alpha\beta},$$

with intercept

$$E \;=\; \frac{1}{1-\beta}\!\left[\,\log\bigl(A(1-\alpha\beta)\bigr)
+ \frac{\beta\alpha}{1-\alpha\beta}\,\log\bigl(A\,\alpha\beta\bigr)\,\right].$$

The steady state solves $k = g(k)$, equivalently $\beta f'(k_{ss}) = 1$:

$$k_{ss} \;=\; \bigl(\alpha\beta A\bigr)^{1/(1-\alpha)},
\qquad c_{ss} \;=\; A k_{ss}^{\alpha} - k_{ss}.$$

The closed form depends on all three assumptions jointly. Drop log utility,
introduce partial depreciation, or replace the production function and the
Ramsey transition still exists, but $g$ and $V$ have to be solved numerically.
That generic case is exactly what VFI is for; the calibration here is the
sharpest available test of whether the solver gets it right.
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
        r"$$(TV)(k) \;=\; \max_{0 < k' < A k^{\alpha}}"
        r"\Bigl\{\, \log\bigl(A k^{\alpha} - k'\bigr) + \beta\, V(k') \,\Bigr\}."
        "$$\n\n"
        "Blackwell's monotonicity and discounting conditions hold, so $T$ is a "
        "contraction with modulus $\\beta$. Successive iterates satisfy "
        "$\\|V_n - V\\|_{\\infty} \\le \\beta^{n}\\|V_0 - V\\|_{\\infty}$, which "
        "fixes the convergence rate and the stopping rule. With $\\beta=" f"{beta}$ "
        "the bound predicts roughly $\\log(\\varepsilon)/\\log(\\beta)$ iterations "
        "to reach tolerance $\\varepsilon$.\n\n"
        "Numerically, $V$ is tabulated on a uniform state grid for $k$. At each "
        "state, the maximizer is searched on a finer grid of candidate next-period "
        "capital values, and the continuation $V(k')$ is recovered by linear "
        "interpolation against the current iterate. Two implementation choices "
        "matter economically: (i) the upper end of the state grid sits well above "
        "$k_{ss}$ so that the policy converges to $g(k)<k$ before hitting the "
        "boundary, and (ii) the inner choice grid is at least as fine as the state "
        "grid, because policy errors propagate directly into the simulated "
        "transition. The closed-form policy is *not* used inside the loop; it is "
        "computed afterwards solely as a benchmark.\n\n"
        "```text\n"
        "Algorithm — Optimal-Growth VFI with continuous k', interpolated continuation\n"
        "Input : capital grid {k_i}_{i=1..N_k}, choice grid size N_kp,\n"
        "        primitives (A, alpha, beta), utility u(c) = log c, tolerance epsilon\n"
        "Output: value V*(k_i), capital policy g(k_i)\n"
        "  initialise V_0(k_i) = u(A k_i^alpha)             # eat-everything guess\n"
        "  for n = 0, 1, 2, ... :\n"
        "      for each state k_i :\n"
        "          y_i    <- A * k_i^alpha\n"
        "          kp_max <- min(y_i, k_max)\n"
        "          kp     <- N_kp points uniform on [k_min, kp_max)\n"
        "          c      <- y_i - kp                         # period consumption\n"
        "          V_cont <- interp(V_n, kp)                  # off-grid continuation\n"
        "          obj    <- log(c) + beta * V_cont\n"
        "          V_{n+1}(k_i) <- max(obj)\n"
        "          g(k_i)       <- argmax(obj)\n"
        "      err <- max_i | V_{n+1}(k_i) - V_n(k_i) |\n"
        "      stop when err < epsilon\n"
        "```\n\n"
        f"With the calibration above the iteration converges in **{info['iterations']} "
        f"steps** to a sup-norm residual of **{info['error']:.2e}**, consistent with the "
        "geometric bound. The Euler equation could also be solved in one pass by "
        "endogenous-grid points or a shooting method, but VFI is what generalizes to "
        "the stochastic and constrained problems later in the catalog."
    )

    # ------------------------------------------------------------------
    # Figure 1 — value function vs closed form
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
        "The value function is increasing and concave because more capital relaxes "
        "the resource constraint while marginal product diminishes. With log utility "
        "and Cobb-Douglas production it is exactly affine in $\\log k$, and the "
        "numerical curve sits on top of the closed-form curve over the whole "
        "economically relevant range. Outside the bottom decile of the grid the "
        f"largest sup-norm gap is **{max_value_error:.2e}**, which is essentially "
        "interpolation noise; the wider deviation near $k=0$ is the usual artifact of "
        "the log singularity on a uniform grid."
    )
    report.add_figure(
        "figures/value-function.png",
        "Numerical value function plotted against the closed-form $E + B\\log k$",
        fig1,
    )

    # ------------------------------------------------------------------
    # Figure 2 — capital policy vs closed form
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
        "The economic content of the model lives in this picture. The policy crosses "
        "the $45^{\\circ}$ line exactly at $k_{ss}$: below the steady state the "
        "planner accumulates ($k' > k$), above it the planner runs capital down "
        "($k' < k$). The slope at the crossing is less than one, which is what makes "
        "the steady state stable and the transition monotone. Under log utility the "
        f"saving rate is the constant $\\alpha\\beta = {alpha*beta:.2f}$ regardless "
        "of the level of capital; off log utility, $g$ would still cross the "
        "$45^{\\circ}$ line at $k_{ss}$ but its curvature would shift with the "
        "intertemporal elasticity. The largest pointwise policy gap outside the "
        f"bottom decile is **{max_policy_error:.2e}**, with a corresponding "
        f"consumption-policy gap of **{max_consumption_error:.2e}**."
    )
    report.add_figure(
        "figures/policy-function.png",
        "Capital policy $g(k)$ versus the closed-form rule $\\alpha\\beta A k^{\\alpha}$",
        fig2,
    )

    # ------------------------------------------------------------------
    # Figure 3 — transition paths
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
        "Iterating $k_{t+1}=g(k_t)$ from $k_0 = 0.1\\,k_{ss}$ traces the Ramsey "
        "transition. Capital rises quickly at first because marginal product is high "
        "when capital is scarce, then convergence slows as $f'(k)$ falls toward "
        "$1/\\beta$. Consumption inherits the same hump-free monotonicity here "
        "because the saving rate is constant; with non-log utility the consumption "
        "path could overshoot or undershoot $c_{ss}$ even when capital does not. "
        "The numerical and closed-form trajectories are visually indistinguishable, "
        f"with sup-norm capital-path error **{max_path_error:.2e}**."
    )
    report.add_figure(
        "figures/simulation.png",
        f"Capital and consumption transitions starting from $k_0={k0:.2f}$",
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
        "The audit table reports both objects at eight representative capital states. "
        "Value-function residuals are uniformly tight; policy residuals are larger "
        "but smooth in $k$ and never reverse sign in a way that would suggest a "
        "spurious local optimum. The relevant diagnostic for downstream simulations "
        "is the policy column, since policies are what get forward-iterated."
    )
    report.add_table(
        "tables/comparison.csv",
        "Numerical vs closed-form solution at selected capital states",
        df,
    )

    report.add_takeaway(
        "Optimal growth is the cake-eating Bellman equation with one extra ingredient: "
        "the resource constraint runs through a production function, so saving today "
        "delivers $f'(k_{t+1})$ extra units of consumption tomorrow. The Euler "
        "equation absorbs that change cleanly, and under log utility, Cobb-Douglas "
        "production, and full depreciation it collapses to a constant saving rate "
        f"$\\alpha\\beta = {alpha*beta:.2f}$ and a closed-form transition toward "
        f"$k_{{ss}} = {kss:.2f}$. VFI recovers that policy to interpolation accuracy, "
        "which is the right calibration to take into the stochastic, partially "
        "depreciated, or constrained settings later in the catalog, where Euler "
        "residuals and equilibrium consistency replace the closed form as the only "
        "available checks."
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
