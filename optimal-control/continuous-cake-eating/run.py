#!/usr/bin/env python3
"""Continuous-time cake eating with Pontryagin's maximum principle.

The tutorial solves the continuous-time version of the finite-resource
cake-eating problem. A planner allocates a fixed stock over time under CRRA
utility, and the costate variable gives the shadow value of leaving one more
unit of cake for the future.

Reference: Acemoglu (2009), Introduction to Modern Economic Growth, Ch. 7.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    rho = 0.05        # Continuous-time discount rate
    sigma = 2.0       # Relative risk aversion; IES is 1 / sigma
    W_0 = 1.0         # Initial cake size
    T = 80.0          # Horizon used only for plotting and numerical checks
    n_eval = 500      # Number of evaluation points

    # Discrete-time counterpart parameters
    beta = np.exp(-rho)  # Equivalent discrete discount factor

    # =========================================================================
    # Exact continuous-time solution
    # =========================================================================
    # The first-order condition is e^{-rho t} c(t)^{-sigma} = lambda.
    # Because W does not enter the Hamiltonian directly, lambda is constant in
    # present value. Differentiating the FOC gives c_dot / c = -rho / sigma,
    # and the resource constraint pins down c(0).

    c_0 = (rho / sigma) * W_0

    t_eval = np.linspace(0, T, n_eval)

    # Exact consumption path
    c_analytical = c_0 * np.exp(-rho * t_eval / sigma)

    # Exact cake remaining: W(t) = integral_t^infinity c(s) ds.
    W_analytical = W_0 * np.exp(-rho * t_eval / sigma)

    # Present-value costate: lambda(t) = c(t)^{-sigma} e^{-rho t}.
    lambda_pv = c_0 ** (-sigma) * np.ones_like(t_eval)
    # Current-value shadow price: mu(t) = lambda e^{rho t} = c(t)^{-sigma}.
    lambda_cv = c_analytical ** (-sigma)

    # =========================================================================
    # Numerical ODE integration, used as a check against the exact solution
    # =========================================================================
    # System: dW/dt = -c(t), dc/dt = -(rho/sigma)*c(t)
    def ode_system(t, y):
        W, c = y
        dW_dt = -c
        dc_dt = -(rho / sigma) * c
        return [dW_dt, dc_dt]

    sol = solve_ivp(
        ode_system,
        [0, T],
        [W_0, c_0],
        method="RK45",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    W_numerical = sol.y[0]
    c_numerical = sol.y[1]

    # Compute max error
    max_W_error = np.max(np.abs(W_numerical - W_analytical))
    max_c_error = np.max(np.abs(c_numerical - c_analytical))

    # =========================================================================
    # Discrete-time reference path
    # =========================================================================
    # The finite-period counterpart consumes a geometric sequence. It is useful
    # context for readers coming from dynamic-programming/cake-eating.

    gamma_discrete = beta ** (1.0 / sigma)  # consumption growth factor
    c0_discrete = (1 - gamma_discrete) * W_0
    T_discrete = 60
    t_discrete = np.arange(T_discrete)
    c_discrete = c0_discrete * gamma_discrete ** t_discrete
    W_discrete = np.zeros(T_discrete)
    W_discrete[0] = W_0
    for t in range(1, T_discrete):
        W_discrete[t] = W_discrete[t - 1] - c_discrete[t - 1]

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Continuous-Time Cake Eating and Shadow Prices",
        "Optimal depletion of a finite resource, solved with Pontryagin's maximum principle.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Cake eating is the stripped-down resource allocation problem: a planner starts with "
        "a fixed stock $W_0$ and must decide how quickly to turn it into consumption. In "
        "discrete time this appears as a Bellman equation, as in "
        "[Finite-Resource Cake Eating](../../dynamic-programming/cake-eating/). Here the "
        "same economics is written in continuous time, so the central object is the path "
        "$c(t)$ rather than a sequence of grid choices.\n\n"
        "Consuming more now lowers the stock available later, while CRRA utility rewards "
        "smoothing. Pontryagin's maximum principle turns that tradeoff into a shadow price "
        "for the remaining cake. Because the cake itself does not enter utility except "
        "through feasible consumption, the present-value shadow price is constant and "
        "consumption declines smoothly rather than dropping to zero in finite time."
    )

    report.add_equations(
        r"""
For $\sigma \neq 1$, flow utility is
$$u(c)=\frac{c^{1-\sigma}}{1-\sigma}, \qquad u'(c)=c^{-\sigma},$$
with the log case obtained as $\sigma \to 1$. The continuous-time problem is
$$\max_{\{c(t)\}_{t\geq 0}} \int_0^\infty e^{-\rho t} u(c(t)) \, dt$$

subject to
$$\dot{W}(t)=-c(t), \qquad W(0)=W_0, \qquad c(t)\geq 0, \qquad W(t)\geq 0.$$

The present-value Hamiltonian is
$$\mathcal{H}(c,W,\lambda,t)=e^{-\rho t}u(c)-\lambda c.$$

The first-order and costate equations are
$$e^{-\rho t}c(t)^{-\sigma}=\lambda(t), \qquad
\dot{\lambda}(t)=-\frac{\partial \mathcal{H}}{\partial W}=0.$$

Since $\lambda(t)$ is constant in present value, differentiating the first-order
condition gives the Euler equation
$$\frac{\dot c(t)}{c(t)}=-\frac{\rho}{\sigma}.$$

The no-waste condition $\int_0^\infty c(t)\,dt=W_0$ pins down the initial
consumption rate:
$$c(t)=\frac{\rho}{\sigma}W_0 e^{-\rho t/\sigma}, \qquad
W(t)=W_0 e^{-\rho t/\sigma}.$$

The current-value shadow price is
$$\mu(t)=e^{\rho t}\lambda=c(t)^{-\sigma},$$
so $\dot{\mu}(t)/\mu(t)=\rho$. Scarcity grows in current-value terms even though
the present-value costate is flat.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$    | {rho} | Continuous discount rate |\n"
        f"| $\\sigma$  | {sigma} | Relative risk aversion; IES $=1/\\sigma$ |\n"
        f"| $W_0$     | {W_0} | Initial cake size |\n"
        f"| $\\beta = e^{{-\\rho}}$ | {beta:.4f} | Period-one discount factor used for the discrete reference path |\n"
        f"| $T$       | {T} | Plotting and ODE-check horizon |\n"
        f"| Evaluation points | {n_eval} | Time points for plotting exact and numerical paths |"
    )

    report.add_solution_method(
        "Pontryagin's principle attaches a price to relaxing the stock constraint by "
        "one unit. The planner chooses $c(t)$ so that discounted marginal utility "
        "equals that price at every instant. Since $W$ has no direct payoff, the "
        "present-value price does not drift; all movement in consumption comes from "
        "discounting and curvature.\n\n"
        "```text\n"
        "Inputs: rho, sigma, W0, evaluation grid {t_m}\n"
        "1. Write the present-value Hamiltonian H = exp(-rho t) u(c) - lambda c.\n"
        "2. Use the FOC exp(-rho t) c(t)^(-sigma) = lambda(t).\n"
        "3. Use the costate equation lambda_dot(t) = -H_W = 0.\n"
        "4. Differentiate the FOC to obtain c_dot(t) / c(t) = -rho / sigma.\n"
        "5. Use integral_0^infinity c(t) dt = W0 to set c(0) = (rho / sigma) W0.\n"
        "6. Evaluate c(t), W(t), and mu(t) = c(t)^(-sigma) on {t_m}.\n"
        "Output: exact consumption, stock, and shadow-price paths.\n"
        "```\n\n"
        "The ODE integration is a numerical check, not the source of the solution. It "
        "integrates $\\dot{W}=-c$ and $\\dot{c}=-(\\rho/\\sigma)c$ from the analytical "
        "initial condition and compares the resulting path with the exact one. The "
        "period-one discrete path is a nearby benchmark for the dynamic programming "
        "version of the same allocation problem.\n\n"
        f"**Verification:** Max absolute error in $W(t)$: {max_W_error:.2e}, "
        f"in $c(t)$: {max_c_error:.2e}."
    )

    # --- Figure 1: Optimal Consumption Path ---
    fig1, ax1 = plt.subplots()
    mark_every = slice(None, None, 35)
    ax1.plot(t_eval, c_analytical, color="#1f4e79", linewidth=2.2,
             label="Exact continuous path")
    ax1.plot(t_eval[mark_every], c_numerical[mark_every], "o", color="#222222",
             markersize=3, alpha=0.65, label="RK45 ODE check")
    ax1.step(t_discrete, c_discrete, color="#b23a48", linestyle="--",
             linewidth=1.5, alpha=0.8, where="post",
             label="Period-one discrete path")
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("Consumption rate $c(t)$")
    ax1.set_title("Smooth Depletion of a Fixed Resource")
    ax1.legend()
    ax1.set_xlim(0, 60)
    report.add_figure(
        "figures/consumption-path.png",
        "Exact continuous consumption path with ODE and discrete-time comparisons",
        fig1,
        description="Consumption starts at $(\\rho/\\sigma)W_0$ and then falls at the "
        "constant proportional rate $\\rho/\\sigma$. The black markers show that direct "
        "ODE integration recovers the exact path. The stepped line is the period-one "
        "discrete allocation, which connects this continuous-time problem to the "
        "finite-resource dynamic-programming tutorial."
    )

    # --- Figure 2: Cake Remaining ---
    fig2, ax2 = plt.subplots()
    ax2.plot(t_eval, W_analytical, color="#1f4e79", linewidth=2.2,
             label="Exact continuous stock")
    ax2.plot(t_eval[mark_every], W_numerical[mark_every], "o", color="#222222",
             markersize=3, alpha=0.65, label="RK45 ODE check")
    ax2.step(t_discrete, W_discrete, color="#b23a48", linestyle="--",
             linewidth=1.5, alpha=0.8, where="post",
             label="Period-one discrete stock")
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("Cake remaining $W(t)$")
    ax2.set_title("Cake Remaining Along the Optimal Path")
    ax2.legend()
    ax2.set_xlim(0, 60)
    report.add_figure(
        "figures/cake-remaining.png",
        "Exact cake stock with ODE and discrete-time comparisons",
        fig2,
        description="The stock is depleted only asymptotically. That is not a numerical "
        "artifact: with CRRA utility and an infinite horizon, the planner keeps a tail of "
        "future consumption alive because marginal utility becomes large as consumption "
        "approaches zero."
    )

    # --- Figure 3: Shadow Price ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    ax3a.plot(t_eval, lambda_pv, color="#1f4e79", linewidth=2.2)
    ax3a.set_xlabel("Time $t$")
    ax3a.set_ylabel("$\\lambda(t)$")
    ax3a.set_title("Present-Value Shadow Price")
    ax3a.set_ylim(0, lambda_pv[0] * 1.5)

    ax3b.plot(t_eval, lambda_cv, color="#1f4e79", linewidth=2.2)
    ax3b.set_xlabel("Time $t$")
    ax3b.set_ylabel("$\\mu(t) = c(t)^{-\\sigma}$")
    ax3b.set_title("Current-Value Shadow Price")
    ax3b.set_xlim(0, 60)
    fig3.tight_layout()
    report.add_figure(
        "figures/shadow-price.png",
        "Present-value and current-value shadow prices for cake",
        fig3,
        description="The present-value costate is flat because the resource stock has no "
        "direct payoff term. In current-value units the shadow price rises at rate "
        "$\\rho$: one unit of cake left for a later date is scarce in utility terms "
        "even though its discounted value is equalized along the optimum."
    )

    # --- Table ---
    sample_t = np.array([0, 5, 10, 20, 30, 50])
    sample_idx = np.searchsorted(t_eval, sample_t)
    table_data = {
        "t": [f"{t_eval[i]:.0f}" for i in sample_idx],
        "c(t) exact": [f"{c_analytical[i]:.6f}" for i in sample_idx],
        "c(t) RK45": [f"{c_numerical[i]:.6f}" for i in sample_idx],
        "c error": [f"{abs(c_numerical[i] - c_analytical[i]):.1e}" for i in sample_idx],
        "W(t) exact": [f"{W_analytical[i]:.6f}" for i in sample_idx],
        "W(t) RK45": [f"{W_numerical[i]:.6f}" for i in sample_idx],
        "W error": [f"{abs(W_numerical[i] - W_analytical[i]):.1e}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/comparison.csv",
        "Selected Checks Against the Exact Continuous-Time Path",
        df,
        description="The exact solution is a benchmark for the numerical ODE path. "
        "The small errors below are solver error, not approximation error from a grid over "
        "choices."
    )

    report.add_takeaway(
        "The costate is the intertemporal price of the remaining resource, not just "
        "formal machinery. In this problem the present-value price is constant, so "
        "optimality requires a declining consumption path that keeps discounted marginal "
        "utility equal across dates.\n\n"
        "Higher impatience raises the depletion rate, while higher risk aversion slows it "
        "through the smoothing motive. The continuous-time formulation also clarifies what "
        "the discrete cake-eating Bellman problem is approximating: a smooth shadow-price "
        "condition, not merely a search over feasible consumption grid points."
    )

    report.add_references([
        "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 7.",
        "Kamien, M. and Schwartz, N. (2012). *Dynamic Optimization*. Dover, 2nd edition.",
        "Chiang, A. (1992). *Elements of Dynamic Optimization*. Waveland Press.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
