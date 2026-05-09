#!/usr/bin/env python3
"""Fixed-resource consumption with Pontryagin's maximum principle.

The tutorial studies a planner who owns one fixed stock of a consumption good
and must choose how fast to use it over continuous time. Pontryagin's maximum
principle converts the depletion problem into a path for consumption and a
shadow price for the remaining stock.

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
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Fixed-Resource Consumption and Pontryagin Shadow Prices",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A planner owns a fixed stock $W_0$ of a consumption good. Consuming more today "
        "leaves less for every future date.\n\n"
        "The object is a continuous-time consumption path $c(t)$ and remaining stock "
        "$W(t)$. CRRA utility rewards smoothing, so the path should decline gradually.\n\n"
        "The computation turns an infinite-horizon control problem into paths that can "
        "be evaluated and checked. Pontryagin's maximum principle supplies the costate "
        "equation and the shadow price."
    )

    report.add_equations(
        r"""
For $\sigma > 0$ and $\sigma \neq 1$, flow utility is
$$u(c)=\frac{c^{1-\sigma}}{1-\sigma}, \qquad u'(c)=c^{-\sigma},$$
and the planner solves
$$\max_{\{c(t)\}_{t\geq 0}} \int_0^\infty e^{-\rho t} u(c(t)) \, dt$$

subject to
$$\dot{W}(t)=-c(t), \qquad W(0)=W_0, \qquad c(t)\geq 0, \qquad W(t)\geq 0.$$

The present-value Hamiltonian is
$$\mathcal{H}(c,W,\lambda,t)=e^{-\rho t}u(c)-\lambda c.$$

The first-order condition and costate equation are
$$e^{-\rho t}c(t)^{-\sigma}=\lambda(t), \qquad
\dot{\lambda}(t)=-\frac{\partial \mathcal{H}}{\partial W}=0.$$

Because $W$ has no direct payoff, the present-value costate is constant.
Differentiating the first-order condition gives
$$\frac{\dot c(t)}{c(t)}=-\frac{\rho}{\sigma}.$$

The no-waste condition $\int_0^\infty c(t)\,dt=W_0$ pins down the initial
consumption rate and therefore the full path:
$$c(t)=\frac{\rho}{\sigma}W_0 e^{-\rho t/\sigma}, \qquad
W(t)=W_0 e^{-\rho t/\sigma}.$$

The current-value shadow price is
$$\mu(t)=e^{\rho t}\lambda=c(t)^{-\sigma},$$
so $\dot{\mu}(t)/\mu(t)=\rho$. Current-value scarcity rises even though the
present-value costate is flat.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$    | {rho} | Continuous discount rate |\n"
        f"| $\\sigma$  | {sigma} | Relative risk aversion; IES $=1/\\sigma$ |\n"
        f"| $W_0$     | {W_0} | Initial resource stock |\n"
        f"| $T$       | {T} | Plotting horizon; the economic problem has an infinite horizon |\n"
        f"| Evaluation points | {n_eval} | Time points for exact paths and ODE checks |"
    )

    report.add_solution_method(
        "Pontryagin's principle attaches a price to one extra unit of stock. The "
        "planner equates discounted marginal utility with that price at each instant. "
        "Because $W$ has no direct payoff, the present-value price does not drift. "
        "Discounting and curvature then set the decline in consumption.\n\n"
        "```text\n"
        "Inputs: rho, sigma, W0, evaluation grid {t_m}\n"
        "1. Write the present-value Hamiltonian H = exp(-rho t) u(c) - lambda c.\n"
        "2. Use the FOC exp(-rho t) c(t)^(-sigma) = lambda(t).\n"
        "3. Use the costate equation lambda_dot(t) = -H_W = 0.\n"
        "4. Differentiate the FOC to obtain c_dot(t) / c(t) = -rho / sigma.\n"
        "5. Use integral_0^infinity c(t) dt = W0 to set c(0) = (rho / sigma) W0.\n"
        "6. Evaluate c(t), W(t), and mu(t) = c(t)^(-sigma) on {t_m}.\n"
        "Output: consumption, remaining stock, and shadow-price paths.\n"
        "```\n\n"
        "The numerical check integrates two ODEs from the implied initial consumption "
        "rate. It then compares the ODE solution with the closed-form path.\n\n"
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
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("Consumption rate $c(t)$")
    ax1.set_title("Smooth Depletion of a Fixed Resource")
    ax1.legend()
    ax1.set_xlim(0, 60)
    report.add_figure(
        "figures/consumption-path.png",
        "Continuous consumption path with ODE check",
        fig1,
        description="Consumption starts at $(\\rho/\\sigma)W_0$ and falls at the "
        "constant proportional rate $\\rho/\\sigma$.\n\n"
        "The ODE markers sit on the closed-form path. The costate equations and ODE "
        "system imply the same allocation."
    )

    # --- Figure 2: Cake Remaining ---
    fig2, ax2 = plt.subplots()
    ax2.plot(t_eval, W_analytical, color="#1f4e79", linewidth=2.2,
             label="Exact continuous stock")
    ax2.plot(t_eval[mark_every], W_numerical[mark_every], "o", color="#222222",
             markersize=3, alpha=0.65, label="RK45 ODE check")
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("Cake remaining $W(t)$")
    ax2.set_title("Cake Remaining Along the Optimal Path")
    ax2.legend()
    ax2.set_xlim(0, 60)
    report.add_figure(
        "figures/cake-remaining.png",
        "Exact cake stock with ODE check",
        fig2,
        description="The stock is depleted only asymptotically. With CRRA utility and an "
        "infinite horizon, the planner preserves future consumption because marginal "
        "utility rises near zero."
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
        "Present-value and current-value shadow prices for the resource stock",
        fig3,
        description="The present-value costate is flat because the resource stock has no "
        "direct payoff term. The current-value shadow price rises at rate $\\rho$. "
        "Later consumption is scarce in utility terms."
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
        "Selected Checks Against the Continuous-Time Path",
        df,
        description="The exact solution benchmarks the ODE path. The errors come from "
        "solver tolerances."
    )

    report.add_takeaway(
        "The costate is the intertemporal price of the remaining resource. Here the "
        "present-value price is constant. Optimal consumption declines at rate "
        "$\\rho/\\sigma$ and keeps discounted marginal utility equal across dates.\n\n"
        "Higher impatience raises the depletion rate. Higher risk aversion slows it "
        "through the smoothing motive."
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
