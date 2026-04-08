#!/usr/bin/env python3
"""Continuous-Time Cake Eating via Pontryagin's Maximum Principle.

Solves the continuous-time cake-eating problem using optimal control theory.
The agent maximizes the present discounted value of CRRA utility from consuming
a non-renewable resource, subject to the law of motion dW/dt = -c(t).

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
    sigma = 2.0       # CRRA coefficient (elasticity of marginal utility)
    W_0 = 1.0         # Initial cake size
    T = 80.0          # Time horizon for numerical integration
    n_eval = 500      # Number of evaluation points

    # Discrete-time counterpart parameters
    beta = np.exp(-rho)  # Equivalent discrete discount factor

    # =========================================================================
    # Analytical Solution (Continuous Time)
    # =========================================================================
    # From Pontryagin's principle:
    #   Hamiltonian: H = e^{-rho*t} * u(c) + lambda * (-c)
    #   FOC: e^{-rho*t} * u'(c) = lambda  =>  c^{-sigma} * e^{-rho*t} = lambda
    #   Costate: dlambda/dt = 0  (since dH/dW = 0; W does not appear in H)
    #
    # Since lambda is constant, differentiating the FOC:
    #   dc/dt = (rho / sigma) * c(t)  ... wait, let's be careful with signs.
    #
    # From c^{-sigma} * e^{-rho*t} = lambda (constant):
    #   -sigma * c^{-sigma-1} * dc/dt * e^{-rho*t} + c^{-sigma} * (-rho) * e^{-rho*t} = 0
    #   -sigma * (dc/dt)/c - rho = 0
    #   dc/dt = -(rho/sigma) * c
    #
    # So c(t) = c_0 * exp(-rho*t/sigma), consumption declines over time.
    #
    # Resource constraint: integral_0^inf c(t) dt = W_0
    #   c_0 * integral_0^inf exp(-rho*t/sigma) dt = W_0
    #   c_0 * (sigma / rho) = W_0
    #   c_0 = (rho / sigma) * W_0

    c_0 = (rho / sigma) * W_0

    t_eval = np.linspace(0, T, n_eval)

    # Analytical consumption path
    c_analytical = c_0 * np.exp(-rho * t_eval / sigma)

    # Analytical cake remaining: W(t) = integral_t^inf c(s) ds
    #   = c_0 * integral_t^inf exp(-rho*s/sigma) ds
    #   = c_0 * (sigma/rho) * exp(-rho*t/sigma)
    #   = W_0 * exp(-rho*t/sigma)
    W_analytical = W_0 * np.exp(-rho * t_eval / sigma)

    # Shadow price (costate variable): lambda = c_0^{-sigma} * e^{-rho*0} = c_0^{-sigma}
    # In current-value terms: mu(t) = lambda * e^{rho*t} = c(t)^{-sigma}
    # Present-value costate: lambda(t) = c(t)^{-sigma} * e^{-rho*t} = c_0^{-sigma} (constant)
    lambda_pv = c_0 ** (-sigma) * np.ones_like(t_eval)
    # Current-value shadow price
    lambda_cv = c_analytical ** (-sigma)

    # =========================================================================
    # Numerical Solution via ODE (verification)
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
    # Discrete-Time Comparison
    # =========================================================================
    # Discrete-time cake eating: c_t = (1-beta)*W_t, W_{t+1} = beta*W_t
    # With log utility (sigma=1): c_t = (1-beta)*beta^t * W_0
    # With general CRRA: c_t = (1 - beta^{1/sigma}) * W_t (for geometric discounting)
    # More precisely, the Euler equation gives c_{t+1}/c_t = beta^{1/sigma}
    # and budget: sum_{t=0}^inf c_t = W_0 => c_0 = (1 - beta^{1/sigma}) * W_0

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
        "Continuous-Time Cake Eating",
        "Optimal consumption of a finite resource in continuous time via Pontryagin's maximum principle.",
    )

    report.add_overview(
        "This model extends the classic cake-eating problem to continuous time. Instead of "
        "choosing consumption each discrete period, the agent selects a continuous consumption "
        "path $c(t)$ to maximize the integral of discounted utility. The solution method uses "
        "Pontryagin's maximum principle rather than dynamic programming.\n\n"
        "The continuous-time formulation yields a clean analytical solution: consumption declines "
        "exponentially at rate $\\rho/\\sigma$, and the cake is asymptotically depleted."
    )

    report.add_equations(
        r"""
$$\max_{c(t)} \int_0^\infty e^{-\rho t} \, u(c(t)) \, dt$$

subject to $\dot{W}(t) = -c(t)$, $W(0) = W_0$, $W(t) \ge 0$.

**Hamiltonian (present value):**
$$\mathcal{H} = e^{-\rho t} \, u(c) + \lambda \cdot (-c)$$

**First-order conditions:**
$$\frac{\partial \mathcal{H}}{\partial c} = 0 \implies e^{-\rho t} \, c^{-\sigma} = \lambda$$

**Costate equation:** $\dot{\lambda} = -\frac{\partial \mathcal{H}}{\partial W} = 0$ (so $\lambda$ is constant)

**Optimal consumption path:**
$$c(t) = c_0 \, e^{-\rho t / \sigma}, \qquad c_0 = \frac{\rho}{\sigma} \, W_0$$

**Cake remaining:**
$$W(t) = W_0 \, e^{-\rho t / \sigma}$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$    | {rho} | Continuous discount rate |\n"
        f"| $\\sigma$  | {sigma} | CRRA coefficient |\n"
        f"| $W_0$     | {W_0} | Initial cake size |\n"
        f"| $\\beta = e^{{-\\rho}}$ | {beta:.4f} | Equivalent discrete discount factor |\n"
        f"| $T$       | {T} | Integration horizon |"
    )

    report.add_solution_method(
        "**Pontryagin's Maximum Principle:** The present-value Hamiltonian is formed and "
        "optimized pointwise over the control $c(t)$. The first-order condition pins down "
        "the consumption rule, and the costate equation confirms $\\lambda$ is constant "
        "(since cake size $W$ does not appear in the Hamiltonian directly).\n\n"
        "The ODE system $\\dot{W} = -c$, $\\dot{c} = -(\\rho/\\sigma)c$ is integrated "
        "numerically using `scipy.integrate.solve_ivp` (RK45) to verify the analytical solution.\n\n"
        f"**Verification:** Max absolute error in $W(t)$: {max_W_error:.2e}, "
        f"in $c(t)$: {max_c_error:.2e}."
    )

    # --- Figure 1: Optimal Consumption Path ---
    fig1, ax1 = plt.subplots()
    ax1.plot(t_eval, c_analytical, "b-", linewidth=2, label="Continuous: $c(t) = c_0 e^{-\\rho t/\\sigma}$")
    ax1.step(t_discrete, c_discrete, "r--", linewidth=1.5, alpha=0.7, where="post",
             label=f"Discrete: $c_t = c_0 \\beta^{{t/\\sigma}}$")
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("Consumption $c(t)$")
    ax1.set_title("Optimal Consumption Path")
    ax1.legend()
    ax1.set_xlim(0, 60)
    report.add_figure(
        "figures/consumption-path.png",
        "Optimal consumption path: continuous vs discrete time",
        fig1,
        description="Consumption declines exponentially because the shadow price of cake is "
        "constant in present value: an optimizing agent equates the marginal value across "
        "all time periods after discounting. The discrete-time path closely tracks the "
        "continuous solution, converging as the period length shrinks.",
    )

    # --- Figure 2: Cake Remaining ---
    fig2, ax2 = plt.subplots()
    ax2.plot(t_eval, W_analytical, "b-", linewidth=2, label="Continuous: $W(t) = W_0 e^{-\\rho t/\\sigma}$")
    ax2.step(t_discrete, W_discrete, "r--", linewidth=1.5, alpha=0.7, where="post",
             label="Discrete: $W_t$")
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("Cake remaining $W(t)$")
    ax2.set_title("Cake Depletion Over Time")
    ax2.legend()
    ax2.set_xlim(0, 60)
    report.add_figure(
        "figures/cake-remaining.png",
        "Cake depletion: continuous exponential decay vs discrete geometric decay",
        fig2,
        description="The cake is never fully consumed in finite time -- it decays asymptotically "
        "to zero. Higher risk aversion (sigma) slows depletion because the agent values "
        "consumption smoothing more and is reluctant to let future consumption fall too low.",
    )

    # --- Figure 3: Shadow Price ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    ax3a.plot(t_eval, lambda_pv, "b-", linewidth=2)
    ax3a.set_xlabel("Time $t$")
    ax3a.set_ylabel("$\\lambda(t)$")
    ax3a.set_title("Present-Value Shadow Price (constant)")
    ax3a.set_ylim(0, lambda_pv[0] * 1.5)

    ax3b.plot(t_eval, lambda_cv, "b-", linewidth=2)
    ax3b.set_xlabel("Time $t$")
    ax3b.set_ylabel("$\\mu(t) = c(t)^{-\\sigma}$")
    ax3b.set_title("Current-Value Shadow Price (rising)")
    ax3b.set_xlim(0, 60)
    fig3.tight_layout()
    report.add_figure(
        "figures/shadow-price.png",
        "Shadow price of cake: present-value (constant) and current-value (rising as cake becomes scarce)",
        fig3,
        description="The constant present-value costate is the key optimality result from "
        "Pontryagin's principle: since cake does not appear in the Hamiltonian directly, "
        "its shadow price is time-invariant. The rising current-value shadow price reflects "
        "increasing scarcity as the resource is depleted.",
    )

    # --- Table ---
    sample_t = np.array([0, 5, 10, 20, 30, 50])
    sample_idx = np.searchsorted(t_eval, sample_t)
    table_data = {
        "t": [f"{t_eval[i]:.0f}" for i in sample_idx],
        "c(t) analytical": [f"{c_analytical[i]:.6f}" for i in sample_idx],
        "c(t) numerical": [f"{c_numerical[i]:.6f}" for i in sample_idx],
        "W(t) analytical": [f"{W_analytical[i]:.6f}" for i in sample_idx],
        "W(t) numerical": [f"{W_numerical[i]:.6f}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/comparison.csv",
        "Analytical vs Numerical Solution at Selected Time Points",
        df,
        description="The near-zero difference between analytical and numerical columns validates "
        "the ODE integration. This problem has a closed-form solution, making it an ideal "
        "test case for verifying numerical methods before applying them to harder problems.",
    )

    report.add_takeaway(
        "The continuous-time cake-eating problem illustrates the power of Pontryagin's "
        "maximum principle as an alternative to dynamic programming.\n\n"
        "**Key insights:**\n"
        "- Consumption declines exponentially at rate $\\rho/\\sigma$. Higher impatience "
        "($\\rho$) speeds depletion; higher risk aversion ($\\sigma$) smooths consumption "
        "and slows depletion.\n"
        "- The shadow price $\\lambda$ is constant in present value: the marginal value of "
        "cake is the same at every instant (once discounted). In current-value terms, the "
        "shadow price *rises* over time as the resource becomes scarcer.\n"
        "- The continuous-time solution converges to the discrete-time solution as the period "
        "length shrinks. With $\\beta = e^{-\\rho}$, both yield the same consumption-to-wealth "
        "ratio in the limit.\n"
        "- Unlike the discrete case, the continuous formulation avoids grid discretization "
        "entirely --- the ODE system has a closed-form solution."
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
