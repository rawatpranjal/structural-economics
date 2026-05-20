#!/usr/bin/env python3
"""DMP search-and-matching tutorial.

Compute the Diamond-Mortensen-Pissarides equilibrium under the Shimer (2005)
calibration. A local rule and a finite-state nonlinear fixed point compute
labor-market tightness. A small table varies the flow value of unemployment to
show how surplus calibration drives amplification.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import rouwenhorst
from lib.plotting import setup_style, save_figure, save_thumbnail


def calibrate_vacancy_cost(
    beta: float,
    separation_rate: float,
    matching_efficiency: float,
    benefit: float,
    bargaining_power: float,
    z_bar: float,
) -> float:
    """Vacancy cost that makes deterministic-steady-state tightness equal one."""
    numerator = beta * matching_efficiency * (1.0 - bargaining_power) * (z_bar - benefit)
    denominator = (
        1.0
        - beta * (1.0 - separation_rate)
        + beta * matching_efficiency * bargaining_power
    )
    return numerator / denominator


def tightness_elasticity(
    beta: float,
    rho: float,
    separation_rate: float,
    matching_efficiency: float,
    matching_elasticity: float,
    bargaining_power: float,
    vacancy_cost: float,
) -> tuple[float, float, float]:
    """Linearized elasticity of tightness with respect to log productivity."""
    a_coeff = (
        matching_elasticity
        * vacancy_cost
        / ((1.0 - bargaining_power) * beta * matching_efficiency)
    )
    b_coeff = (
        beta * a_coeff * (1.0 - separation_rate)
        - bargaining_power * vacancy_cost / (1.0 - bargaining_power)
    )
    elasticity = rho / (a_coeff - b_coeff * rho)
    return elasticity, a_coeff, b_coeff


def solve_nonlinear_tightness(
    beta: float,
    rho: float,
    shock_sigma: float,
    separation_rate: float,
    matching_efficiency: float,
    matching_elasticity: float,
    benefit: float,
    bargaining_power: float,
    vacancy_cost: float,
    z_bar: float,
    n_z: int = 41,
    tol: float = 1e-11,
    max_iter: int = 10_000,
) -> dict[str, np.ndarray | float | int | bool]:
    """Finite-state nonlinear DMP free-entry fixed point.

    Iterates J(z) = (1-gamma)(z-b) - gamma*k*theta(z) + beta*(1-sigma)*E[J(z')|z],
    with theta(z) = (beta*chi*E[J(z')|z]/k)^(1/(1-eta)) substituted from free
    entry inside each sweep. The operator is a contraction with modulus
    beta*(1-sigma) in the sup norm.
    """
    zhat_grid_jax, transition_jax, _ = rouwenhorst(n_z, 0.0, shock_sigma, rho)
    zhat_grid = np.asarray(zhat_grid_jax, dtype=float).ravel()
    transition = np.asarray(transition_jax, dtype=float)
    z_grid = z_bar * np.exp(zhat_grid)

    job_value = np.full(n_z, vacancy_cost / (beta * matching_efficiency), dtype=float)
    error = np.inf

    for iteration in range(1, max_iter + 1):
        expected_job_value = transition @ job_value
        theta = (
            beta
            * matching_efficiency
            * np.maximum(expected_job_value, 0.0)
            / vacancy_cost
        ) ** (1.0 / (1.0 - matching_elasticity))
        new_job_value = (
            (1.0 - bargaining_power) * (z_grid - benefit)
            - bargaining_power * vacancy_cost * theta
            + beta * (1.0 - separation_rate) * expected_job_value
        )
        error = float(np.max(np.abs(new_job_value - job_value)))
        job_value = new_job_value
        if error < tol:
            break

    expected_job_value = transition @ job_value
    theta = (
        beta
        * matching_efficiency
        * np.maximum(expected_job_value, 0.0)
        / vacancy_cost
    ) ** (1.0 / (1.0 - matching_elasticity))

    return {
        "zhat_grid": zhat_grid,
        "z_grid": z_grid,
        "theta": theta,
        "job_value": job_value,
        "transition": transition,
        "iterations": iteration,
        "error": error,
        "converged": error < tol,
    }


def amplification_by_surplus(
    beta: float,
    rho: float,
    separation_rate: float,
    matching_efficiency: float,
    matching_elasticity: float,
    bargaining_power: float,
    z_bar: float,
    benefits: np.ndarray,
) -> pd.DataFrame:
    """Tightness elasticity C as a function of the unemployment flow value b.

    For each b the vacancy cost is recalibrated so that theta_ss = 1, then the
    log-linear elasticity of tightness w.r.t. productivity is recomputed. The
    closed-form mapping makes the Hagedorn-Manovskii lever explicit: amplification
    is large precisely when the surplus z - b is small.
    """
    rows = []
    for b in benefits:
        k_b = calibrate_vacancy_cost(
            beta, separation_rate, matching_efficiency, b, bargaining_power, z_bar,
        )
        c_b, _, _ = tightness_elasticity(
            beta, rho, separation_rate, matching_efficiency, matching_elasticity,
            bargaining_power, k_b,
        )
        rows.append(
            {
                "Flow value b": f"{b:.2f}",
                "Surplus z-b": f"{z_bar - b:.2f}",
                "Vacancy cost k": f"{k_b:.4f}",
                "Tightness elasticity C": f"{c_b:.2f}",
            }
        )
    return pd.DataFrame(rows)


def simulate_productivity(
    rho: float,
    shock_sigma: float,
    periods: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate log-productivity deviations from the AR(1) law of motion."""
    rng = np.random.default_rng(seed)
    zhat = np.zeros(periods)
    for t in range(1, periods):
        zhat[t] = rho * zhat[t - 1] + rng.normal(0.0, shock_sigma)
    return zhat


def simulate_unemployment(
    theta: np.ndarray,
    separation_rate: float,
    matching_efficiency: float,
    matching_elasticity: float,
    u0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate unemployment and vacancies given a tightness sequence."""
    job_finding = matching_efficiency * theta**matching_elasticity
    unemployment = np.empty_like(theta)
    unemployment[0] = u0

    for t in range(len(theta) - 1):
        unemployment[t + 1] = (
            separation_rate * (1.0 - unemployment[t])
            + (1.0 - job_finding[t]) * unemployment[t]
        )

    vacancies = theta * unemployment
    return unemployment, vacancies


def cycle_stats(
    variables: list[tuple[str, np.ndarray]],
    productivity: np.ndarray,
) -> pd.DataFrame:
    """Standard log-deviation business-cycle moments relative to productivity."""
    productivity_sd = float(np.std(np.log(productivity)))
    rows = []
    log_productivity = np.log(productivity)

    for label, series in variables:
        log_series = np.log(series)
        sd = float(np.std(log_series))
        rows.append(
            {
                "Variable": label,
                "Mean": f"{np.mean(series):.4f}",
                "Std. log dev.": f"{sd:.4f}",
                "Std./Std. z": f"{sd / productivity_sd:.2f}",
                "Corr. with z": f"{np.corrcoef(log_series, log_productivity)[0, 1]:.3f}",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    # Shimer (2005) monthly calibration.
    beta = 0.996
    rho = 0.949
    shock_sigma = 0.0065
    separation_rate = 0.034
    matching_efficiency = 0.49
    benefit = 0.40
    bargaining_power = 0.72
    matching_elasticity = 0.72
    z_bar = 1.0

    vacancy_cost = calibrate_vacancy_cost(
        beta,
        separation_rate,
        matching_efficiency,
        benefit,
        bargaining_power,
        z_bar,
    )
    elasticity, a_coeff, b_coeff = tightness_elasticity(
        beta,
        rho,
        separation_rate,
        matching_efficiency,
        matching_elasticity,
        bargaining_power,
        vacancy_cost,
    )

    theta_ss = 1.0
    q_ss = matching_efficiency * theta_ss ** (matching_elasticity - 1.0)
    f_ss = matching_efficiency * theta_ss**matching_elasticity
    u_ss = separation_rate / (separation_rate + f_ss)
    wage_ss = (
        bargaining_power * (z_bar + vacancy_cost * theta_ss)
        + (1.0 - bargaining_power) * benefit
    )
    surplus_ss = z_bar - benefit
    linear_term_modulus = beta * (1.0 - separation_rate)
    # Effective contraction modulus of the FULL nonlinear operator. The linear
    # term contributes beta*(1-sigma); the theta-feedback term theta(EJ) inside
    # the Bellman adds a negative correction. The total derivative of J_new
    # w.r.t. EJ at the steady state (EJ_ss = k/(beta*chi), theta_ss = 1) is:
    expected_job_value_ss = vacancy_cost / (beta * matching_efficiency)
    theta_feedback_term = (
        bargaining_power
        * vacancy_cost
        * (1.0 / (1.0 - matching_elasticity))
        * (beta * matching_efficiency / vacancy_cost)
        ** (1.0 / (1.0 - matching_elasticity))
        * expected_job_value_ss ** (matching_elasticity / (1.0 - matching_elasticity))
    )
    # ||P||_inf = 1 for a row-stochastic transition matrix, so the effective
    # Lipschitz constant is the absolute value of this derivative.
    effective_modulus = abs(linear_term_modulus - theta_feedback_term)

    # Coarse nonlinear solver (tutorial run) and finer benchmark.
    n_z_coarse = 41
    n_z_fine = 121
    nonlinear = solve_nonlinear_tightness(
        beta, rho, shock_sigma, separation_rate, matching_efficiency,
        matching_elasticity, benefit, bargaining_power, vacancy_cost, z_bar,
        n_z=n_z_coarse,
    )
    nonlinear_fine = solve_nonlinear_tightness(
        beta, rho, shock_sigma, separation_rate, matching_efficiency,
        matching_elasticity, benefit, bargaining_power, vacancy_cost, z_bar,
        n_z=n_z_fine,
    )
    if not (nonlinear["converged"] and nonlinear_fine["converged"]):
        raise RuntimeError("Nonlinear DMP fixed point did not converge.")

    zhat_grid = nonlinear["zhat_grid"]
    z_grid = nonlinear["z_grid"]
    theta_nonlinear_grid = nonlinear["theta"]
    theta_linear_grid = np.exp(elasticity * zhat_grid)
    theta_fine_on_coarse = np.interp(
        zhat_grid, nonlinear_fine["zhat_grid"], nonlinear_fine["theta"]
    )
    max_policy_gap = float(
        np.max(np.abs(theta_nonlinear_grid - theta_linear_grid) / theta_nonlinear_grid)
    )
    max_grid_gap = float(
        np.max(np.abs(theta_nonlinear_grid - theta_fine_on_coarse) / theta_fine_on_coarse)
    )

    # Long simulation under the nonlinear tightness rule.
    periods = 5_000
    burn = 500
    zhat = simulate_productivity(rho, shock_sigma, periods)
    productivity = z_bar * np.exp(zhat)
    theta_linear = np.exp(elasticity * zhat)
    theta_nonlinear = np.interp(zhat, zhat_grid, theta_nonlinear_grid)
    unemployment, vacancies = simulate_unemployment(
        theta_nonlinear,
        separation_rate,
        matching_efficiency,
        matching_elasticity,
        u_ss,
    )

    z_plot = productivity[burn:]
    u_plot = unemployment[burn:]
    v_plot = vacancies[burn:]
    theta_plot = theta_nonlinear[burn:]
    theta_linear_plot = theta_linear[burn:]
    t_axis = np.arange(len(z_plot))

    df_stats = cycle_stats(
        [
            ("Productivity z", z_plot),
            ("Unemployment u", u_plot),
            ("Vacancies v", v_plot),
            ("Tightness theta", theta_plot),
            ("Tightness theta, log-linear", theta_linear_plot),
        ],
        z_plot,
    )

    benefit_grid = np.array([0.40, 0.55, 0.71, 0.85, 0.95])
    df_amp = amplification_by_surplus(
        beta, rho, separation_rate, matching_efficiency, matching_elasticity,
        bargaining_power, z_bar, benefit_grid,
    )

    print(
        f"Steady state: u={u_ss:.4f}, theta={theta_ss:.4f}, "
        f"wage={wage_ss:.4f}, k={vacancy_cost:.4f}"
    )
    print(
        f"Log-linear elasticity C = {elasticity:.4f}; "
        f"nonlinear (N_z={n_z_coarse}) iters = {nonlinear['iterations']}; "
        f"nonlinear (N_z={n_z_fine}) iters = {nonlinear_fine['iterations']}"
    )

    setup_style()

    fig1, ax1 = plt.subplots()
    ax1.plot(
        z_grid,
        theta_nonlinear_grid,
        color="black",
        linewidth=2.4,
        label=f"Nonlinear free entry, $N_z={n_z_coarse}$",
    )
    ax1.plot(
        z_grid,
        theta_fine_on_coarse,
        color="tab:green",
        linestyle=":",
        linewidth=1.6,
        label=f"Nonlinear benchmark, $N_z={n_z_fine}$",
    )
    ax1.plot(
        z_grid,
        theta_linear_grid,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="Log-linear rule",
    )
    ax1.scatter(
        z_plot[::20],
        theta_plot[::20],
        s=6,
        alpha=0.20,
        color="steelblue",
        label="Simulated months",
    )
    ax1.axvline(z_bar, color="0.45", linestyle=":", linewidth=1.0)
    ax1.axhline(theta_ss, color="0.45", linestyle=":", linewidth=1.0)
    ax1.set_xlabel("Productivity $z_t$")
    ax1.set_ylabel("Labor-market tightness $\\theta_t$")
    ax1.set_title("Productivity Determines Vacancy Creation")
    ax1.legend(loc="upper left")
    save_figure(fig1, "figures/productivity-tightness.png", dpi=150)

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax2a.plot(t_axis[:1_000], u_plot[:1_000], color="tab:blue", linewidth=1.2)
    ax2a.axhline(u_ss, color="0.45", linestyle=":", linewidth=1.0)
    ax2a.set_ylabel("Unemployment rate $u_t$")
    ax2a.set_title("Unemployment Falls When Matching Gets Easier")

    ax2b.plot(t_axis[:1_000], v_plot[:1_000], color="tab:red", linewidth=1.2)
    ax2b.set_ylabel("Vacancy rate $v_t$")
    ax2b.set_xlabel("Month")
    ax2b.set_title("Vacancies Rise With Firm Entry")
    fig2.tight_layout()
    save_figure(fig2, "figures/unemployment-vacancies.png", dpi=150)

    fig3, ax3 = plt.subplots()
    ax3.scatter(u_plot, v_plot, s=3, alpha=0.25, color="steelblue", edgecolors="none")
    ax3.scatter([u_ss], [theta_ss * u_ss], s=45, color="black", label="Steady state")
    ax3.set_xlabel("Unemployment rate $u_t$")
    ax3.set_ylabel("Vacancy rate $v_t$")
    ax3.set_title("A Beveridge Curve From Matching Frictions")
    ax3.legend()
    save_figure(fig3, "figures/beveridge-curve.png", dpi=150)

    # Thumbnail
    save_thumbnail("figures/productivity-tightness.png", "figures/thumb.png")

    # =========================================================================
    # Tables
    # =========================================================================
    Path("tables").mkdir(parents=True, exist_ok=True)

    df_stats.to_csv("tables/business-cycle-stats.csv", index=False)
    df_amp.to_csv("tables/amplification-by-surplus.csv", index=False)

    df_diagnostics = pd.DataFrame(
        {
            "Quantity": [
                "Coarse-grid policy gap vs. log-linear",
                "Coarse-grid interpolation gap vs. fine grid",
                "Coarse-grid fixed-point iterations",
                "Fine-grid fixed-point iterations",
            ],
            "policy_gap_pct": [
                f"{100.0 * max_policy_gap:.4f}",
                "",
                "",
                "",
            ],
            "grid_gap_pct": [
                "",
                f"{100.0 * max_grid_gap:.6f}",
                "",
                "",
            ],
            "iterations": [
                "",
                "",
                f"{nonlinear['iterations']}",
                f"{nonlinear_fine['iterations']}",
            ],
        }
    )
    df_diagnostics.to_csv("tables/solver-diagnostics.csv", index=False)

    print(
        f"Generated figures and tables."
    )


if __name__ == "__main__":
    main()
