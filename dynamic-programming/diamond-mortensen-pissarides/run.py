#!/usr/bin/env python3
"""DMP search-and-matching tutorial.

The script computes a Shimer-style DMP calibration, compares the local
log-linear tightness rule with a finite-state nonlinear free-entry fixed point,
and regenerates README.md, figures, tables, and the catalog thumbnail.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import rouwenhorst
from lib.output import ModelReport
from lib.plotting import setup_style


def calibrate_vacancy_cost(
    beta: float,
    separation_rate: float,
    matching_efficiency: float,
    benefit: float,
    bargaining_power: float,
    z_bar: float,
) -> float:
    """Choose vacancy cost so steady-state tightness is one."""
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
    """Return the local elasticity of tightness with respect to productivity."""
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
    """Solve the finite-state nonlinear DMP free-entry fixed point."""
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


def simulate_productivity(
    rho: float,
    shock_sigma: float,
    periods: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate log productivity deviations."""
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
    """Propagate unemployment and vacancies from a tightness sequence."""
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
    """Summarize simulated business-cycle moments in log deviations."""
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
    job_value_ss = vacancy_cost / (beta * q_ss)
    surplus_ss = z_bar - benefit

    nonlinear = solve_nonlinear_tightness(
        beta,
        rho,
        shock_sigma,
        separation_rate,
        matching_efficiency,
        matching_elasticity,
        benefit,
        bargaining_power,
        vacancy_cost,
        z_bar,
    )
    if not nonlinear["converged"]:
        raise RuntimeError("Nonlinear DMP fixed point did not converge.")

    zhat_grid = nonlinear["zhat_grid"]
    z_grid = nonlinear["z_grid"]
    theta_nonlinear_grid = nonlinear["theta"]
    theta_linear_grid = np.exp(elasticity * zhat_grid)
    max_policy_gap = float(
        np.max(np.abs(theta_nonlinear_grid - theta_linear_grid) / theta_nonlinear_grid)
    )

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

    print(
        f"Steady state: u={u_ss:.4f}, theta={theta_ss:.4f}, "
        f"wage={wage_ss:.4f}, k={vacancy_cost:.4f}"
    )
    print(
        f"Log-linear tightness elasticity = {elasticity:.4f}; "
        f"nonlinear fixed point iterations = {nonlinear['iterations']}"
    )

    setup_style()

    report = ModelReport(
        "DMP Search, Vacancies, and Unemployment",
        "Matching frictions, vacancy creation, and the Shimer volatility puzzle.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The Diamond-Mortensen-Pissarides model asks why unemployment and "
        "vacancies move together the way they do over the business cycle. Firms "
        "pay to post vacancies, workers search while unemployed, and jobs are "
        "created only when the matching technology brings the two sides "
        "together. The central price-like object is labor-market tightness, "
        "$\\theta_t=v_t/u_t$: a high value means vacancies are plentiful "
        "relative to unemployed workers.\n\n"
        "This tutorial uses the Shimer (2005) calibration to make the model's "
        "main tension visible. Productivity shocks move job surplus and vacancy "
        "creation, so the model generates a Beveridge curve. But with the "
        "standard surplus implied by $b=0.40$, tightness and unemployment are "
        "much less volatile than in U.S. data. The example extends the "
        "[McCall search tutorial](../job-search-mccall/) from a worker's "
        "reservation rule to an equilibrium labor market with endogenous "
        "vacancies, and it connects to the [RBC tutorial](../rbc/) through the "
        "use of persistent aggregate productivity shocks."
    )

    report.add_equations(
        rf"""
Let $u_t$ be unemployment, $v_t$ vacancies, and
$\theta_t=v_t/u_t$ labor-market tightness. Matches are produced by

$$m(u_t,v_t)=\chi u_t^{{1-\eta}}v_t^\eta,$$

so the worker job-finding rate and firm vacancy-filling rate are

$$f(\theta_t)=\chi\theta_t^\eta,\qquad
q(\theta_t)=\chi\theta_t^{{\eta-1}}.$$

Productivity follows

$$\hat z_{{t+1}}=\rho \hat z_t+\epsilon_{{t+1}},\qquad
\epsilon_{{t+1}}\sim N(0,\sigma_\epsilon^2),\qquad
z_t=\bar z\exp(\hat z_t).$$

With Nash bargaining weight $\gamma$ for the worker, the wage rule is

$$w_t=\gamma(z_t+k\theta_t)+(1-\gamma)b.$$

The value $J_t$ of a filled job satisfies

$$J_t=z_t-w_t+\beta(1-\sigma)\mathbb{{E}}_t[J_{{t+1}}],$$

and free entry into vacancy posting imposes

$$k=\beta q(\theta_t)\mathbb{{E}}_t[J_{{t+1}}].$$

Unemployment evolves mechanically once $\theta_t$ is known:

$$u_{{t+1}}=\sigma(1-u_t)+[1-f(\theta_t)]u_t.$$

The local solution writes $\hat\theta_t=C\hat z_t$. For this timing convention,
the linearized free-entry condition gives

$$C=\frac{{\rho}}{{A-B\rho}},\qquad
A=\frac{{\eta k}}{{(1-\gamma)\beta\chi}},\qquad
B=\beta A(1-\sigma)-\frac{{\gamma k}}{{1-\gamma}}.$$

At the baseline calibration, $A={a_coeff:.4f}$ and $B={b_coeff:.4f}$.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| Discount factor $\\beta$ | {beta:.3f} | Monthly discounting |\n"
        f"| Productivity persistence $\\rho$ | {rho:.3f} | AR(1) persistence of $\\hat z_t$ |\n"
        f"| Innovation s.d. $\\sigma_\\epsilon$ | {shock_sigma:.4f} | Monthly productivity shock scale |\n"
        f"| Separation rate $\\sigma$ | {separation_rate:.3f} | Exogenous job destruction |\n"
        f"| Matching efficiency $\\chi$ | {matching_efficiency:.2f} | Level of the matching function |\n"
        f"| Matching elasticity $\\eta$ | {matching_elasticity:.2f} | Vacancy elasticity in $m(u,v)$ |\n"
        f"| Worker bargaining weight $\\gamma$ | {bargaining_power:.2f} | Worker share in Nash bargaining |\n"
        f"| Flow value of unemployment $b$ | {benefit:.2f} | Outside option while searching |\n"
        f"| Vacancy cost $k$ | {vacancy_cost:.4f} | Calibrated so $\\theta_{{ss}}=1$ |\n"
        f"| Steady-state unemployment $u_{{ss}}$ | {u_ss:.4f} | Implied by separations and job finding |\n"
        f"| Steady-state wage $w_{{ss}}$ | {wage_ss:.4f} | Nash wage at $z=\\bar z$ |\n"
        f"| Job surplus $\\bar z-b$ | {surplus_ss:.2f} | Baseline surplus before vacancy costs |"
    )

    report.add_solution_method(
        "There are two computations. The report uses the nonlinear finite-state "
        "solution as a check on the log-linear tightness rule, then simulates "
        "unemployment from the nonlinear rule. The approximation gap is small "
        "in this calibration. The economic point is that the standard surplus "
        "calibration leaves little amplification to begin with.\n\n"
        "```text\n"
        "Algorithm: log-linear DMP rule\n"
        "Input: beta, rho, sigma, chi, eta, gamma, b, z_bar\n"
        "Output: elasticity C and simulated theta_t\n"
        "Choose k so the deterministic steady state has theta_ss = 1\n"
        "Compute the linearized free-entry coefficients A and B\n"
        "Set C = rho / (A - B*rho)\n"
        "For each simulated productivity deviation zhat_t:\n"
        "    theta_t = exp(C * zhat_t)\n"
        "    f_t = chi * theta_t^eta\n"
        "    u_{t+1} = sigma*(1 - u_t) + (1 - f_t)*u_t\n"
        "```\n\n"
        "```text\n"
        "Algorithm: nonlinear finite-state free-entry check\n"
        "Input: Rouwenhorst grid for zhat, transition matrix P, calibrated k\n"
        "Output: job values J_i and tightness theta_i at each productivity state i\n"
        "Initialize J_i at its steady-state value\n"
        "repeat:\n"
        "    EJ_i = sum_j P_ij J_j\n"
        "    theta_i = (beta * chi * EJ_i / k)^(1 / (1 - eta))\n"
        "    J_i_new = (1 - gamma)*(z_i - b) - gamma*k*theta_i\n"
        "              + beta*(1 - sigma)*EJ_i\n"
        "    error = max_i |J_i_new - J_i|\n"
        "    set J_i = J_i_new\n"
        "until error < epsilon\n"
        "```\n\n"
        f"The local rule implies $C={elasticity:.3f}$: a one percent productivity "
        f"increase raises tightness by about {elasticity:.2f} percent. The "
        f"finite-state fixed point converged in **{nonlinear['iterations']} "
        f"iterations** with sup-norm error **{nonlinear['error']:.2e}**. Across "
        f"the productivity grid, the largest proportional gap between nonlinear "
        f"and log-linear tightness is **{100.0 * max_policy_gap:.2f}%**."
    )

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1a.plot(t_axis[:1_000], u_plot[:1_000], color="tab:blue", linewidth=1.2)
    ax1a.axhline(u_ss, color="0.45", linestyle=":", linewidth=1.2)
    ax1a.set_ylabel("Unemployment rate $u_t$")
    ax1a.set_title("Unemployment Falls When Matching Gets Easier")

    ax1b.plot(t_axis[:1_000], v_plot[:1_000], color="tab:red", linewidth=1.2)
    ax1b.set_ylabel("Vacancy rate $v_t$")
    ax1b.set_xlabel("Month")
    ax1b.set_title("Vacancies Rise With Firm Entry")
    fig1.tight_layout()
    report.add_figure(
        "figures/unemployment-vacancies.png",
        "Simulated unemployment and vacancies under the nonlinear tightness rule.",
        fig1,
        description=(
            "The simulated path shows the stock-flow logic of the model. A good "
            "productivity spell raises the surplus from a match, firms post more "
            "vacancies, and the job-finding rate rises. Unemployment then moves "
            "with a lag because it is a stock: today's hiring changes tomorrow's "
            "pool of searchers."
        ),
    )

    fig2, ax2 = plt.subplots()
    ax2.scatter(u_plot, v_plot, s=3, alpha=0.25, color="steelblue", edgecolors="none")
    ax2.scatter([u_ss], [theta_ss * u_ss], s=45, color="black", label="Steady state")
    ax2.set_xlabel("Unemployment rate $u_t$")
    ax2.set_ylabel("Vacancy rate $v_t$")
    ax2.set_title("A Beveridge Curve From Matching Frictions")
    ax2.legend()
    report.add_figure(
        "figures/beveridge-curve.png",
        "Beveridge curve generated by the simulated DMP economy.",
        fig2,
        description=(
            "The Beveridge curve appears because vacancy posting and "
            "unemployment respond on opposite sides of the matching market. "
            "The cloud is tight here because the only shock is aggregate "
            "productivity. A shock to matching efficiency $\\chi$ would shift "
            "the curve rather than just move the economy along it."
        ),
    )

    fig3, ax3 = plt.subplots()
    ax3.plot(
        z_grid,
        theta_nonlinear_grid,
        color="black",
        linewidth=2.2,
        label="Nonlinear finite-state rule",
    )
    ax3.plot(
        z_grid,
        theta_linear_grid,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="Log-linear rule",
    )
    ax3.scatter(
        z_plot[::20],
        theta_plot[::20],
        s=6,
        alpha=0.20,
        color="steelblue",
        label="Simulated months",
    )
    ax3.axvline(z_bar, color="0.45", linestyle=":", linewidth=1.1)
    ax3.axhline(theta_ss, color="0.45", linestyle=":", linewidth=1.1)
    ax3.set_xlabel("Productivity $z_t$")
    ax3.set_ylabel("Labor-market tightness $\\theta_t$")
    ax3.set_title("Productivity and Vacancy Creation")
    ax3.legend()
    report.add_figure(
        "figures/productivity-tightness.png",
        "Nonlinear and log-linear tightness rules by productivity.",
        fig3,
        description=(
            "Over this shock range, the log-linear rule tracks the nonlinear "
            "free-entry fixed point closely. The failure is economic: "
            f"even the nonlinear rule moves tightness only about "
            f"{df_stats.loc[3, 'Std./Std. z']} times as much as productivity, "
            "well below the volatility of labor-market tightness emphasized by "
            "Shimer."
        ),
    )

    report.add_table(
        "tables/business-cycle-stats.csv",
        "Simulated business-cycle moments",
        df_stats,
        description=(
            "The table reports log-deviation moments after the burn-in. "
            "Tightness is strongly procyclical and unemployment is strongly "
            "countercyclical, so the model has the right signs. The volatility "
            "ratios show the Shimer puzzle: the canonical calibration produces "
            "far too little amplification relative to productivity."
        ),
    )

    report.add_takeaway(
        "The DMP model gives an equilibrium account of the Beveridge curve: "
        "productivity raises match surplus, vacancy posting increases, and "
        "unemployment falls through the job-finding rate. The same computation "
        "also makes Shimer's result sharp. With $b=0.40$, the surplus "
        "from a match is not small enough for modest productivity shocks to "
        "create large swings in vacancy creation. Changing the numerical method "
        "from a local rule to a finite-state nonlinear fixed point does not "
        "remove that tension; resolving it requires changing the economic "
        "surplus or other primitives."
    )

    report.add_references(
        [
            "Diamond, P. (1982). \"Aggregate Demand Management in Search Equilibrium.\" "
            "*Journal of Political Economy*, 90(5), 881-894.",
            "Mortensen, D. and Pissarides, C. (1994). \"Job Creation and Job Destruction "
            "in the Theory of Unemployment.\" *Review of Economic Studies*, 61(3), 397-415.",
            "Shimer, R. (2005). \"The Cyclical Behavior of Equilibrium Unemployment and "
            "Vacancies.\" *American Economic Review*, 95(1), 25-49.",
            "Hagedorn, M. and Manovskii, I. (2008). \"The Cyclical Behavior of Equilibrium "
            "Unemployment and Vacancies Revisited.\" *American Economic Review*, 98(4), 1692-1706.",
            "Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press.",
        ]
    )

    report.write("README.md")
    print(
        f"Generated README.md, {len(report._figures)} figures, "
        f"and {len(report._tables)} table."
    )


if __name__ == "__main__":
    main()
