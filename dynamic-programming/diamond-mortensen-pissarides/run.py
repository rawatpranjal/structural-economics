#!/usr/bin/env python3
"""DMP search-and-matching tutorial.

Compute the Diamond-Mortensen-Pissarides equilibrium under the Shimer (2005)
calibration. Two solvers run in parallel: a closed-form local rule built from the
linearized free-entry condition, and a finite-state nonlinear free-entry fixed
point on a Rouwenhorst grid. A finer Rouwenhorst grid serves as the
discretization benchmark, and a small sensitivity table varies the flow value of
unemployment to make the Hagedorn-Manovskii (2008) resolution of the
amplification puzzle visible.
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
    contraction_modulus = beta * (1.0 - separation_rate)

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

    report = ModelReport(
        "DMP Search, Vacancies, and Unemployment",
        "Free entry, Nash bargaining, and the Shimer amplification puzzle.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "An unemployed worker meets a posted vacancy through a matching "
        "technology, the resulting match generates surplus $z_t-b$ which is "
        "split by Nash bargaining, and free entry of vacancies pins down how "
        "many firms find it profitable to post. The single equilibrium price is "
        "labor-market tightness $\\theta_t=v_t/u_t$: high tightness means firms "
        "are competing for workers, raising the worker's job-finding rate and "
        "lowering the firm's vacancy-filling rate.\n\n"
        "The mechanism is direct. A productivity shock changes the present "
        "value $J_t$ of a filled job, which through the free-entry condition "
        "$k=\\beta q(\\theta_t)\\mathbb{E}_t[J_{t+1}]$ pulls $\\theta_t$ in the "
        "same direction. Higher tightness then raises $f(\\theta_t)$, drains "
        "unemployment through the matching rate, and produces the downward-"
        "sloping Beveridge curve in $(u,v)$-space.\n\n"
        "Quantitatively the model has a problem. With Shimer's (2005) "
        f"calibration the surplus $z-b={surplus_ss:.2f}$ is large relative to "
        f"the vacancy cost $k={vacancy_cost:.3f}$, so a small productivity "
        "innovation barely moves the present-value calculus that drives entry. "
        "The simulated standard deviation ratio of tightness to productivity "
        "is roughly $1.7$ here, while in U.S. data it is closer to $19$. "
        "Hagedorn and Manovskii (2008) trace the amplification gap to this "
        "single calibration choice: with $b$ near $\\bar z$ the surplus is "
        "small, the elasticity coefficient $C$ explodes, and the model can "
        "match observed labor-market volatility.\n\n"
        "Two solvers compute the equilibrium and confirm that the puzzle is "
        "*economic*, not numerical. The log-linear local rule writes "
        "$\\hat\\theta_t=C\\hat z_t$ with $C$ derived analytically from the "
        "linearized free-entry condition. The nonlinear rule discretizes "
        "$\\hat z$ on a Rouwenhorst grid and iterates the free-entry fixed "
        "point at every node. Both produce nearly the same equilibrium because "
        "shocks are small and the relevant nonlinearity sits in $\\theta\\to "
        "q(\\theta)$, which is nearly log-linear over the ergodic range.\n\n"
        "Two cross-references frame this tutorial. The "
        "[McCall search tutorial](../job-search-mccall/) keeps only the worker "
        "side: a known wage offer distribution and an optimal-stopping "
        "reservation rule. Here vacancy posting is endogenous and the wage is "
        "bargained, so the worker's outside option matters in equilibrium. "
        "The [RBC tutorial](../rbc/) drives a representative-agent business "
        "cycle with the same persistent productivity shock; in DMP that shock "
        "propagates through match formation rather than capital accumulation."
    )

    report.add_equations(
        rf"""
**Matching technology.** Let $u_t$ be the unemployment rate, $v_t$ the vacancy
rate, and $\theta_t=v_t/u_t$ tightness. Constant-returns matching gives

$$m(u_t,v_t)=\chi u_t^{{1-\eta}}v_t^\eta,\qquad
f(\theta_t)=\chi\theta_t^{{\eta}},\qquad
q(\theta_t)=\chi\theta_t^{{\eta-1}},$$

where $f$ is the worker's job-finding rate and $q$ is the firm's
vacancy-filling rate. Both depend on $\theta_t$ only.

**Productivity.** Aggregate productivity is a stationary AR(1) in logs,

$$\hat z_{{t+1}}=\rho\hat z_t+\epsilon_{{t+1}},\quad
\epsilon_{{t+1}}\sim\mathcal{{N}}(0,\sigma_\epsilon^2),\quad
z_t=\bar z\exp(\hat z_t).$$

**Wage rule.** Nash bargaining with worker weight $\gamma$ splits joint
surplus and yields the equilibrium wage

$$w_t=\gamma(z_t+k\theta_t)+(1-\gamma)b,$$

where $b$ is the flow value of unemployment (benefits, leisure, home
production) and $k$ is the per-period cost of an open vacancy. The term
$k\theta_t$ enters because tighter markets are more costly in expectation
for the worker to walk away from.

**Job value and free entry.** A filled job satisfies the recursive identity

$$J_t=z_t-w_t+\beta(1-\sigma)\,\mathbb{{E}}_t[J_{{t+1}}],$$

with $\sigma$ the exogenous separation rate. Free entry of vacancies equates
the expected discounted job value with the cost of one open vacancy,

$$k=\beta\,q(\theta_t)\,\mathbb{{E}}_t[J_{{t+1}}].$$

This is the equilibrium condition that pins down $\theta_t$.

**Stock dynamics.** Once $\theta_t$ is determined, unemployment evolves
mechanically and vacancies follow as a residual:

$$u_{{t+1}}=\sigma(1-u_t)+\bigl(1-f(\theta_t)\bigr)u_t,\qquad
v_t=\theta_t u_t.$$

The deterministic steady state has $u_{{ss}}=\sigma/\bigl(\sigma+f(\theta_{{ss}})\bigr)$,
the textbook Beveridge relation.

**Local linearization.** Writing $\hat\theta_t=\log\theta_t-\log\theta_{{ss}}$ and
linearizing the free-entry condition at $\theta_{{ss}}=1$ delivers a closed-form
elasticity rule $\hat\theta_t=C\hat z_t$ with

$$C=\frac{{\rho}}{{A-B\rho}},\qquad
A=\frac{{\eta k}}{{(1-\gamma)\beta\chi}},\qquad
B=\beta A(1-\sigma)-\frac{{\gamma k}}{{1-\gamma}}.$$

At the baseline calibration $A={a_coeff:.4f}$ and $B={b_coeff:.4f}$, so a one-percent
productivity innovation raises tightness by $C={elasticity:.2f}$ percent.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| Discount factor $\\beta$ | {beta:.3f} | Monthly time preference |\n"
        f"| Productivity persistence $\\rho$ | {rho:.3f} | AR(1) coefficient on $\\hat z_t$ |\n"
        f"| Innovation s.d. $\\sigma_\\epsilon$ | {shock_sigma:.4f} | Monthly productivity shock |\n"
        f"| Separation rate $\\sigma$ | {separation_rate:.3f} | Exogenous job destruction |\n"
        f"| Matching efficiency $\\chi$ | {matching_efficiency:.2f} | Level of $m(u,v)$ |\n"
        f"| Matching elasticity $\\eta$ | {matching_elasticity:.2f} | Vacancy elasticity in $m$ |\n"
        f"| Worker bargaining weight $\\gamma$ | {bargaining_power:.2f} | Nash share |\n"
        f"| Flow value of unemployment $b$ | {benefit:.2f} | Outside option |\n"
        f"| Vacancy cost $k$ | {vacancy_cost:.4f} | Calibrated for $\\theta_{{ss}}=1$ |\n"
        f"| Steady-state unemployment $u_{{ss}}$ | {u_ss:.4f} | $\\sigma/(\\sigma+f(\\theta_{{ss}}))$ |\n"
        f"| Steady-state wage $w_{{ss}}$ | {wage_ss:.4f} | Nash wage at $z=\\bar z$ |\n"
        f"| Surplus $\\bar z-b$ | {surplus_ss:.2f} | Match surplus before vacancy costs |\n"
        f"| Coarse grid $N_z$ | {n_z_coarse} | Rouwenhorst nodes (tutorial run) |\n"
        f"| Fine-grid benchmark $N_z$ | {n_z_fine} | Discretization audit |\n"
        f"| Simulation length | {periods - burn} months | Post-burn-in moments |"
    )

    report.add_solution_method(
        "Two solvers run in parallel.\n\n"
        "**Log-linear local rule.** Linearizing the free-entry condition and "
        "the productivity AR(1) around the deterministic steady state delivers "
        "the closed-form elasticity $C=\\rho/(A-B\\rho)$ above. The mapping "
        "$\\theta_t=\\exp(C\\hat z_t)$ is exact to first order in $\\hat z$ and "
        "carries no discretization error. Its weakness is that it forces a "
        "single elasticity at every productivity level, which would matter if "
        "the underlying $\\theta(z)$ were strongly curved over the ergodic set.\n\n"
        "**Nonlinear free-entry fixed point.** Discretize $\\hat z_t$ on a "
        f"Rouwenhorst grid with $N_z={n_z_coarse}$ nodes and transition matrix "
        "$P_{ij}=\\Pr(\\hat z_{t+1}=\\hat z_j\\mid\\hat z_t=\\hat z_i)$. "
        "Substitute the free-entry condition for $\\theta_i$ inside the job-"
        "value Bellman to get a recursion in $J_i=J(z_i)$ alone:\n\n"
        "$$J_i=(1-\\gamma)(z_i-b)-\\gamma k\\theta_i+\\beta(1-\\sigma)\\sum_j P_{ij}J_j,\\qquad "
        "\\theta_i=\\Bigl(\\tfrac{\\beta\\chi}{k}\\sum_j P_{ij}J_j\\Bigr)^{1/(1-\\eta)}.$$\n\n"
        "The right-hand side defines a contraction in the sup norm with "
        f"modulus $\\beta(1-\\sigma)={contraction_modulus:.4f}$, so iterates "
        "converge geometrically at that rate independent of $N_z$.\n\n"
        "```text\n"
        "Algorithm 1: Log-linear local rule\n"
        "Inputs    primitives (β, σ, χ, η, γ, b, z̄, ρ); shock series {ẑ_t}\n"
        "Outputs   elasticity C; tightness {θ_t}; unemployment {u_t}, vacancies {v_t}\n"
        "\n"
        "1. Calibrate k from θ_ss = 1 at z = z̄\n"
        "2. Compute  A = η k / [(1−γ) β χ],  B = β A (1−σ) − γ k / (1−γ)\n"
        "3. Set C = ρ / (A − B ρ)\n"
        "4. For each ẑ_t in the simulation:\n"
        "       θ_t  ← exp(C · ẑ_t)\n"
        "       f_t  ← χ θ_t^η\n"
        "       u_{t+1} ← σ (1 − u_t) + (1 − f_t) u_t\n"
        "       v_t  ← θ_t · u_t\n"
        "```\n\n"
        "```text\n"
        "Algorithm 2: Nonlinear finite-state free-entry fixed point\n"
        "Inputs    primitives; Rouwenhorst grid {ẑ_i}_{i=1..N_z};\n"
        "          transition matrix P; calibrated k; tolerance ε\n"
        "Outputs   job value J_i and tightness θ_i at each productivity state z_i\n"
        "\n"
        "Initialise   J_i ← k / (β χ)               # value if free entry binds today\n"
        "repeat n = 0, 1, 2, ...:\n"
        "    EJ_i  ← Σ_j P_{ij} J_j                 # one mat-vec multiply\n"
        "    θ_i   ← (β χ EJ_i / k)^{1/(1−η)}       # invert free entry\n"
        "    J_i^new ← (1−γ)(z_i − b) − γ k θ_i + β(1−σ) EJ_i\n"
        "    err   ← max_i |J_i^new − J_i|\n"
        "    J_i   ← J_i^new\n"
        "until err < ε\n"
        "```\n\n"
        "**Discretization audit.** The same nonlinear solver is rerun with "
        f"$N_z={n_z_fine}$ nodes on a wider Rouwenhorst grid that brackets the "
        "coarse one. Interpolating the fine solution onto the coarse $\\hat z$ "
        f"grid gives a max relative gap in $\\theta(z)$ of "
        f"**{100.0 * max_grid_gap:.2e}%**, so the $N_z={n_z_coarse}$ tutorial "
        "run is essentially the discretization-free answer.\n\n"
        f"At baseline calibration: log-linear $C={elasticity:.3f}$; nonlinear "
        f"$N_z={n_z_coarse}$ converged in **{nonlinear['iterations']} "
        f"iterations** to sup-norm error **{nonlinear['error']:.2e}**; "
        f"nonlinear $N_z={n_z_fine}$ converged in "
        f"**{nonlinear_fine['iterations']} iterations** to "
        f"**{nonlinear_fine['error']:.2e}**. The maximum relative gap between "
        f"the nonlinear coarse solution and the log-linear rule is "
        f"**{100.0 * max_policy_gap:.2f}%** across the productivity grid."
    )

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
    report.add_figure(
        "figures/productivity-tightness.png",
        "Tightness as a function of productivity: log-linear, nonlinear coarse, and nonlinear fine-grid benchmark, with simulated months overlaid.",
        fig1,
        description=(
            "The nonlinear free-entry rule (black) and the log-linear local "
            "rule (red dashed) trace nearly the same path through "
            "$(z,\\theta)$-space, and the fine-grid benchmark (green dotted) "
            "sits indistinguishably on top of the coarse-grid solution. The "
            "blue cloud is where the simulated economy actually spends "
            f"time. Reading off the cloud, $\\theta_t$ moves about "
            f"{df_stats.loc[3, 'Std./Std. z']} times more than $z_t$ in log "
            "deviations — well below the order-of-magnitude amplification that "
            "Shimer (2005) measures in U.S. labor-market data. The puzzle is "
            "economic, not numerical: switching from the local rule to the "
            "global solver leaves the volatility ratio essentially unchanged."
        ),
    )

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
    report.add_figure(
        "figures/unemployment-vacancies.png",
        "Simulated unemployment and vacancy paths under the nonlinear tightness rule.",
        fig2,
        description=(
            "Once tightness is determined, the rest is mechanics. Vacancy "
            "posting jumps because free entry makes $v_t$ the firm's contemporaneous "
            "choice variable; unemployment moves with a lag because it is a "
            "stock — today's hires only show up as a smaller pool of "
            "searchers tomorrow. The result is a co-movement of low-frequency "
            "$u$ and high-frequency $v$ that produces the cyclical "
            "asymmetry the Beveridge curve below traces out."
        ),
    )

    fig3, ax3 = plt.subplots()
    ax3.scatter(u_plot, v_plot, s=3, alpha=0.25, color="steelblue", edgecolors="none")
    ax3.scatter([u_ss], [theta_ss * u_ss], s=45, color="black", label="Steady state")
    ax3.set_xlabel("Unemployment rate $u_t$")
    ax3.set_ylabel("Vacancy rate $v_t$")
    ax3.set_title("A Beveridge Curve From Matching Frictions")
    ax3.legend()
    report.add_figure(
        "figures/beveridge-curve.png",
        "Simulated unemployment and vacancy pairs trace a downward-sloping Beveridge curve around the steady state.",
        fig3,
        description=(
            "Productivity shocks move the economy along the curve; what makes "
            "the locus downward-sloping is the matching technology, which "
            "trades unemployment for vacancies through the constant-returns "
            "matching function. The cluster is one-dimensional because $z_t$ "
            "is the only shock — separations and matching efficiency are held "
            "fixed. Reallocation shocks to $\\sigma$ or $\\chi$ would shift "
            "this curve outward or inward instead of moving along it, which is "
            "the standard empirical decomposition of Beveridge-curve "
            "movements."
        ),
    )

    report.add_table(
        "tables/business-cycle-stats.csv",
        "Simulated business-cycle moments",
        df_stats,
        description=(
            "The signs are right: tightness and vacancies are strongly "
            "procyclical, unemployment is strongly countercyclical, and the "
            "log-linear and nonlinear simulations agree closely. The "
            "amplification ratios are wrong: Shimer (2005) reports an "
            "empirical $\\sigma_\\theta/\\sigma_z\\approx 19$, this calibration "
            "delivers about $1.7$. Switching solvers does not move that "
            "number."
        ),
    )

    report.add_table(
        "tables/amplification-by-surplus.csv",
        "Tightness elasticity by flow value of unemployment",
        df_amp,
        description=(
            "Where the surplus comes from is the entire computational story. "
            "Holding the matching technology and bargaining weight fixed and "
            "raising $b$ from Shimer's $0.40$ to Hagedorn-Manovskii's $0.95$ "
            "shrinks the surplus $\\bar z-b$ by an order of magnitude and "
            "drives the elasticity $C$ from about $1.55$ to roughly $19$ — "
            "the empirical volatility ratio Shimer measures. A small-surplus "
            "economy amplifies productivity shocks because the vacancy cost "
            "$k$ is then a substantial fraction of expected match value, so "
            "a small move in $z$ is a large move in the ratio "
            "$\\beta\\chi\\,\\mathbb{E}[J']/k$ that drives free entry."
        ),
    )

    report.add_takeaway(
        "DMP gives an equilibrium account of the Beveridge curve. Productivity "
        "raises match surplus, vacancy posting expands through free entry, and "
        "unemployment falls as the job-finding rate rises. Switching from a "
        "log-linear local rule to a global nonlinear free-entry fixed point "
        "leaves the volatility ratios essentially unchanged, so the Shimer "
        "(2005) amplification puzzle is *not* a numerical artefact: with "
        f"$b=0.40$ the surplus $\\bar z-b={surplus_ss:.2f}$ is large enough that "
        "modest productivity shocks are barely transmitted through the firm's "
        "free-entry decision. The sensitivity table makes the lever explicit: "
        "amplification depends mechanically on the size of the surplus, "
        "exactly the channel Hagedorn and Manovskii (2008) exploit. Two "
        "natural extensions: replacing exogenous separations by an endogenous "
        "match-destruction margin (Mortensen-Pissarides 1994) gives the "
        "Beveridge curve a second source of variation; embedding the same "
        "search block inside the heterogeneous-agent saving problem in "
        "[Aiyagari](../aiyagari/) lets idiosyncratic income risk and "
        "frictional unemployment interact."
    )

    report.add_references(
        [
            "Diamond, P. (1982). \"Aggregate Demand Management in Search Equilibrium.\" "
            "*Journal of Political Economy*, 90(5), 881-894.",
            "Mortensen, D. and Pissarides, C. (1994). \"Job Creation and Job Destruction "
            "in the Theory of Unemployment.\" *Review of Economic Studies*, 61(3), 397-415.",
            "Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press, 2nd edition.",
            "Shimer, R. (2005). \"The Cyclical Behavior of Equilibrium Unemployment and "
            "Vacancies.\" *American Economic Review*, 95(1), 25-49.",
            "Hagedorn, M. and Manovskii, I. (2008). \"The Cyclical Behavior of Equilibrium "
            "Unemployment and Vacancies Revisited.\" *American Economic Review*, 98(4), 1692-1706.",
        ]
    )

    report.write("README.md")
    print(
        f"Generated README.md, {len(report._figures)} figures, "
        f"and {len(report._tables)} tables."
    )


if __name__ == "__main__":
    main()
