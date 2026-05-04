#!/usr/bin/env python3
"""Dynamic Discrete Choice: Rust-Style Bus Engine Replacement.

Solves and estimates a small dynamic discrete choice model with Type-I extreme
value shocks. The state is bus mileage, and the action is whether to replace the
engine. The tutorial compares full-solution maximum likelihood with a simple
Hotz-Miller conditional-choice-probability estimator.

Reference: Rust (1987); Hotz and Miller (1993).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import save_thumbnail, setup_style


EULER_GAMMA = 0.5772156649015329


def build_transition_matrices(x: np.ndarray, delta_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Build replacement and no-replacement mileage transition matrices."""
    n = len(x)
    keep = np.zeros((n, n))
    for i, current in enumerate(x):
        increments = x - current
        valid = increments >= -1e-12
        keep[i, valid] = np.exp(-np.maximum(increments[valid], 0.0)) * (1.0 - np.exp(-delta_x))
        keep[i, -1] += max(0.0, 1.0 - keep[i].sum())
        keep[i] = keep[i] / keep[i].sum()

    replace = np.tile(keep[0], (n, 1))
    return replace, keep


def solve_ddc(
    theta: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    tol: float = 1e-10,
    max_iter: int = 5_000,
) -> dict[str, np.ndarray | float | int | bool]:
    """Solve the conditional value function contraction."""
    flow_replace = np.zeros_like(x)
    flow_keep = theta[0] + theta[1] * x
    values = np.zeros((len(x), 2))

    for iteration in range(1, max_iter + 1):
        inclusive = logsumexp(values, axis=1) + EULER_GAMMA
        next_replace = F_replace @ inclusive
        next_keep = F_keep @ inclusive
        values_new = np.column_stack([
            flow_replace + beta * next_replace,
            flow_keep + beta * next_keep,
        ])
        error = float(np.max(np.abs(values_new - values)))
        values = values_new
        if error < tol:
            break

    log_denom = logsumexp(values, axis=1)
    p_replace = np.exp(values[:, 0] - log_denom)
    return {
        "values": values,
        "p_replace": p_replace,
        "flow_keep": flow_keep,
        "iterations": iteration,
        "error": error,
        "converged": error < tol,
    }


def draw_next_state(rng: np.random.Generator, row: np.ndarray) -> int:
    """Draw a next-state index from a transition row."""
    return int(rng.choice(len(row), p=row))


def simulate_buses(
    theta: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    n_buses: int,
    n_periods: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate panel data from the solved DDC model."""
    solution = solve_ddc(theta, x, F_replace, F_keep, beta)
    p_replace = np.asarray(solution["p_replace"])
    rng = np.random.default_rng(seed)

    x_index = np.zeros((n_buses, n_periods), dtype=int)
    mileage = np.zeros((n_buses, n_periods))
    replace = np.zeros((n_buses, n_periods), dtype=int)

    current = np.zeros(n_buses, dtype=int)
    for t in range(n_periods):
        x_index[:, t] = current
        mileage[:, t] = x[current]
        draws = rng.uniform(size=n_buses)
        replace[:, t] = draws < p_replace[current]

        next_state = np.empty(n_buses, dtype=int)
        for i in range(n_buses):
            row = F_replace[current[i]] if replace[i, t] else F_keep[current[i]]
            next_state[i] = draw_next_state(rng, row)
        current = next_state

    return {
        "x_index": x_index,
        "mileage": mileage,
        "replace": replace,
        "p_replace": p_replace,
    }


def panel_log_likelihood(p_replace: np.ndarray, data: dict[str, np.ndarray]) -> float:
    """Choice log likelihood for observed replacement decisions."""
    p = np.clip(p_replace[data["x_index"]], 1e-8, 1.0 - 1e-8)
    y = data["replace"]
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def estimate_full_solution(
    data: dict[str, np.ndarray],
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate structural parameters by full-solution maximum likelihood."""
    bounds = [(0.5, 3.5), (-0.35, -0.03)]

    def objective(theta: np.ndarray) -> float:
        solution = solve_ddc(theta, x, F_replace, F_keep, beta, tol=1e-8)
        return -panel_log_likelihood(np.asarray(solution["p_replace"]), data)

    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80})
    return {"theta": result.x, "objective": result.fun, "success": result.success}


def estimate_first_stage_logit(data: dict[str, np.ndarray]) -> np.ndarray:
    """Fit a reduced-form logit replacement rule in mileage and mileage squared."""
    mileage = data["mileage"].ravel()
    y = data["replace"].ravel()
    X = np.column_stack([np.ones_like(mileage), mileage, mileage**2])

    def objective(gamma: np.ndarray) -> float:
        eta = X @ gamma
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(objective, np.zeros(3), method="BFGS")
    return result.x


def hotz_miller_ccp(
    theta: np.ndarray,
    p_hat: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Map structural parameters and first-stage CCPs into model-implied CCPs."""
    p_replace = np.clip(p_hat, 1e-5, 1.0 - 1e-5)
    p_keep = 1.0 - p_replace
    flow_keep = theta[0] + theta[1] * x

    expected_flow = (
        p_replace * (-np.log(p_replace))
        + p_keep * (flow_keep - np.log(p_keep))
        + EULER_GAMMA
    )
    policy_transition = p_replace[:, None] * F_replace + p_keep[:, None] * F_keep
    ex_ante_value = np.linalg.solve(np.eye(len(x)) - beta * policy_transition, expected_flow)

    value_replace = beta * (F_replace @ ex_ante_value)
    value_keep = flow_keep + beta * (F_keep @ ex_ante_value)
    return expit(value_replace - value_keep)


def estimate_ccp(
    data: dict[str, np.ndarray],
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate structural parameters using a Hotz-Miller-style CCP step."""
    gamma = estimate_first_stage_logit(data)
    design_grid = np.column_stack([np.ones_like(x), x, x**2])
    p_hat = expit(design_grid @ gamma)
    bounds = [(0.5, 3.5), (-0.35, -0.03)]

    def objective(theta: np.ndarray) -> float:
        p_model = hotz_miller_ccp(theta, p_hat, x, F_replace, F_keep, beta)
        return -panel_log_likelihood(p_model, data)

    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80})
    return {"theta": result.x, "objective": result.fun, "success": result.success, "p_hat": p_hat}


def main() -> None:
    beta = 0.90
    theta_true = np.array([2.00, -0.15])
    x_min, x_max, delta_x = 0.0, 15.0, 0.25
    n_buses, n_periods = 1_500, 35

    x = np.arange(x_min, x_max + delta_x, delta_x)
    F_replace, F_keep = build_transition_matrices(x, delta_x)
    solution_true = solve_ddc(theta_true, x, F_replace, F_keep, beta)
    data = simulate_buses(theta_true, x, F_replace, F_keep, beta, n_buses, n_periods, seed=12)

    full_est = estimate_full_solution(data, x, F_replace, F_keep, beta, start=np.array([1.7, -0.12]))
    ccp_est = estimate_ccp(data, x, F_replace, F_keep, beta, start=np.array([1.7, -0.12]))
    solution_full = solve_ddc(np.asarray(full_est["theta"]), x, F_replace, F_keep, beta)
    solution_ccp = solve_ddc(np.asarray(ccp_est["theta"]), x, F_replace, F_keep, beta)

    repair_rate = float(data["replace"].mean())
    average_mileage = float(data["mileage"].mean())
    print("Dynamic discrete choice tutorial")
    print(f"  VFI converged in {solution_true['iterations']} iterations")
    print(f"  Simulated repair rate: {repair_rate:.3f}")
    print(f"  Full-solution estimate: {full_est['theta']}")
    print(f"  CCP estimate: {ccp_est['theta']}")

    setup_style()
    report = ModelReport(
        "Dynamic Discrete Choice",
        "Rust-style bus engine replacement with full-solution and CCP estimation.",
    )

    report.add_overview(
        "Dynamic discrete choice models are used when agents make repeated discrete "
        "decisions and today's action changes tomorrow's state. The canonical example "
        "is Rust's bus engine replacement problem. A bus operator observes mileage and "
        "chooses whether to replace the engine. In this normalization, replacement gives "
        "up the current keep payoff but resets future mileage and reduces future maintenance "
        "costs.\n\n"
        "This tutorial solves the model, simulates panel data, and estimates the payoff "
        "parameters two ways: full-solution maximum likelihood and a Hotz-Miller "
        "conditional-choice-probability estimator."
    )

    report.add_equations(
        r"""
State $x$ is mileage. Action $a=1$ replaces the engine, while $a=0$ keeps it.
Replacement utility is normalized to zero:

$$u(x,1) = 0,$$

and the flow payoff from keeping the engine is:

$$u(x,0) = \theta_0 + \theta_1 x, \qquad \theta_1 < 0.$$

With Type-I extreme value shocks, the conditional value functions satisfy:

$$v_a(x) = u(x,a) + \beta \sum_{x'} F_a(x' \mid x)
\left[\log\left(\exp(v_1(x')) + \exp(v_0(x'))\right) + \gamma\right].$$

The replacement probability is:

$$P(a=1 \mid x) =
\frac{\exp(v_1(x))}{\exp(v_1(x))+\exp(v_0(x))}.$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $\\theta_0$ | {theta_true[0]:.2f} | Keep-engine payoff intercept |\n"
        f"| $\\theta_1$ | {theta_true[1]:.2f} | Mileage cost slope |\n"
        f"| Mileage states | {len(x)} | Grid from {x_min:g} to {x_max:g} |\n"
        f"| Buses | {n_buses} | Simulated panel units |\n"
        f"| Periods | {n_periods} | Observations per bus |"
    )

    report.add_solution_method(
        "**Full-solution ML** solves the value function for every candidate parameter "
        "vector and evaluates the likelihood of observed replacement decisions.\n\n"
        "**CCP estimation** first estimates the policy rule nonparametrically with a "
        "logit in mileage and mileage squared. The Hotz-Miller inversion then recovers "
        "ex ante values implied by those estimated CCPs, avoiding a nested value-function "
        "solve inside the structural objective."
    )

    values = np.asarray(solution_true["values"])
    p_true = np.asarray(solution_true["p_replace"])
    p_full = np.asarray(solution_full["p_replace"])
    p_ccp = np.asarray(solution_ccp["p_replace"])

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1a.plot(x, values[:, 0], label="Replace")
    ax1a.plot(x, values[:, 1], label="Keep")
    ax1a.set_xlabel("Mileage")
    ax1a.set_ylabel("Conditional value")
    ax1a.set_title("Conditional Value Functions")
    ax1a.legend()
    ax1b.plot(x, p_true, label="True")
    ax1b.plot(x, np.asarray(ccp_est["p_hat"]), "--", label="First-stage logit")
    ax1b.set_xlabel("Mileage")
    ax1b.set_ylabel("Replacement probability")
    ax1b.set_title("Conditional Choice Probability")
    ax1b.legend()
    fig1.tight_layout()
    report.add_figure(
        "figures/value-and-ccp.png",
        "Value functions and replacement probabilities",
        fig1,
        description=(
            "The keep option becomes less attractive as mileage rises. The replacement "
            "probability is therefore low at fresh-engine states and high at worn-engine states."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    periods = np.arange(1, n_periods + 1)
    for bus in range(6):
        ax2.plot(periods, data["mileage"][bus], alpha=0.85, linewidth=1.6)
        repair_periods = periods[data["replace"][bus] == 1]
        ax2.scatter(repair_periods, data["mileage"][bus, data["replace"][bus] == 1], s=25)
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Mileage")
    ax2.set_title("Simulated Bus Histories")
    report.add_figure(
        "figures/simulated-histories.png",
        "Mileage histories for six simulated buses",
        fig2,
        description=(
            "Mileage drifts upward when the operator keeps the old engine and resets after "
            "replacement. The scattered points mark replacement decisions."
        ),
    )

    fig3, ax3 = plt.subplots()
    ax3.plot(x, p_true, label="True")
    ax3.plot(x, p_full, "--", label="Full-solution ML")
    ax3.plot(x, p_ccp, ":", label="CCP")
    ax3.set_xlabel("Mileage")
    ax3.set_ylabel("Replacement probability")
    ax3.set_title("Estimated Policy Rules")
    ax3.legend()
    report.add_figure(
        "figures/estimated-policies.png",
        "Policy rules implied by the true and estimated parameters",
        fig3,
        description=(
            "Both estimators recover the main economic pattern: replacement becomes more "
            "likely as mileage increases. Differences are largest in high-mileage states "
            "that are rarely observed because buses are usually replaced before reaching them."
        ),
    )

    estimate_table = pd.DataFrame({
        "Parameter": ["theta_0", "theta_1"],
        "True": theta_true,
        "Full-solution ML": np.asarray(full_est["theta"]),
        "CCP": np.asarray(ccp_est["theta"]),
    })
    report.add_table(
        "tables/parameter-estimates.csv",
        "Structural parameter estimates",
        estimate_table,
        description="The full-solution estimator nests a value-function solve inside the likelihood. The CCP estimator uses an estimated policy rule to reduce that computational burden.",
    )

    moments = pd.DataFrame({
        "Moment": ["Repair rate", "Average mileage", "VFI iterations", "Full ML success", "CCP success"],
        "Value": [
            repair_rate,
            average_mileage,
            float(solution_true["iterations"]),
            float(bool(full_est["success"])),
            float(bool(ccp_est["success"])),
        ],
    })
    report.add_table(
        "tables/simulation-moments.csv",
        "Simulation and solver diagnostics",
        moments,
        description="The simulated panel gives the estimators repeated choices at many mileage states.",
    )

    report.add_takeaway(
        "The hard part of dynamic discrete choice is the feedback between choices and "
        "future states. Full-solution likelihood is conceptually direct but expensive "
        "because each parameter guess requires solving the dynamic program. CCP methods "
        "trade some first-stage smoothing for speed by using observed choice probabilities "
        "to infer continuation values."
    )

    report.add_references([
        "Rust, J. (1987). Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. Econometrica.",
        "Hotz, V. J. and Miller, R. A. (1993). Conditional choice probabilities and the estimation of dynamic models. Review of Economic Studies.",
        "Aguirregabiria, V. and Mira, P. (2010). Dynamic discrete choice structural models: A survey. Journal of Econometrics.",
    ])
    report.write()
    save_thumbnail("figures/estimated-policies.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
