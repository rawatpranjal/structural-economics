#!/usr/bin/env python3
"""Dynamic Discrete Choice: Rust-Style Bus Engine Replacement.

Solves and estimates a small dynamic discrete choice model with Type-I extreme
value shocks. The state is bus mileage, and the action is whether to replace the
engine. The tutorial compares NFXP, CCP, and MPEC estimators.

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
from lib.plotting import save_figure, save_thumbnail, setup_style


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


def bellman_residual(
    theta: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Stack Bellman residuals for replacement and keep values."""
    flow_keep = theta[0] + theta[1] * x
    inclusive = logsumexp(values, axis=1) + EULER_GAMMA
    replace_rhs = beta * (F_replace @ inclusive)
    keep_rhs = flow_keep + beta * (F_keep @ inclusive)
    return np.column_stack([values[:, 0] - replace_rhs, values[:, 1] - keep_rhs]).ravel()


def pack_mpec(theta: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Pack structural parameters and conditional values for MPEC."""
    return np.concatenate([theta, values.ravel()])


def unpack_mpec(z: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack an MPEC decision vector."""
    return z[:2], z[2:].reshape(n_states, 2)


def estimate_mpec(
    data: dict[str, np.ndarray],
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    theta_start: np.ndarray,
) -> dict[str, object]:
    """Estimate by imposing Bellman equations as equality constraints."""
    start_solution = solve_ddc(theta_start, x, F_replace, F_keep, beta, tol=1e-8)
    z0 = pack_mpec(theta_start, np.asarray(start_solution["values"]))
    bounds = [(0.5, 3.5), (-0.35, -0.03)] + [(None, None)] * (2 * len(x))

    def objective(z: np.ndarray) -> float:
        _, values = unpack_mpec(z, len(x))
        log_denom = logsumexp(values, axis=1)
        p_replace = np.exp(values[:, 0] - log_denom)
        return -panel_log_likelihood(p_replace, data)

    def constraints(z: np.ndarray) -> np.ndarray:
        theta, values = unpack_mpec(z, len(x))
        return bellman_residual(theta, values, x, F_replace, F_keep, beta)

    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": constraints},
        options={"ftol": 1e-7, "maxiter": 80, "disp": False},
    )
    theta_hat, values_hat = unpack_mpec(result.x, len(x))
    residual = constraints(result.x)
    log_denom = logsumexp(values_hat, axis=1)
    p_replace = np.exp(values_hat[:, 0] - log_denom)
    return {
        "theta": theta_hat,
        "values": values_hat,
        "p_replace": p_replace,
        "objective": float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
        "constraint_max": float(np.max(np.abs(residual))),
        "message": str(result.message),
    }


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
    mpec_est = estimate_mpec(
        data,
        x,
        F_replace,
        F_keep,
        beta,
        theta_start=np.asarray(ccp_est["theta"]),
    )
    solution_full = solve_ddc(np.asarray(full_est["theta"]), x, F_replace, F_keep, beta)
    solution_ccp = solve_ddc(np.asarray(ccp_est["theta"]), x, F_replace, F_keep, beta)

    repair_rate = float(data["replace"].mean())
    average_mileage = float(data["mileage"].mean())
    high_mileage_share = float(np.mean(data["mileage"] >= 10.0))
    print("Dynamic discrete choice tutorial")
    print(f"  VFI converged in {solution_true['iterations']} iterations")
    print(f"  Simulated repair rate: {repair_rate:.3f}")
    print(f"  Full-solution estimate: {full_est['theta']}")
    print(f"  CCP estimate: {ccp_est['theta']}")
    print(f"  MPEC estimate: {mpec_est['theta']}")
    print(f"  MPEC Bellman residual: {mpec_est['constraint_max']:.2e}")

    setup_style()

    values = np.asarray(solution_true["values"])
    p_true = np.asarray(solution_true["p_replace"])
    p_full = np.asarray(solution_full["p_replace"])
    p_ccp = np.asarray(solution_ccp["p_replace"])
    p_mpec = np.asarray(mpec_est["p_replace"])
    p_hat = np.asarray(ccp_est["p_hat"])

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1a.plot(x, values[:, 0], label="Replace, $a=1$")
    ax1a.plot(x, values[:, 1], label="Keep, $a=0$")
    ax1a.set_xlabel("Mileage")
    ax1a.set_ylabel("Conditional value")
    ax1a.set_title("Conditional Values")
    ax1a.legend()
    ax1b.plot(x, p_true, label="True")
    ax1b.plot(x, p_hat, "--", label="First-stage logit")
    ax1b.set_xlabel("Mileage")
    ax1b.set_ylabel("Replacement probability")
    ax1b.set_title("Replacement Hazard")
    ax1b.legend()
    fig1.tight_layout()
    save_figure(fig1, "figures/value-and-ccp.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    periods = np.arange(1, n_periods + 1)
    for bus in range(6):
        ax2.plot(periods, data["mileage"][bus], alpha=0.85, linewidth=1.6)
        repair_periods = periods[data["replace"][bus] == 1]
        ax2.scatter(repair_periods, data["mileage"][bus, data["replace"][bus] == 1], s=25)
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Mileage")
    ax2.set_title("Simulated Bus Histories")
    save_figure(fig2, "figures/simulated-histories.png", dpi=150)

    fig3, ax3 = plt.subplots()
    ax3.plot(x, p_true, label="True")
    ax3.plot(x, p_full, "--", label="Full-solution ML")
    ax3.plot(x, p_ccp, ":", label="CCP")
    ax3.plot(x, p_mpec, "-.", label="MPEC")
    ax3.set_xlabel("Mileage")
    ax3.set_ylabel("Replacement probability")
    ax3.set_title("Estimated Replacement Policies")
    ax3.legend()
    save_figure(fig3, "figures/estimated-policies.png", dpi=150)

    theta_full = np.asarray(full_est["theta"])
    theta_ccp = np.asarray(ccp_est["theta"])
    theta_mpec = np.asarray(mpec_est["theta"])
    estimate_table = pd.DataFrame({
        "Parameter": ["theta_0", "theta_1"],
        "True": theta_true,
        "Full-solution ML": theta_full,
        "Full ML error": theta_full - theta_true,
        "CCP": theta_ccp,
        "CCP error": theta_ccp - theta_true,
        "MPEC": theta_mpec,
        "MPEC error": theta_mpec - theta_true,
    })
    Path("tables/parameter-estimates.csv").parent.mkdir(parents=True, exist_ok=True)
    estimate_table.round(5).to_csv("tables/parameter-estimates.csv", index=False)

    moments = pd.DataFrame({
        "Moment": [
            "Repair rate",
            "Average mileage",
            "Share with mileage >= 10",
            "VFI iterations",
            "Full ML success",
            "CCP success",
            "MPEC success",
            "MPEC iterations",
            "MPEC max Bellman residual",
        ],
        "Value": [
            repair_rate,
            average_mileage,
            high_mileage_share,
            float(solution_true["iterations"]),
            float(bool(full_est["success"])),
            float(bool(ccp_est["success"])),
            float(bool(mpec_est["success"])),
            float(mpec_est["iterations"]),
            float(mpec_est["constraint_max"]),
        ],
    })
    Path("tables/simulation-moments.csv").parent.mkdir(parents=True, exist_ok=True)
    moments.to_csv("tables/simulation-moments.csv", index=False)

    save_thumbnail("figures/estimated-policies.png", "figures/thumb.png")
    print("\nFigures and tables written.")


if __name__ == "__main__":
    main()
