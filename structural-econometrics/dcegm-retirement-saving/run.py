#!/usr/bin/env python3
"""Discrete-continuous EGM for retirement and saving.

The tutorial solves a finite-horizon life-cycle model with an absorbing
retirement choice and a continuous saving decision. It compares DC-EGM against
a smaller brute-force grid-search benchmark.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


@dataclass(frozen=True)
class RetirementPrimitives:
    """Calibration for the retirement-saving model."""

    beta: float = 0.96
    r: float = 0.02
    gamma: float = 2.0
    start_age: int = 55
    end_age: int = 70
    asset_min: float = 0.0
    asset_max: float = 22.0
    n_assets: int = 420
    audit_assets: int = 150
    pension: float = 0.78
    retire_amenity: float = 0.00
    bequest_weight: float = 1.15
    bequest_floor: float = 1.0
    taste_scale: float = 0.045
    initial_asset_mean: float = 2.8
    initial_asset_sigma: float = 0.42
    simulation_agents: int = 8_000
    simulation_seed: int = 619

    @property
    def R(self) -> float:
        return 1.0 + self.r

    @property
    def ages(self) -> np.ndarray:
        return np.arange(self.start_age, self.end_age + 1)

    @property
    def horizon(self) -> int:
        return len(self.ages)


def utility(consumption: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """CRRA utility with a small positivity guard."""
    c = np.maximum(consumption, 1.0e-10)
    if abs(gamma - 1.0) < 1.0e-12:
        return np.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def marginal_utility(consumption: np.ndarray, gamma: float) -> np.ndarray:
    """CRRA marginal utility."""
    return np.maximum(consumption, 1.0e-10) ** (-gamma)


def inverse_marginal_utility(mu: np.ndarray, gamma: float) -> np.ndarray:
    """Inverse CRRA marginal utility."""
    return np.maximum(mu, 1.0e-12) ** (-1.0 / gamma)


def working_income(age: int) -> float:
    """Deterministic labor income by age."""
    return 1.42 - 0.012 * (age - 55) - 0.006 * max(age - 62, 0) ** 2


def work_disutility(age: int) -> float:
    """Age-varying utility cost of working."""
    return 0.16 + 0.024 * (age - 55) + 0.010 * max(age - 62, 0) ** 2


def branch_income(age: int, choice: str, p: RetirementPrimitives) -> float:
    """Income associated with a discrete branch."""
    if choice == "work":
        return working_income(age)
    if choice == "retire":
        return p.pension
    raise ValueError(f"Unknown choice: {choice}")


def branch_shift(age: int, choice: str, p: RetirementPrimitives) -> float:
    """Nonconsumption utility term for a branch."""
    if choice == "work":
        return -work_disutility(age)
    if choice == "retire":
        return p.retire_amenity
    raise ValueError(f"Unknown choice: {choice}")


def terminal_value(asset_grid: np.ndarray, p: RetirementPrimitives) -> np.ndarray:
    """Terminal bequest value."""
    return p.bequest_weight * utility(asset_grid + p.bequest_floor, p.gamma)


def value_derivative(value: np.ndarray, asset_grid: np.ndarray) -> np.ndarray:
    """Numerical derivative of a value function, clipped to positive values."""
    derivative = np.gradient(value, asset_grid, edge_order=2)
    return np.maximum(derivative, 1.0e-8)


def interpolate_value(asset_grid: np.ndarray, value: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Linear interpolation with flat extrapolation at grid endpoints."""
    return np.interp(points, asset_grid, value, left=value[0], right=value[-1])


def egm_branch(
    age: int,
    choice: str,
    next_value: np.ndarray,
    asset_grid: np.ndarray,
    p: RetirementPrimitives,
) -> dict[str, np.ndarray]:
    """Solve one choice-specific branch by Euler inversion."""
    y = branch_income(age, choice, p)
    shift = branch_shift(age, choice, p)
    vprime_next = value_derivative(next_value, asset_grid)
    c_endog = inverse_marginal_utility(p.beta * p.R * vprime_next, p.gamma)
    a_next = asset_grid.copy()
    current_asset_endog = (c_endog + a_next - y) / p.R
    branch_value_endog = utility(c_endog, p.gamma) + shift + p.beta * next_value

    order = np.argsort(current_asset_endog)
    x = np.maximum.accumulate(current_asset_endog[order])
    c_sorted = c_endog[order]
    a_next_sorted = a_next[order]
    value_sorted = branch_value_endog[order]

    # Drop repeated x values created by the monotonicity repair.
    keep = np.r_[True, np.diff(x) > 1.0e-10]
    x = x[keep]
    c_sorted = c_sorted[keep]
    a_next_sorted = a_next_sorted[keep]
    value_sorted = value_sorted[keep]

    policy_next = np.interp(asset_grid, x, a_next_sorted, left=p.asset_min, right=a_next_sorted[-1])
    consumption = p.R * asset_grid + y - policy_next
    branch_value = np.interp(asset_grid, x, value_sorted, left=value_sorted[0], right=value_sorted[-1])

    constrained = asset_grid < x[0]
    if np.any(constrained):
        policy_next[constrained] = p.asset_min
        consumption[constrained] = p.R * asset_grid[constrained] + y - p.asset_min
        branch_value[constrained] = (
            utility(consumption[constrained], p.gamma)
            + shift
            + p.beta * next_value[0]
        )

    return {
        "value": branch_value,
        "consumption": np.maximum(consumption, 1.0e-10),
        "assets_next": np.clip(policy_next, p.asset_min, asset_grid[-1]),
    }


def solve_dcegm(
    asset_grid: np.ndarray,
    p: RetirementPrimitives,
) -> tuple[dict[str, np.ndarray], float]:
    """Solve the absorbing retirement model by DC-EGM."""
    start = time.perf_counter()
    n_t = p.horizon
    n_a = len(asset_grid)
    value = np.zeros((n_t + 1, 2, n_a))
    consumption = np.zeros((n_t, 2, n_a))
    assets_next = np.zeros((n_t, 2, n_a))
    retire_choice = np.zeros((n_t, n_a), dtype=bool)
    retire_gap = np.zeros((n_t, n_a))
    work_value = np.full((n_t, n_a), np.nan)
    retire_value_active = np.full((n_t, n_a), np.nan)
    work_consumption = np.full((n_t, n_a), np.nan)
    retire_consumption = np.full((n_t, n_a), np.nan)
    work_assets_next = np.full((n_t, n_a), np.nan)
    retire_assets_next = np.full((n_t, n_a), np.nan)

    value[n_t, 0, :] = terminal_value(asset_grid, p)
    value[n_t, 1, :] = terminal_value(asset_grid, p)

    for t in range(n_t - 1, -1, -1):
        age = int(p.ages[t])

        retired_branch = egm_branch(age, "retire", value[t + 1, 1], asset_grid, p)
        value[t, 1] = retired_branch["value"]
        consumption[t, 1] = retired_branch["consumption"]
        assets_next[t, 1] = retired_branch["assets_next"]

        work_branch = egm_branch(age, "work", value[t + 1, 0], asset_grid, p)
        retire_branch = egm_branch(age, "retire", value[t + 1, 1], asset_grid, p)
        choose_retire = retire_branch["value"] >= work_branch["value"]

        value[t, 0] = np.where(choose_retire, retire_branch["value"], work_branch["value"])
        consumption[t, 0] = np.where(
            choose_retire,
            retire_branch["consumption"],
            work_branch["consumption"],
        )
        assets_next[t, 0] = np.where(
            choose_retire,
            retire_branch["assets_next"],
            work_branch["assets_next"],
        )
        retire_choice[t] = choose_retire
        retire_gap[t] = retire_branch["value"] - work_branch["value"]
        work_value[t] = work_branch["value"]
        retire_value_active[t] = retire_branch["value"]
        work_consumption[t] = work_branch["consumption"]
        retire_consumption[t] = retire_branch["consumption"]
        work_assets_next[t] = work_branch["assets_next"]
        retire_assets_next[t] = retire_branch["assets_next"]

    result = {
        "value": value,
        "consumption": consumption,
        "assets_next": assets_next,
        "retire_choice": retire_choice,
        "retire_gap": retire_gap,
        "work_value": work_value,
        "retire_value_active": retire_value_active,
        "work_consumption": work_consumption,
        "retire_consumption": retire_consumption,
        "work_assets_next": work_assets_next,
        "retire_assets_next": retire_assets_next,
    }
    return result, time.perf_counter() - start


def solve_bruteforce(
    asset_grid: np.ndarray,
    p: RetirementPrimitives,
) -> tuple[dict[str, np.ndarray], float]:
    """Small brute-force VFI benchmark over next-period assets."""
    start = time.perf_counter()
    n_t = p.horizon
    n_a = len(asset_grid)
    value = np.zeros((n_t + 1, 2, n_a))
    consumption = np.zeros((n_t, 2, n_a))
    assets_next = np.zeros((n_t, 2, n_a))
    retire_choice = np.zeros((n_t, n_a), dtype=bool)
    retire_gap = np.zeros((n_t, n_a))
    value[n_t, 0] = terminal_value(asset_grid, p)
    value[n_t, 1] = terminal_value(asset_grid, p)

    for t in range(n_t - 1, -1, -1):
        age = int(p.ages[t])
        for status in (1, 0):
            for i, asset in enumerate(asset_grid):
                candidate_values = []
                candidate_consumption = []
                candidate_next_assets = []
                candidate_choices = ["retire"] if status == 1 else ["work", "retire"]
                for choice in candidate_choices:
                    next_status = 0 if choice == "work" else 1
                    y = branch_income(age, choice, p)
                    shift = branch_shift(age, choice, p)
                    cash = p.R * asset + y
                    feasible_next = asset_grid[asset_grid <= cash + 1.0e-10]
                    c = cash - feasible_next
                    vals = utility(c, p.gamma) + shift + p.beta * interpolate_value(
                        asset_grid,
                        value[t + 1, next_status],
                        feasible_next,
                    )
                    best = int(np.argmax(vals))
                    candidate_values.append(float(vals[best]))
                    candidate_consumption.append(float(c[best]))
                    candidate_next_assets.append(float(feasible_next[best]))

                best_choice = int(np.argmax(candidate_values))
                value[t, status, i] = candidate_values[best_choice]
                consumption[t, status, i] = candidate_consumption[best_choice]
                assets_next[t, status, i] = candidate_next_assets[best_choice]
                if status == 0:
                    retire_gap[t, i] = candidate_values[-1] - candidate_values[0]
                    retire_choice[t, i] = candidate_choices[best_choice] == "retire"

    result = {
        "value": value,
        "consumption": consumption,
        "assets_next": assets_next,
        "retire_choice": retire_choice,
        "retire_gap": retire_gap,
    }
    return result, time.perf_counter() - start


def retirement_boundary(asset_grid: np.ndarray, retire_choice: np.ndarray, p: RetirementPrimitives) -> pd.DataFrame:
    """Compute the lowest asset level at which retirement is chosen."""
    rows = []
    for t, age in enumerate(p.ages):
        idx = np.flatnonzero(retire_choice[t])
        if len(idx) == 0:
            threshold = np.nan
        else:
            threshold = float(asset_grid[idx[0]])
        rows.append({"age": int(age), "retirement_asset_threshold": threshold})
    return pd.DataFrame(rows)


def simulate_life_cycles(
    asset_grid: np.ndarray,
    solution: dict[str, np.ndarray],
    p: RetirementPrimitives,
) -> pd.DataFrame:
    """Simulate households using a smooth retirement rule around the value gap."""
    rng = np.random.default_rng(p.simulation_seed)
    assets = np.clip(
        rng.lognormal(
            mean=np.log(p.initial_asset_mean),
            sigma=p.initial_asset_sigma,
            size=p.simulation_agents,
        ),
        0.2,
        13.5,
    )
    retired = np.zeros(p.simulation_agents, dtype=bool)
    rows = []
    retire_age = np.full(p.simulation_agents, np.nan)

    for t, age in enumerate(p.ages):
        gap = np.interp(assets, asset_grid, solution["retire_gap"][t])
        retire_prob = 1.0 / (1.0 + np.exp(-np.clip(gap / p.taste_scale, -35.0, 35.0)))
        new_retire = (~retired) & (rng.uniform(size=p.simulation_agents) < retire_prob)
        retire_age[new_retire & np.isnan(retire_age)] = age
        retired = retired | new_retire

        c = np.empty_like(assets)
        a_next = np.empty_like(assets)
        active_work = ~retired
        if np.any(active_work):
            c[active_work] = np.interp(
                assets[active_work],
                asset_grid,
                solution["work_consumption"][t],
            )
            a_next[active_work] = np.interp(
                assets[active_work],
                asset_grid,
                solution["work_assets_next"][t],
            )
        if np.any(retired):
            c[retired] = np.interp(
                assets[retired],
                asset_grid,
                solution["consumption"][t, 1],
            )
            a_next[retired] = np.interp(
                assets[retired],
                asset_grid,
                solution["assets_next"][t, 1],
            )

        rows.append(
            pd.DataFrame(
                {
                    "agent": np.arange(p.simulation_agents),
                    "age": int(age),
                    "assets": assets,
                    "consumption": c,
                    "retired": retired,
                    "retire_prob": retire_prob,
                }
            )
        )
        assets = np.clip(a_next, p.asset_min, p.asset_max)

    panel = pd.concat(rows, ignore_index=True)
    panel["retire_age"] = np.repeat(retire_age, p.horizon)
    return panel


def compare_with_bruteforce(
    main_grid: np.ndarray,
    dcegm: dict[str, np.ndarray],
    audit_grid: np.ndarray,
    brute: dict[str, np.ndarray],
    p: RetirementPrimitives,
) -> pd.DataFrame:
    """Compare DC-EGM policies to brute-force VFI on the audit grid."""
    rows = []
    for target_age in [58, 62, 66, 70]:
        t = int(target_age - p.start_age)
        c_dcegm = np.interp(audit_grid, main_grid, dcegm["consumption"][t, 0])
        a_dcegm = np.interp(audit_grid, main_grid, dcegm["assets_next"][t, 0])
        retire_dcegm = np.interp(audit_grid, main_grid, dcegm["retire_choice"][t].astype(float)) >= 0.5
        c_brute = brute["consumption"][t, 0]
        a_brute = brute["assets_next"][t, 0]
        retire_brute = brute["retire_choice"][t]
        rows.append(
            {
                "age": target_age,
                "max_abs_consumption_gap": float(np.max(np.abs(c_dcegm - c_brute))),
                "max_abs_saving_gap": float(np.max(np.abs(a_dcegm - a_brute))),
                "retirement_policy_agreement": float(np.mean(retire_dcegm == retire_brute)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = RetirementPrimitives()
    asset_grid = np.linspace(p.asset_min, p.asset_max, p.n_assets)
    audit_grid = np.linspace(p.asset_min, p.asset_max, p.audit_assets)

    print("Solving DC-EGM retirement model...")
    dcegm, dcegm_time = solve_dcegm(asset_grid, p)
    print(f"  DC-EGM solve time: {dcegm_time:.3f} seconds")
    print("Solving brute-force audit model...")
    brute, brute_time = solve_bruteforce(audit_grid, p)
    print(f"  brute-force solve time: {brute_time:.3f} seconds")

    boundaries = retirement_boundary(asset_grid, dcegm["retire_choice"], p)
    panel = simulate_life_cycles(asset_grid, dcegm, p)
    comparison = compare_with_bruteforce(asset_grid, dcegm, audit_grid, brute, p)

    retire_summary = (
        panel.groupby("age", as_index=False)
        .agg(
            mean_assets=("assets", "mean"),
            mean_consumption=("consumption", "mean"),
            retired_share=("retired", "mean"),
            mean_retire_prob=("retire_prob", "mean"),
        )
    )
    observed_retire_age = panel.groupby("agent")["retire_age"].first().dropna()
    moments = pd.DataFrame(
        {
            "Moment": [
                "Mean simulated retirement age",
                "Share retired by age 62",
                "Share retired by age 67",
                "Mean assets at age 55",
                "Mean assets at age 70",
                "DC-EGM runtime seconds",
                "Brute-force runtime seconds",
                "Brute-force asset points",
                "DC-EGM asset points",
            ],
            "Value": [
                float(observed_retire_age.mean()),
                float(retire_summary.loc[retire_summary["age"] == 62, "retired_share"].iloc[0]),
                float(retire_summary.loc[retire_summary["age"] == 67, "retired_share"].iloc[0]),
                float(retire_summary.loc[retire_summary["age"] == 55, "mean_assets"].iloc[0]),
                float(retire_summary.loc[retire_summary["age"] == 70, "mean_assets"].iloc[0]),
                dcegm_time,
                brute_time,
                float(p.audit_assets),
                float(p.n_assets),
            ],
        }
    )

    setup_style()

    age_plot = 63
    t_plot = age_plot - p.start_age
    fig1, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(asset_grid, dcegm["work_consumption"][t_plot], label="Work branch")
    ax1.plot(asset_grid, dcegm["retire_consumption"][t_plot], label="Retire branch")
    ax1.plot(asset_grid, dcegm["consumption"][t_plot, 0], color="black", linestyle="--", label="Upper envelope")
    ax1.set_xlabel("Assets at start of age")
    ax1.set_ylabel("Consumption")
    ax1.set_title(f"Branch Consumption Policies at Age {age_plot}")
    ax1.legend()
    save_figure(fig1, "figures/branch-consumption.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
    valid_boundaries = boundaries.dropna()
    ax2.plot(valid_boundaries["age"], valid_boundaries["retirement_asset_threshold"], marker="o")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Lowest asset level choosing retirement")
    ax2.set_title("Retirement Boundary")
    save_figure(fig2, "figures/retirement-boundary.png", dpi=150)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax3a.plot(retire_summary["age"], retire_summary["retired_share"], marker="o")
    ax3a.set_xlabel("Age")
    ax3a.set_ylabel("Retired share")
    ax3a.set_ylim(0.0, 1.05)
    ax3a.set_title("Simulated Retirement")
    ax3b.plot(retire_summary["age"], retire_summary["mean_assets"], marker="o", label="Assets")
    ax3b.plot(retire_summary["age"], retire_summary["mean_consumption"], marker="s", label="Consumption")
    ax3b.set_xlabel("Age")
    ax3b.set_title("Mean Assets and Consumption")
    ax3b.legend()
    fig3.tight_layout()
    save_figure(fig3, "figures/simulated-life-cycles.png", dpi=150)

    fig4, ax4 = plt.subplots(figsize=(8.5, 4.8))
    t_audit = 62 - p.start_age
    ax4.plot(asset_grid, dcegm["retire_choice"][t_audit].astype(float), label="DC-EGM")
    ax4.step(audit_grid, brute["retire_choice"][t_audit].astype(float), where="mid", label="Brute force", alpha=0.8)
    ax4.set_xlabel("Assets at age 62")
    ax4.set_ylabel("Retire indicator")
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_title("Retirement Rule Audit at Age 62")
    ax4.legend()
    save_figure(fig4, "figures/bruteforce-audit.png", dpi=150)

    comparison_out = comparison.copy()
    comparison_out[["max_abs_consumption_gap", "max_abs_saving_gap", "retirement_policy_agreement"]] = (
        comparison_out[["max_abs_consumption_gap", "max_abs_saving_gap", "retirement_policy_agreement"]].round(4)
    )
    comparison_out = comparison_out.rename(
        columns={
            "age": "Age",
            "max_abs_consumption_gap": "Largest consumption-policy gap",
            "max_abs_saving_gap": "Largest next-asset-policy gap",
            "retirement_policy_agreement": "Retirement decision agreement",
        }
    )
    Path("tables").mkdir(parents=True, exist_ok=True)
    comparison_out.to_csv("tables/bruteforce-comparison.csv", index=False)

    moments_out = moments.copy()
    moments_out["Value"] = moments_out["Value"].round(4)
    moments_out.to_csv("tables/lifecycle-moments.csv", index=False)

    save_thumbnail("figures/branch-consumption.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
