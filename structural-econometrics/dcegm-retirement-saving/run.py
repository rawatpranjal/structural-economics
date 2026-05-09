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
from lib.output import ModelReport
from lib.plotting import setup_style


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
    report = ModelReport(
        "Retirement and Saving by Discrete-Continuous EGM",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "An older worker chooses whether to keep working or retire. The same household "
        "also chooses how much to save. Retirement is discrete and absorbing. Saving is "
        "continuous.\n\n"
        "The economic object is the retirement boundary: at each age, which asset levels "
        "make the household leave work? The saving policy matters because assets insure "
        "retirement consumption.\n\n"
        "A plain grid search treats every current asset and every next asset as a nested "
        "maximization. DC-EGM avoids that inner search. It solves the continuous saving "
        "problem separately for work and retirement, then keeps the upper envelope of the "
        "choice-specific value functions."
    )

    report.add_equations(
        r"""
At age $t$, the household enters with assets $a_t$ and retirement status
$m_t \in \lbrace 0,1 \rbrace$. Status $m_t=0$ means still active, and $m_t=1$ means already
retired. An active household can choose work or retire:

$$
d_t \in D(m_t), \qquad
D(0)=\lbrace \mathrm{work},\mathrm{retire} \rbrace, \qquad
D(1)=\lbrace \mathrm{retire} \rbrace.
$$

The next retirement status is absorbing:

$$
m'(\mathrm{work})=0, \qquad m'(\mathrm{retire})=1.
$$

Preferences are CRRA, and the terminal value is a bequest value:

$$
u(c)=\frac{c^{1-\gamma}-1}{1-\gamma}, \qquad
u'(c)=c^{-\gamma}, \qquad
V_T^m(a)=\omega_B u(a+\bar b).
$$

Let calendar age be $\alpha_t=55+t$. The branch income and nonconsumption
utility terms are

$$
\begin{aligned}
y_t(\mathrm{work}) &=
1.42-0.012(\alpha_t-55)
\quad -0.006\max \lbrace \alpha_t-62,0 \rbrace^2,\\
y_t(\mathrm{retire}) &= \bar y^R,\\
\psi_t(\mathrm{work}) &=
{}-\left[0.16+0.024(\alpha_t-55)
\quad +0.010\max \lbrace \alpha_t-62,0 \rbrace^2\right],\\
\psi_t(\mathrm{retire}) &= \chi_R.
\end{aligned}
$$

The budget constraint is

$$
c_t + a_{t+1} = R a_t + y_t(d_t),
\qquad a_{t+1} \geq \underline a .
$$

Resources are split between current consumption and next assets. Throughout the branch problems below, $a^{+}$ denotes the same next-period assets as $a_{t+1}$ in the budget constraint, written without a time subscript to mark it as the free variable of the branch maximization.

For any branch $d$, define the branch value

$$
V_t^d(a) =
\max_{a^{+} \geq \underline a}
\left[
\underbrace{u(Ra+y_t(d)-a^{+})}_{\text{utility from consumption}} +
\underbrace{\psi_t(d)}_{\text{work cost or retirement amenity}} +
\underbrace{\beta V_{t+1}^{m'(d)}(a^{+})}_{\text{continuation value under next status}}
\right].
$$

This branch Bellman equation solves work and retirement as separate
continuous-saving problems.

The branch objects are

$$
\begin{aligned}
c_t^d(a,a^{+}) &= R a + y_t(d) - a^{+}, \\
\widetilde V_t^d(a,a^{+}) &=
u(c_t^d(a,a^{+}))+\psi_t(d)+\beta V_{t+1}^{m'(d)}(a^{+}).
\end{aligned}
$$

The branch maximization is over feasible next assets with positive
consumption, so infeasible choices are discarded.

The active and retired value functions are then

$$
V_t^1(a) =
V_t^{\mathrm{retire}}(a),
\qquad
V_t^0(a)=
\max \lbrace V_t^{\mathrm{work}}(a), V_t^{\mathrm{retire}}(a) \rbrace.
$$

The upper envelope chooses retirement where the retirement branch value exceeds
the work branch value.

This upper envelope is the central DC-EGM object. It preserves the discrete
retirement kink instead of forcing the value function to be globally concave.

On a fixed branch, the continuous saving problem has the Euler equation

$$
u'(c_t^d(a^{+})) =
\beta R
\frac{\partial V_{t+1}^{m'(d)}(a^{+})}{\partial a^{+}}.
$$

Write $a_i^{+}$ for a candidate next-period asset point on the exogenous grid.
Write $\mu_{t+1}^{m'(d)}(a_i^{+})$ for the next-period marginal value evaluated
at that candidate asset.

$$
\mu_{t+1}^{m'(d)}(a_i^{+})
=
\frac{\partial V_{t+1}^{m'(d)}(a_i^{+})}{\partial a^{+}}.
$$

EGM works backward from next assets: choose a grid point for tomorrow's assets,
compute the marginal value of arriving there, then invert marginal utility to
recover today's consumption.

$$
c_{t,i}^d =
(\beta R \mu_{t+1}^{m'(d)}(a_i^{+}))^{-1/\gamma}.
$$

The endogenous current asset attached to that next-asset point is

$$
a_{t,i}^{\mathrm{endo},d} =
\frac{c_{t,i}^d + a_i^{+} - y_t(d)}{R}.
$$

Each branch produces its own endogenous grid and value curve:

Here $a^{+} = a_i^{+}$ is fixed at each grid point, so $\widetilde V_t^d(a_{t,i}^{\mathrm{endo},d})$ is shorthand for $\widetilde V_t^d(a_{t,i}^{\mathrm{endo},d}, a_i^{+})$.

$$
\widetilde V_t^d(a_{t,i}^{\mathrm{endo},d}) =
u(c_{t,i}^d)+\psi_t(d)+\beta V_{t+1}^{m'(d)}(a_i^{+}).
$$

After sorting and dropping repeated endogenous assets, DC-EGM interpolates the
branch curve back to the common current-asset grid:

$$
g_t^d(a)=\mathrm{interp}\left(a;\,
a_{t,i}^{\mathrm{endo},d}, a_i^{+}\right),
\qquad
V_t^d(a)=\mathrm{interp}\left(a;\,
a_{t,i}^{\mathrm{endo},d}, \widetilde V_t^d(a_{t,i}^{\mathrm{endo},d})\right).
$$

For current assets below the first endogenous grid point, the borrowing
constraint binds:

$$
g_t^d(a)=\underline a,\qquad
c_t^d(a)=R a+y_t(d)-\underline a.
$$

In the constraint-binding region, $c_t^d(a)$ abbreviates $c_t^d(a,\underline a)$ with $a^{+}=\underline a$; elsewhere, $c_t^d(a)$ abbreviates $c_t^d(a,g_t^d(a))$ with $a^{+}$ at the optimal next asset.

The final active policies copy the winning branch:

$$
\begin{aligned}
d_t^{\ast}(a) &=
\arg\max_{d \in \lbrace \mathrm{work},\mathrm{retire} \rbrace} V_t^d(a),\\
g_t^0(a) &= g_t^{d_t^{\ast}(a)}(a),\\
c_t^0(a) &= c_t^{d_t^{\ast}(a)}(a).
\end{aligned}
$$
"""
    )

    report.add_model_setup(
        f"| Symbol | Calibration | Meaning |\n"
        f"|---|---:|---|\n"
        f"| $t$ | ages {p.start_age}-{p.end_age} | Finite-horizon retirement window |\n"
        f"| $a_t$ | grid on [{p.asset_min:.1f}, {p.asset_max:.1f}] | Assets at the start of age $t$ |\n"
        f"| $m_t$ | $0$ active, $1$ retired | Absorbing retirement status |\n"
        f"| $d_t$ | $\\mathrm{{work}}$ or $\\mathrm{{retire}}$ | Discrete labor-supply choice |\n"
        f"| $c_t$ | residual from budget | Consumption after choosing next assets |\n"
        f"| $a_i^{{+}}$ | {p.n_assets} points | Exogenous next-asset grid used by DC-EGM |\n"
        f"| $a^{{\\mathrm{{endo}},d}}_{{t,i}}$ | branch-specific | Current asset implied by Euler inversion on branch $d$ |\n"
        f"| $\\beta$ | {p.beta:.2f} | Discount factor |\n"
        f"| $R=1+r$ | {p.R:.2f} | Gross asset return |\n"
        f"| $\\gamma$ | {p.gamma:.1f} | CRRA curvature |\n"
        f"| $y_t(\\mathrm{{retire}})$ | {p.pension:.2f} | Pension income after retirement |\n"
        f"| $\\psi_t(\\mathrm{{retire}})$ | {p.retire_amenity:.2f} | Retirement amenity relative to work cost |\n"
        f"| $\\omega_B$ | {p.bequest_weight:.2f} | Terminal bequest weight |\n"
        f"| $\\bar b$ | {p.bequest_floor:.1f} | Bequest utility floor |\n"
        f"| $\\underline a$ | {p.asset_min:.1f} | Borrowing limit on next assets |\n"
        f"| Brute-force audit grid | {p.audit_assets} assets | Smaller benchmark grid for exhaustive search |\n"
        f"| Synthetic panel | {p.simulation_agents:,} households | Simulated with initial assets centered at {p.initial_asset_mean:.1f} |"
    )

    report.add_solution_method(
        r"""
The continuous decision is solved on a next-asset grid. The discrete choice is
handled after each branch has produced its own value function.

For each branch $d$, DC-EGM constructs points

$$
(a_{t,i}^{\mathrm{endo},d}, c_{t,i}^d, a_i^{+}, \widetilde V_t^d(a_{t,i}^{\mathrm{endo},d})).
$$

Interpolation converts those branch-specific points into functions on the
common current-asset grid. The active policy is then

$$
d_t^{\ast}(a)=
\mathrm{work} \quad \text{if } V_t^{\mathrm{work}}(a) \geq V_t^{\mathrm{retire}}(a),
\quad \text{and } \mathrm{retire} \text{ otherwise}.
$$

The selected consumption and saving policies are copied from the winning branch.

```text
Algorithm: DC-EGM for retirement and saving
Input:
    current asset grid A = {a_j}_{j=1}^J
    next asset grid A^+ = {a_i^+}_{i=1}^J
    ages t = 0,...,T-1
    primitives beta, R, gamma, y_t(d), psi_t(d), borrowing limit a_min

Initialize:
    for every asset a in A:
        V_T^0(a) = V_T^1(a) = omega_B * u(a + b_bar)

Subroutine SOLVE_BRANCH(t, d, next_status):
    y = y_t(d)
    psi = psi_t(d)
    compute mu_i = d V_{t+1}^{next_status}(a_i^+) / d a^+
        at every next-asset grid point a_i^+
    clip mu_i to a small positive value if a numerical derivative is nonpositive

    for each grid point a_i^+ in A^+:
        c_i = (beta * R * mu_i)^(-1 / gamma)
        a_i_endo = (c_i + a_i^+ - y) / R
        V_i_endo = u(c_i) + psi + beta * V_{t+1}^{next_status}(a_i^+)

    sort rows by a_i_endo
    repair monotonicity by replacing a_i_endo with its running maximum
    drop repeated a_i_endo values created by the monotonicity repair

    interpolate a_i^+ on a_i_endo to get the branch saving rule g_t^d(a)
    interpolate V_i_endo on a_i_endo to get the branch value V_t^d(a)

    for current assets a below the first endogenous point:
        set g_t^d(a) = a_min
        set c_t^d(a) = R * a + y - a_min
        set V_t^d(a) = u(c_t^d(a)) + psi
                         + beta * V_{t+1}^{next_status}(a_min)

    for all other current assets a:
        set c_t^d(a) = R * a + y - g_t^d(a)

    clip g_t^d(a) to the feasible asset grid
    return V_t^d(a), c_t^d(a), g_t^d(a)

Backward recursion:
for t = T-1, T-2, ..., 0:
    # already retired: only the retirement branch is feasible
    V_retired, c_retired, g_retired = SOLVE_BRANCH(t, retire, 1)
    store V_t^1(a) = V_retired(a), c_t^1(a) = c_retired(a),
          g_t^1(a) = g_retired(a)

    # active household: solve both feasible branches before the discrete choice
    V_work, c_work, g_work = SOLVE_BRANCH(t, work, 0)
    V_retire, c_retire, g_retire = SOLVE_BRANCH(t, retire, 1)

    for each current asset a in A:
        retirement_gap(a) = V_retire(a) - V_work(a)
        if retirement_gap(a) >= 0:
            choose retire
            V_t^0(a) = V_retire(a)
            c_t^0(a) = c_retire(a)
            g_t^0(a) = g_retire(a)
        else:
            choose work
            V_t^0(a) = V_work(a)
            c_t^0(a) = c_work(a)
            g_t^0(a) = g_work(a)

Simulation after solving:
    draw initial assets for each household
    for each age:
        interpolate the active retirement gap at the household asset
        convert the gap into a smooth retirement probability
        once retired, keep status retired forever
        interpolate the selected saving and consumption policies
        record age, status, assets, next assets, and consumption
```

The brute-force audit solves the same model on a smaller asset grid by checking
every feasible next asset at every current asset. It is slower and coarser, but
it provides a direct benchmark for the branch policies and the retirement
boundary.
"""
    )

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
    report.add_figure(
        "figures/branch-consumption.png",
        "Work and retirement consumption branches",
        fig1,
        description=(
            "The work and retirement branches solve ordinary continuous saving problems. "
            "The selected active policy follows the branch with the larger value. The "
            "switch point is where the discrete choice creates a kink."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
    valid_boundaries = boundaries.dropna()
    ax2.plot(valid_boundaries["age"], valid_boundaries["retirement_asset_threshold"], marker="o")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Lowest asset level choosing retirement")
    ax2.set_title("Retirement Boundary")
    report.add_figure(
        "figures/retirement-boundary.png",
        "Asset threshold for retirement by age",
        fig2,
        description=(
            "The threshold falls with age as work becomes less attractive and the horizon "
            "for earning labor income shrinks. At high ages, even lower-asset households "
            "prefer the retired branch."
        ),
    )

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
    report.add_figure(
        "figures/simulated-life-cycles.png",
        "Simulated retirement, assets, and consumption",
        fig3,
        description=(
            "The simulated panel translates the policy functions into life-cycle moments. "
            "Retirement rises gradually because households start with different assets "
            "and the simulation smooths the deterministic boundary with small taste shocks."
        ),
    )

    fig4, ax4 = plt.subplots(figsize=(8.5, 4.8))
    t_audit = 62 - p.start_age
    ax4.plot(asset_grid, dcegm["retire_choice"][t_audit].astype(float), label="DC-EGM")
    ax4.step(audit_grid, brute["retire_choice"][t_audit].astype(float), where="mid", label="Brute force", alpha=0.8)
    ax4.set_xlabel("Assets at age 62")
    ax4.set_ylabel("Retire indicator")
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_title("Retirement Rule Audit at Age 62")
    ax4.legend()
    report.add_figure(
        "figures/bruteforce-audit.png",
        "Retirement rule from DC-EGM and brute-force VFI",
        fig4,
        description=(
            "The brute-force rule uses a smaller grid and searches over all feasible "
            "next assets. The comparison checks whether the upper envelope chooses the "
            "same retirement region."
        ),
    )

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
    report.add_table(
        "tables/bruteforce-comparison.csv",
        "DC-EGM versus brute-force audit",
        comparison_out,
        description=(
            "The policy gaps are measured after interpolating the DC-EGM policy onto "
            "the smaller brute-force grid. Agreement is the share of audit-grid assets "
            "with the same retire/work decision."
        ),
    )

    moments_out = moments.copy()
    moments_out["Value"] = moments_out["Value"].round(4)
    report.add_table(
        "tables/lifecycle-moments.csv",
        "Simulation and runtime moments",
        moments_out,
        description=(
            "The runtime comparison is deliberately uneven: DC-EGM uses the larger main "
            "grid, while brute force uses the smaller audit grid. The point is the order "
            "of the computational bottleneck."
        ),
    )

    report.add_takeaway(
        "DC-EGM is useful when a structural labor model combines a discrete margin with "
        "continuous saving. Each branch remains an Euler-equation problem, so EGM avoids "
        "the inner root search or grid search. The discrete retirement option then enters "
        "through the upper envelope. That envelope is the economic policy boundary and "
        "the numerical source of the kink."
    )

    report.add_references(
        [
            "[Iskhakov, F., Jorgensen, T. H., Rust, J., and Schjerning, B. (2017). The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with or without Taste Shocks. *Quantitative Economics*, 8(2), 317-365.](https://doi.org/10.3982/QE643)",
            "[Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.](https://doi.org/10.1016/j.econlet.2005.09.013)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
