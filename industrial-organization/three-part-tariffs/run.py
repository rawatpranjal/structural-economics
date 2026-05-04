#!/usr/bin/env python3
"""Three-part tariffs and forward-looking broadband usage."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


PLANS = {
    "Metered": {"fixed_fee": 16.0, "allowance": 25.0, "overage_price": 0.70, "speed": 80.0},
    "Three-part": {"fixed_fee": 46.0, "allowance": 85.0, "overage_price": 1.60, "speed": 200.0},
    "Unlimited": {"fixed_fee": 52.0, "allowance": 10_000.0, "overage_price": 0.0, "speed": 320.0},
}


def daily_utility(usage: np.ndarray, taste: float, satiation: float = 0.34) -> np.ndarray:
    return taste * np.log1p(usage) - 0.5 * satiation * usage**2


def solve_usage_dp(plan: dict[str, float], taste: float, T: int = 30, step: float = 0.5, max_usage: float = 180.0) -> dict[str, np.ndarray]:
    grid = np.arange(0.0, max_usage + step, step)
    actions = np.arange(0.0, 6.0 + step, step)
    V = np.zeros((T + 1, len(grid)))
    policy = np.zeros((T, len(grid)))
    allowance = plan["allowance"]
    overage_price = plan["overage_price"]

    for t in range(T - 1, -1, -1):
        for i, cumulative in enumerate(grid):
            best_value = -1e15
            best_action = 0.0
            for usage in actions:
                next_cumulative = min(cumulative + usage, grid[-1])
                j = int(round(next_cumulative / step))
                overage = max(0.0, cumulative + usage - allowance) - max(0.0, cumulative - allowance)
                value = daily_utility(np.array([usage]), taste)[0] - overage_price * overage + V[t + 1, j]
                if value > best_value:
                    best_value = value
                    best_action = usage
            V[t, i] = best_value
            policy[t, i] = best_action
    return {"grid": grid, "policy": policy, "value": V}


def simulate_policy(dp: dict[str, np.ndarray], plan: dict[str, float], taste: float, T: int = 30, step: float = 0.5) -> dict[str, float | np.ndarray]:
    grid = np.asarray(dp["grid"])
    policy = np.asarray(dp["policy"])
    cumulative = 0.0
    usage_path = []
    overage_payment = 0.0
    gross_utility = 0.0
    for t in range(T):
        idx = int(round(min(cumulative, grid[-1]) / step))
        usage = float(policy[t, idx])
        usage_path.append(usage)
        gross_utility += float(daily_utility(np.array([usage]), taste)[0])
        overage = max(0.0, cumulative + usage - plan["allowance"]) - max(0.0, cumulative - plan["allowance"])
        overage_payment += plan["overage_price"] * overage
        cumulative += usage
    speed_value = 2.6 * np.log(plan["speed"])
    revenue = plan["fixed_fee"] + overage_payment
    consumer_value = gross_utility + speed_value - revenue
    return {
        "usage_path": np.array(usage_path),
        "total_usage": cumulative,
        "overage_payment": overage_payment,
        "revenue": revenue,
        "consumer_value": consumer_value,
        "gross_utility": gross_utility + speed_value,
    }


def plan_choice_summary(tastes: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for taste, weight in zip(tastes, weights):
        best_plan = None
        best_outcome = None
        for name, plan in PLANS.items():
            dp = solve_usage_dp(plan, float(taste))
            outcome = simulate_policy(dp, plan, float(taste))
            if best_outcome is None or outcome["consumer_value"] > best_outcome["consumer_value"]:
                best_plan = name
                best_outcome = outcome
        assert best_plan is not None and best_outcome is not None
        row = {"Taste": taste, "Weight": weight, "Chosen plan": best_plan}
        row.update({k: v for k, v in best_outcome.items() if not isinstance(v, np.ndarray)})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    focal_taste = 4.4
    focal_dp = solve_usage_dp(PLANS["Three-part"], focal_taste)
    focal_outcome = simulate_policy(focal_dp, PLANS["Three-part"], focal_taste)

    tastes = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.6, 6.2])
    weights = np.array([0.10, 0.14, 0.18, 0.20, 0.17, 0.13, 0.08])
    choices = plan_choice_summary(tastes, weights)
    summary_rows = []
    for plan_name in PLANS:
        group = choices[choices["Chosen plan"] == plan_name]
        if group.empty:
            summary_rows.append({
                "Chosen plan": plan_name,
                "Share": 0.0,
                "Average usage": 0.0,
                "Average revenue": 0.0,
                "Average consumer value": 0.0,
            })
            continue
        summary_rows.append({
            "Chosen plan": plan_name,
            "Share": group["Weight"].sum(),
            "Average usage": np.average(group["total_usage"], weights=group["Weight"]),
            "Average revenue": np.average(group["revenue"], weights=group["Weight"]),
            "Average consumer value": np.average(group["consumer_value"], weights=group["Weight"]),
        })
    plan_summary = pd.DataFrame(summary_rows)

    print("Three-part tariffs tutorial")
    print(plan_summary.to_string(index=False))

    report = ModelReport(
        "Three-Part Tariffs and Forward-Looking Broadband Demand",
        "Usage allowances, overage prices, and dynamic consumption within a billing cycle.",
    )

    report.add_overview(
        "Usage-based broadband pricing is a three-part tariff: a fixed monthly fee, "
        "an included allowance, and an overage price for usage above the cap. Consumers "
        "are forward-looking within the billing cycle because using data today changes "
        "the remaining allowance tomorrow.\n\n"
        "The tutorial solves a finite-horizon dynamic program for daily usage. It then "
        "lets heterogeneous consumers choose among metered, three-part, and unlimited plans."
    )

    report.add_equations(r"""
Daily utility from usage $c_t$ is:
$$u(c_t;h) = h\log(1+c_t) - \frac{\psi}{2}c_t^2$$

Cumulative usage evolves as:
$$C_t = C_{t-1} + c_t$$

The dynamic value under plan $k$ is:
$$V_{kt}(C_{t-1}) = \max_{c_t} u(c_t;h) - p_k^{over}\Delta O_t + V_{k,t+1}(C_t)$$

where $\Delta O_t$ is the incremental overage usage created by today's consumption.
""")

    report.add_model_setup(
        "| Plan | Fixed fee | Allowance | Overage price | Speed |\n"
        "|------|-----------|-----------|---------------|-------|\n"
        f"| Metered | {PLANS['Metered']['fixed_fee']:.0f} | {PLANS['Metered']['allowance']:.0f} GB | {PLANS['Metered']['overage_price']:.2f} | {PLANS['Metered']['speed']:.0f} Mbps |\n"
        f"| Three-part | {PLANS['Three-part']['fixed_fee']:.0f} | {PLANS['Three-part']['allowance']:.0f} GB | {PLANS['Three-part']['overage_price']:.2f} | {PLANS['Three-part']['speed']:.0f} Mbps |\n"
        f"| Unlimited | {PLANS['Unlimited']['fixed_fee']:.0f} | uncapped | {PLANS['Unlimited']['overage_price']:.2f} | {PLANS['Unlimited']['speed']:.0f} Mbps |"
    )

    report.add_solution_method(
        "Backward induction solves the monthly usage problem for each plan and consumer "
        "type. The state is cumulative usage so far in the billing cycle. The policy "
        "function shows how much to consume today as a function of the remaining allowance "
        "and days left."
    )

    policy = np.asarray(focal_dp["policy"])
    grid = np.asarray(focal_dp["grid"])
    remaining = PLANS["Three-part"]["allowance"] - grid
    keep = np.where((remaining >= -25) & (remaining <= 95))[0][::-1]
    remaining_kept = remaining[keep]
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    im = ax1.imshow(
        policy[:, keep].T,
        origin="lower",
        aspect="auto",
        extent=[1, 30, remaining_kept.min(), remaining_kept.max()],
        cmap="viridis",
    )
    ax1.axhline(0, color="white", linestyle="--", linewidth=1)
    ax1.set_xlabel("Day of billing cycle")
    ax1.set_ylabel("Remaining allowance")
    ax1.set_title("Forward-Looking Usage Policy Under a Three-Part Tariff")
    fig1.colorbar(im, ax=ax1, label="Daily usage")
    report.add_figure(
        "figures/usage-policy.png",
        "Daily usage policy by day and remaining allowance",
        fig1,
        description="Near the allowance, consumers conserve usage early in the cycle because "
        "today's usage raises the chance of overage payments later. The constraint relaxes near "
        "the end of the month.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    days = np.arange(1, 31)
    cumulative = np.cumsum(np.asarray(focal_outcome["usage_path"]))
    ax2.plot(days, cumulative, label="Cumulative usage", color="#4C78A8")
    ax2.axhline(PLANS["Three-part"]["allowance"], color="#E45756", linestyle="--", label="Allowance")
    ax2.bar(days, focal_outcome["usage_path"], alpha=0.25, label="Daily usage", color="#F58518")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("GB")
    ax2.set_title("Simulated Billing-Cycle Usage")
    ax2.legend()
    report.add_figure(
        "figures/billing-cycle-usage.png",
        "Cumulative and daily usage under the three-part plan",
        fig2,
        description="The consumer manages usage around the allowance. Overage risk creates "
        "intertemporal substitution even though daily tastes are constant.",
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plan_summary))
    ax3.bar(x - 0.25, plan_summary["Share"], 0.25, label="Plan share", color="#4C78A8")
    ax3.bar(x, plan_summary["Average usage"] / 100, 0.25, label="Usage / 100", color="#54A24B")
    ax3.bar(x + 0.25, plan_summary["Average revenue"] / 100, 0.25, label="Revenue / 100", color="#F58518")
    ax3.set_xticks(x)
    ax3.set_xticklabels(plan_summary["Chosen plan"])
    ax3.set_title("Plan Choice, Usage, and Revenue")
    ax3.legend()
    report.add_figure(
        "figures/plan-comparison.png",
        "Plan shares, average usage, and average revenue",
        fig3,
        description="Heterogeneous consumers sort across contracts. Low-usage types prefer "
        "metered plans, middle types value the allowance, and high-usage types value unlimited access.",
    )

    table = plan_summary.rename(columns={"Chosen plan": "Plan"}).copy()
    for col in ["Share", "Average usage", "Average revenue", "Average consumer value"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    report.add_table("tables/plan-summary.csv", "Plan-choice summary across consumer types", table)

    report.add_takeaway(
        "The allowance makes demand dynamic. A consumer far below the cap treats usage as "
        "almost free, while a consumer near the cap faces the shadow value of preserving "
        "allowance for later. Three-part tariffs therefore affect both plan choice and "
        "within-month usage timing."
    )

    report.add_references([
        "Nevo, A., Turner, J., and Williams, J. (2016). Usage-Based Pricing and Demand for Residential Broadband. *Econometrica*, 84(2), 411-443.",
        "Lecture 18 Slides 2023: Three-part tariffs and forward-looking broadband demand.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
