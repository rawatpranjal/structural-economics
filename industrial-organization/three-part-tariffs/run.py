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
BILLING_DAYS = 30
SATIATION = 0.34
DAILY_ACTION_MAX = 6.0


def daily_utility(usage: np.ndarray, taste: float, satiation: float = SATIATION) -> np.ndarray:
    return taste * np.log1p(usage) - 0.5 * satiation * usage**2


def solve_usage_dp(
    plan: dict[str, float],
    taste: float,
    T: int = BILLING_DAYS,
    step: float = 0.5,
    max_usage: float = 180.0,
) -> dict[str, np.ndarray]:
    grid = np.arange(0.0, max_usage + step, step)
    actions = np.arange(0.0, DAILY_ACTION_MAX + step, step)
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


def simulate_policy(
    dp: dict[str, np.ndarray],
    plan: dict[str, float],
    taste: float,
    T: int = BILLING_DAYS,
    step: float = 0.5,
) -> dict[str, float | np.ndarray]:
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


def plan_choice_summary(tastes: np.ndarray, weights: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    value_rows = []
    for taste, weight in zip(tastes, weights):
        best_plan = None
        best_outcome = None
        for name, plan in PLANS.items():
            dp = solve_usage_dp(plan, float(taste))
            outcome = simulate_policy(dp, plan, float(taste))
            value_rows.append({
                "Taste": taste,
                "Plan": name,
                "Consumer value": outcome["consumer_value"],
            })
            if best_outcome is None or outcome["consumer_value"] > best_outcome["consumer_value"]:
                best_plan = name
                best_outcome = outcome
        assert best_plan is not None and best_outcome is not None
        row = {"Taste": taste, "Weight": weight, "Chosen plan": best_plan}
        row.update({k: v for k, v in best_outcome.items() if not isinstance(v, np.ndarray)})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(value_rows)


def main() -> None:
    setup_style()
    focal_taste = 4.4
    focal_dp = solve_usage_dp(PLANS["Three-part"], focal_taste)
    focal_outcome = simulate_policy(focal_dp, PLANS["Three-part"], focal_taste)
    fine_step = 0.25
    fine_dp = solve_usage_dp(PLANS["Three-part"], focal_taste, step=fine_step)
    fine_outcome = simulate_policy(fine_dp, PLANS["Three-part"], focal_taste, step=fine_step)
    focal_cumulative = np.cumsum(np.asarray(focal_outcome["usage_path"]))
    fine_cumulative = np.cumsum(np.asarray(fine_outcome["usage_path"]))
    max_cumulative_gap = float(np.max(np.abs(focal_cumulative - fine_cumulative)))
    total_usage_gap = float(abs(focal_outcome["total_usage"] - fine_outcome["total_usage"]))
    consumer_value_gap = float(abs(focal_outcome["consumer_value"] - fine_outcome["consumer_value"]))
    consumer_value_relative_gap = consumer_value_gap / abs(float(focal_outcome["consumer_value"]))

    tastes = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.6, 6.2])
    weights = np.array([0.10, 0.14, 0.18, 0.20, 0.17, 0.13, 0.08])
    choices, plan_values = plan_choice_summary(tastes, weights)
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
        "How data caps make monthly broadband demand a dynamic choice problem.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Residential broadband contracts often combine three instruments: a fixed fee, "
        "an included data allowance, and a per-GB overage price. The allowance is not just "
        "a nonlinear price schedule. It creates a state variable inside the month. A GB "
        "used on day 3 lowers the remaining allowance on day 4, so the relevant marginal "
        "price includes the option value of keeping data for later.\n\n"
        "The tutorial keeps the demand side deliberately small. A consumer solves a "
        "finite-horizon usage problem within the billing cycle, then heterogeneous types "
        "choose among a low-fee metered plan, a middle three-part plan, and an unlimited "
        "plan. The dynamic-choice logic is close to the continuation-value reasoning in "
        "[bus replacement](../dynamic-discrete-choice/), while the fixed-fee role connects "
        "to the two-part-tariff discussion in "
        "[vertical relationships](../vertical-relationships/)."
    )

    report.add_equations(r"""
Let $t=1,\ldots,T$ index days in a billing cycle and let $C_{t-1}$ be cumulative
usage before day $t$. Under plan $k$, the consumer pays a fixed fee $F_k$, has
allowance $A_k$, and pays overage price $q_k$ per GB above the allowance.

Daily usage is $c_t \geq 0$. Type $h$ has gross daily utility

$$u(c_t;h) = h\log(1+c_t) - \frac{\psi}{2}c_t^2,$$

with $\psi>0$. Cumulative usage follows

$$C_t=C_{t-1}+c_t,\qquad C_0=0.$$

The incremental overage quantity created on day $t$ is

$$\Delta O_k(C_{t-1},c_t)
=\max\{0,C_{t-1}+c_t-A_k\}-\max\{0,C_{t-1}-A_k\}.$$

For a given plan and type, the within-cycle value function is

$$V_{k,t}(C_{t-1};h)=
\max_{c_t\in[0,\bar c]}
\left[
u(c_t;h)-q_k\Delta O_k(C_{t-1},c_t)
+V_{k,t+1}(C_t;h)
\right],$$

with terminal value $V_{k,T+1}(\cdot;h)=0$. The policy
$g_{k,t}(C_{t-1};h)$ gives daily usage.

Plan choice adds the fixed fee and the value of speed $B(s_k)$:

$$W_i(k)=V_{k,1}(0;h_i)+B(s_k)-F_k,\qquad
d_i=\arg\max_k W_i(k).$$
""")

    report.add_model_setup(
        f"The calibration uses a {BILLING_DAYS}-day billing cycle, "
        f"$\\psi={SATIATION}$, daily choices $c_t\\in[0,{DAILY_ACTION_MAX:.0f}]$, "
        "and the speed shifter $B(s_k)=2.6\\log(s_k)$. Consumer heterogeneity is a "
        "small discrete type distribution; the weights are used only for plan shares and "
        "average outcomes.\n\n"
        "| Plan | Fixed fee | Allowance | Overage price | Speed |\n"
        "|------|-----------|-----------|---------------|-------|\n"
        f"| Metered | {PLANS['Metered']['fixed_fee']:.0f} | {PLANS['Metered']['allowance']:.0f} GB | {PLANS['Metered']['overage_price']:.2f} | {PLANS['Metered']['speed']:.0f} Mbps |\n"
        f"| Three-part | {PLANS['Three-part']['fixed_fee']:.0f} | {PLANS['Three-part']['allowance']:.0f} GB | {PLANS['Three-part']['overage_price']:.2f} | {PLANS['Three-part']['speed']:.0f} Mbps |\n"
        f"| Unlimited | {PLANS['Unlimited']['fixed_fee']:.0f} | uncapped | {PLANS['Unlimited']['overage_price']:.2f} | {PLANS['Unlimited']['speed']:.0f} Mbps |\n\n"
        "| Taste type $h_i$ | Weight |\n"
        "|------------------|--------|\n"
        + "\n".join(f"| {taste:.1f} | {weight:.2f} |" for taste, weight in zip(tastes, weights))
    )

    report.add_solution_method(
        "For each type-plan pair, backward induction solves the finite-horizon usage "
        "problem on a grid for cumulative monthly usage. The fixed fee is excluded from "
        "the daily Bellman recursion because it is sunk after the plan is chosen; it enters "
        "only when comparing plans. The overage price enters inside the recursion because "
        "today's usage can move the consumer closer to the cap or past it.\n\n"
        "```text\n"
        "Algorithm: finite-horizon usage and plan choice\n"
        "Input: plans (F_k, A_k, q_k, s_k), type distribution (h_i, omega_i), usage grid C\n"
        "Output: daily policies g_{k,t}(C; h_i), chosen plans d_i, plan shares\n"
        "for each type h_i and plan k:\n"
        "    set V_{k,T+1}(C; h_i) = 0 for every cumulative-usage state C\n"
        "    for t = T, T-1, ..., 1:\n"
        "        for each state C on the cumulative-usage grid:\n"
        "            for each feasible daily usage c in [0, c_bar]:\n"
        "                C_next = C + c\n"
        "                overage_increment = max(0, C_next - A_k) - max(0, C - A_k)\n"
        "                payoff = u(c; h_i) - q_k * overage_increment + V_{k,t+1}(C_next; h_i)\n"
        "            choose c that maximizes payoff and record g_{k,t}(C; h_i)\n"
        "    compute W_i(k) = V_{k,1}(0; h_i) + B(s_k) - F_k\n"
        "choose d_i = argmax_k W_i(k), then aggregate shares with weights omega_i\n"
        "```\n\n"
        "The focal policy uses a 0.5 GB grid. For the billing-cycle path, the same model is "
        "also solved on a 0.25 GB grid as a numerical benchmark."
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
        description=(
            "The policy surface shows the shadow price of the remaining allowance. Early in "
            "the month, a consumer near the cap cuts usage because each GB raises the chance "
            "of paying overage charges later. Near the end of the cycle, the same remaining "
            "allowance has less option value, so the policy relaxes."
        ),
    )

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    days = np.arange(1, BILLING_DAYS + 1)
    ax2a.plot(days, focal_cumulative, label="0.5 GB grid", color="#4C78A8")
    ax2a.plot(days, fine_cumulative, label="0.25 GB grid", color="#E45756", linestyle="--")
    ax2a.axhline(PLANS["Three-part"]["allowance"], color="gray", linestyle=":", label="Allowance")
    ax2a.set_xlabel("Day")
    ax2a.set_ylabel("Cumulative GB")
    ax2a.set_title("Cumulative Usage")
    ax2a.legend()
    ax2b.plot(days, focal_outcome["usage_path"], label="0.5 GB grid", color="#4C78A8")
    ax2b.plot(days, fine_outcome["usage_path"], label="0.25 GB grid", color="#E45756", linestyle="--")
    ax2b.set_xlabel("Day")
    ax2b.set_ylabel("Daily GB")
    ax2b.set_title("Daily Usage")
    ax2b.legend()
    fig2.tight_layout()
    report.add_figure(
        "figures/billing-cycle-usage.png",
        "Billing-cycle usage under the baseline grid and a finer-grid benchmark",
        fig2,
        description=(
            "The simulated path stays close to the allowance without treating it as a hard "
            "constraint. The finer-grid benchmark matches total usage within "
            f"**{total_usage_gap:.2f} GB**, while the largest cumulative-path gap is "
            f"**{max_cumulative_gap:.2f} GB**. The dynamics come from the nonlinear contract, "
            "not from time-varying daily tastes."
        ),
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for plan_name, group in plan_values.groupby("Plan"):
        ax3.plot(group["Taste"], group["Consumer value"], marker="o", linewidth=2, label=plan_name)
    chosen = choices.merge(plan_values, left_on=["Taste", "Chosen plan"], right_on=["Taste", "Plan"])
    ax3.scatter(
        chosen["Taste"],
        chosen["Consumer value"],
        s=80,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        label="Chosen plan",
        zorder=5,
    )
    ax3.set_xlabel("Taste for usage $h$")
    ax3.set_ylabel("Net consumer value")
    ax3.set_title("Plan Value by Consumer Type")
    ax3.legend()
    report.add_figure(
        "figures/plan-comparison.png",
        "Net consumer value by type and plan",
        fig3,
        description=(
            "Plan choice is a sorting problem. Low-usage types choose the low fixed fee, "
            "middle types value the allowance, and high-usage types pay for unlimited access. "
            "The circled points are the contracts selected by the discrete type distribution."
        ),
    )

    table = plan_summary.rename(columns={"Chosen plan": "Plan"}).copy()
    for col in ["Share", "Average usage", "Average revenue", "Average consumer value"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    report.add_table("tables/plan-summary.csv", "Plan-choice summary across consumer types", table)

    report.add_takeaway(
        "A three-part tariff changes demand before the cap is actually hit. The allowance "
        "has a shadow value because it can be spent later in the billing cycle, so a "
        "forward-looking consumer reacts to expected overage risk rather than only to the "
        "current marginal price. Across types, the same contract menu sorts consumers by "
        "usage intensity: low types avoid the fixed fee, middle types buy the allowance, "
        "and high types choose unlimited access. The numerical benchmark check suggests "
        f"that the focal path is not driven by the coarse grid: net consumer value differs "
        f"from the finer-grid solution by **{consumer_value_gap:.3f}**, about "
        f"**{consumer_value_relative_gap:.1%}** of the baseline value."
    )

    report.add_references([
        "Nevo, A., Turner, J., and Williams, J. (2016). Usage-Based Pricing and Demand for Residential Broadband. *Econometrica*, 84(2), 411-443.",
        "Lecture 18 Slides 2023: Three-part tariffs and forward-looking broadband demand.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
