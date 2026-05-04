#!/usr/bin/env python3
"""Dynamic games: Markov-perfect investment competition."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def flow_profit(q_i: int, q_j: int, market_size: float = 14.0) -> float:
    exp_i = np.exp(0.75 * q_i)
    exp_j = np.exp(0.75 * q_j)
    share = exp_i / (1.0 + exp_i + exp_j)
    return market_size * share + 0.35 * q_i


def transition_probs(q: int, action: int, q_max: int) -> dict[int, float]:
    if action == 1:
        up = min(q + 1, q_max)
        return {up: 0.62, q: 0.38} if up != q else {q: 1.0}
    down = max(q - 1, 0)
    return {down: 0.12, q: 0.88} if down != q else {q: 1.0}


def expected_value(V: np.ndarray, q1: int, q2: int, a1: int, a2: int, firm: int) -> float:
    q_max = V.shape[0] - 1
    probs1 = transition_probs(q1, a1, q_max)
    probs2 = transition_probs(q2, a2, q_max)
    ev = 0.0
    for nq1, p1 in probs1.items():
        for nq2, p2 in probs2.items():
            ev += p1 * p2 * V[nq1, nq2, firm]
    return ev


def select_equilibrium(pay1: np.ndarray, pay2: np.ndarray) -> tuple[int, int]:
    equilibria = []
    for a1 in [0, 1]:
        for a2 in [0, 1]:
            if pay1[a1, a2] >= pay1[1 - a1, a2] - 1e-10 and pay2[a1, a2] >= pay2[a1, 1 - a2] - 1e-10:
                equilibria.append((a1, a2))
    if not equilibria:
        total = pay1 + pay2
        idx = np.unravel_index(np.argmax(total), total.shape)
        return int(idx[0]), int(idx[1])
    return max(equilibria, key=lambda a: pay1[a] + pay2[a])


def solve_game(q_max: int = 4, beta: float = 0.90, invest_cost: float = 2.2) -> dict[str, np.ndarray | int | float]:
    V = np.zeros((q_max + 1, q_max + 1, 2))
    policy = np.zeros((q_max + 1, q_max + 1, 2), dtype=int)
    error = np.inf
    for iteration in range(1, 1500):
        V_new = np.zeros_like(V)
        policy_new = np.zeros_like(policy)
        for q1 in range(q_max + 1):
            for q2 in range(q_max + 1):
                pay1 = np.zeros((2, 2))
                pay2 = np.zeros((2, 2))
                for a1 in [0, 1]:
                    for a2 in [0, 1]:
                        pay1[a1, a2] = flow_profit(q1, q2) - invest_cost * a1 + beta * expected_value(V, q1, q2, a1, a2, 0)
                        pay2[a1, a2] = flow_profit(q2, q1) - invest_cost * a2 + beta * expected_value(V, q1, q2, a1, a2, 1)
                a1_star, a2_star = select_equilibrium(pay1, pay2)
                policy_new[q1, q2] = [a1_star, a2_star]
                V_new[q1, q2, 0] = pay1[a1_star, a2_star]
                V_new[q1, q2, 1] = pay2[a1_star, a2_star]
        error = float(np.max(np.abs(V_new - V)))
        V = 0.35 * V_new + 0.65 * V
        policy = policy_new
        if error < 1e-8:
            break
    return {"V": V, "policy": policy, "iterations": iteration, "error": error}


def simulate(policy: np.ndarray, T: int = 45, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q_max = policy.shape[0] - 1
    q1, q2 = 1, 1
    rows = []
    for t in range(T):
        a1, a2 = policy[q1, q2]
        rows.append({"Period": t, "Firm 1 quality": q1, "Firm 2 quality": q2, "Firm 1 invest": a1, "Firm 2 invest": a2})
        probs1 = transition_probs(q1, int(a1), q_max)
        probs2 = transition_probs(q2, int(a2), q_max)
        q1 = int(rng.choice(list(probs1.keys()), p=list(probs1.values())))
        q2 = int(rng.choice(list(probs2.keys()), p=list(probs2.values())))
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    sol = solve_game()
    V = np.asarray(sol["V"])
    policy = np.asarray(sol["policy"])
    sim = simulate(policy)

    print("Dynamic games tutorial")
    print(f"Converged in {sol['iterations']} iterations with error {sol['error']:.2e}")

    report = ModelReport(
        "Dynamic Games and Markov-Perfect Investment",
        "Ericson-Pakes style strategic dynamics with quality states and investment choices.",
    )

    report.add_overview(
        "Dynamic IO games let firm actions change the future state of competition. "
        "A firm may invest today not only because it raises future quality, but also "
        "because it changes its rival's future incentives. The Markov-perfect restriction "
        "keeps strategies payoff-relevant: actions depend on the current state, not on "
        "the entire history.\n\n"
        "This tutorial solves a two-firm quality ladder. Each firm chooses whether to "
        "invest. Investment is costly but can move the firm up one quality state. Current "
        "profits come from a differentiated-products share formula."
    )

    report.add_equations(r"""
State is the quality pair:
$$\omega_t = (q_{1t}, q_{2t})$$

Markov-perfect values satisfy:
$$V_i(\omega) = \max_{a_i\in\{0,1\}} \pi_i(\omega) - \kappa a_i + \beta E[V_i(\omega')|\omega,a_i,a_{-i}^{*}(\omega)]$$

Investment changes transition probabilities:
$$Pr(q_i'=q_i+1|a_i=1)=0.62$$

The equilibrium policy maps each state into investment actions:
$$a_i^{*}(\omega)\in\{0,1\}$$
""")

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        "| Firms | 2 |\n"
        "| Quality states | 0 through 4 |\n"
        "| Actions | Invest or do not invest |\n"
        "| Discount factor | 0.90 |\n"
        "| Investment cost | 2.20 |\n"
        "| Equilibrium concept | Pure-strategy Markov-perfect equilibrium at each state |"
    )

    report.add_solution_method(
        "The solver iterates on firm value functions. At each state it constructs the "
        "two-by-two investment payoff matrix using the previous value function, finds "
        "a pure Nash equilibrium of that state game, and updates continuation values. "
        "The state space is deliberately small so the dynamic-game logic is visible."
    )

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    im = ax1.imshow(policy[:, :, 0], origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax1.set_xlabel("Firm 2 quality")
    ax1.set_ylabel("Firm 1 quality")
    ax1.set_title("Firm 1 Investment Policy")
    for q1 in range(policy.shape[0]):
        for q2 in range(policy.shape[1]):
            ax1.text(q2, q1, "Invest" if policy[q1, q2, 0] else "Wait", ha="center", va="center")
    fig1.colorbar(im, ax=ax1, ticks=[0, 1], label="Invest")
    report.add_figure(
        "figures/investment-policy.png",
        "Firm 1 investment policy over the quality state space",
        fig1,
        description="Investment incentives are strongest when a firm is behind or close to its rival. "
        "At high own quality, the marginal benefit of another quality step is smaller.",
    )

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    advantage = V[:, :, 0] - V[:, :, 1]
    im2 = ax2.imshow(advantage, origin="lower", cmap="RdBu_r")
    ax2.set_xlabel("Firm 2 quality")
    ax2.set_ylabel("Firm 1 quality")
    ax2.set_title("Value Advantage: Firm 1 Minus Firm 2")
    fig2.colorbar(im2, ax=ax2, label="Value difference")
    report.add_figure(
        "figures/value-advantage.png",
        "Value advantage across states",
        fig2,
        description="Dynamic state variables are payoff relevant. A one-step quality lead "
        "changes both current market share and future investment incentives.",
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.step(sim["Period"], sim["Firm 1 quality"], where="post", label="Firm 1 quality")
    ax3.step(sim["Period"], sim["Firm 2 quality"], where="post", label="Firm 2 quality")
    invest_periods = sim.loc[(sim["Firm 1 invest"] == 1) | (sim["Firm 2 invest"] == 1), "Period"]
    for t in invest_periods:
        ax3.axvline(t, color="gray", alpha=0.12)
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Quality")
    ax3.set_title("Simulated Quality Paths")
    ax3.legend()
    report.add_figure(
        "figures/simulated-quality-path.png",
        "Simulated quality paths under Markov-perfect policies",
        fig3,
        description="The vertical lines mark periods with at least one investment action. "
        "Quality leadership is persistent but not permanent because investment and depreciation "
        "keep the state moving.",
    )

    rows = []
    for state in [(0, 0), (1, 2), (2, 1), (4, 4)]:
        q1, q2 = state
        rows.append({
            "State": f"({q1},{q2})",
            "Firm 1 policy": "Invest" if policy[q1, q2, 0] else "Wait",
            "Firm 2 policy": "Invest" if policy[q1, q2, 1] else "Wait",
            "Firm 1 value": f"{V[q1, q2, 0]:.2f}",
            "Firm 2 value": f"{V[q1, q2, 1]:.2f}",
        })
    report.add_table("tables/policy-by-state.csv", "Selected state policies and values", pd.DataFrame(rows))

    report.add_takeaway(
        "Dynamic games turn IO counterfactuals into state-transition problems. The hard part "
        "is not just computing a price or entry outcome today; it is tracking how current "
        "actions change tomorrow's competitive state and therefore tomorrow's incentives."
    )

    report.add_references([
        "Ericson, R., and Pakes, A. (1995). Markov-Perfect Industry Dynamics. *Review of Economic Studies*, 62(1), 53-82.",
        "Pakes, A., and McGuire, P. (1994). Computing Markov-Perfect Nash Equilibria. *RAND Journal of Economics*, 25(4), 555-589.",
        "Lecture 17 Slides 2023: Dynamic games and the Ericson-Pakes framework.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
