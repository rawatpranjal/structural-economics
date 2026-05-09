#!/usr/bin/env python3
"""Dynamic games: finite-state Markov-perfect quality investment."""
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


def payoff_matrices(V: np.ndarray, q1: int, q2: int, beta: float, invest_cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the two firms' state-game payoff matrices."""
    pay1 = np.zeros((2, 2))
    pay2 = np.zeros((2, 2))
    for a1 in [0, 1]:
        for a2 in [0, 1]:
            pay1[a1, a2] = flow_profit(q1, q2) - invest_cost * a1 + beta * expected_value(V, q1, q2, a1, a2, 0)
            pay2[a1, a2] = flow_profit(q2, q1) - invest_cost * a2 + beta * expected_value(V, q1, q2, a1, a2, 1)
    return pay1, pay2


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
                pay1, pay2 = payoff_matrices(V, q1, q2, beta, invest_cost)
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


def compute_deviation_gains(
    V: np.ndarray,
    policy: np.ndarray,
    beta: float = 0.90,
    invest_cost: float = 2.2,
) -> np.ndarray:
    """Compute each firm's one-step gain from deviating at the reported policy."""
    q_max = policy.shape[0] - 1
    gains = np.zeros((q_max + 1, q_max + 1, 2))
    for q1 in range(q_max + 1):
        for q2 in range(q_max + 1):
            a1_star, a2_star = map(int, policy[q1, q2])
            pay1, pay2 = payoff_matrices(V, q1, q2, beta, invest_cost)
            gains[q1, q2, 0] = max(pay1[:, a2_star]) - pay1[a1_star, a2_star]
            gains[q1, q2, 1] = max(pay2[a1_star, :]) - pay2[a1_star, a2_star]
    return gains


def simulate(policy: np.ndarray, T: int = 45, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q_max = policy.shape[0] - 1
    q1, q2 = 1, 1
    rows = []
    for t in range(T):
        a1, a2 = policy[q1, q2]
        rows.append({
            "Period": t,
            "Firm 1 quality": q1,
            "Firm 2 quality": q2,
            "Firm 1 invest": a1,
            "Firm 2 invest": a2,
        })
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
    deviation_gains = compute_deviation_gains(V, policy)
    sim = simulate(policy)

    print("Dynamic games tutorial")
    print(f"Converged in {sol['iterations']} iterations with error {sol['error']:.2e}")
    print(f"Largest one-step deviation gain: {np.max(deviation_gains):.2e}")

    report = ModelReport(
        "Quality Investment in Dynamic Oligopoly",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Two firms sell differentiated products whose quality carries over. "
        "A firm can invest today to raise tomorrow's quality, while its rival "
        "moves too.\n\n"
        "The model is a two-firm Ericson-Pakes quality ladder. The state is the "
        "pair of quality levels. The policy maps each state into wait or invest.\n\n"
        "The policy needs a fixed point. Today's best response uses tomorrow's "
        "value, and tomorrow's value depends on today's equilibrium actions."
    )

    report.add_equations(r"""
Let the industry state be the quality pair
$\omega_t=(q_{1t},q_{2t})$, with $q_{it}\in\{0,\ldots,Q\}$.
Each firm chooses $a_{it}\in\{0,1\}$, where $a_{it}=1$ means invest.
Flow profit uses a logit-share reduced form:

$$\pi_i(q_i,q_j) = M\frac{\exp(\eta q_i)} {1+\exp(\eta q_i)+\exp(\eta q_j)} +\lambda q_i .$$

Investment raises the chance of moving one rung up the ladder. Waiting avoids
the investment cost but leaves a small depreciation risk:

$$\Pr(q_i'=\min\{q_i+1,Q\}\mid q_i,a_i=1)=0.62, \qquad \Pr(q_i'=\max\{q_i-1,0\}\mid q_i,a_i=0)=0.12 .$$

Given candidate continuation values $V_i$, each state contains a simultaneous
two-action game. Firm $i$'s payoff from action profile $(a_1,a_2)$ is

$$G_i(a_i,a_j;\omega,V) = \pi_i(\omega)-\kappa a_i +\beta\sum_{\omega'} P(\omega'\mid \omega,a_i,a_j)V_i(\omega').$$

A pure-strategy Markov-perfect equilibrium is a policy
$a^{*}(\omega)=(a_1^{*}(\omega),a_2^{*}(\omega))$ and values satisfying

$$G_i(a_i^{*},a_j^{*};\omega,V)\geq G_i(a_i,a_j^{*};\omega,V) \quad\text{for all }a_i\in\{0,1\},$$

with $V_i(\omega)=G_i(a_i^{*},a_j^{*};\omega,V)$ at every state.
""")

    report.add_model_setup(
        "| Primitive | Value | Role |\n"
        "|-----------|-------|------|\n"
        "| Firms | 2 | Symmetric oligopolists |\n"
        "| Quality ladder | $q_i=0,\ldots,4$ | Payoff-relevant industry state |\n"
        "| Actions | $a_i\\in\\{0,1\\}$ | Wait or invest |\n"
        "| Discount factor | $\\beta=0.90$ | Continuation-value weight |\n"
        "| Investment cost | $\\kappa=2.20$ | Current cost of attempting to improve quality |\n"
        "| Market size | $M=14$ | Scale of current profits |\n"
        "| Quality in demand | $\\eta=0.75$ | How quality shifts product share |\n"
        "| Direct quality payoff | $\\lambda=0.35$ | Extra payoff from own quality |\n"
        "| Equilibrium concept | Pure-strategy MPE | Nash equilibrium in each state game |"
    )

    report.add_solution_method(
        "The fixed point is over Markov strategies. Start with values for each "
        "firm at each quality pair. At one state, those values define a "
        "two-by-two game. Payoffs combine current profit, investment cost, and "
        "expected continuation value. Solve that state game, update values, and "
        "repeat.\n\n"
        "```text\n"
        "Inputs: quality cap Q, discount factor beta, investment cost kappa,\n"
        "        transition kernel P(q' | q, a), tolerance epsilon\n"
        "Initialize V_i^0(q_1,q_2)=0 for both firms and all quality pairs.\n"
        "For n = 0,1,2,...:\n"
        "  For each state omega=(q_1,q_2):\n"
        "    Build G_i^n(a_1,a_2; omega) from profit, cost, and continuation value.\n"
        "    Find pure Nash equilibria of the 2-by-2 state game.\n"
        "    Select the equilibrium with the largest joint payoff if there is a tie.\n"
        "    Set T_i V^n(omega) equal to the selected equilibrium payoff.\n"
        "  Update V_i^{n+1} = lambda T_i V^n + (1-lambda) V_i^n.\n"
        "  Stop when max_{i,omega} |T_i V^n(omega)-V_i^n(omega)| < epsilon.\n"
        "Output: MPE policy a_i^{*}(omega), values V_i(omega), and deviation gains.\n"
        "```\n\n"
        "After convergence, compute one-step deviation gains. At the reported "
        "policy, every gain should be zero."
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
        description="Firm 1 invests at every interior quality state. It waits at "
        "the top rung because the ladder cap binds.",
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
        description="The diagonal is symmetric. Off-diagonal states show the value "
        "of a quality lead. The lead affects current share and continuation value.",
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
        description="The simulation turns the policy into a stochastic industry "
        "path. Light vertical lines mark investment periods. Quality leadership "
        "persists, but depreciation and catch-up investment create turnover.",
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
            "Value advantage": f"{V[q1, q2, 0] - V[q1, q2, 1]:.2f}",
            "Max deviation gain": f"{np.max(deviation_gains[q1, q2]):.2e}",
        })
    report.add_table(
        "tables/policy-by-state.csv",
        "Selected state policies, values, and deviation checks",
        pd.DataFrame(rows),
        description="Symmetric states have symmetric values. States with a quality "
        "lead show a value advantage. Deviation gains are zero at the reported "
        "actions.",
    )

    report.add_takeaway(
        "The quality-ladder game makes investment rivalry a state-transition "
        "problem. The computed policy maps quality states into actions and "
        "continuation values. In this calibration, firms invest until the cap "
        "binds. Off-diagonal states show why a quality lead is valuable."
    )

    report.add_references([
        "Ericson, R., and Pakes, A. (1995). Markov-Perfect Industry Dynamics. "
        "*Review of Economic Studies*, 62(1), 53-82.",
        "Pakes, A., and McGuire, P. (1994). Computing Markov-Perfect Nash Equilibria. "
        "*RAND Journal of Economics*, 25(4), 555-589.",
        "Lecture 17 Slides 2023: Dynamic games and the Ericson-Pakes framework.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
