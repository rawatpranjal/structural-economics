#!/usr/bin/env python3
"""Dynamic games: finite-state Markov-perfect quality investment."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


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


def select_equilibrium(
    pay1: np.ndarray, pay2: np.ndarray, fallback_log: list[bool] | None = None,
) -> tuple[int, int]:
    """Return an action profile for the 2-by-2 state game.

    Prefers a pure-strategy Nash equilibrium and, when several exist,
    selects the one with the largest joint payoff. While value iteration
    is still moving, an intermediate continuation-value guess can produce
    a state game with no pure NE; in that case the joint-payoff-maximising
    profile is used as a fallback. ``fallback_log`` records whether the
    fallback fired, so the caller can check that it does not fire at the
    converged equilibrium.
    """
    equilibria = []
    for a1 in [0, 1]:
        for a2 in [0, 1]:
            if pay1[a1, a2] >= pay1[1 - a1, a2] - 1e-10 and pay2[a1, a2] >= pay2[a1, 1 - a2] - 1e-10:
                equilibria.append((a1, a2))
    if not equilibria:
        if fallback_log is not None:
            fallback_log.append(True)
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

    # Verify the converged output is a genuine pure-strategy MPE: re-solve
    # every state game at the converged values and confirm the no-pure-NE
    # fallback never fires. It can fire on intermediate value guesses while
    # iteration is still moving, but it must not fire at the fixed point.
    converged_fallback: list[bool] = []
    for q1 in range(q_max + 1):
        for q2 in range(q_max + 1):
            pay1, pay2 = payoff_matrices(V, q1, q2, beta, invest_cost)
            select_equilibrium(pay1, pay2, fallback_log=converged_fallback)
    assert not converged_fallback, (
        "no pure-strategy Nash equilibrium at the converged values"
    )
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

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    im = ax1.imshow(policy[:, :, 0], origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax1.set_xlabel("Firm 2 quality")
    ax1.set_ylabel("Firm 1 quality")
    ax1.set_title("Firm 1 Investment Policy")
    for q1 in range(policy.shape[0]):
        for q2 in range(policy.shape[1]):
            ax1.text(q2, q1, "Invest" if policy[q1, q2, 0] else "Wait", ha="center", va="center")
    fig1.colorbar(im, ax=ax1, ticks=[0, 1], label="Invest")
    save_figure(fig1, "figures/investment-policy.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    advantage = V[:, :, 0] - V[:, :, 1]
    im2 = ax2.imshow(advantage, origin="lower", cmap="RdBu_r")
    ax2.set_xlabel("Firm 2 quality")
    ax2.set_ylabel("Firm 1 quality")
    ax2.set_title("Value Advantage: Firm 1 Minus Firm 2")
    fig2.colorbar(im2, ax=ax2, label="Value difference")
    save_figure(fig2, "figures/value-advantage.png", dpi=150)

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
    save_figure(fig3, "figures/simulated-quality-path.png", dpi=150)

    rows = []
    for state in [(0, 0), (1, 2), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]:
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
    Path("tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("tables/policy-by-state.csv", index=False)

    save_thumbnail("figures/investment-policy.png", "figures/thumb.png")
    print(f"Figures and tables written.")


if __name__ == "__main__":
    main()
