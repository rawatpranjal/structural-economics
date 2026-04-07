#!/usr/bin/env python3
"""Static Games: Nash Equilibrium in Normal-Form Games.

Computes pure and mixed strategy Nash equilibria for classic 2x2 games
(Prisoner's Dilemma, Matching Pennies, Battle of the Sexes, Cournot),
demonstrates best-response functions, and applies to a tennis serve game.

Reference: Osborne and Rubinstein (1994), "A Course in Game Theory."
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


# =========================================================================
# Game Theory Functions
# =========================================================================

def find_pure_nash(A: np.ndarray, B: np.ndarray) -> list[tuple[int, int]]:
    """Find pure strategy Nash equilibria in a bimatrix game (A, B)."""
    m, n = A.shape
    equilibria = []
    for i in range(m):
        for j in range(n):
            # Check if i is best response to j
            if A[i, j] == np.max(A[:, j]):
                # Check if j is best response to i
                if B[i, j] == np.max(B[i, :]):
                    equilibria.append((i, j))
    return equilibria


def find_mixed_nash_2x2(A: np.ndarray, B: np.ndarray) -> tuple[float, float] | None:
    """Find mixed strategy Nash equilibrium in a 2x2 game.
    Returns (p, q) where p = P(row plays action 0), q = P(col plays action 0).
    """
    # Player 2 mixes to make Player 1 indifferent:
    # p1 payoff from row 0: A[0,0]*q + A[0,1]*(1-q)
    # p1 payoff from row 1: A[1,0]*q + A[1,1]*(1-q)
    # Set equal and solve for q
    denom_q = (A[0, 0] - A[1, 0]) - (A[0, 1] - A[1, 1])
    if abs(denom_q) < 1e-10:
        return None
    q = (A[1, 1] - A[0, 1]) / denom_q

    # Player 1 mixes to make Player 2 indifferent:
    denom_p = (B[0, 0] - B[0, 1]) - (B[1, 0] - B[1, 1])
    if abs(denom_p) < 1e-10:
        return None
    p = (B[1, 1] - B[1, 0]) / denom_p

    if 0 <= p <= 1 and 0 <= q <= 1:
        return (p, q)
    return None


def best_response_1(A: np.ndarray, q: float) -> int:
    """Player 1's best response to Player 2 mixing with prob q on action 0."""
    payoffs = A @ np.array([q, 1 - q])
    return int(np.argmax(payoffs))


def cournot_best_response(q_other: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Cournot best response: q_i = (a - c - b*q_j) / (2b)."""
    return np.maximum(0, (a - c - b * q_other) / (2 * b))


def main():
    # =========================================================================
    # Classic 2x2 Games
    # =========================================================================
    games = {
        "Prisoner's Dilemma": {
            "A": np.array([[-1, -3], [0, -2]]),
            "B": np.array([[-1, 0], [-3, -2]]),
            "actions": (["Cooperate", "Defect"], ["Cooperate", "Defect"]),
        },
        "Matching Pennies": {
            "A": np.array([[1, -1], [-1, 1]]),
            "B": np.array([[-1, 1], [1, -1]]),
            "actions": (["Heads", "Tails"], ["Heads", "Tails"]),
        },
        "Battle of the Sexes": {
            "A": np.array([[3, 0], [0, 2]]),
            "B": np.array([[2, 0], [0, 3]]),
            "actions": (["Opera", "Football"], ["Opera", "Football"]),
        },
    }

    results = []
    for name, game in games.items():
        A, B = game["A"], game["B"]
        pure = find_pure_nash(A, B)
        mixed = find_mixed_nash_2x2(A, B)
        acts = game["actions"]
        pure_str = ", ".join(f"({acts[0][i]}, {acts[1][j]})" for i, j in pure) if pure else "None"
        mixed_str = f"p={mixed[0]:.2f}, q={mixed[1]:.2f}" if mixed else "None"
        results.append({
            "Game": name,
            "Pure NE": pure_str,
            "Mixed NE": mixed_str,
        })

    # =========================================================================
    # Cournot Duopoly
    # =========================================================================
    a, b, c = 10, 1, 2  # Demand: P = a - b*Q, Cost: c*q_i
    q_range = np.linspace(0, 8, 200)
    br1 = cournot_best_response(q_range, a, b, c)
    br2 = cournot_best_response(q_range, a, b, c)
    q_nash = (a - c) / (3 * b)  # Symmetric NE
    p_nash = a - 2 * b * q_nash
    q_monopoly = (a - c) / (2 * b)
    q_competitive = (a - c) / b

    # =========================================================================
    # Tennis Serve Game (Mixed Strategy Application)
    # =========================================================================
    # Server chooses Left/Right, Receiver chooses Left/Right
    # Payoffs: probability server wins the point
    A_tennis = np.array([[0.3, 0.8], [0.9, 0.2]])  # Server payoffs
    B_tennis = 1 - A_tennis  # Zero-sum: receiver payoffs
    mixed_tennis = find_mixed_nash_2x2(A_tennis, B_tennis)

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Static Games and Nash Equilibrium",
        "Pure and mixed strategy Nash equilibria in normal-form games with applications to IO.",
    )

    report.add_overview(
        "Static (simultaneous-move) games are the foundation of strategic analysis in "
        "industrial organization. Players choose strategies simultaneously, and the outcome "
        "depends on the combination of all players' choices. A Nash equilibrium is a strategy "
        "profile where no player can improve their payoff by unilaterally deviating.\n\n"
        "This module covers: classic 2x2 games (Prisoner's Dilemma, Matching Pennies, "
        "Battle of the Sexes), Cournot duopoly with best-response functions, and a tennis "
        "serve application of mixed strategy equilibrium."
    )

    report.add_equations(r"""
**Nash Equilibrium:** A strategy profile $(s_1^*, s_2^*)$ is a NE if:
$$u_1(s_1^*, s_2^*) \geq u_1(s_1, s_2^*) \quad \forall s_1, \qquad u_2(s_1^*, s_2^*) \geq u_2(s_1^*, s_2) \quad \forall s_2$$

**Mixed Strategy NE (2x2):** Player 2 mixes with probability $q$ on action 0 to make Player 1 indifferent:
$$q = \frac{A_{11} - A_{01}}{(A_{00} - A_{10}) - (A_{01} - A_{11})}$$

**Cournot Best Response:** Given linear demand $P = a - bQ$ and constant MC $c$:
$$q_i^*(q_j) = \frac{a - c - bq_j}{2b}, \qquad q^{NE} = \frac{a-c}{3b}$$
""")

    report.add_model_setup(
        "**Cournot parameters:** $a = 10$ (demand intercept), $b = 1$ (slope), $c = 2$ (marginal cost)\n\n"
        "**Tennis serve game:** Server wins with probabilities:\n\n"
        "| | Receiver Left | Receiver Right |\n"
        "|---|---|---|\n"
        "| **Server Left** | 0.30 | 0.80 |\n"
        "| **Server Right** | 0.90 | 0.20 |"
    )

    report.add_solution_method(
        "**Pure NE:** Enumerate all strategy profiles and check mutual best response.\n\n"
        "**Mixed NE (2x2):** Solve the indifference conditions analytically.\n\n"
        "**Cournot:** Find the intersection of best-response functions.\n\n"
        f"**Tennis NE:** Server plays Left with probability {mixed_tennis[0]:.2f}, "
        f"Receiver plays Left with probability {mixed_tennis[1]:.2f}."
    )

    # --- Figure 1: Cournot Best Responses ---
    fig1, ax1 = plt.subplots()
    ax1.plot(q_range, br1, "b-", linewidth=2, label="Firm 1 BR: $q_1^*(q_2)$")
    ax1.plot(br2, q_range, "r-", linewidth=2, label="Firm 2 BR: $q_2^*(q_1)$")
    ax1.plot(q_nash, q_nash, "ko", markersize=10, label=f"Nash: $q^*={q_nash:.2f}$")
    ax1.plot(q_monopoly / 2, q_monopoly / 2, "gs", markersize=8, label=f"Joint monopoly: $q={q_monopoly/2:.2f}$")
    ax1.set_xlabel("$q_2$ (Firm 2 output)")
    ax1.set_ylabel("$q_1$ (Firm 1 output)")
    ax1.set_title("Cournot Duopoly: Best Response Functions")
    ax1.legend()
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 8)
    report.add_figure("figures/cournot-best-response.png", "Cournot best responses intersect at Nash equilibrium", fig1)

    # --- Figure 2: Best response correspondence for Matching Pennies ---
    q_grid = np.linspace(0, 1, 200)
    A_mp = games["Matching Pennies"]["A"]
    br1_mp = np.array([A_mp @ np.array([q, 1 - q]) for q in q_grid])
    p_br = np.where(br1_mp[:, 0] > br1_mp[:, 1], 1.0, np.where(br1_mp[:, 0] < br1_mp[:, 1], 0.0, 0.5))

    fig2, ax2 = plt.subplots()
    # Player 1's best response
    ax2.step(q_grid, p_br, "b-", linewidth=2, where="mid", label="P1 best response $p^*(q)$")
    # Player 2's best response (symmetric in matching pennies)
    ax2.step(p_br, q_grid, "r-", linewidth=2, where="mid", label="P2 best response $q^*(p)$")
    ax2.plot(0.5, 0.5, "ko", markersize=10, label="Mixed NE: (0.5, 0.5)")
    ax2.set_xlabel("$q$ (Player 2: prob of Heads)")
    ax2.set_ylabel("$p$ (Player 1: prob of Heads)")
    ax2.set_title("Matching Pennies: Best Response Correspondences")
    ax2.legend()
    report.add_figure("figures/matching-pennies-br.png", "Matching Pennies: only equilibrium is mixed (0.5, 0.5)", fig2)

    # --- Figure 3: Cournot output and welfare ---
    fig3, ax3 = plt.subplots()
    q_total = np.linspace(0, a / b, 200)
    price = a - b * q_total
    cs = 0.5 * b * q_total ** 2
    ps = (price - c) * q_total
    ax3.plot(q_total, cs, "b-", linewidth=2, label="Consumer Surplus")
    ax3.plot(q_total, ps, "r-", linewidth=2, label="Producer Surplus")
    ax3.plot(q_total, cs + ps, "k--", linewidth=1.5, label="Total Welfare")
    ax3.axvline(2 * q_nash, color="steelblue", linestyle=":", alpha=0.7, label=f"Cournot: Q={2*q_nash:.1f}")
    ax3.axvline(q_monopoly, color="coral", linestyle=":", alpha=0.7, label=f"Monopoly: Q={q_monopoly:.1f}")
    ax3.axvline(q_competitive, color="green", linestyle=":", alpha=0.7, label=f"Competitive: Q={q_competitive:.1f}")
    ax3.set_xlabel("Total Output $Q$")
    ax3.set_ylabel("Surplus")
    ax3.set_title("Welfare Analysis: Monopoly vs Cournot vs Competition")
    ax3.legend(fontsize=8)
    report.add_figure("figures/welfare-analysis.png", "Cournot output lies between monopoly and competitive levels", fig3)

    # --- Tables ---
    df_games = pd.DataFrame(results)
    report.add_table("tables/classic-games.csv", "Nash Equilibria in Classic 2x2 Games", df_games)

    cournot_table = pd.DataFrame({
        "Market Structure": ["Monopoly", "Cournot Duopoly", "Perfect Competition"],
        "Total Output Q": [f"{q_monopoly:.2f}", f"{2*q_nash:.2f}", f"{q_competitive:.2f}"],
        "Price": [f"{a - b*q_monopoly:.2f}", f"{p_nash:.2f}", f"{c:.2f}"],
        "Profit per firm": [
            f"{(a - b*q_monopoly - c)*q_monopoly:.2f}",
            f"{(p_nash - c)*q_nash:.2f}",
            "0.00",
        ],
    })
    report.add_table("tables/cournot-comparison.csv", "Cournot vs Monopoly vs Competition", cournot_table)

    report.add_takeaway(
        "Static games provide the micro-foundations for industrial organization:\n\n"
        "**Key insights:**\n"
        "- **Prisoner's Dilemma**: Both firms defect (compete aggressively) even though "
        "cooperation would be mutually beneficial — this is why cartels are unstable.\n"
        "- **Mixed strategies**: In games with no pure NE (like Matching Pennies or tennis serves), "
        "players randomize to keep opponents indifferent. The mixing probabilities depend on "
        "the OTHER player's payoffs, not your own.\n"
        "- **Cournot**: Duopoly output lies between monopoly and competition. As the number of "
        f"firms increases, output approaches the competitive level Q={q_competitive:.0f}.\n"
        "- **Welfare**: Cournot creates deadweight loss relative to perfect competition, "
        "but less than monopoly. This is the quantitative basis for antitrust policy."
    )

    report.add_references([
        "Nash, J. (1950). \"Equilibrium Points in N-Person Games.\" *Proceedings of the National Academy of Sciences*, 36(1).",
        "Osborne, M. and Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.",
        "Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 5.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
