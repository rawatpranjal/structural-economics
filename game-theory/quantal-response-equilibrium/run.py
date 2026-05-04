#!/usr/bin/env python3
"""Quantal response equilibrium for a simple entry game.

Solves logit QRE by finding the root of the fixed-point residual. The example
is intentionally small so the equilibrium concept is visible in the code.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def softmax(values: np.ndarray, precision: float) -> np.ndarray:
    """Logit choice probabilities."""
    scaled = precision * values
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def logit_best_response(
    own_payoffs: np.ndarray,
    opponent_prob_enter: float,
    precision: float,
) -> float:
    """Probability of Enter under logit best response."""
    opponent_mix = np.array([opponent_prob_enter, 1.0 - opponent_prob_enter])
    expected_values = own_payoffs @ opponent_mix
    return float(softmax(expected_values, precision)[0])


def solve_symmetric_entry_qre(
    precision: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[float, int, float]:
    """Solve the symmetric entry-game QRE by bisection.

    The symmetric fixed point is p = QBR(p; lambda). The residual
    f(p) = p - QBR(p; lambda) is strictly increasing in this entry game, so
    bisection is robust even when naive fixed-point iteration cycles.
    """
    low = 0.0
    high = 1.0

    def response(prob_enter: float) -> float:
        enter_payoff = 2.0 - 3.0 * prob_enter
        stay_out_payoff = 0.0
        return float(softmax(np.array([enter_payoff, stay_out_payoff]), precision)[0])

    def residual(prob_enter: float) -> float:
        return prob_enter - response(prob_enter)

    mid = 0.5
    for it in range(1, max_iter + 1):
        mid = 0.5 * (low + high)
        mid_residual = residual(mid)
        if abs(mid_residual) < tol or high - low < tol:
            break
        if mid_residual < 0.0:
            low = mid
        else:
            high = mid

    return float(mid), it, abs(residual(mid))


def main() -> None:
    # Actions are Enter, Stay Out. If both enter, congestion makes entry costly.
    row_payoffs = np.array([[-1.0, 2.0], [0.0, 0.0]])
    col_payoffs = np.array([[-1.0, 0.0], [2.0, 0.0]])
    precisions = np.linspace(0.0, 8.0, 81)
    mixed_nash_enter_prob = 2.0 / 3.0

    qre_rows = []
    p_path = []
    for precision in precisions:
        p_entry, iterations, residual = solve_symmetric_entry_qre(float(precision))
        p_row = p_entry
        p_col = p_entry
        p_path.append((p_row, p_col))
        if precision in {0.0, 1.0, 2.0, 4.0, 8.0}:
            qre_rows.append({
                "Precision lambda": f"{precision:.1f}",
                "Row Pr(Enter)": f"{p_row:.4f}",
                "Column Pr(Enter)": f"{p_col:.4f}",
                "Iterations": iterations,
                "Residual": f"{residual:.2e}",
            })

    p_path_arr = np.array(p_path)

    setup_style()
    report = ModelReport(
        "Quantal Response Equilibrium",
        "Noisy best responses solved as a fixed point.",
    )

    report.add_overview(
        "Quantal response equilibrium relaxes the exact best-response assumption. Players are "
        "more likely to choose actions with higher expected payoffs, but they can still make "
        "mistakes. The logit precision parameter controls how sharply choice probabilities "
        "respond to payoff differences."
    )

    report.add_equations(r"""
For each action $a_i$, player $i$ assigns logit probability
$$
\sigma_i(a_i) =
\frac{\exp(\lambda E[u_i(a_i, a_{-i})])}
{\sum_{a'_i}\exp(\lambda E[u_i(a'_i, a_{-i})])}.
$$

A logit quantal response equilibrium is a fixed point:
$$
\sigma_i = QBR_i(\sigma_{-i}; \lambda)
\qquad \text{for each player } i.
$$

As $\lambda \to 0$, choices approach uniform randomization. As $\lambda$ rises,
choices put more weight on higher-payoff actions.
""")

    report.add_model_setup(
        "The example is a two-player entry game. Each player chooses Enter or Stay Out.\n\n"
        "| | Column Enter | Column Stay Out |\n"
        "|---|---:|---:|\n"
        "| **Row Enter** | -1, -1 | 2, 0 |\n"
        "| **Row Stay Out** | 0, 2 | 0, 0 |\n\n"
        "The exact mixed Nash equilibrium has each player entering with probability $2/3$."
    )

    report.add_solution_method(
        "For each precision value $\\lambda$, solve the symmetric fixed-point equation "
        "$p = QBR(p; \\lambda)$ by bisection on the residual $p - QBR(p; \\lambda)$. "
        "This keeps the implementation low-code while avoiding the cycling that naive "
        "iteration can produce at high precision."
    )

    fig, ax = plt.subplots()
    ax.plot(precisions, p_path_arr[:, 0], label="Row Pr(Enter)")
    ax.plot(precisions, p_path_arr[:, 1], linestyle="--", label="Column Pr(Enter)")
    ax.axhline(mixed_nash_enter_prob, color="black", linestyle=":", label="Mixed Nash: 2/3")
    ax.set_xlabel("Precision $\\lambda$")
    ax.set_ylabel("Entry probability")
    ax.set_title("Logit QRE Path")
    ax.legend()
    report.add_figure(
        "figures/qre-path.png",
        "QRE entry probabilities approach the mixed Nash benchmark",
        fig,
        description=(
            "At zero precision, players randomize 50-50. As precision rises, the logit fixed "
            "point moves toward the exact mixed Nash entry probability."
        ),
    )

    precision = 4.0
    p_entry, _, _ = solve_symmetric_entry_qre(precision=precision)
    p_row = p_entry
    p_col = p_entry
    opponent_probs = np.linspace(0, 1, 200)
    row_br = np.array([logit_best_response(row_payoffs, q, precision) for q in opponent_probs])

    fig2, ax2 = plt.subplots()
    ax2.plot(opponent_probs, row_br, label="Logit best response")
    ax2.plot(opponent_probs, opponent_probs, color="black", linestyle="--", linewidth=1, label="45-degree line")
    ax2.scatter(p_col, p_row, color="crimson", zorder=5, label="QRE fixed point")
    ax2.set_xlabel("Opponent Pr(Enter)")
    ax2.set_ylabel("Own Pr(Enter)")
    ax2.set_title("QRE as a Fixed Point")
    ax2.legend()
    report.add_figure(
        "figures/fixed-point-map.png",
        "Logit QRE is a fixed point of noisy best responses",
        fig2,
        description="The fixed point is where the noisy best-response curve crosses the 45-degree line.",
    )

    report.add_table(
        "tables/qre-summary.csv",
        "QRE Summary",
        pd.DataFrame(qre_rows),
    )

    final_p, _, final_residual = solve_symmetric_entry_qre(float(precisions[-1]))
    report.add_table(
        "tables/final-diagnostic.csv",
        "Final Fixed-Point Diagnostic",
        pd.DataFrame([{
            "Precision lambda": f"{precisions[-1]:.1f}",
            "Row Pr(Enter)": f"{final_p:.6f}",
            "Column Pr(Enter)": f"{final_p:.6f}",
            "Fixed-point residual": f"{final_residual:.2e}",
        }]),
    )

    report.add_takeaway(
        "QRE is useful when exact best response is too sharp or behavior is noisy. Computationally, "
        "it is just another fixed-point problem: probabilities must equal the logit best responses "
        "to the probabilities chosen by opponents. This makes it a low-code bridge between finite "
        "games and stochastic choice models."
    )

    report.add_references([
        "McKelvey, R. D. and Palfrey, T. R. (1995). Quantal Response Equilibria for Normal Form Games. *Games and Economic Behavior*, 10(1).",
        "Goeree, J. K., Holt, C. A., and Palfrey, T. R. (2016). *Quantal Response Equilibrium*. Princeton University Press.",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
