#!/usr/bin/env python3
"""Quantal response equilibrium for a simple entry game.

The script follows the symmetric logit-QRE branch in a two-player entry game
and compares it with the exact symmetric mixed Nash benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def entry_payoff_gap(opponent_prob_enter: float | np.ndarray) -> float | np.ndarray:
    """Expected payoff from Enter minus Stay Out in the entry game."""
    return 2.0 - 3.0 * opponent_prob_enter


def logit_entry_response(
    opponent_prob_enter: float | np.ndarray,
    precision: float,
) -> float | np.ndarray:
    """Probability of Enter under the logit best response."""
    payoff_gap = entry_payoff_gap(opponent_prob_enter)
    return 1.0 / (1.0 + np.exp(-precision * payoff_gap))


def mixed_nash_entry_probability() -> float:
    """Symmetric mixed Nash entry probability for the exact entry game."""
    return 2.0 / 3.0


def solve_symmetric_entry_qre(
    precision: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[float, int, float]:
    """Solve the symmetric entry-game logit QRE by bisection.

    The symmetric fixed point is p = QBR(p; lambda). In this entry game,
    f(p) = p - QBR(p; lambda) is strictly increasing, so bisection is reliable
    even when direct fixed-point iteration would be poorly behaved.
    """
    low = 0.0
    high = 1.0

    def residual(prob_enter: float) -> float:
        return prob_enter - float(logit_entry_response(prob_enter, precision))

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
    precisions = np.linspace(0.0, 32.0, 129)
    summary_precisions = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    focal_precision = 4.0
    mixed_nash_enter_prob = mixed_nash_entry_probability()

    p_path: list[float] = []
    for precision in precisions:
        p_entry, _, _ = solve_symmetric_entry_qre(float(precision))
        p_path.append(p_entry)

    qre_rows = []
    for precision in summary_precisions:
        p_entry, _, residual = solve_symmetric_entry_qre(precision)
        qre_rows.append({
            "Precision lambda": f"{precision:.1f}",
            "QRE Pr(Enter)": f"{p_entry:.4f}",
            "Mixed Nash Pr(Enter)": f"{mixed_nash_enter_prob:.4f}",
            "Gap to Nash": f"{p_entry - mixed_nash_enter_prob:+.4f}",
            "Residual": f"{residual:.2e}",
        })

    p_path_arr = np.array(p_path)

    setup_style()
    report = ModelReport(
        "Market Entry with Quantal Response Equilibrium",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Two firms decide whether to enter a small market. Entry pays when only one firm "
        "enters. Joint entry makes both firms lose money.\n\n"
        "Quantal response equilibrium keeps payoff comparison but allows mistakes. More "
        "profitable actions receive higher probability. The precision parameter controls "
        "how sharply firms respond to payoff gaps.\n\n"
        "The unknown is a symmetric entry probability. It must equal the logit response to "
        "itself. That condition is a fixed point."
    )

    report.add_equations(r"""
Each player chooses $E$ (Enter) or $O$ (Stay Out). Let $p_i$ be player $i$'s
entry probability. If the rival enters with probability $q$, the payoff gap is:

$$
\Delta(q)
= \mathbb{E}[u_i(E,a_{-i})]-\mathbb{E}[u_i(O,a_{-i})]
= 2(1-q)-q
= 2-3q.
$$

The exact symmetric mixed Nash equilibrium sets $\Delta(q)=0$:

$$
p^{N} = \frac{2}{3}.
$$

Logit QRE smooths the exact best response:

$$
\begin{aligned}
QBR(q;\lambda)
&= \frac{\exp(\lambda \Delta(q))}{1+\exp(\lambda \Delta(q))} \\
&= [1+\exp(-\lambda(2-3q))]^{-1}.
\end{aligned}
$$

A symmetric logit-QRE is a fixed point:

$$
p = QBR(p;\lambda).
$$

At $\lambda=0$, both actions receive probability one half. As $\lambda$ rises,
$p(\lambda)$ moves toward the mixed Nash probability $p^{N}=2/3$.
""")

    report.add_model_setup(
        "These payoffs create excess entry pressure when the rival is unlikely to enter.\n\n"
        "| | Column Enter | Column Stay Out |\n"
        "|---|---:|---:|\n"
        "| **Row Enter** | -1, -1 | 2, 0 |\n"
        "| **Row Stay Out** | 0, 2 | 0, 0 |\n\n"
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        f"| Symmetric mixed Nash $p^N$ | {mixed_nash_enter_prob:.4f} | Exact benchmark for symmetric entry |\n"
        f"| Precision grid | {precisions[0]:.0f} to {precisions[-1]:.0f} | Strength of payoff sensitivity |\n"
        f"| Focal fixed-point plot | $\\lambda={focal_precision:.1f}$ | One logit response map |"
    )

    report.add_solution_method(
        "The symmetric QRE reduces to a one-dimensional root search. For candidate "
        "probability $p$, define $G_{\\lambda}(p)=p-QBR(p;\\lambda)$. A fixed point sets "
        "this residual to zero. Bisection is enough because $G_{\\lambda}$ rises on "
        "$[0,1]$ and changes sign across the bracket.\n\n"
        "```text\n"
        "Algorithm: symmetric logit-QRE path in the entry game\n"
        "Inputs: precision grid Lambda, payoff gap Delta(p)=2-3p, tolerance epsilon\n"
        "Outputs: QRE entry probabilities p(lambda), residuals, gaps to p^N\n\n"
        "1. Compute the exact symmetric mixed Nash benchmark p^N from Delta(p^N)=0.\n"
        "2. For each lambda in Lambda, define QBR(p;lambda) = [1+exp(-lambda Delta(p))]^{-1}.\n"
        "3. Set the initial bracket [low, high] = [0, 1].\n"
        "4. Bisect the bracket on G_lambda(p)=p-QBR(p;lambda).\n"
        "5. Stop when |G_lambda(p)| or the bracket width is below epsilon.\n"
        "6. Report p(lambda), |G_lambda(p(lambda))|, and p(lambda)-p^N.\n"
        "```"
    )

    fig, ax = plt.subplots()
    ax.plot(precisions, p_path_arr, linewidth=2.3, label="Symmetric logit QRE")
    ax.axhline(
        mixed_nash_enter_prob,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label="Exact mixed Nash: 2/3",
    )
    ax.set_xlabel("Precision $\\lambda$")
    ax.set_ylabel("Entry probability")
    ax.set_title("Symmetric Entry Probability Along the Logit-QRE Branch")
    ax.set_ylim(0.48, 0.69)
    ax.legend()
    report.add_results(
        "At zero precision, firms enter with probability one half. As precision rises, "
        "entry probability moves toward two thirds. The dotted line is the mixed Nash "
        "benchmark."
    )
    report.add_figure(
        "figures/qre-path.png",
        "Symmetric logit-QRE entry probability and exact mixed Nash benchmark",
        fig,
    )

    p_entry, _, _ = solve_symmetric_entry_qre(precision=focal_precision)
    opponent_probs = np.linspace(0, 1, 200)
    row_br = logit_entry_response(opponent_probs, focal_precision)

    fig2, ax2 = plt.subplots()
    ax2.plot(opponent_probs, row_br, linewidth=2.3, label="Logit best response")
    ax2.plot(
        opponent_probs,
        opponent_probs,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="45-degree line",
    )
    ax2.axvline(
        mixed_nash_enter_prob,
        color="0.35",
        linestyle=":",
        linewidth=1.8,
        label="Mixed Nash benchmark",
    )
    ax2.scatter(p_entry, p_entry, color="crimson", s=60, zorder=5, label="QRE fixed point")
    ax2.set_xlabel("Opponent Pr(Enter)")
    ax2.set_ylabel("Own Pr(Enter)")
    ax2.set_title(f"Noisy Best Response at $\\lambda={focal_precision:.1f}$")
    ax2.legend()
    report.add_results(
        f"At $\\lambda={focal_precision:.1f}$, the response curve slopes down. A higher "
        "rival entry probability lowers the payoff from entry. The QRE is the crossing "
        "with the 45-degree line."
    )
    report.add_figure(
        "figures/fixed-point-map.png",
        "Noisy best-response map and symmetric QRE fixed point",
        fig2,
    )

    report.add_table(
        "tables/qre-summary.csv",
        "QRE Path Summary",
        pd.DataFrame(qre_rows),
        description=(
            "The residual is numerical error. The gap to Nash is finite-precision distance "
            "from exact mixed Nash."
        ),
    )

    report.add_takeaway(
        "QRE keeps mutual consistency but softens exact best response. In this entry game, "
        "higher precision moves entry toward mixed Nash. The residual checks computation. "
        "The Nash gap measures finite-precision behavior."
    )

    report.add_references([
        "[McKelvey, R. D. and Palfrey, T. R. (1995). Quantal Response Equilibria for Normal Form Games. *Games and Economic Behavior*, 10(1), 6-38.](https://doi.org/10.1006/game.1995.1023)",
        "[Goeree, J. K., Holt, C. A., and Palfrey, T. R. (2016). *Quantal Response Equilibrium: A Stochastic Theory of Games*. Princeton University Press.](https://doi.org/10.23943/princeton/9780691124230.001.0001)",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
