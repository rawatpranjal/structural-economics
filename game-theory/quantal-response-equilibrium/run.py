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
        p_entry, iterations, residual = solve_symmetric_entry_qre(precision)
        qre_rows.append({
            "Precision lambda": f"{precision:.1f}",
            "QRE Pr(Enter)": f"{p_entry:.4f}",
            "Mixed Nash Pr(Enter)": f"{mixed_nash_enter_prob:.4f}",
            "Gap to Nash": f"{p_entry - mixed_nash_enter_prob:+.4f}",
            "Iterations": iterations,
            "Residual": f"{residual:.2e}",
        })

    p_path_arr = np.array(p_path)

    setup_style()
    report = ModelReport(
        "Market Entry with Quantal Response Equilibrium",
        "How payoff-sensitive entry mistakes trace a fixed-point path toward mixed Nash.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Two firms are deciding whether a small market can support entry. One entrant earns "
        "positive profit. If both firms enter, competition makes entry unprofitable. Exact "
        "Nash equilibrium predicts either one entrant for sure or a symmetric mixed strategy "
        "that makes each firm indifferent.\n\n"
        "Quantal response equilibrium asks a nearby behavioral question. Suppose firms still "
        "compare expected payoffs, but they make payoff-sensitive mistakes. More profitable "
        "actions are chosen more often, yet lower-payoff actions can still occur. The "
        "calculation below follows the symmetric logit-QRE branch in this entry game and "
        "compares each finite-precision entry probability with the exact mixed Nash "
        "benchmark. The computation is a fixed-point problem because each firm's noisy entry "
        "probability must agree with the noisy response induced by the rival's probability."
    )

    report.add_equations(r"""
Each player chooses $E$ (Enter) or $O$ (Stay Out). Let $p_i$ be player $i$'s
probability of entry. If the rival enters with probability $q$, the expected
payoff difference between entering and staying out is

$$
\Delta(q)
= E[u_i(E,a_{-i})]-E[u_i(O,a_{-i})]
= 2(1-q)-q
= 2-3q.
$$

The exact symmetric mixed Nash equilibrium sets this difference to zero:

$$
p^{N} = \frac{2}{3}.
$$

Logit QRE replaces the discontinuous best response with

$$
QBR(q;\lambda)
=
\frac{\exp(\lambda \Delta(q))}
     {1+\exp(\lambda \Delta(q))}
=
\left[1+\exp(-\lambda(2-3q))\right]^{-1}.
$$

A symmetric logit-QRE is a fixed point

$$
p = QBR(p;\lambda).
$$

The precision parameter $\lambda \geq 0$ controls how strongly payoff gaps move
choice probabilities. At $\lambda=0$, both actions receive probability one half.
As $\lambda$ rises, the symmetric QRE branch $p(\lambda)$ moves toward the mixed
Nash probability $p^{N}=2/3$.
""")

    report.add_model_setup(
        "The payoff table gives the economic tension directly. Entry pays when the rival "
        "stays out. Joint entry destroys profits for both firms.\n\n"
        "| | Column Enter | Column Stay Out |\n"
        "|---|---:|---:|\n"
        "| **Row Enter** | -1, -1 | 2, 0 |\n"
        "| **Row Stay Out** | 0, 2 | 0, 0 |\n\n"
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        "| Exact pure Nash profiles | $(E,O)$ and $(O,E)$ | One entrant serves the market |\n"
        f"| Symmetric mixed Nash $p^N$ | {mixed_nash_enter_prob:.4f} | Exact benchmark for symmetric entry |\n"
        f"| Precision grid | {precisions[0]:.0f} to {precisions[-1]:.0f} | Strength of payoff sensitivity |\n"
        f"| Focal fixed-point plot | $\\lambda={focal_precision:.1f}$ | One logit response map |"
    )

    report.add_solution_method(
        "The symmetric branch turns the equilibrium calculation into a one-dimensional root "
        "search at each precision value. For a candidate entry probability $p$, define "
        "$G_{\\lambda}(p)=p-QBR(p;\\lambda)$. A QRE sets this residual to zero. In this "
        "entry game, $G_{\\lambda}$ is strictly increasing on $[0,1]$ and changes sign "
        "between the endpoints, so bisection finds the fixed point without tuning a step "
        "size.\n\n"
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
        "```\n\n"
        "For larger normal-form games, QRE becomes a system of fixed-point equations over "
        "mixed strategies. The one-dimensional version here keeps the entry object visible: "
        "a noisy entry probability that must be consistent with the noisy response it gives "
        "the other firm."
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
        "At zero precision, firms ignore payoffs and enter with probability one half. As "
        "precision rises, the symmetric QRE entry probability moves upward. The reason is "
        "economic: entry has positive expected payoff whenever the rival enters with "
        "probability below $2/3$. The dotted line comes from the exact indifference "
        "condition, not from the QRE root search."
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
        f"At $\\lambda={focal_precision:.1f}$, the noisy best-response curve is smooth but "
        "still strategic. A higher rival entry probability lowers the payoff from entry, "
        "so the response curve slopes down. The QRE is the crossing with the 45-degree line. "
        "The exact mixed Nash benchmark sits to the right because finite precision still "
        "puts weight on the lower-payoff action."
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
            "The residual column is numerical root-finding error. The gap to Nash is economic: "
            "it is the distance between finite-precision behavior and the exact symmetric "
            "mixed equilibrium."
        ),
    )

    final_p, final_iterations, final_residual = solve_symmetric_entry_qre(float(precisions[-1]))
    report.add_table(
        "tables/final-diagnostic.csv",
        "High-Precision Diagnostic",
        pd.DataFrame([{
            "Precision lambda": f"{precisions[-1]:.1f}",
            "QRE Pr(Enter)": f"{final_p:.6f}",
            "Mixed Nash Pr(Enter)": f"{mixed_nash_enter_prob:.6f}",
            "Absolute gap": f"{abs(final_p - mixed_nash_enter_prob):.2e}",
            "Iterations": final_iterations,
            "Fixed-point residual": f"{final_residual:.2e}",
        }]),
        description=(
            "The high-precision endpoint is close to, but still below, the mixed Nash limit. "
            "That distinction matters: QRE at a finite precision is a behavioral model, not "
            "a failed Nash computation."
        ),
    )

    report.add_takeaway(
        "QRE keeps mutual consistency but softens the exact best-response rule. In this "
        "entry game, higher precision moves the symmetric QRE toward the mixed Nash "
        "probability. The fixed-point residual checks the computation, while the gap to "
        "Nash measures finite-precision behavior. Those are different objects, and the "
        "tables keep them separate."
    )

    report.add_references([
        "[McKelvey, R. D. and Palfrey, T. R. (1995). Quantal Response Equilibria for Normal Form Games. *Games and Economic Behavior*, 10(1), 6-38.](https://doi.org/10.1006/game.1995.1023)",
        "[Goeree, J. K., Holt, C. A., and Palfrey, T. R. (2016). *Quantal Response Equilibrium: A Stochastic Theory of Games*. Princeton University Press.](https://doi.org/10.23943/princeton/9780691124230.001.0001)",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
