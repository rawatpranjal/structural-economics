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
from lib.plotting import setup_style, save_figure, save_thumbnail


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
    save_figure(fig, "figures/qre-path.png", dpi=150)

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
    save_figure(fig2, "figures/fixed-point-map.png", dpi=150)

    df = pd.DataFrame(qre_rows)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/qre-summary.csv", index=False)

    save_thumbnail("figures/qre-path.png", "figures/thumb.png")
    print("Done: figures/qre-path.png, figures/fixed-point-map.png, tables/qre-summary.csv")


if __name__ == "__main__":
    main()
