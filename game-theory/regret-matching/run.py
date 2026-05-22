#!/usr/bin/env python3
"""Hart-Mas-Colell regret matching on three normal-form games.

Two variants are implemented: exact regret matching (counterfactual values
computed against the opponent's current mixed strategy) and sampled regret
matching (counterfactual values computed against a single drawn action).
The exact variant is the default and converges faster to Nash.

Games studied:
  - Rock-Paper-Scissors: zero-sum 3x3; symmetric Nash at (1/3, 1/3, 1/3).
  - Stag Hunt: coordination 2x2; two pure Nash and one mixed Nash at (3/4, 1/4).
  - Battle of the Sexes: non-zero-sum 2x2; regret matching vs fictitious play.

Figures:
  figures/cumulative-regret.png          -- RPS player 1 cumulative regret per action
  figures/time-average-convergence.png   -- time-average strategy for all three games
  figures/regret-vs-fictitious.png       -- BoS comparison of the two algorithms
  figures/thumb.png                      -- from time-average-convergence
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED: int = 0
T_RPS: int = 10_000
T_STAG: int = 10_000
T_BOS: int = 10_000

# ---------------------------------------------------------------------------
# Payoff matrices
# ---------------------------------------------------------------------------

# Rock-Paper-Scissors: row player payoffs.
#   Actions: R=0, P=1, S=2
RPS_PAYOFF = np.array(
    [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ],
    dtype=float,
)

# Stag Hunt: row player payoffs.
#   Actions: Stag=0, Hare=1
STAG_PAYOFF = np.array(
    [
        [4, 0],
        [3, 3],
    ],
    dtype=float,
)

# Battle of the Sexes: separate payoffs for each player.
#   Actions: Opera=0, Football=1
BOS_PAYOFF_1 = np.array(
    [
        [3, 0],
        [0, 2],
    ],
    dtype=float,
)
BOS_PAYOFF_2 = np.array(
    [
        [2, 0],
        [0, 3],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Core regret matching
# ---------------------------------------------------------------------------


def regret_match_strategy(cumulative_regret: np.ndarray) -> np.ndarray:
    """Return a mixed strategy from cumulative regret via regret matching.

    Probability is proportional to positive cumulative regret. Falls back to
    uniform when all entries are non-positive.

    Args:
        cumulative_regret: 1-D array of length n_actions.

    Returns:
        Probability vector of length n_actions.
    """
    positive = np.maximum(cumulative_regret, 0.0)
    total = positive.sum()
    if total > 0.0:
        return positive / total
    return np.full(len(cumulative_regret), 1.0 / len(cumulative_regret))


def regret_matching_exact(
    payoff_1: np.ndarray,
    payoff_2: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> dict:
    """Two-player exact regret matching on a normal-form game.

    Counterfactual values are computed as expected payoff over the opponent's
    current mixed strategy, so no sampling noise is introduced. This is the
    form studied by Hart and Mas-Colell (2000) and converges to a correlated
    equilibrium in time-average play.

    Args:
        payoff_1: (n1, n2) payoff matrix for player 1.
        payoff_2: (n1, n2) payoff matrix for player 2.
        n_iter:   number of iterations T.
        rng:      NumPy random generator (used for action sampling to record
                  the sampled play path; does not affect regret updates).

    Returns:
        Dict with keys:
          regret_history_1 -- (T, n1) cumulative regret per iteration
          regret_history_2 -- (T, n2) cumulative regret per iteration
          avg_history_1    -- (T, n1) time-average strategy per iteration
          avg_history_2    -- (T, n2) time-average strategy per iteration
          final_avg_1      -- (n1,) final time-average strategy player 1
          final_avg_2      -- (n2,) final time-average strategy player 2
    """
    n1, n2 = payoff_1.shape
    R1 = np.zeros(n1)
    R2 = np.zeros(n2)
    S1 = np.zeros(n1)  # cumulative strategy mass for time averaging
    S2 = np.zeros(n2)

    regret_history_1 = np.zeros((n_iter, n1))
    regret_history_2 = np.zeros((n_iter, n2))
    avg_history_1 = np.zeros((n_iter, n1))
    avg_history_2 = np.zeros((n_iter, n2))

    for t in range(n_iter):
        pi1 = regret_match_strategy(R1)
        pi2 = regret_match_strategy(R2)

        # Expected counterfactual utility for each action of player 1:
        #   u1(a, pi2) = sum_{b} pi2[b] * payoff_1[a, b]
        cf1 = payoff_1 @ pi2
        actual1 = pi1 @ cf1
        R1 += cf1 - actual1

        # Symmetric for player 2:
        #   u2(pi1, b) = sum_{a} pi1[a] * payoff_2[a, b]
        cf2 = payoff_2.T @ pi1
        actual2 = pi2 @ cf2
        R2 += cf2 - actual2

        S1 += pi1
        S2 += pi2

        regret_history_1[t] = R1.copy()
        regret_history_2[t] = R2.copy()
        avg_history_1[t] = S1 / (t + 1)
        avg_history_2[t] = S2 / (t + 1)

    final_avg_1 = S1 / n_iter
    final_avg_2 = S2 / n_iter
    return {
        "regret_history_1": regret_history_1,
        "regret_history_2": regret_history_2,
        "avg_history_1": avg_history_1,
        "avg_history_2": avg_history_2,
        "final_avg_1": final_avg_1,
        "final_avg_2": final_avg_2,
    }


def regret_matching_sampled(
    payoff_1: np.ndarray,
    payoff_2: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> dict:
    """Two-player sampled regret matching on a normal-form game.

    Actions are drawn from the current mixed strategies and regret is updated
    against the realized opponent action rather than its expectation. Converges
    to correlated equilibrium in expectation but with higher variance.

    Args:
        payoff_1: (n1, n2) payoff matrix for player 1.
        payoff_2: (n1, n2) payoff matrix for player 2.
        n_iter:   number of iterations T.
        rng:      NumPy random generator for action sampling.

    Returns:
        Same key structure as regret_matching_exact.
    """
    n1, n2 = payoff_1.shape
    R1 = np.zeros(n1)
    R2 = np.zeros(n2)
    S1 = np.zeros(n1)
    S2 = np.zeros(n2)

    avg_history_1 = np.zeros((n_iter, n1))
    avg_history_2 = np.zeros((n_iter, n2))

    for t in range(n_iter):
        pi1 = regret_match_strategy(R1)
        pi2 = regret_match_strategy(R2)

        a1 = int(rng.choice(n1, p=pi1))
        a2 = int(rng.choice(n2, p=pi2))

        # Counterfactual regret against the realized opponent action.
        cf1 = payoff_1[:, a2]
        actual1 = float(payoff_1[a1, a2])
        R1 += cf1 - actual1

        cf2 = payoff_2[a1, :]
        actual2 = float(payoff_2[a1, a2])
        R2 += cf2 - actual2

        S1 += pi1
        S2 += pi2

        avg_history_1[t] = S1 / (t + 1)
        avg_history_2[t] = S2 / (t + 1)

    return {
        "avg_history_1": avg_history_1,
        "avg_history_2": avg_history_2,
        "final_avg_1": S1 / n_iter,
        "final_avg_2": S2 / n_iter,
    }


# ---------------------------------------------------------------------------
# Fictitious play
# ---------------------------------------------------------------------------


def fictitious_play(
    payoff_1: np.ndarray,
    payoff_2: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> dict:
    """Two-player fictitious play on a normal-form game.

    Each player best-responds to the empirical frequency of the opponent's
    past actions. Ties are broken by drawing uniformly from the argmax set.

    Args:
        payoff_1: (n1, n2) payoff matrix for player 1.
        payoff_2: (n1, n2) payoff matrix for player 2.
        n_iter:   number of iterations T.
        rng:      NumPy random generator for tie-breaking.

    Returns:
        Dict with avg_history_1, avg_history_2, final_avg_1, final_avg_2.
    """
    n1, n2 = payoff_1.shape
    count1 = np.ones(n1)  # Laplace smoothing: start with uniform prior
    count2 = np.ones(n2)

    avg_history_1 = np.zeros((n_iter, n1))
    avg_history_2 = np.zeros((n_iter, n2))

    for t in range(n_iter):
        freq1 = count1 / count1.sum()
        freq2 = count2 / count2.sum()

        # Best response: argmax expected payoff against opponent's empirical freq.
        br1_vals = payoff_1 @ freq2
        br2_vals = payoff_2.T @ freq1

        best1 = np.flatnonzero(br1_vals == br1_vals.max())
        best2 = np.flatnonzero(br2_vals == br2_vals.max())

        a1 = int(rng.choice(best1))
        a2 = int(rng.choice(best2))

        count1[a1] += 1
        count2[a2] += 1

        avg_history_1[t] = count1 / count1.sum()
        avg_history_2[t] = count2 / count2.sum()

    return {
        "avg_history_1": avg_history_1,
        "avg_history_2": avg_history_2,
        "final_avg_1": avg_history_1[-1],
        "final_avg_2": avg_history_2[-1],
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def normalized_max_regret(cumulative_regret: np.ndarray, n_iter: int) -> float:
    """Average regret bound: max positive cumulative regret divided by T.

    This is the standard convergence metric for regret matching. Hart and
    Mas-Colell (2000) show it converges to zero, guaranteeing the time-average
    strategy is an approximate correlated equilibrium.

    Args:
        cumulative_regret: 1-D array of cumulative regret at termination.
        n_iter:            number of iterations T.

    Returns:
        max(R^T, 0).max() / T, the average regret per round.
    """
    return float(np.maximum(cumulative_regret, 0.0).max()) / n_iter


def best_response_gap_zerosum(
    avg_1: np.ndarray, avg_2: np.ndarray, payoff: np.ndarray
) -> float:
    """Exploitability gap for player 1 in a zero-sum game.

    Measures how much player 1 could gain by best-responding to avg_2 instead
    of playing avg_1. A value near zero confirms avg_1 is close to Nash.

    Args:
        avg_1:  time-average strategy of player 1.
        avg_2:  time-average strategy of player 2.
        payoff: (n1, n2) payoff matrix for player 1.

    Returns:
        max_{a} u1(a, avg_2) - u1(avg_1, avg_2), clipped to non-negative.
    """
    expected_per_action = payoff @ avg_2
    best = float(expected_per_action.max())
    current = float(avg_1 @ expected_per_action)
    return max(best - current, 0.0)



# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_COLORS = ["C0", "C1", "C2", "C3"]


def _log_ticks(n_iter: int) -> np.ndarray:
    """Integer x-axis indices for a log-spaced convergence plot."""
    return np.arange(1, n_iter + 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    setup_style()

    Path("figures").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Game 1: Rock-Paper-Scissors
    # ------------------------------------------------------------------
    rps = regret_matching_exact(RPS_PAYOFF, -RPS_PAYOFF, T_RPS, rng)
    rps_nash = np.array([1 / 3, 1 / 3, 1 / 3])

    print("=== Rock-Paper-Scissors ===")
    print(f"  Final time-average (P1):   {rps['final_avg_1']}")
    print(f"  Nash benchmark:            {rps_nash}")
    rps_norm_regret = normalized_max_regret(rps["regret_history_1"][-1], T_RPS)
    print(f"  Normalized max regret R/T: {rps_norm_regret:.6f}")
    rps_gap = best_response_gap_zerosum(rps["final_avg_1"], rps["final_avg_2"], RPS_PAYOFF)
    print(f"  BR gap (P1 vs avg_2):      {rps_gap:.6f}")

    # ------------------------------------------------------------------
    # Game 2: Stag Hunt
    # ------------------------------------------------------------------
    stag = regret_matching_exact(STAG_PAYOFF, STAG_PAYOFF.T, T_STAG, rng)
    stag_mixed_nash = np.array([3 / 4, 1 / 4])

    print("\n=== Stag Hunt ===")
    print(f"  Final time-average (P1):   {stag['final_avg_1']}")
    print(f"  Mixed Nash benchmark:      {stag_mixed_nash}")
    stag_norm_regret = normalized_max_regret(stag["regret_history_1"][-1], T_STAG)
    print(f"  Normalized max regret R/T: {stag_norm_regret:.6f}")

    # ------------------------------------------------------------------
    # Game 3: Battle of the Sexes
    # ------------------------------------------------------------------
    bos_rm = regret_matching_exact(BOS_PAYOFF_1, BOS_PAYOFF_2, T_BOS, rng)
    bos_fp = fictitious_play(BOS_PAYOFF_1, BOS_PAYOFF_2, T_BOS, rng)
    bos_mixed_nash_1 = np.array([3 / 5, 2 / 5])  # P1: Opera with prob 3/5
    bos_mixed_nash_2 = np.array([2 / 5, 3 / 5])  # P2: Opera with prob 2/5

    print("\n=== Battle of the Sexes ===")
    print(f"  RM  final time-average (P1): {bos_rm['final_avg_1']}")
    print(f"  FP  final time-average (P1): {bos_fp['final_avg_1']}")
    print(f"  Mixed Nash benchmark (P1):   {bos_mixed_nash_1}")
    print(f"  RM  final time-average (P2): {bos_rm['final_avg_2']}")
    print(f"  FP  final time-average (P2): {bos_fp['final_avg_2']}")
    print(f"  Mixed Nash benchmark (P2):   {bos_mixed_nash_2}")

    iters = _log_ticks(T_RPS)

    # ------------------------------------------------------------------
    # Figure 1: Cumulative regret per action (RPS, player 1)
    # ------------------------------------------------------------------
    action_labels = ["Rock", "Paper", "Scissors"]
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    for j, label in enumerate(action_labels):
        ax1.plot(iters, rps["regret_history_1"][:, j], color=_COLORS[j], label=label)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xscale("log")
    ax1.set_xlabel("Iteration $t$")
    ax1.set_ylabel("Cumulative regret $R^t(a)$")
    ax1.set_title("Cumulative Regret per Action - Rock-Paper-Scissors, Player 1")
    ax1.legend()
    save_figure(fig1, "figures/cumulative-regret.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 2: Time-average strategy convergence (3 panels)
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel (a): RPS
    ax = axes[0]
    rps_targets = [1 / 3, 1 / 3, 1 / 3]
    for j, label in enumerate(action_labels):
        ax.plot(iters, rps["avg_history_1"][:, j], color=_COLORS[j], label=label)
        ax.axhline(rps_targets[j], color=_COLORS[j], linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("Time-average probability")
    ax.set_title("Rock-Paper-Scissors")
    ax.legend(fontsize=8)

    # Panel (b): Stag Hunt
    ax = axes[1]
    stag_labels = ["Stag", "Hare"]
    stag_targets = [3 / 4, 1 / 4]
    for j, label in enumerate(stag_labels):
        ax.plot(iters[:T_STAG], stag["avg_history_1"][:, j], color=_COLORS[j], label=label)
        ax.axhline(stag_targets[j], color=_COLORS[j], linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Iteration $t$")
    ax.set_title("Stag Hunt")
    ax.legend(fontsize=8)

    # Panel (c): Battle of the Sexes (regret matching)
    ax = axes[2]
    bos_labels = ["Opera", "Football"]
    bos_targets_1 = [3 / 5, 2 / 5]
    for j, label in enumerate(bos_labels):
        ax.plot(iters[:T_BOS], bos_rm["avg_history_1"][:, j], color=_COLORS[j], label=label)
        ax.axhline(bos_targets_1[j], color=_COLORS[j], linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Iteration $t$")
    ax.set_title("Battle of the Sexes (RM)")
    ax.legend(fontsize=8)

    fig2.suptitle("Time-Average Strategy Convergence (dotted lines: Nash targets)")
    fig2.tight_layout()
    save_figure(fig2, "figures/time-average-convergence.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3: Regret matching vs fictitious play (BoS)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))

    iters_bos = _log_ticks(T_BOS)
    nash_lines = [bos_mixed_nash_1, bos_mixed_nash_2]

    for player_idx, (ax, player_label, target) in enumerate(
        zip(axes3, ["Player 1", "Player 2"], nash_lines)
    ):
        key = f"avg_history_{player_idx + 1}"
        rm_history = bos_rm[key]
        fp_history = bos_fp[key]
        for j, action in enumerate(bos_labels):
            ax.plot(
                iters_bos,
                rm_history[:, j],
                color=_COLORS[j],
                linestyle="-",
                label=f"RM - {action}",
            )
            ax.plot(
                iters_bos,
                fp_history[:, j],
                color=_COLORS[j],
                linestyle="--",
                label=f"FP - {action}",
            )
            ax.axhline(target[j], color=_COLORS[j], linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Iteration $t$")
        ax.set_ylabel("Time-average probability")
        ax.set_title(f"Battle of the Sexes - {player_label}")
        ax.legend(fontsize=7)

    fig3.suptitle(
        "Regret Matching (solid) vs Fictitious Play (dashed); dotted: mixed Nash"
    )
    fig3.tight_layout()
    save_figure(fig3, "figures/regret-vs-fictitious.png", dpi=150)

    # Thumbnail from figure 2.
    save_thumbnail("figures/time-average-convergence.png", "figures/thumb.png")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: 3 figures + thumb.png in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
