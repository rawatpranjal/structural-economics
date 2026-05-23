#!/usr/bin/env python3
"""Online pricing with revealed-preference bounds.

A seller posts prices on a discrete grid and observes only buy or no-buy from
each arriving customer. Plain bandit algorithms (epsilon-greedy, UCB1,
Thompson sampling) treat each price as an independent arm. Adding economic
structure tightens learning: WARP-style monotonicity converts each
observation into a valuation bound, the bounds imply lower and upper demand
at every price, and dominated prices are dropped from the active set. The
UCB with partial identification (UCB-PI) hybrid combines this dominance
filter with the standard UCB1 index.

Reference: Auer, Cesa-Bianchi and Fischer (2002), "Finite-time Analysis of
the Multiarmed Bandit Problem," Machine Learning 47, 235-256. Manski
(2003), Partial Identification of Probability Distributions. Lattimore and
Szepesvari (2020), Bandit Algorithms, Cambridge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


def oracle_curve(prices: np.ndarray, valuations: np.ndarray) -> np.ndarray:
    """Expected per-round revenue at each price, averaged over arrivals."""
    demand = np.array([np.mean(valuations >= p) for p in prices])
    return prices * demand


def cumulative_regret(rewards: np.ndarray, oracle_revenue: float) -> np.ndarray:
    """Cumulative regret against an oracle fixed price."""
    t = np.arange(1, len(rewards) + 1)
    return oracle_revenue * t - np.cumsum(rewards)


def epsilon_greedy(prices, T, segments, valuations, rng, eps=0.10):
    K = len(prices)
    counts = np.zeros(K)
    means = np.zeros(K)
    rewards = np.zeros(T)
    picks = np.zeros(T, dtype=int)
    for t in range(T):
        if rng.random() < eps:
            k = int(rng.integers(K))
        else:
            k = int(np.argmax(means))
        p = prices[k]
        s = segments[t]
        buy = int(valuations[s] >= p)
        r = p * buy
        counts[k] += 1
        means[k] += (r - means[k]) / counts[k]
        rewards[t] = r
        picks[t] = k
    return rewards, picks


def learn_then_earn(prices, T, segments, valuations, rng, T_explore=400):
    K = len(prices)
    counts = np.zeros(K)
    means = np.zeros(K)
    rewards = np.zeros(T)
    picks = np.zeros(T, dtype=int)
    for t in range(T):
        if t < T_explore:
            k = int(rng.integers(K))
        else:
            k = int(np.argmax(means))
        p = prices[k]
        s = segments[t]
        buy = int(valuations[s] >= p)
        r = p * buy
        counts[k] += 1
        means[k] += (r - means[k]) / counts[k]
        rewards[t] = r
        picks[t] = k
    return rewards, picks


def ucb1(prices, T, segments, valuations, rng):
    K = len(prices)
    counts = np.zeros(K)
    means = np.zeros(K)
    rewards = np.zeros(T)
    picks = np.zeros(T, dtype=int)
    max_reward = float(prices.max())
    for t in range(min(K, T)):
        k = t
        p = prices[k]
        s = segments[t]
        buy = int(valuations[s] >= p)
        r = p * buy
        counts[k] += 1
        means[k] = r
        rewards[t] = r
        picks[t] = k
    for t in range(K, T):
        bonus = max_reward * np.sqrt(2.0 * np.log(t + 1) / np.maximum(counts, 1))
        k = int(np.argmax(means + bonus))
        p = prices[k]
        s = segments[t]
        buy = int(valuations[s] >= p)
        r = p * buy
        counts[k] += 1
        means[k] += (r - means[k]) / counts[k]
        rewards[t] = r
        picks[t] = k
    return rewards, picks


def thompson_sampling(prices, T, segments, valuations, rng):
    """Beta posterior on the buy probability at each price."""
    K = len(prices)
    alpha = np.ones(K)
    beta = np.ones(K)
    rewards = np.zeros(T)
    picks = np.zeros(T, dtype=int)
    for t in range(T):
        theta = rng.beta(alpha, beta)
        k = int(np.argmax(prices * theta))
        p = prices[k]
        s = segments[t]
        buy = int(valuations[s] >= p)
        r = p * buy
        alpha[k] += buy
        beta[k] += 1 - buy
        rewards[t] = r
        picks[t] = k
    return rewards, picks


def ucb_partial_id(prices, T, segments, valuations, rng, V_max,
                   checkpoints=None):
    """UCB1 over an active price set shrunk by WARP-bound dominance.

    State per round:
      - per-segment valuation bounds (v_s^L, v_s^U)
      - per-price UCB statistics (counts, empirical revenue mean)
      - active mask (True if not yet dominated)
    """
    if checkpoints is None:
        checkpoints = []
    K = len(prices)
    S = len(valuations)
    v_lower = np.zeros(S)
    v_upper = np.full(S, float(V_max))
    counts = np.zeros(K)
    means = np.zeros(K)
    rewards = np.zeros(T)
    picks = np.zeros(T, dtype=int)
    active = np.ones(K, dtype=bool)
    active_over_time = np.zeros(T, dtype=int)
    max_reward = float(prices.max())
    diagnostics = {}

    def demand_and_profit_bounds():
        D_L = np.array([np.mean(v_lower >= p) for p in prices])
        D_U = np.array([np.mean(v_upper >= p) for p in prices])
        pi_L = prices * D_L
        pi_U = prices * D_U
        max_pi_L = pi_L.max()
        mask = pi_U >= max_pi_L
        return mask, pi_L, pi_U, D_L, D_U

    def play(t, k):
        p = prices[k]
        s = segments[t]
        buy = int(valuations[s] >= p)
        r = p * buy
        counts[k] += 1
        means[k] += (r - means[k]) / counts[k]
        rewards[t] = r
        picks[t] = k
        if buy:
            v_lower[s] = max(v_lower[s], p)
        else:
            v_upper[s] = min(v_upper[s], p)

    t = 0
    while t < T and counts[active].min() < 1.0:
        unseen_active = np.where(active & (counts == 0))[0]
        if unseen_active.size == 0:
            break
        k = int(unseen_active[0])
        play(t, k)
        active, pi_L, pi_U, D_L, D_U = demand_and_profit_bounds()
        active_over_time[t] = int(active.sum())
        if (t + 1) in checkpoints:
            diagnostics[t + 1] = {
                "pi_L": pi_L.copy(), "pi_U": pi_U.copy(),
                "v_lower": v_lower.copy(), "v_upper": v_upper.copy(),
                "active": active.copy(),
            }
        t += 1

    while t < T:
        n_safe = np.maximum(counts, 1)
        bonus = max_reward * np.sqrt(2.0 * np.log(t + 1) / n_safe)
        score = means + bonus
        score[~active] = -np.inf
        k = int(np.argmax(score))
        play(t, k)
        active, pi_L, pi_U, D_L, D_U = demand_and_profit_bounds()
        active_over_time[t] = int(active.sum())
        if (t + 1) in checkpoints:
            diagnostics[t + 1] = {
                "pi_L": pi_L.copy(), "pi_U": pi_U.copy(),
                "v_lower": v_lower.copy(), "v_upper": v_upper.copy(),
                "active": active.copy(),
            }
        t += 1

    return rewards, picks, active_over_time, diagnostics


def main():
    setup_style()

    seed = 0
    S = 4
    V_max = 10.0
    valuations = np.array([2.5, 4.5, 6.5, 8.5])
    K = 20
    prices = np.linspace(0.5, V_max - 0.5, K)
    T = 5000
    checkpoints = [100, 500, 1000, 5000]

    rng = np.random.default_rng(seed)
    segments = rng.integers(S, size=T)

    revenue_curve = oracle_curve(prices, valuations)
    oracle_idx = int(np.argmax(revenue_curve))
    oracle_price = float(prices[oracle_idx])
    oracle_revenue = float(revenue_curve[oracle_idx])

    algos = {}
    rng = np.random.default_rng(seed + 1)
    algos["epsilon-greedy"] = epsilon_greedy(prices, T, segments, valuations, rng)
    rng = np.random.default_rng(seed + 2)
    algos["learn-then-earn"] = learn_then_earn(prices, T, segments, valuations, rng, T_explore=400)
    rng = np.random.default_rng(seed + 3)
    algos["UCB1"] = ucb1(prices, T, segments, valuations, rng)
    rng = np.random.default_rng(seed + 4)
    algos["Thompson"] = thompson_sampling(prices, T, segments, valuations, rng)
    rng = np.random.default_rng(seed + 5)
    rewards_pi, picks_pi, active_pi, diagnostics = ucb_partial_id(
        prices, T, segments, valuations, rng, V_max, checkpoints
    )

    regrets = {
        name: cumulative_regret(result[0], oracle_revenue)
        for name, result in algos.items()
    }
    regrets["UCB-PI"] = cumulative_regret(rewards_pi, oracle_revenue)
    picks_all = {name: result[1] for name, result in algos.items()}
    picks_all["UCB-PI"] = picks_pi

    out = Path(__file__).parent
    figs = out / "figures"
    tabs = out / "tables"
    tabs.mkdir(exist_ok=True)

    colors = {
        "epsilon-greedy": "#888888",
        "learn-then-earn": "#cc7722",
        "UCB1": "#1166aa",
        "Thompson": "#22aa44",
        "UCB-PI": "#cc2255",
    }

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    t_axis = np.arange(1, T + 1)
    for name, r in regrets.items():
        clipped = np.maximum(r, 1e-2)
        ax.loglog(t_axis, clipped, label=name, color=colors[name], lw=1.8)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Cumulative regret vs an oracle fixed price")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    save_figure(fig, figs / "regret-comparison.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(t_axis, active_pi, color=colors["UCB-PI"], lw=1.8, label="UCB-PI active set")
    ax.axhline(K, color=colors["UCB1"], ls="--", lw=1.2, label="UCB1 active set (no elimination)")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Number of active prices")
    ax.set_title("Active-price set under UCB-PI shrinks as bounds tighten")
    ax.set_ylim(0, K + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save_figure(fig, figs / "active-prices.png")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True, sharey=True)
    for ax, t_check in zip(axes.flat, checkpoints):
        d = diagnostics[t_check]
        ax.fill_between(prices, d["pi_L"], d["pi_U"], color="#cc2255", alpha=0.20,
                        label="Profit bound band")
        ax.plot(prices, d["pi_L"], color="#cc2255", lw=1.0, ls=":")
        ax.plot(prices, d["pi_U"], color="#cc2255", lw=1.0, ls="-")
        ax.plot(prices, revenue_curve, color="black", lw=1.5, label="True revenue")
        ax.axvline(oracle_price, color="black", ls=":", lw=1.0, alpha=0.5)
        active_mask = d["active"]
        ax.scatter(prices[active_mask], revenue_curve[active_mask],
                   color="#cc2255", s=28, zorder=3,
                   label=f"Active ({int(active_mask.sum())})")
        if (~active_mask).any():
            ax.scatter(prices[~active_mask], revenue_curve[~active_mask],
                       color="lightgray", s=24, zorder=2,
                       label=f"Eliminated ({int((~active_mask).sum())})")
        ax.set_title(f"Round $t = {t_check}$")
        ax.grid(True, alpha=0.3)
        if t_check == checkpoints[0]:
            ax.legend(fontsize=7, loc="upper left")
    fig.supxlabel("Price $p$")
    fig.supylabel("Profit per round")
    fig.suptitle("Profit bounds and elimination across rounds")
    fig.tight_layout()
    save_figure(fig, figs / "profit-bounds.png")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    n_diag = len(checkpoints)
    width = 0.18
    for i, t_check in enumerate(checkpoints):
        d = diagnostics[t_check]
        y = np.arange(S) + (i - (n_diag - 1) / 2) * width
        color = plt.cm.viridis(i / max(1, n_diag - 1))
        ax.hlines(y, d["v_lower"], d["v_upper"], color=color, lw=4,
                  label=f"$t = {t_check}$")
    ax.scatter(valuations, np.arange(S), color="black", s=70, zorder=5,
               label="True $v_s$", marker="D")
    ax.set_yticks(np.arange(S))
    ax.set_yticklabels([f"Segment {s + 1}" for s in range(S)])
    ax.set_xlabel("Valuation")
    ax.set_xlim(0, V_max)
    ax.set_title("Per-segment valuation bounds tighten with observations")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figs / "valuation-intervals.png")

    final_rows = []
    for name in ["epsilon-greedy", "learn-then-earn", "UCB1", "Thompson", "UCB-PI"]:
        rewards = algos[name][0] if name != "UCB-PI" else rewards_pi
        picks = picks_all[name]
        final_r = float(regrets[name][-1])
        avg_r = final_r / T
        last_window = picks[-500:]
        most_played_idx = int(np.argmax(np.bincount(last_window, minlength=K)))
        most_played_price = float(prices[most_played_idx])
        active_final = int(active_pi[-1]) if name == "UCB-PI" else K
        final_rows.append({
            "Algorithm": name,
            "Final cumulative regret": round(final_r, 2),
            "Average regret per round": round(avg_r, 4),
            "Oracle price": round(oracle_price, 2),
            "Most-played price (last 500 rounds)": round(most_played_price, 2),
            "Active prices at T": active_final,
        })
    pd.DataFrame(final_rows).to_csv(tabs / "final-regret.csv", index=False)

    elim_rows = []
    for t_check in checkpoints:
        d = diagnostics[t_check]
        active_count = int(d["active"].sum())
        elim_rows.append({
            "Round": t_check,
            "Active prices": active_count,
            "Eliminated prices": K - active_count,
            "Active fraction": round(active_count / K, 3),
        })
    pd.DataFrame(elim_rows).to_csv(tabs / "elimination-diagnostics.csv", index=False)

    save_thumbnail(figs / "active-prices.png", figs / "thumb.png")

    print(f"Oracle price: {oracle_price:.2f} | Oracle revenue per round: {oracle_revenue:.3f}")
    for name in ["epsilon-greedy", "learn-then-earn", "UCB1", "Thompson", "UCB-PI"]:
        final_r = float(regrets[name][-1])
        ap = int(active_pi[-1]) if name == "UCB-PI" else K
        print(f"  {name:18s} final regret = {final_r:8.2f} | active prices at T = {ap}")


if __name__ == "__main__":
    main()
