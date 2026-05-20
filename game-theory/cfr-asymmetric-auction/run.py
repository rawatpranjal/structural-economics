#!/usr/bin/env python3
"""Asymmetric first-price auction solved by counterfactual regret minimization.

Two bidders draw values from different uniform distributions, so the symmetric
closed-form bid rule no longer applies. Vanilla CFR runs regret matching at
each information set (one per type) and converges to the Bayesian Nash
equilibrium of the discretized game. The implementation is sanity-checked on
the symmetric uniform case where the closed form is known.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def make_uniform_grid(low: float, high: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced grid on [low, high] with a uniform PMF."""
    grid = np.linspace(low, high, n_nodes)
    pmf = np.full(n_nodes, 1.0 / n_nodes)
    return grid, pmf


def opponent_bid_pmf(strategy: np.ndarray, type_pmf: np.ndarray) -> np.ndarray:
    """Marginal bid distribution: q(b) = sum_v P(v) sigma(b | v)."""
    return type_pmf @ strategy


def win_probability(opp_bid_pmf: np.ndarray) -> np.ndarray:
    """Probability of winning at each bid in the grid under uniform tie-break."""
    cum_below = np.zeros_like(opp_bid_pmf)
    cum_below[1:] = np.cumsum(opp_bid_pmf[:-1])
    return cum_below + 0.5 * opp_bid_pmf


def counterfactual_values(
    values: np.ndarray,
    bids: np.ndarray,
    opponent_strategy: np.ndarray,
    type_pmf_self: np.ndarray,
    type_pmf_opp: np.ndarray,
) -> np.ndarray:
    """Counterfactual value at each (v_i, b) for the active player.

    Includes the chance reach probability P(v_i) on the self side, following the
    standard CFR convention.
    """
    opp_pmf = opponent_bid_pmf(opponent_strategy, type_pmf_opp)
    win = win_probability(opp_pmf)
    payoff = values[:, None] - bids[None, :]
    return type_pmf_self[:, None] * payoff * win[None, :]


def regret_matching(regret: np.ndarray) -> np.ndarray:
    """Regret matching at each information set (a row of `regret`)."""
    positive = np.maximum(regret, 0.0)
    total = positive.sum(axis=-1, keepdims=True)
    uniform = np.full_like(regret, 1.0 / regret.shape[-1])
    return np.where(total > 0, positive / np.where(total > 0, total, 1.0), uniform)


def average_strategy(cumulative: np.ndarray) -> np.ndarray:
    """Normalize cumulative strategy mass to a probability distribution per type."""
    total = cumulative.sum(axis=-1, keepdims=True)
    uniform = np.full_like(cumulative, 1.0 / cumulative.shape[-1])
    return np.where(total > 0, cumulative / np.where(total > 0, total, 1.0), uniform)


def expected_bid(strategy: np.ndarray, bids: np.ndarray) -> np.ndarray:
    """Expected bid given each type: sum_b sigma(b | v) b."""
    return strategy @ bids


def exploitability(
    strat_1: np.ndarray,
    strat_2: np.ndarray,
    values_1: np.ndarray,
    values_2: np.ndarray,
    type_pmf_1: np.ndarray,
    type_pmf_2: np.ndarray,
    bids: np.ndarray,
) -> tuple[float, float, float]:
    """Sum of best-response payoff gains, plus per-player gains."""
    eps = []
    for self_vals, self_pmf, opp_strat, opp_pmf, self_strat in [
        (values_1, type_pmf_1, strat_2, type_pmf_2, strat_1),
        (values_2, type_pmf_2, strat_1, type_pmf_1, strat_2),
    ]:
        opp_bid = opponent_bid_pmf(opp_strat, opp_pmf)
        win = win_probability(opp_bid)
        payoff = (self_vals[:, None] - bids[None, :]) * win[None, :]
        current = (self_strat * payoff).sum(axis=-1)
        best = payoff.max(axis=-1)
        eps.append(float((self_pmf * (best - current)).sum()))
    return eps[0] + eps[1], eps[0], eps[1]


def vanilla_cfr(
    values_1: np.ndarray,
    values_2: np.ndarray,
    bids: np.ndarray,
    type_pmf_1: np.ndarray,
    type_pmf_2: np.ndarray,
    n_iter: int,
    log_iters: np.ndarray,
) -> dict:
    """Vanilla CFR with simultaneous regret updates and uniform averaging."""
    NV1, NV2, NB = len(values_1), len(values_2), len(bids)
    R1 = np.zeros((NV1, NB))
    R2 = np.zeros((NV2, NB))
    S1 = np.zeros((NV1, NB))
    S2 = np.zeros((NV2, NB))

    log_set = set(int(t) for t in log_iters)
    iters_logged: list[int] = []
    expl_logged: list[float] = []

    for t in range(1, n_iter + 1):
        sigma1 = regret_matching(R1)
        sigma2 = regret_matching(R2)

        cf1 = counterfactual_values(values_1, bids, sigma2, type_pmf_1, type_pmf_2)
        cf2 = counterfactual_values(values_2, bids, sigma1, type_pmf_2, type_pmf_1)

        avg_v1 = (sigma1 * cf1).sum(axis=-1, keepdims=True)
        avg_v2 = (sigma2 * cf2).sum(axis=-1, keepdims=True)

        R1 += cf1 - avg_v1
        R2 += cf2 - avg_v2

        S1 += sigma1
        S2 += sigma2

        if t in log_set:
            avg1 = average_strategy(S1)
            avg2 = average_strategy(S2)
            eps_total, _, _ = exploitability(
                avg1, avg2, values_1, values_2, type_pmf_1, type_pmf_2, bids
            )
            iters_logged.append(t)
            expl_logged.append(eps_total)

    return {
        "average_strategy_1": average_strategy(S1),
        "average_strategy_2": average_strategy(S2),
        "iterations": np.array(iters_logged),
        "exploitability": np.array(expl_logged),
    }


def log_iteration_grid(n_iter: int, n_points: int = 40) -> np.ndarray:
    """Logarithmically spaced iteration checkpoints (always including 1 and n_iter)."""
    return np.unique(np.geomspace(1, n_iter, n_points).round().astype(int))


def asymmetric_bne_odes(b: float, phi: np.ndarray) -> list[float]:
    """MMRS inverse-bid ODEs for asymmetric FPA with F_1=U[0,1], F_2=U[0,2].

    For uniform F_i on [0, M_i], the hazard ratio F_i / f_i = v, so the ODEs
    simplify to dphi_1/db = phi_1 / (phi_2 - b) and dphi_2/db = phi_2 / (phi_1 - b),
    where phi_i(b) is the type of bidder i that bids b in equilibrium.
    """
    p1, p2 = phi
    return [p1 / (p2 - b), p2 / (p1 - b)]


def solve_asymmetric_bne(
    b_0: float = 1e-3,
    alpha_lo: float = 1.0,
    alpha_hi: float = 1.9,
    n_grid: int = 4001,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the MMRS boundary-value problem by forward shooting on alpha.

    Near b = 0, both inverse bid functions admit the asymptotic series
    phi_1(b) = 2b - alpha b^3 + O(b^5) and phi_2(b) = 2b + alpha b^3 + O(b^5),
    where alpha is a free coefficient that pins down the asymmetry. Forward
    integrate from a small b_0 with this asymptotic; stop when phi_1 reaches
    M_1 = 1 and read off the upper bid bbar and phi_2(bbar). Bisect alpha
    until phi_2(bbar) hits M_2 = 2. Returns the upper bid, the bid grid, both
    inverse bid functions, and the bisected coefficient alpha.
    """
    def shoot(alpha: float) -> tuple[float, float, object]:
        phi0 = [2.0 * b_0 - alpha * b_0**3, 2.0 * b_0 + alpha * b_0**3]

        def event(b: float, phi: np.ndarray) -> float:
            return phi[0] - 1.0
        event.terminal = True
        event.direction = 1

        sol = solve_ivp(
            asymmetric_bne_odes, [b_0, 5.0], phi0,
            events=event, rtol=1e-12, atol=1e-14, max_step=2e-4,
        )
        if len(sol.t_events[0]) == 0:
            return float("nan"), float("nan"), sol
        bbar = float(sol.t_events[0][0])
        phi2 = float(sol.y_events[0][0][1])
        return bbar, phi2, sol

    def residual(alpha: float) -> float:
        _, phi2, _ = shoot(alpha)
        return phi2 - 2.0

    alpha_opt = brentq(residual, alpha_lo, alpha_hi, xtol=1e-12)
    bbar, _, _ = shoot(alpha_opt)

    phi0 = [2.0 * b_0 - alpha_opt * b_0**3, 2.0 * b_0 + alpha_opt * b_0**3]
    t_eval = np.linspace(b_0, bbar, n_grid)
    sol = solve_ivp(
        asymmetric_bne_odes, [b_0, bbar], phi0,
        t_eval=t_eval, rtol=1e-12, atol=1e-14, max_step=2e-4,
    )
    b_grid = np.concatenate([[0.0], sol.t])
    phi1 = np.concatenate([[0.0], sol.y[0]])
    phi2 = np.concatenate([[0.0], sol.y[1]])
    return float(bbar), b_grid, phi1, phi2, float(alpha_opt)


def bne_bid_at_values(
    values: np.ndarray, b_grid: np.ndarray, phi: np.ndarray, bbar: float,
) -> np.ndarray:
    """Invert phi(b) to get b(v): the bid that type v plays in BNE."""
    return np.interp(values, phi, b_grid, left=0.0, right=bbar)


def main() -> None:
    n_types = 21
    n_bids = 41
    n_iter = 5000

    values_strong, pmf_strong = make_uniform_grid(0.0, 2.0, n_types)
    values_weak, pmf_weak = make_uniform_grid(0.0, 1.0, n_types)
    bids = np.linspace(0.0, 1.0, n_bids)
    log_iters = log_iteration_grid(n_iter)

    asym = vanilla_cfr(
        values_weak, values_strong, bids, pmf_weak, pmf_strong, n_iter, log_iters,
    )
    sym_values, sym_pmf = make_uniform_grid(0.0, 1.0, n_types)
    sym = vanilla_cfr(
        sym_values, sym_values, bids, sym_pmf, sym_pmf, n_iter, log_iters,
    )

    closed_form_sym = 0.5 * sym_values
    sym_bid = expected_bid(sym["average_strategy_1"], bids)
    sym_residual = float(np.max(np.abs(sym_bid - closed_form_sym)))

    asym_bid_weak = expected_bid(asym["average_strategy_1"], bids)
    asym_bid_strong = expected_bid(asym["average_strategy_2"], bids)

    bbar_bne, bne_b_grid, bne_phi1, bne_phi2, alpha_bne = solve_asymmetric_bne()
    bne_bid_weak = bne_bid_at_values(values_weak, bne_b_grid, bne_phi1, bbar_bne)
    bne_bid_strong = bne_bid_at_values(values_strong, bne_b_grid, bne_phi2, bbar_bne)
    asym_residual_weak = float(np.max(np.abs(asym_bid_weak - bne_bid_weak)))
    asym_residual_strong = float(np.max(np.abs(asym_bid_strong - bne_bid_strong)))

    setup_style()

    fig1, ax1 = plt.subplots()
    bne_v1_curve = np.linspace(0.0, 1.0, 400)
    bne_v2_curve = np.linspace(0.0, 2.0, 400)
    bne_b1_curve = bne_bid_at_values(bne_v1_curve, bne_b_grid, bne_phi1, bbar_bne)
    bne_b2_curve = bne_bid_at_values(bne_v2_curve, bne_b_grid, bne_phi2, bbar_bne)
    ax1.plot(bne_v1_curve, bne_b1_curve, color="C0", linestyle=":", linewidth=2.2,
             label="Weak BNE (MMRS ODE)")
    ax1.plot(bne_v2_curve, bne_b2_curve, color="C3", linestyle=":", linewidth=2.2,
             label="Strong BNE (MMRS ODE)")
    ax1.plot(values_weak, asym_bid_weak, marker="o", linestyle="-", color="C0",
             markersize=5, label="Weak CFR average, $v_1 \\sim U[0,1]$")
    ax1.plot(values_strong, asym_bid_strong, marker="s", linestyle="-", color="C3",
             markersize=5, label="Strong CFR average, $v_2 \\sim U[0,2]$")
    ax1.plot(values_strong, values_strong, color="black", linestyle="--", linewidth=1.0,
             label="Truthful bid")
    ax1.set_xlabel("Value $v$")
    ax1.set_ylabel("Expected bid $E[b \\mid v]$")
    ax1.set_title("Asymmetric Bid Functions: CFR vs MMRS BNE")
    ax1.legend(fontsize=8)
    save_figure(fig1, "figures/bid-functions-asymmetric.png", dpi=150)

    fig2, ax2 = plt.subplots()
    ax2.loglog(asym["iterations"], asym["exploitability"], color="C0")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Exploitability")
    ax2.set_title("Exploitability of the Average Strategy on the Asymmetric Game")
    save_figure(fig2, "figures/exploitability.png", dpi=150)

    fig3, ax3 = plt.subplots()
    ax3.plot(sym_values, closed_form_sym, color="black", linestyle="--", linewidth=1.4, label="Closed form $v / 2$")
    ax3.plot(sym_values, sym_bid, marker="o", linestyle="-", color="C0", label="Vanilla CFR average")
    ax3.set_xlabel("Value $v$")
    ax3.set_ylabel("Expected bid $E[b \\mid v]$")
    ax3.set_title("Symmetric Sanity Check Against the $v / 2$ Closed Form")
    ax3.legend()
    save_figure(fig3, "figures/bid-functions-symmetric.png", dpi=150)

    summary_table = pd.DataFrame([
        {
            "Quantity": "Symmetric residual (max CFR bid error vs $v / 2$)",
            "Value": f"{sym_residual:.3e}",
        },
        {
            "Quantity": "Asymmetric residual: weak bidder (max CFR bid error vs MMRS BNE)",
            "Value": f"{asym_residual_weak:.3e}",
        },
        {
            "Quantity": "Asymmetric residual: strong bidder (max CFR bid error vs MMRS BNE)",
            "Value": f"{asym_residual_strong:.3e}",
        },
        {
            "Quantity": "MMRS upper bid $\\bar{b}$",
            "Value": f"{bbar_bne:.4f}",
        },
        {
            "Quantity": "MMRS shooting coefficient $\\alpha$",
            "Value": f"{alpha_bne:.4f}",
        },
        {
            "Quantity": "Asymmetric exploitability (final iteration)",
            "Value": f"{asym['exploitability'][-1]:.3e}",
        },
        {
            "Quantity": "CFR iterations",
            "Value": f"{n_iter:,}",
        },
    ])
    Path("tables").mkdir(parents=True, exist_ok=True)
    summary_table.to_csv("tables/methods-summary.csv", index=False)

    convergence_table = pd.DataFrame({
        "Iteration": asym["iterations"],
        "Exploitability": [f"{x:.3e}" for x in asym["exploitability"]],
    })
    convergence_table.to_csv("tables/asymmetric-exploitability.csv", index=False)

    save_thumbnail("figures/bid-functions-asymmetric.png", "figures/thumb.png")
    print(f"Done: 3 figures + 2 tables")
    print(f"Symmetric residual (CFR vs v/2): {sym_residual:.3e}")
    print(f"MMRS upper bid bbar: {bbar_bne:.6f}")
    print(f"Asymmetric residual weak (CFR vs MMRS BNE): {asym_residual_weak:.3e}")
    print(f"Asymmetric residual strong (CFR vs MMRS BNE): {asym_residual_strong:.3e}")
    print(f"Asymmetric exploitability (final): {asym['exploitability'][-1]:.3e}")


if __name__ == "__main__":
    main()
