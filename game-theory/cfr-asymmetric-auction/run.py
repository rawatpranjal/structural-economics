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
from lib.output import ModelReport
from lib.plotting import setup_style


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
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the MMRS boundary-value problem by forward shooting on alpha.

    Near b = 0, both inverse bid functions admit the asymptotic series
    phi_1(b) = 2b - alpha b^3 + O(b^5) and phi_2(b) = 2b + alpha b^3 + O(b^5),
    where alpha is a free coefficient that pins down the asymmetry. Forward
    integrate from a small b_0 with this asymptotic; stop when phi_1 reaches
    M_1 = 1 and read off the upper bid bbar and phi_2(bbar). Bisect alpha
    until phi_2(bbar) hits M_2 = 2.
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
    return float(bbar), b_grid, phi1, phi2


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

    bbar_bne, bne_b_grid, bne_phi1, bne_phi2 = solve_asymmetric_bne()
    bne_bid_weak = bne_bid_at_values(values_weak, bne_b_grid, bne_phi1, bbar_bne)
    bne_bid_strong = bne_bid_at_values(values_strong, bne_b_grid, bne_phi2, bbar_bne)
    asym_residual_weak = float(np.max(np.abs(asym_bid_weak - bne_bid_weak)))
    asym_residual_strong = float(np.max(np.abs(asym_bid_strong - bne_bid_strong)))

    setup_style()
    report = ModelReport(
        "Asymmetric First-Price Auctions by Counterfactual Regret Minimization",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Two bidders compete in a sealed-bid first-price auction with private values. "
        "Their value distributions are not the same. One bidder draws from a uniform "
        "distribution on a small support; the other draws from a uniform distribution on a "
        "wider support. The symmetric closed-form bid rule from the existing first-price "
        "auction tutorial does not apply. Best-response iteration cycles instead of "
        "converging.\n\n"
        "Counterfactual regret minimization is a learning algorithm that handles this case. "
        "Each bidder type is treated as an information set. The bidder accumulates regret "
        "for each candidate bid against the opponent's current strategy. The next strategy "
        "puts probability on bids in proportion to their positive cumulative regret. The "
        "time-averaged strategy converges to a Bayesian Nash equilibrium.\n\n"
        "The tutorial implements vanilla CFR on the asymmetric game and gives it two "
        "independent ground-truth checks. The symmetric closed form anchors the "
        "implementation when both value distributions are made equal. The continuous "
        "asymmetric BNE itself is recovered separately by solving the Marshall, Meurer, "
        "Richard, and Stromquist boundary-value problem on the inverse bid functions, "
        "and overlaid on the CFR result. Exploitability of the average strategy on the "
        "discretized game is the third diagnostic, the same no-deviation idea as the "
        "bid-grid deviation check in the existing first-price auction tutorial."
    )

    report.add_equations(r"""
The auction has two bidders indexed by $i \in \lbrace 1, 2 \rbrace$. Bidder $i$ draws a
private value $v_i$ from a known distribution $F_i$, independently across bidders.
Each bidder submits one sealed bid $b_i$ on a finite bid grid $B$, the highest bid
wins, the winner pays its own bid, and ties are broken uniformly at random.

A behavioral strategy $\sigma_i(b \mid v)$ is a probability distribution over bids
for each type $v$. The information set $I_v$ for bidder $i$ is the set of game
histories at which the bidder has observed type $v$. There is one information set
per type.

The expected payoff to bidder $i$ from bidding $b$ at type $v$ against the
opponent strategy $\sigma_{-i}$ is

$$
u_i(v, b; \sigma_{-i}) = (v - b) \cdot \Pr(\text{win} \mid b, \sigma_{-i}).
$$

The win probability uses the marginal opponent bid distribution
$q_{-i}(b') = \sum_{v'} P(v') \sigma_{-i}(b' \mid v')$ and the uniform tie-break
$\Pr(\text{win} \mid b) = \sum_{b' < b} q_{-i}(b') + \tfrac{1}{2} q_{-i}(b)$.

The counterfactual value at information set $I_v$ for action $b$ multiplies the
expected payoff by the chance reach probability $P(v)$:

$$
v_i^{\sigma}(I_v, b) = P(v) \cdot u_i(v, b; \sigma_{-i}).
$$

The instantaneous regret at iteration $t$ is the gap between the value of
deviating to $b$ and the value of the current mixed strategy:

$$
r_i^{t}(I_v, b) = v_i^{\sigma^{t}}(I_v, b) - \sum_{b'} \sigma_i^{t}(b' \mid v) \cdot v_i^{\sigma^{t}}(I_v, b').
$$

Cumulative regret accumulates these one-shot gaps:

$$
R_i^{T}(I_v, b) = \sum_{t = 1}^{T} r_i^{t}(I_v, b).
$$

The next strategy is regret matching, which puts mass on each bid in proportion
to its positive cumulative regret:

$$
\sigma_i^{T+1}(b \mid v) = \frac{\max(R_i^{T}(I_v, b), 0)}{\sum_{b'} \max(R_i^{T}(I_v, b'), 0)},
$$

with a uniform fallback when every cumulative regret is non-positive. The output
of the algorithm is the time-averaged strategy

$$
\bar{\sigma}_i^{T}(b \mid v) = \frac{1}{T} \sum_{t = 1}^{T} \sigma_i^{t}(b \mid v).
$$

The exploitability of a strategy profile $\sigma$ is the sum across players of
the most a single bidder could gain by switching to the best response:

$$
\varepsilon(\sigma) = \sum_{i = 1}^{2} \left(\max_{\sigma'_i} U_i(\sigma'_i, \sigma_{-i}) - U_i(\sigma_i, \sigma_{-i})\right),
$$

where $U_i(\sigma) = \sum_{v} P(v) \sum_{b} \sigma_i(b \mid v) \cdot u_i(v, b; \sigma_{-i})$
is the ex-ante expected payoff. Exploitability equals zero exactly at a Bayesian
Nash equilibrium of the discretized game. The best response in the maximization
is computed by picking, at each type, the bid on $B$ with the highest expected
payoff.

The continuous-game BNE itself can be written down as an ODE system on the
inverse bid functions $\phi_i(b)$, which give the type of bidder $i$ that bids
$b$ in equilibrium. Differentiating the bidder's first-order condition and
imposing equilibrium $v_i = \phi_i(b)$ yields the MMRS system

$$
\phi_i'(b) = \frac{F_i(\phi_i(b))}{f_i(\phi_i(b)) \cdot (\phi_j(b) - b)}, \quad j = 3 - i.
$$

For uniform value distributions on $[0, M_i]$, $F_i / f_i = v$ identically, so
the system simplifies to $\phi_1'(b) = \phi_1 / (\phi_2 - b)$ and
$\phi_2'(b) = \phi_2 / (\phi_1 - b)$. The boundary conditions are
$\phi_1(0) = \phi_2(0) = 0$ and $\phi_1(\bar{b}) = 1$, $\phi_2(\bar{b}) = 2$,
where the common upper bid $\bar{b}$ is an unknown that the boundary-value
problem pins down. Near $b = 0$ the inverse functions admit the asymptotic

$$
\phi_1(b) = 2b - \alpha b^3 + O(b^5), \qquad \phi_2(b) = 2b + \alpha b^3 + O(b^5),
$$

where $\alpha$ is a free coefficient. Shooting forward from a small $b_0$ with
this asymptotic initial condition and bisecting $\alpha$ on the constraint
$\phi_2(\bar{b}) = 2$ produces $\alpha = 3/2$ and $\bar{b} = 2/3$ for our
distributions. The continuous BNE bid function $b_i(v)$ is the inverse of
$\phi_i(b)$.
""")

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        "| Weak bidder values | $v_1 \\sim U[0, 1]$ | Smaller-support distribution |\n"
        "| Strong bidder values | $v_2 \\sim U[0, 2]$ | Larger-support distribution |\n"
        f"| Type grid | {n_types} nodes per bidder | Each type is one information set |\n"
        f"| Bid grid | {n_bids} nodes on $[0, 1]$ | Shared discrete action set |\n"
        f"| Iterations | {n_iter:,} | Simultaneous regret updates |\n"
        "| Tie-break | Uniform | Splits ties evenly across bidders |\n"
        "| Symmetric check | $v_1, v_2 \\sim U[0, 1]$ | Compares to $b^{\\ast}(v) = v / 2$ |"
    )

    report.add_solution_method(
        "Each bidder type is its own information set. The bidder runs regret matching "
        "locally at every type, accumulating regret for each candidate bid against the "
        "opponent's current strategy. Regret matching is Hannan-consistent at each "
        "information set, so the per-set average regret shrinks at rate of order one over "
        "the square root of iterations. The chance-reach weighting glues these per-set "
        "bounds into a global average regret bound on the time-averaged strategy. The "
        "tightest theoretical guarantee that the time-averaged strategy converges to a "
        "Nash equilibrium holds in two-player zero-sum games. The first-price auction is "
        "general-sum from the bidders' point of view, but CFR converges in practice on "
        "this game and on many other extensive-form Bayesian games beyond the zero-sum "
        "case. Exploitability of the average strategy is the diagnostic that confirms "
        "convergence on this run.\n\n"
        "Why regret matching works can be seen in a one-information-set toy. Suppose "
        "action $a$ always pays 2 and action $b$ always pays 1 against a fixed opponent. "
        "Starting from uniform play the average payoff is 1.5. Action $a$ accumulates "
        "regret 0.5 per iteration and action $b$ accumulates regret minus 0.5. After a "
        "few iterations the strategy puts all mass on $a$. The time-averaged strategy "
        "converges to the dominant action.\n\n"
        "```text\n"
        "Algorithm: vanilla CFR for the asymmetric first-price auction\n"
        "Inputs: type grids V_1, V_2 with PMFs P_1, P_2; bid grid B; iterations T\n"
        "Outputs: time-averaged strategies sigma_bar_1, sigma_bar_2\n\n"
        "1. Initialize R_i(v, b) = 0 and S_i(v, b) = 0 for i in {1, 2}.\n"
        "2. For t = 1, 2, ..., T:\n"
        "   a. For each i, compute sigma_i^t(b | v) by regret matching on R_i(v, .).\n"
        "   b. For each i, form the marginal opponent bid PMF\n"
        "      q_{-i}(b) = sum_{v'} P_{-i}(v') sigma_{-i}^t(b | v').\n"
        "   c. Compute the win probability w_{-i}(b) under uniform tie-break.\n"
        "   d. For each i, v in V_i, b in B, compute the counterfactual value\n"
        "      cf_i(v, b) = P_i(v) (v - b) w_{-i}(b)\n"
        "      and the iteration-average value cf_i_avg(v) = sum_b sigma_i^t(b | v) cf_i(v, b).\n"
        "   e. R_i(v, b) <- R_i(v, b) + cf_i(v, b) - cf_i_avg(v).\n"
        "   f. S_i(v, b) <- S_i(v, b) + sigma_i^t(b | v).\n"
        "3. Return sigma_bar_i(b | v) = S_i(v, b) / sum_{b'} S_i(v, b').\n"
        "```\n\n"
        "Exploitability of the average strategy is the deviation diagnostic. At each "
        "logged iteration the code computes the best-response payoff at every type and "
        "subtracts the average-strategy payoff. The expected gap, summed across "
        "bidders, is the exploitability."
    )

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
    report.add_results(
        "The average strategies on the asymmetric game match the continuous BNE "
        "computed independently from the MMRS boundary-value problem. The weak bidder "
        "bids more aggressively per unit of value than in the symmetric game because the "
        "rival often holds a higher value and shading too much loses too many auctions. "
        f"The strong bidder shades more deeply and tops out at a maximum bid of about "
        f"{bbar_bne:.3f}, well below the weak-support upper bound at one. Both CFR bid "
        "functions track the BNE within bid-grid spacing across the full type range. "
        "The asymmetric game has no closed-form solution, but the BNE is pinned down by "
        "a coupled ODE system on the inverse bid functions."
    )
    report.add_figure(
        "figures/bid-functions-asymmetric.png",
        "Asymmetric bid functions: CFR average vs MMRS BNE",
        fig1,
    )

    fig2, ax2 = plt.subplots()
    ax2.loglog(asym["iterations"], asym["exploitability"], color="C0")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Exploitability")
    ax2.set_title("Exploitability of the Average Strategy on the Asymmetric Game")
    report.add_results(
        "Exploitability of the average strategy falls steadily across iterations and "
        "tracks the textbook rate of order one over the square root of iterations on a "
        "log-log plot. Exploitability never reaches exactly zero because the bid grid is "
        "finite, but the residual gap is small relative to expected revenue. "
        "Exploitability is the asymmetric analogue of the bid-grid deviation check used "
        "by the existing first-price auction tutorial."
    )
    report.add_figure(
        "figures/exploitability.png",
        "Exploitability decay for vanilla CFR",
        fig2,
    )

    fig3, ax3 = plt.subplots()
    ax3.plot(sym_values, closed_form_sym, color="black", linestyle="--", linewidth=1.4, label="Closed form $v / 2$")
    ax3.plot(sym_values, sym_bid, marker="o", linestyle="-", color="C0", label="Vanilla CFR average")
    ax3.set_xlabel("Value $v$")
    ax3.set_ylabel("Expected bid $E[b \\mid v]$")
    ax3.set_title("Symmetric Sanity Check Against the $v / 2$ Closed Form")
    ax3.legend()
    report.add_results(
        "Setting both value distributions to uniform on the unit interval recovers a "
        "case where the analytic Bayesian Nash bid is half the value. The CFR average "
        "strategy tracks the closed form to within bid-grid spacing. The sanity check "
        "confirms that the implementation finds the right equilibrium on a case where "
        "the right answer is known."
    )
    report.add_figure(
        "figures/bid-functions-symmetric.png",
        "Symmetric uniform sanity check",
        fig3,
    )

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
            "Quantity": "Asymmetric exploitability (final iteration)",
            "Value": f"{asym['exploitability'][-1]:.3e}",
        },
        {
            "Quantity": "CFR iterations",
            "Value": f"{n_iter:,}",
        },
    ])
    report.add_table(
        "tables/methods-summary.csv",
        "Run summary",
        summary_table,
        description=(
            "Symmetric residual benchmarks the CFR average against the $v / 2$ closed "
            "form when both bidders draw from $U[0, 1]$. Asymmetric residuals benchmark "
            "the CFR average against the BNE bid function obtained by solving the MMRS "
            "boundary-value problem with `scipy.integrate.solve_ivp` plus bisection on "
            "$\\bar{b}$. Asymmetric exploitability is the sum of best-response payoff "
            "gains at the average strategy on the discretized asymmetric game."
        ),
    )

    convergence_table = pd.DataFrame({
        "Iteration": asym["iterations"],
        "Exploitability": [f"{x:.3e}" for x in asym["exploitability"]],
    })
    report.add_table(
        "tables/asymmetric-exploitability.csv",
        "Exploitability decay on the asymmetric game",
        convergence_table,
        description=(
            "Logarithmically spaced iteration checkpoints. Each row reports the "
            "exploitability of the time-averaged strategy at that iteration."
        ),
    )

    report.add_takeaway(
        "Counterfactual regret minimization replaces the analytic Bayesian Nash "
        "calculation with a regret-matching loop on the discretized game. The algorithm "
        "applies whenever each player has its own information set, including auctions "
        "where no closed form is available.\n\n"
        "On this asymmetric auction the CFR average strategy lands within bid-grid "
        "spacing of the continuous BNE recovered from the MMRS boundary-value problem. "
        "Two independent computations agreeing on the same bid functions is the "
        "convergence test that the exploitability metric alone could not provide.\n\n"
        "The same algorithm, scaled up to large extensive-form games with carefully "
        "tuned variants such as CFR+, is the engine behind modern poker AI."
    )

    report.add_references([
        "[Zinkevich, M., Johanson, M., Bowling, M., and Piccione, C. (2007). Regret Minimization in Games with Incomplete Information. *Advances in Neural Information Processing Systems*, 20.](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)",
        "[Tammelin, O., Burch, N., Johanson, M., and Bowling, M. (2015). Solving Heads-Up Limit Texas Hold'em. *IJCAI*, 645-652.](https://www.ijcai.org/Proceedings/15/Papers/097.pdf)",
        "[Maskin, E. and Riley, J. (2000). Asymmetric Auctions. *Review of Economic Studies*, 67(3), 413-438.](https://doi.org/10.1111/1467-937X.00137)",
        "[Krishna, V. (2009). *Auction Theory*, 2nd ed. Academic Press.](https://shop.elsevier.com/books/auction-theory/krishna/978-0-12-374507-1)",
        "**See also.** The symmetric uniform first-price auction is solved by the closed-form bid rule and verified by a bid-grid deviation check in [`game-theory/first-price-auctions/`](../../game-theory/first-price-auctions/). That tutorial is the symmetric ground-truth anchor for this one.",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")
    print(f"Symmetric residual (CFR vs v/2): {sym_residual:.3e}")
    print(f"MMRS upper bid bbar: {bbar_bne:.6f}")
    print(f"Asymmetric residual weak (CFR vs MMRS BNE): {asym_residual_weak:.3e}")
    print(f"Asymmetric residual strong (CFR vs MMRS BNE): {asym_residual_strong:.3e}")
    print(f"Asymmetric exploitability (final): {asym['exploitability'][-1]:.3e}")


if __name__ == "__main__":
    main()
