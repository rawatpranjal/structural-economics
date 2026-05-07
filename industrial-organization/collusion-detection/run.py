#!/usr/bin/env python3
"""Repeated-game cartels and price-screen diagnostics.

Models collusion as a repeated Cournot game with trigger strategies.
Computes critical discount factors for cartel sustainability and shows how
price and margin breaks look in a stylized antitrust screen.

References: Stigler (1964), Porter (1983), Harrington (2008),
Igami and Sugaya (2021).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


# =============================================================================
# Cournot duopoly: analytical profit calculations
# =============================================================================

def cournot_nash_profits(n, a, c):
    """Compute per-firm Nash equilibrium profits in symmetric Cournot.

    Inverse demand: P = a - Q, where Q = sum of all firms' quantities.
    Constant marginal cost c for all firms.

    Returns per-firm quantity, price, and profit.
    """
    q_i = (a - c) / (n + 1)
    Q = n * q_i
    P = a - Q
    pi_i = (a - c) ** 2 / (n + 1) ** 2
    return q_i, P, pi_i


def collusion_profits(n, a, c):
    """Compute per-firm profits under perfect collusion (joint monopoly).

    Firms split the monopoly output equally.
    """
    Q_m = (a - c) / 2       # monopoly total quantity
    q_i = Q_m / n            # each firm's share
    P_m = a - Q_m            # monopoly price
    pi_i = q_i * (P_m - c)  # per-firm profit = (1/n) * monopoly profit
    return q_i, P_m, pi_i


def deviation_profits(n, a, c):
    """Compute the one-shot deviation profit for a single firm.

    One firm deviates optimally while the other (n-1) firms produce their
    collusive quantity q_collude = (a-c)/(2n).
    The deviator best-responds to the others' collusive output.
    """
    q_collude_each = (a - c) / (2 * n)
    Q_others = (n - 1) * q_collude_each

    # Deviator maximizes: q_d * (a - Q_others - q_d - c)
    # FOC: a - Q_others - 2*q_d - c = 0
    q_d = (a - c - Q_others) / 2
    P_d = a - Q_others - q_d
    pi_d = q_d * (P_d - c)
    return q_d, P_d, pi_d


def critical_discount_factor(n, a, c):
    """Compute delta* for grim trigger strategy in symmetric Cournot.

    delta* = (pi_deviate - pi_collude) / (pi_deviate - pi_compete)
    Collusion is sustainable iff delta >= delta*.
    """
    _, _, pi_compete = cournot_nash_profits(n, a, c)
    _, _, pi_collude = collusion_profits(n, a, c)
    _, _, pi_deviate = deviation_profits(n, a, c)

    delta_star = (pi_deviate - pi_collude) / (pi_deviate - pi_compete)
    return delta_star


def maximum_sustainable_firms(delta, a, c, max_n=500):
    """Largest symmetric firm count with delta >= delta*(n)."""
    feasible = [
        n for n in range(2, max_n + 1)
        if critical_discount_factor(n, a, c) <= delta
    ]
    return max(feasible) if feasible else None


# =============================================================================
# Price-screen simulation
# =============================================================================

def simulate_price_series(T_compete, T_collude, T_detect, a, c, n, sigma_noise,
                          rng=None):
    """Simulate a price series with regime changes.

    Phases:
      1. Competition (T_compete periods): prices near Nash equilibrium
      2. Collusion (T_collude periods): prices near monopoly level
      3. Detection / reversion (T_detect periods): prices drop back to Nash

    Returns time index, prices, marginal costs, and regime labels.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    _, P_nash, _ = cournot_nash_profits(n, a, c)
    _, P_mono, _ = collusion_profits(n, a, c)

    T_total = T_compete + T_collude + T_detect
    t = np.arange(T_total)
    prices = np.zeros(T_total)
    regimes = np.empty(T_total, dtype=object)

    # Phase 1: competition
    prices[:T_compete] = P_nash + rng.normal(0, sigma_noise, T_compete)
    regimes[:T_compete] = "Competition"

    # Phase 2: collusion -- prices gradually rise then stabilise
    ramp_up = min(5, T_collude)
    for i in range(T_collude):
        frac = min(1.0, (i + 1) / ramp_up)
        base = P_nash + frac * (P_mono - P_nash)
        prices[T_compete + i] = base + rng.normal(0, sigma_noise * 0.7)
    regimes[T_compete:T_compete + T_collude] = "Collusion"

    # Phase 3: detection -- prices crash back
    ramp_down = min(3, T_detect)
    for i in range(T_detect):
        frac = min(1.0, (i + 1) / ramp_down)
        base = P_mono - frac * (P_mono - P_nash)
        prices[T_compete + T_collude + i] = base + rng.normal(0, sigma_noise)
    regimes[T_compete + T_collude:] = "Post-Detection"

    mc = np.full(T_total, c)
    return t, prices, mc, regimes


# =============================================================================
# Main
# =============================================================================

def main():
    setup_style()

    # -------------------------------------------------------------------------
    # Parameters for the theoretical model
    # -------------------------------------------------------------------------
    a = 100       # demand intercept (P = a - Q)
    c = 40        # marginal cost
    n_base = 2    # baseline: duopoly

    # -------------------------------------------------------------------------
    # 1. Compute profits under three scenarios for n=2
    # -------------------------------------------------------------------------
    q_compete, P_compete, pi_compete = cournot_nash_profits(n_base, a, c)
    q_collude, P_collude, pi_collude = collusion_profits(n_base, a, c)
    q_deviate, P_deviate, pi_deviate = deviation_profits(n_base, a, c)

    print("=== Symmetric Cournot Duopoly (a=100, c=40) ===")
    print(f"  Nash:      q={q_compete:.2f}, P={P_compete:.2f}, pi={pi_compete:.2f}")
    print(f"  Collusion: q={q_collude:.2f}, P={P_collude:.2f}, pi={pi_collude:.2f}")
    print(f"  Deviation: q={q_deviate:.2f}, P={P_deviate:.2f}, pi={pi_deviate:.2f}")

    delta_star_2 = critical_discount_factor(n_base, a, c)
    print(f"  Critical discount factor: delta* = {delta_star_2:.4f}")

    # -------------------------------------------------------------------------
    # 2. Critical discount factor as function of number of firms
    # -------------------------------------------------------------------------
    delta_reference = 0.9
    n_range = np.arange(2, 51)
    delta_stars = np.array([critical_discount_factor(n, a, c) for n in n_range])
    max_n_delta = maximum_sustainable_firms(delta_reference, a, c, max_n=500)

    # -------------------------------------------------------------------------
    # 3. Simulate price series with structural break
    # -------------------------------------------------------------------------
    T_compete, T_collude, T_detect = 30, 25, 20
    t_sim, prices_sim, mc_sim, regimes_sim = simulate_price_series(
        T_compete, T_collude, T_detect, a, c, n_base, sigma_noise=1.5
    )

    # Price-cost margin over time
    pcm_sim = (prices_sim - mc_sim) / prices_sim
    pcm_compete = (P_compete - c) / P_compete
    pcm_collude = (P_collude - c) / P_collude

    # =========================================================================
    # Generate Report
    # =========================================================================
    report = ModelReport(
        "Repeated-Game Cartels and Price Screens",
        "A price break becomes informative when it is tied to a repeated-game discipline condition.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "In many cartel cases the first visible fact is a price path. Prices rise "
        "together for a while, margins look unusually high, and prices fall after an "
        "investigation or a breakdown in coordination. That pattern matters, but the "
        "economic object is the agreement firms would have to maintain period after "
        "period. Each firm must prefer future cooperation to today's temptation to "
        "expand output while rivals keep output low.\n\n"
        "The tutorial uses a symmetric Cournot market so those incentives can be "
        "computed exactly. The calculation compares three payoffs: competitive "
        "Cournot profit, joint-monopoly profit split across firms, and the one-period "
        "deviation payoff. Those payoffs give the critical discount factor. A simulated "
        "price and margin path then shows how an empirical screen would look when the "
        "break is generated by the same competitive and monopoly benchmarks. The "
        "neighboring [HHI tutorial](../effective-hhi/) is a static concentration "
        "screen; here the screen is tied to a repeated-game model of conduct."
    )

    report.add_equations(
        r"""
Firms $i=1,\ldots,n$ choose quantities $q_i$. Total quantity is
$Q=\sum_i q_i$, inverse demand is $P(Q)=a-Q$, and all firms have constant
marginal cost $c<a$. Let $\delta\in(0,1)$ be the common discount factor.

| Regime | Per-firm quantity | Per-firm profit |
|---|---:|---:|
| Cournot-Nash | $q^N=\dfrac{a-c}{n+1}$ | $\pi^N=\left(\dfrac{a-c}{n+1}\right)^2$ |
| Joint monopoly split equally | $q^M=\dfrac{a-c}{2n}$ | $\pi^M=\dfrac{(a-c)^2}{4n}$ |
| One firm deviates while others collude | $q^D=\dfrac{(n+1)(a-c)}{4n}$ | $\pi^D=\dfrac{(n+1)^2(a-c)^2}{16n^2}$ |

A grim-trigger cartel colludes until a deviation is detected and then reverts to
Cournot-Nash forever. The value of staying in the cartel is

$$
V^M=\frac{\pi^M}{1-\delta},
$$

while the value of deviating once is

$$
V^D=\pi^D+\frac{\delta\pi^N}{1-\delta}.
$$

The incentive constraint $V^M\geq V^D$ is equivalent to

$$
\delta\geq
\delta^{*}
=\frac{\pi^D-\pi^M}{\pi^D-\pi^N}
=\frac{(n+1)^2}{n^2+6n+1}.
$$

The price-screen simulation uses the same exact benchmarks. It observes

$$
P_t=P^{r_t}+\eta_t,\qquad
r_t\in\{N,M,N\},
$$

where $P^N$ is the Cournot price, $P^M$ is the joint-monopoly price, and the
regime $r_t$ moves from competition to cartel conduct and then back after
detection. The reported margin is $m_t=(P_t-c)/P_t$.
"""
    )

    report.add_model_setup(
        "Think of a homogeneous input market where firms have similar costs and an "
        "investigator observes a long price series. The calibration keeps the "
        "incentives auditable by hand: the repeated-game object is analytical, while "
        "the simulated time series shows the price and margin breaks an empirical "
        "screen would flag.\n\n"
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| Demand intercept $a$ | {a} | Sets the competitive and monopoly price benchmarks |\n"
        f"| Marginal cost $c$ | {c} | Common cost used in profits and margins |\n"
        f"| Baseline firms | {n_base} | Duopoly used for the simulated price path |\n"
        f"| Firm-count grid | 2 to {n_range[-1]} | Exact cartel-stability thresholds by $n$ |\n"
        f"| Reference patience | $\\delta={delta_reference:.1f}$ | Used to mark which firm counts are sustainable |\n"
        f"| Regimes | {T_compete}+{T_collude}+{T_detect} periods | Competition, cartel, post-detection |\n"
        f"| Price noise | $\\sigma=1.5$ | Adds sampling noise around the exact regime price |\n\n"
        f"In the baseline duopoly, the Cournot price is {P_compete:.0f}, the "
        f"joint-monopoly price is {P_collude:.0f}, and a deviating firm earns "
        f"{pi_deviate:.1f} for one period if its rival stays with the cartel quantity. "
        "The screen is useful because it maps an observed break back to these model "
        "benchmarks rather than treating every high-price episode as equally suspicious."
    )

    report.add_solution_method(
        "The computation has two layers. The first layer evaluates the repeated-game "
        "incentive constraint across possible numbers of firms. The second layer uses "
        "the resulting Cournot and monopoly prices as reference lines for a simulated "
        "screen. This keeps the antitrust interpretation tied to the economic model: "
        "the price break is compared with prices that come from profit maximization, "
        "not with an arbitrary before-after average.\n\n"
        "```text\n"
        "Algorithm: repeated-Cournot cartel screen\n"
        "Input: demand intercept a, marginal cost c, firm-count grid N, discount factor delta\n"
        "Output: delta*(n), sustainability flags, price and margin benchmarks\n"
        "1. For each n in N, compute the symmetric Cournot payoff pi^N(n).\n"
        "2. Compute the equal-split joint-monopoly payoff pi^M(n).\n"
        "3. Let one firm best respond to the other n-1 firms' collusive quantities;\n"
        "   record the one-shot deviation payoff pi^D(n).\n"
        "4. Evaluate delta*(n) = [pi^D(n)-pi^M(n)] / [pi^D(n)-pi^N(n)].\n"
        "5. Mark collusion sustainable when delta >= delta*(n).\n"
        "6. For the baseline duopoly, simulate prices around P^N, then P^M,\n"
        "   then P^N again; compute margins m_t = (P_t-c)/P_t.\n"
        "```\n\n"
        f"With $\\delta={delta_reference:.1f}$, the exact threshold in this calibration "
        f"allows at most {max_n_delta} symmetric firms. The simulated path is a "
        "benchmark for reading a clean structural break before adding demand shocks, "
        "capacity constraints, procurement rules, or richer monitoring technology."
    )

    # --- Figure 1: Profits under three scenarios ---
    n_plot = np.arange(2, 11)
    pi_N = np.array([cournot_nash_profits(n, a, c)[2] for n in n_plot])
    pi_M = np.array([collusion_profits(n, a, c)[2] for n in n_plot])
    pi_D = np.array([deviation_profits(n, a, c)[2] for n in n_plot])

    fig1, ax1 = plt.subplots()
    ax1.plot(n_plot, pi_D, "rs-", markersize=6, label="Deviation $\\pi^D$")
    ax1.plot(n_plot, pi_M, "go-", markersize=6, label="Collusion $\\pi^M$")
    ax1.plot(n_plot, pi_N, "b^-", markersize=6, label="Nash $\\pi^N$")
    ax1.set_xlabel("Number of firms $n$")
    ax1.set_ylabel("Per-firm profit")
    ax1.set_title("Payoffs behind the cartel incentive constraint")
    ax1.legend()
    ax1.set_xticks(n_plot)
    report.add_figure(
        "figures/profits-by-regime.png",
        "Per-firm Nash, collusive, and deviation profits by firm count",
        fig1,
        description="The two sides of the incentive constraint are visible directly. "
        "The distance from $\\pi^M$ up to $\\pi^D$ is the short-run gain from cheating. "
        "The distance from $\\pi^N$ up to $\\pi^M$ is the per-period rent lost "
        "after punishment. Adding members dilutes the monopoly rent faster than it "
        "shrinks the deviation opportunity.",
    )

    # --- Figure 2: Critical discount factor vs number of firms ---
    fig2, ax2 = plt.subplots()
    ax2.plot(n_range, delta_stars, "ko-", markersize=5, linewidth=2)
    ax2.axhline(
        y=delta_reference,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"$\\delta = {delta_reference:.1f}$",
    )
    ax2.axhline(y=0.8, color="orange", linestyle="--", alpha=0.7, label="$\\delta = 0.8$")
    ax2.axvline(
        x=max_n_delta + 0.5,
        color="red",
        linestyle=":",
        alpha=0.5,
        label=f"max $n$ at $\\delta={delta_reference:.1f}$",
    )
    ax2.fill_between(n_range, delta_stars, 1.0, alpha=0.15, color="green",
                     label="Collusion sustainable")
    ax2.fill_between(n_range, 0, delta_stars, alpha=0.10, color="red",
                     label="Collusion breaks down")
    ax2.set_xlabel("Number of firms $n$")
    ax2.set_ylabel("Critical discount factor $\\delta^{*}$")
    ax2.set_title("Exact grim-trigger threshold by firm count")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)
    report.add_figure(
        "figures/critical-discount-factor.png",
        "Exact critical discount factor as a function of the number of firms",
        fig2,
        description=f"The threshold curve is exact for the linear Cournot model. "
        f"At $\\delta={delta_reference:.1f}$, the last sustainable symmetric market has "
        f"{max_n_delta} firms; adding one more pushes the deviation constraint "
        "above the reference discount factor. More firms leave less per-firm cartel "
        "rent to protect the agreement.",
    )

    # --- Figure 3: Simulated price series with structural break ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    # Color background by regime
    ax3.axvspan(0, T_compete - 1, alpha=0.10, color="blue", label="Competition")
    ax3.axvspan(T_compete, T_compete + T_collude - 1, alpha=0.10, color="red",
                label="Collusion")
    ax3.axvspan(T_compete + T_collude, T_compete + T_collude + T_detect - 1,
                alpha=0.10, color="green", label="Post-Detection")
    ax3.plot(t_sim, prices_sim, "k-", linewidth=1.5, label="Observed price")
    ax3.axhline(y=P_compete, color="blue", linestyle=":", alpha=0.8,
                label="Nash price")
    ax3.axhline(y=P_collude, color="red", linestyle=":", alpha=0.8,
                label="Monopoly price")
    ax3.axhline(y=c, color="gray", linestyle="--", alpha=0.4, label="Marginal cost")
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Price")
    ax3.set_title("Price break against exact Cournot benchmarks")
    ax3.legend(fontsize=9, loc="upper right")
    report.add_figure(
        "figures/price-series-structural-break.png",
        "Stylized price series with Nash and monopoly reference prices",
        fig3,
        description="The simulated price path is constructed with known regimes. Before "
        "the cartel, prices fluctuate around the exact Nash benchmark. During the cartel, "
        "they move toward the monopoly benchmark, and after detection they return to Nash. "
        "Real applications replace these clean reference lines with estimated costs, demand, "
        "and counterfactual competitive prices.",
    )

    # --- Figure 4: Price-cost margin over time ---
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    colors = {"Competition": "blue", "Collusion": "red", "Post-Detection": "green"}
    for regime in ["Competition", "Collusion", "Post-Detection"]:
        mask = regimes_sim == regime
        ax4.scatter(t_sim[mask], pcm_sim[mask], c=colors[regime], s=20,
                    label=regime, alpha=0.7)
    ax4.axhline(y=pcm_compete, color="blue", linestyle=":", alpha=0.8,
                label="Nash margin")
    ax4.axhline(y=pcm_collude, color="red", linestyle=":", alpha=0.8,
                label="Monopoly margin")
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Price-cost margin $(P-c)/P$")
    ax4.set_title("Margin break against exact Cournot benchmarks")
    ax4.legend(fontsize=9)
    report.add_figure(
        "figures/price-cost-margin.png",
        "Stylized price-cost margin with Nash and monopoly reference margins",
        fig4,
        description="The margin version of the same screen removes the level of marginal cost "
        "from the price comparison. In this simple run cost is constant, so the margin break "
        "adds no identification by itself. In field data, the margin view is useful because "
        "cartel allegations usually have to separate conduct from cost shocks.",
    )

    # --- Table: Cartel stability conditions for different market structures ---
    table_n = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 33, 34, 40, 50]
    table_data = {
        "n": table_n,
        "pi_N": [f"{cournot_nash_profits(n, a, c)[2]:.1f}" for n in table_n],
        "pi_M": [f"{collusion_profits(n, a, c)[2]:.1f}" for n in table_n],
        "pi_D": [f"{deviation_profits(n, a, c)[2]:.1f}" for n in table_n],
        "delta_star": [f"{critical_discount_factor(n, a, c):.4f}" for n in table_n],
        "delta_0.9_sustains": [
            "yes" if critical_discount_factor(n, a, c) <= delta_reference else "no"
            for n in table_n
        ],
    }
    df_table = pd.DataFrame(table_data)
    report.add_table(
        "tables/cartel-stability.csv",
        "Exact Cartel Stability Conditions ($a=100$, $c=40$)",
        df_table,
        description=f"The table reports exact payoffs and thresholds. For $\\delta={delta_reference:.1f}$, "
        f"the feasibility cutoff lies between {max_n_delta} and {max_n_delta + 1} firms. "
        "The high-$n$ rows show how quickly the incentive constraint tightens as the "
        "cartel has to divide monopoly rents among more members.",
    )

    # --- Economic takeaway ---
    report.add_takeaway(
        "Cartel detection needs both a behavior pattern and an incentive story. In this "
        "simple market, the incentive story reduces to one discipline condition: future "
        "collusive rents must be valuable enough to offset the one-period gain from "
        "cheating. "
        f"In the duopoly, $\\delta^{{*}}={delta_star_2:.4f}$; with ten symmetric firms, "
        f"$\\delta^{{*}}={critical_discount_factor(10, a, c):.4f}$; at "
        f"$\\delta={delta_reference:.1f}$, the cutoff is {max_n_delta} firms. "
        "Price and margin breaks point to periods worth investigating. They still need "
        "cost evidence, demand shocks, monitoring, communication, capacity, procurement "
        "rules, and the legal record before they become a cartel case."
    )

    report.add_references([
        "Stigler, G. (1964). A Theory of Oligopoly. *Journal of Political Economy*, 72(1), 44--61.",
        "Porter, R. (1983). A Study of Cartel Stability: The Joint Executive Committee, 1880--1886. *Bell Journal of Economics*, 14(2), 301--314.",
        "Harrington, J. (2008). Detecting Cartels. In *Handbook of Antitrust Economics*. MIT Press.",
        "Igami, M. and Sugaya, T. (2021). Measuring the Incentive to Collude: The Vitamin Cartels, 1990--1999. *Review of Economic Studies*, 89(3), 1460--1494.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
