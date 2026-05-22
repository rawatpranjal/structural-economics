#!/usr/bin/env python3
"""Pandora's Box: Optimal Sequential Search and the Weitzman Reservation-Value Rule.

A buyer faces J=6 boxes. Each box j has a Gaussian value distribution
V_j ~ N(mu_j, sigma_j^2) and an inspection cost c_j. Weitzman's (1979)
reservation value z_j summarises box j: open boxes in decreasing z_j order
and stop when the current best weakly exceeds every uninspected z_j.

Simulates the optimal Weitzman policy and a myopic benchmark (open in
decreasing mu_j order, same stopping condition) over N_SIM trials and
compares payoff distributions and inspection counts.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

J: int = 6
MUS: list[float] = [3.0, 2.5, 2.8, 2.0, 2.3, 1.8]
SIGMAS: list[float] = [1.0, 1.5, 0.8, 1.2, 0.9, 1.1]
COSTS: list[float] = [0.5, 0.4, 0.6, 0.3, 0.4, 0.2]

N_SIM: int = 1000
SEED: int = 0

# Color palette
COLOR_OPT: str = "#4878CF"
COLOR_MYO: str = "#D65F5F"


# ---------------------------------------------------------------------------
# Reservation-value computation
# ---------------------------------------------------------------------------

def _option_value(z: float, mu: float, sigma: float) -> float:
    """E[max(V - z, 0)] for V ~ N(mu, sigma^2).

    Uses the closed-form: sigma * phi((mu - z)/sigma) + (mu - z) * Phi((z - mu)/sigma complement).
    Written as:
        E[max(V - z, 0)] = sigma * phi(d) + (mu - z) * (1 - Phi(-d))
    where d = (mu - z) / sigma and phi, Phi are the standard normal pdf and cdf.
    """
    d = (mu - z) / sigma
    return sigma * norm.pdf(d) + (mu - z) * norm.cdf(d)


def reservation_value(mu: float, sigma: float, c: float) -> float:
    """Solve c = E[max(V - z, 0)] for z via Brentq.

    The option value E[max(V - z, 0)] is strictly decreasing in z (from
    E[V] - z_low at a large negative z_low toward 0 as z -> +inf), so a
    unique root exists.

    Parameters
    ----------
    mu:    mean of V ~ N(mu, sigma^2)
    sigma: standard deviation of V
    c:     inspection cost

    Returns
    -------
    z: reservation value
    """
    # g(z) = E[max(V - z, 0)] - c; strictly decreasing in z.
    def g(z: float) -> float:
        return _option_value(z, mu, sigma) - c

    # Bracket: option_value(mu - 10*sigma) >> c; option_value(mu + 10*sigma) ~ 0 < c.
    lo = mu - 10.0 * sigma
    hi = mu + 10.0 * sigma
    # Ensure bracket is valid.
    while g(lo) < 0:
        lo -= sigma
    while g(hi) > 0:
        hi += sigma
    return float(brentq(g, lo, hi, xtol=1e-12, rtol=1e-12))


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_optimal_search(
    rng: np.random.Generator,
    mus: np.ndarray,
    sigmas: np.ndarray,
    costs: np.ndarray,
    zs: np.ndarray,
) -> tuple[float, int, list[int]]:
    """Simulate one trial under the Weitzman optimal policy.

    Steps:
      1. Draw V_j ~ N(mu_j, sigma_j) for all j (but reveal only upon opening).
      2. Maintain a priority queue sorted by z_j descending.
      3. At each step: if current best b >= max uninspected z_j, stop.
         Otherwise open the highest-z uninspected box, pay cost, update b.
    Returns (net_payoff, n_opens, opening_order).
    """
    J = len(mus)
    values = rng.normal(mus, sigmas)  # realized values, shape (J,)

    uninspected = list(np.argsort(-zs))  # decreasing z_j order
    best = -np.inf
    total_cost = 0.0
    order: list[int] = []

    while uninspected:
        # Stopping condition: best >= max z over uninspected.
        max_z_uninspected = max(zs[j] for j in uninspected)
        if best >= max_z_uninspected:
            break
        # Open the highest-z uninspected box.
        j = uninspected.pop(0)
        total_cost += costs[j]
        order.append(j)
        if values[j] > best:
            best = values[j]

    net_payoff = best - total_cost
    return float(net_payoff), len(order), order


def simulate_myopic_search(
    rng: np.random.Generator,
    mus: np.ndarray,
    sigmas: np.ndarray,
    costs: np.ndarray,
) -> tuple[float, int, list[int]]:
    """Simulate one trial under the myopic policy.

    Opens boxes in decreasing mu_j order. Stops when the current best b
    weakly exceeds the largest mu_j among all uninspected boxes.
    Returns (net_payoff, n_opens, opening_order).
    """
    J = len(mus)
    values = rng.normal(mus, sigmas)  # realized values, shape (J,)

    uninspected = list(np.argsort(-mus))  # decreasing mu_j order
    best = -np.inf
    total_cost = 0.0
    order: list[int] = []

    while uninspected:
        # Stopping condition: best >= max mu over uninspected.
        max_mu_uninspected = max(mus[j] for j in uninspected)
        if best >= max_mu_uninspected:
            break
        j = uninspected.pop(0)
        total_cost += costs[j]
        order.append(j)
        if values[j] > best:
            best = values[j]

    net_payoff = best - total_cost
    return float(net_payoff), len(order), order


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_reservation_values(
    mus: np.ndarray,
    sigmas: np.ndarray,
    costs: np.ndarray,
    zs: np.ndarray,
    path: str,
) -> None:
    """Bar chart of z_j and c_j per box, sorted by z descending.

    Each bar group shows the reservation value z_j and the inspection cost c_j.
    Box labels annotate mu_j and sigma_j for reference.
    """
    J = len(zs)
    order = np.argsort(-zs)  # decreasing z

    zs_sorted = zs[order]
    cs_sorted = costs[order]
    mus_sorted = mus[order]
    sigs_sorted = sigmas[order]
    labels = [f"Box {order[i] + 1}" for i in range(J)]

    x = np.arange(J)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_z = ax.bar(x - width / 2, zs_sorted, width, label="Reservation value $z_j$",
                    color=COLOR_OPT, alpha=0.85)
    bars_c = ax.bar(x + width / 2, cs_sorted, width, label="Inspection cost $c_j$",
                    color=COLOR_MYO, alpha=0.85)

    # Annotate mu and sigma above each z bar.
    for i, (bar, mu_i, sig_i) in enumerate(zip(bars_z, mus_sorted, sigs_sorted)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03,
            f"mu={mu_i:.1f}\nsig={sig_i:.1f}",
            ha="center", va="bottom", fontsize=8, color="0.3",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Weitzman Reservation Values and Inspection Costs (sorted by $z_j$)")
    ax.legend(frameon=False)
    save_figure(fig, path, dpi=150)


def plot_payoff_distribution(
    payoffs_opt: np.ndarray,
    payoffs_myo: np.ndarray,
    path: str,
) -> None:
    """Overlapping histograms of net payoff under optimal vs myopic policy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(payoffs_opt.min(), payoffs_myo.min()),
        max(payoffs_opt.max(), payoffs_myo.max()),
        40,
    )

    ax.hist(payoffs_opt, bins=bins, alpha=0.55, color=COLOR_OPT,
            label="Optimal (Weitzman)", density=True)
    ax.hist(payoffs_myo, bins=bins, alpha=0.55, color=COLOR_MYO,
            label="Myopic (decreasing mu)", density=True)

    mean_opt = float(payoffs_opt.mean())
    mean_myo = float(payoffs_myo.mean())
    ax.axvline(mean_opt, color=COLOR_OPT, linestyle="--", linewidth=2,
               label=f"Optimal mean = {mean_opt:.3f}")
    ax.axvline(mean_myo, color=COLOR_MYO, linestyle="--", linewidth=2,
               label=f"Myopic mean = {mean_myo:.3f}")

    ax.set_xlabel("Net payoff (best value minus total inspection cost)")
    ax.set_ylabel("Density")
    ax.set_title("Payoff Distribution: Optimal vs Myopic Search")
    ax.legend(frameon=False)
    save_figure(fig, path, dpi=150)


def plot_inspection_counts(
    counts_opt: np.ndarray,
    counts_myo: np.ndarray,
    J: int,
    path: str,
) -> None:
    """Bar chart of mean boxes opened per trial under each policy."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Optimal (Weitzman)", "Myopic (decreasing mu)"]
    means = [float(counts_opt.mean()), float(counts_myo.mean())]
    sems = [
        float(counts_opt.std(ddof=1) / np.sqrt(len(counts_opt))),
        float(counts_myo.std(ddof=1) / np.sqrt(len(counts_myo))),
    ]
    colors = [COLOR_OPT, COLOR_MYO]

    x = np.arange(2)
    bars = ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, width=0.45,
                  capsize=5, error_kw={"linewidth": 1.5})

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean + sems[bars.index(bar)] + 0.04,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean boxes opened per trial")
    ax.set_ylim(0, J + 0.5)
    ax.set_title(f"Mean Inspection Counts over {N_SIM} Trials (J = {J} boxes)")
    save_figure(fig, path, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()
    t0 = time.perf_counter()

    Path("figures").mkdir(parents=True, exist_ok=True)

    mus = np.array(MUS)
    sigmas = np.array(SIGMAS)
    costs = np.array(COSTS)

    # --- Solve reservation values ---
    zs = np.array([reservation_value(mus[j], sigmas[j], costs[j]) for j in range(J)])

    # --- Print box table ---
    print(f"{'Box':>4}  {'mu':>6}  {'sigma':>6}  {'c':>6}  {'z':>8}")
    print("-" * 38)
    for j in range(J):
        print(f"{j + 1:>4}  {mus[j]:>6.2f}  {sigmas[j]:>6.2f}  {costs[j]:>6.2f}  {zs[j]:>8.4f}")
    print()

    # Weitzman opening order (decreasing z).
    opt_order = np.argsort(-zs)
    print("Optimal opening order (decreasing z_j):",
          " -> ".join(f"Box {j + 1} (z={zs[j]:.3f})" for j in opt_order))
    print()

    # --- Simulate ---
    rng = np.random.default_rng(SEED)

    payoffs_opt = np.empty(N_SIM)
    opens_opt = np.empty(N_SIM, dtype=int)

    payoffs_myo = np.empty(N_SIM)
    opens_myo = np.empty(N_SIM, dtype=int)

    for i in range(N_SIM):
        payoff, n, _ = simulate_optimal_search(rng, mus, sigmas, costs, zs)
        payoffs_opt[i] = payoff
        opens_opt[i] = n

        payoff, n, _ = simulate_myopic_search(rng, mus, sigmas, costs)
        payoffs_myo[i] = payoff
        opens_myo[i] = n

    # --- Policy summary ---
    se_opt = float(payoffs_opt.std(ddof=1) / np.sqrt(N_SIM))
    se_myo = float(payoffs_myo.std(ddof=1) / np.sqrt(N_SIM))

    print(f"{'Policy':<26}  {'Mean payoff':>12}  {'SE':>8}  {'Mean opens':>11}")
    print("-" * 65)
    print(f"{'Optimal (Weitzman)':<26}  {payoffs_opt.mean():>12.4f}  {se_opt:>8.4f}  {opens_opt.mean():>11.3f}")
    print(f"{'Myopic (decreasing mu)':<26}  {payoffs_myo.mean():>12.4f}  {se_myo:>8.4f}  {opens_myo.mean():>11.3f}")
    print()

    gain = float(payoffs_opt.mean() - payoffs_myo.mean())
    print(f"Mean payoff gain from optimal policy: {gain:+.4f}")
    print()

    # --- Figures ---
    plot_reservation_values(
        mus, sigmas, costs, zs,
        path="figures/reservation-values.png",
    )
    plot_payoff_distribution(
        payoffs_opt, payoffs_myo,
        path="figures/payoff-distribution.png",
    )
    plot_inspection_counts(
        opens_opt, opens_myo, J,
        path="figures/inspection-counts.png",
    )
    save_thumbnail("figures/reservation-values.png", "figures/thumb.png")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s. Figures written to figures/")


if __name__ == "__main__":
    main()
