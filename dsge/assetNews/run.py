#!/usr/bin/env python3
"""Lucas-tree asset pricing with news shocks.

The accompanying ``model.mod`` spec records a representative-agent Lucas tree
in which dividends are also consumption; a news shock changes the expected
dividend before the dividend itself moves. The spec is documentation only;
``run.py`` does not execute it. The Python report solves the first-order
pricing equation directly, cross-checks the linear coefficients against
Klein (2000) generalized Schur (QZ) decomposition, and compares the linear
response with an exact nonlinear perfect-foresight transition for the same
deterministic shock path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.perturbation import solve_klein
from lib.plotting import save_figure, save_thumbnail, setup_style


@dataclass(frozen=True)
class AssetNewsParams:
    """Calibration shared with the textbook ``model.mod`` spec."""

    beta: float = 0.99
    gamma: float = 2.0
    rho: float = 0.9
    sigma1: float = 0.1
    sigma2: float = 0.1


def read_mod_file(mod_path: Path) -> str:
    """Return the model spec text. Documentation only; not executed."""

    return mod_path.read_text()


def klein_qz_pricing(params: AssetNewsParams) -> dict[str, float]:
    """Cross-check the linear pricing coefficients with Klein-style QZ.

    State ordering ``s_t = (x_t, n_t, q_t)`` with the dividend log ``x`` and
    the i.i.d. news draw ``n`` predetermined and the price gap ``q``
    forward-looking.
    """
    beta = params.beta
    gamma = params.gamma
    rho = params.rho
    sigma1 = params.sigma1
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [(1.0 - beta - gamma), 0.0, beta],
        ]
    )
    B = np.array(
        [
            [rho, sigma1, 0.0],
            [0.0, 0.0, 0.0],
            [-gamma, 0.0, 1.0],
        ]
    )
    sol = solve_klein(A, B, n_predetermined=2)
    return {
        "A": float(sol.P[0, 0]),
        "B": float(sol.P[0, 1]),
        "blanchard_kahn": sol.bk_message,
        "eigenvalues": sol.eigenvalues,
    }


def steady_state(params: AssetNewsParams) -> dict[str, float]:
    """Compute the deterministic steady state of the Lucas tree."""

    d_ss = 1.0
    p_ss = params.beta / (1.0 - params.beta)
    return {
        "d": d_ss,
        "p": p_ss,
        "pd_ratio": p_ss / d_ss,
        "gross_return": 1.0 / params.beta,
    }


def linear_price_coefficients(params: AssetNewsParams) -> dict[str, float]:
    """Solve q_t = A x_t + B n_t for the first-order price response.

    Here x_t = log d_t and q_t = log(p_t / p_ss). Current news n_t is known at
    date t and shifts x_{t+1} by sigma1.
    """

    beta = params.beta
    gamma = params.gamma
    rho = params.rho
    sigma1 = params.sigma1

    a_x = (gamma + rho * (1.0 - beta - gamma)) / (1.0 - beta * rho)
    b_news = sigma1 * (beta * a_x + 1.0 - beta - gamma)
    return {"A": a_x, "B": b_news}


def dividend_news_state(
    params: AssetNewsParams,
    shock_type: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the deterministic dividend and news-state paths for an IRF."""

    x = np.zeros(horizon)
    n = np.zeros(horizon)

    if shock_type == "surprise":
        x[0] = params.sigma2
    elif shock_type == "news":
        n[0] = 1.0
    else:
        raise ValueError(f"Unknown shock type: {shock_type}")

    for t in range(1, horizon):
        x[t] = params.rho * x[t - 1] + params.sigma1 * n[t - 1]

    return x, n


def nonlinear_price_benchmark(
    params: AssetNewsParams,
    x: np.ndarray,
    extra_periods: int = 260,
) -> np.ndarray:
    """Exact nonlinear perfect-foresight price path for a deterministic x path."""

    ss = steady_state(params)
    x_long = np.zeros(len(x) + extra_periods)
    x_long[: len(x)] = x
    for t in range(len(x), len(x_long)):
        x_long[t] = params.rho * x_long[t - 1]

    d = np.exp(x_long)
    p = np.zeros_like(d)
    p[-1] = ss["p"] * d[-1]

    for t in range(len(d) - 2, -1, -1):
        sdf = params.beta * (d[t + 1] / d[t]) ** (-params.gamma)
        p[t] = sdf * (p[t + 1] + d[t + 1])

    return np.log(p[: len(x)] / ss["p"])


def compute_irf(
    params: AssetNewsParams,
    shock_type: str,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Compute first-order and nonlinear benchmark responses to one shock."""

    ss = steady_state(params)
    coeffs = linear_price_coefficients(params)
    x, n = dividend_news_state(params, shock_type, horizon)
    q_linear = coeffs["A"] * x + coeffs["B"] * n
    q_nonlinear = nonlinear_price_benchmark(params, x)

    return {
        "x": x,
        "n": n,
        "q_linear": q_linear,
        "q_nonlinear": q_nonlinear,
        "d": np.exp(x),
        "p_linear": ss["p"] * np.exp(q_linear),
        "pd_linear": q_linear - x,
    }


def simulate_paths(
    params: AssetNewsParams,
    horizon: int = 220,
    seed: int = 20260504,
) -> dict[str, np.ndarray]:
    """Simulate the first-order model with both surprise and news shocks."""

    rng = np.random.default_rng(seed)
    z = rng.normal(size=horizon)
    n = rng.normal(size=horizon)
    x = np.zeros(horizon)
    coeffs = linear_price_coefficients(params)

    x[0] = params.sigma2 * z[0]
    for t in range(1, horizon):
        x[t] = params.rho * x[t - 1] + params.sigma1 * n[t - 1] + params.sigma2 * z[t]

    q = coeffs["A"] * x + coeffs["B"] * n
    return {
        "x": x,
        "q": q,
        "z": z,
        "n": n,
        "news_contribution": params.sigma1 * n,
        "surprise_contribution": params.sigma2 * z,
    }


def signed_percent(value: float) -> str:
    """Format a log deviation as percent."""

    return f"{100.0 * value:.3f}"


def main() -> None:
    mod_dir = Path(__file__).resolve().parent
    mod_text = read_mod_file(mod_dir / "model.mod")
    params = AssetNewsParams()
    ss = steady_state(params)
    coeffs = linear_price_coefficients(params)
    qz = klein_qz_pricing(params)
    qz_diff = max(abs(coeffs["A"] - qz["A"]), abs(coeffs["B"] - qz["B"]))

    print("Read model.mod (Lucas-tree spec; not executed by run.py).")
    print(f"  Source length: {len(mod_text.splitlines())} lines")
    print(f"  Steady-state price-dividend ratio: {ss['pd_ratio']:.2f}")
    print(f"  First-order price rule: q_t = {coeffs['A']:.4f} x_t + {coeffs['B']:.4f} n_t")
    print(f"  Klein QZ ({qz['blanchard_kahn']}): max abs diff = {qz_diff:.2e}")

    horizon = 40
    surprise = compute_irf(params, "surprise", horizon)
    news = compute_irf(params, "news", horizon)
    sim = simulate_paths(params)

    setup_style()
    periods = np.arange(horizon)

    fig1, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    shock_panels = [
        ("Surprise shock", surprise),
        ("News shock", news),
    ]
    for col, (title, irf) in enumerate(shock_panels):
        ax = axes[0, col]
        ax.plot(periods, 100.0 * irf["x"], color="#1f77b4", linewidth=2.5)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
        ax.set_title(title)
        ax.set_ylabel("Dividend, percent log dev.")
        if title == "News shock":
            ax.axvline(1, color="gray", linestyle=":", linewidth=1.1)

        ax = axes[1, col]
        ax.plot(
            periods,
            100.0 * irf["q_linear"],
            color="#b2182b",
            linewidth=2.5,
            label="First-order price",
        )
        ax.plot(
            periods,
            100.0 * irf["q_nonlinear"],
            color="black",
            linewidth=1.8,
            linestyle="--",
            label="Nonlinear benchmark",
        )
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
        if title == "News shock":
            ax.axvline(1, color="gray", linestyle=":", linewidth=1.1)
        ax.set_xlabel("Quarters after shock")
        ax.set_ylabel("Price, percent log dev.")

    axes[1, 0].legend(frameon=False, loc="upper right")
    fig1.suptitle("Dividend Timing and Asset Prices", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig1, "figures/irf-surprise-vs-news.png", dpi=150)

    component_values = pd.DataFrame(
        {
            "Channel": [
                "Continuation price",
                "Next dividend payoff",
                "Marginal utility discounting",
                "Net impact",
            ],
            "Contribution": [
                params.beta * coeffs["A"] * params.sigma1,
                (1.0 - params.beta) * params.sigma1,
                -params.gamma * params.sigma1,
                coeffs["B"],
            ],
        }
    )

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    colors = ["#1b6ca8" if v >= 0 else "#b2182b" for v in component_values["Contribution"]]
    ax2.bar(component_values["Channel"], 100.0 * component_values["Contribution"], color=colors)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Date-0 log price contribution, percent")
    ax2.set_title("Why Good Dividend News Need Not Raise Today's Price")
    ax2.tick_params(axis="x", labelrotation=18)
    for idx, value in enumerate(100.0 * component_values["Contribution"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.35 if value >= 0 else -0.35
        ax2.text(idx, value + offset, f"{value:.2f}", ha="center", va=va, fontsize=9)
    fig2.tight_layout()
    save_figure(fig2, "figures/price-dynamics.png", dpi=150)

    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t_sim = np.arange(len(sim["x"]))
    axes3[0].plot(
        t_sim,
        100.0 * sim["x"],
        color="#1f77b4",
        linewidth=1.4,
        label="Dividend",
    )
    axes3[0].plot(
        t_sim,
        100.0 * sim["q"],
        color="#b2182b",
        linewidth=1.4,
        label="Asset price",
    )
    axes3[0].axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    axes3[0].set_ylabel("Percent log dev.")
    axes3[0].set_title("Simulated Dividend and Price Deviations")
    axes3[0].legend(frameon=False, loc="upper right")

    axes3[1].plot(
        t_sim,
        100.0 * sim["surprise_contribution"],
        color="#4b8f29",
        linewidth=1.0,
        alpha=0.8,
        label=r"Surprise contribution $\sigma_2 z_t$",
    )
    axes3[1].plot(
        t_sim,
        100.0 * sim["news_contribution"],
        color="#6f4aa8",
        linewidth=1.0,
        alpha=0.8,
        label=r"News contribution $\sigma_1 n_t$ to $x_{t+1}$",
    )
    axes3[1].axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    axes3[1].set_xlabel("Quarter")
    axes3[1].set_ylabel("Percent log points")
    axes3[1].set_title("Innovations Feeding the Dividend Process")
    axes3[1].legend(frameon=False, loc="upper right")
    fig3.tight_layout()
    save_figure(fig3, "figures/simulated-paths.png", dpi=150)

    impact_table = pd.DataFrame(
        [
            {
                "Object": "Dividend log deviation",
                "Surprise t=0": signed_percent(surprise["x"][0]),
                "News t=0": signed_percent(news["x"][0]),
                "News t=1": signed_percent(news["x"][1]),
            },
            {
                "Object": "Price log deviation, first order",
                "Surprise t=0": signed_percent(surprise["q_linear"][0]),
                "News t=0": signed_percent(news["q_linear"][0]),
                "News t=1": signed_percent(news["q_linear"][1]),
            },
            {
                "Object": "Price log deviation, nonlinear benchmark",
                "Surprise t=0": signed_percent(surprise["q_nonlinear"][0]),
                "News t=0": signed_percent(news["q_nonlinear"][0]),
                "News t=1": signed_percent(news["q_nonlinear"][1]),
            },
            {
                "Object": "Price-dividend ratio log deviation",
                "Surprise t=0": signed_percent(surprise["pd_linear"][0]),
                "News t=0": signed_percent(news["pd_linear"][0]),
                "News t=1": signed_percent(news["pd_linear"][1]),
            },
        ]
    )

    Path("tables").mkdir(parents=True, exist_ok=True)
    impact_table.to_csv("tables/impact-responses.csv", index=False)

    # Thumbnail from first figure
    save_thumbnail("figures/irf-surprise-vs-news.png", "figures/thumb.png")

    print(f"Saved 3 figures and 1 table.")


if __name__ == "__main__":
    main()
