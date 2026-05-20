#!/usr/bin/env python3
"""Perturbation around a macro steady state.

The tutorial compares first-, second-, and third-order local approximations to
a nonlinear state transition. The example stays small so readers can see how
local Taylor terms change impulse responses near a steady state.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


RHO = 0.82
GAMMA = 0.45
ETA = 0.80
KAPPA = 0.35
SHOCK_SIZE = 0.18


def exact_transition(x: np.ndarray | float) -> np.ndarray | float:
    """Nonlinear transition rule around steady state x=0."""
    z = np.asarray(x)
    return RHO * z + GAMMA * z**2 - ETA * z**3 + KAPPA * z**4


def perturbation_transition(x: np.ndarray | float, order: int) -> np.ndarray | float:
    """Taylor approximation to the transition rule around x=0."""
    z = np.asarray(x)
    out = RHO * z
    if order >= 2:
        out = out + GAMMA * z**2
    if order >= 3:
        out = out - ETA * z**3
    return out


def impulse_response(initial_state: float, order: int | None, periods: int = 28) -> np.ndarray:
    """Iterate the exact or approximated transition after an initial shock."""
    path = np.empty(periods, dtype=float)
    path[0] = initial_state
    for t in range(1, periods):
        if order is None:
            path[t] = exact_transition(path[t - 1])
        else:
            path[t] = perturbation_transition(path[t - 1], order)
    return path


def local_error_table(x_grid: np.ndarray) -> pd.DataFrame:
    """Summarize approximation errors over local and wider domains."""
    rows = []
    exact = exact_transition(x_grid)
    domains = {
        "local abs(x) <= 0.20": np.abs(x_grid) <= 0.20,
        "wide abs(x) <= 0.60": np.abs(x_grid) <= 0.60,
    }
    for order in [1, 2, 3]:
        approx = perturbation_transition(x_grid, order)
        error = np.abs(approx - exact)
        positive_irf = impulse_response(SHOCK_SIZE, order)
        negative_irf = impulse_response(-SHOCK_SIZE, order)
        exact_positive = impulse_response(SHOCK_SIZE, None)
        exact_negative = impulse_response(-SHOCK_SIZE, None)
        for domain, mask in domains.items():
            rows.append(
                {
                    "Order": order,
                    "Domain": domain,
                    "Max map error": np.max(error[mask]),
                    "Median map error": np.median(error[mask]),
                    "Positive IRF RMSE": np.sqrt(np.mean((positive_irf - exact_positive) ** 2)),
                    "Negative IRF RMSE": np.sqrt(np.mean((negative_irf - exact_negative) ** 2)),
                }
            )
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format accuracy table."""
    out = df.copy()
    for col in out.columns:
        if col in {"Order", "Domain"}:
            continue
        out[col] = out[col].map(lambda x: f"{float(x):.2e}")
    return out


def main() -> None:
    setup_style()
    x_grid = np.linspace(-0.6, 0.6, 500)
    exact = exact_transition(x_grid)
    table = local_error_table(x_grid)
    periods = 28
    positive_paths = {
        "exact": impulse_response(SHOCK_SIZE, None, periods),
        "first order": impulse_response(SHOCK_SIZE, 1, periods),
        "second order": impulse_response(SHOCK_SIZE, 2, periods),
        "third order": impulse_response(SHOCK_SIZE, 3, periods),
    }
    negative_paths = {
        "exact": impulse_response(-SHOCK_SIZE, None, periods),
        "first order": impulse_response(-SHOCK_SIZE, 1, periods),
        "second order": impulse_response(-SHOCK_SIZE, 2, periods),
        "third order": impulse_response(-SHOCK_SIZE, 3, periods),
    }

    print("Perturbation and linearization tutorial")
    print(f"  shock size={SHOCK_SIZE:.2f}")
    print(
        "  first-order positive IRF RMSE="
        f"{np.sqrt(np.mean((positive_paths['first order'] - positive_paths['exact']) ** 2)):.2e}"
    )
    print(
        "  third-order positive IRF RMSE="
        f"{np.sqrt(np.mean((positive_paths['third order'] - positive_paths['exact']) ** 2)):.2e}"
    )

    fig1, ax1 = plt.subplots()
    ax1.plot(x_grid, exact, color="black", linewidth=2.4, label="exact")
    for order in [1, 2, 3]:
        ax1.plot(x_grid, perturbation_transition(x_grid, order), label=f"order {order}")
    ax1.axvspan(-0.2, 0.2, color="gray", alpha=0.12, label="local region")
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("State x")
    ax1.set_ylabel("Next state")
    ax1.set_title("Adjustment Map Near the Steady State")
    ax1.legend()
    save_figure(fig1, "figures/local-approximations.png", dpi=150)

    fig2, ax2 = plt.subplots()
    for order in [1, 2, 3]:
        error = np.abs(perturbation_transition(x_grid, order) - exact)
        ax2.semilogy(np.abs(x_grid), np.maximum(error, 1e-14), label=f"order {order}")
    ax2.set_xlabel("Distance from steady state")
    ax2.set_ylabel("Absolute map error")
    ax2.set_title("Approximation Error by Distance")
    ax2.legend()
    save_figure(fig2, "figures/local-errors.png", dpi=150)

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    time = np.arange(periods)
    styles = {
        "exact": {"color": "black", "linewidth": 2.4},
        "first order": {"color": "tab:blue"},
        "second order": {"color": "tab:orange"},
        "third order": {"color": "tab:green"},
    }
    for label, path in positive_paths.items():
        axes3[0].plot(time, path, label=label, **styles[label])
    for label, path in negative_paths.items():
        axes3[1].plot(time, path, label=label, **styles[label])
    axes3[0].set_title("Positive Shock")
    axes3[1].set_title("Negative Shock")
    for ax in axes3:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Period")
    axes3[0].set_ylabel("State response")
    axes3[1].legend()
    fig3.tight_layout()
    save_figure(fig3, "figures/impulse-responses.png", dpi=150)

    asymmetry = positive_paths["exact"] + negative_paths["exact"]
    fig4, ax4 = plt.subplots()
    ax4.plot(time, asymmetry, color="black", label="exact")
    for label in ["first order", "second order", "third order"]:
        ax4.plot(time, positive_paths[label] + negative_paths[label], label=label)
    ax4.axhline(0.0, color="black", linewidth=0.8)
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Positive IRF + negative IRF")
    ax4.set_title("Asymmetry Diagnostic")
    ax4.legend()
    save_figure(fig4, "figures/asymmetry.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    format_table(table).to_csv("tables/approximation-errors.csv", index=False)

    save_thumbnail("figures/local-approximations.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
