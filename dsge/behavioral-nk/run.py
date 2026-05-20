#!/usr/bin/env python3
"""Behavioral New Keynesian tutorial with cognitive discounting.

The page compares the rational three-equation New Keynesian model with a
behavioral variant in which agents discount expected future output and
inflation. The code is deliberately small: coefficient matching solves the
current-shock experiment, and a finite-horizon recursion solves forward
guidance news.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


@dataclass(frozen=True)
class Calibration:
    """Quarterly calibration for the linearized NK block."""

    sigma: float = 1.0
    beta: float = 0.99
    kappa: float = 0.10
    phi_pi: float = 1.50
    phi_x: float = 0.125
    rho_v: float = 0.50
    shock_size: float = 0.01
    irf_horizon: int = 32
    news_horizon: int = 20


@dataclass(frozen=True)
class AttentionSetting:
    """Cognitive-discounting parameters for households and firms."""

    label: str
    m: float
    m_f: float
    color: str


def solve_policy_coefficients(
    calibration: Calibration,
    setting: AttentionSetting,
) -> dict[str, float]:
    """Solve responses to an AR(1) Taylor-rule wedge by coefficient matching.

    Guess ``x_t = psi_x v_t`` and ``pi_t = psi_pi v_t``. Since
    ``E_t v_{t+1} = rho_v v_t``, the IS curve and Phillips curve reduce to a
    two-equation linear system in ``psi_x`` and ``psi_pi``.
    """
    c = calibration
    rho = c.rho_v

    lhs = np.array(
        [
            [1.0 - setting.m * rho + c.sigma * c.phi_x, c.sigma * (c.phi_pi - rho)],
            [-c.kappa, 1.0 - c.beta * setting.m_f * rho],
        ]
    )
    rhs = np.array([-c.sigma, 0.0])
    psi_x, psi_pi = np.linalg.solve(lhs, rhs)
    psi_i = c.phi_pi * psi_pi + c.phi_x * psi_x + 1.0

    return {
        "psi_x": float(psi_x),
        "psi_pi": float(psi_pi),
        "psi_i": float(psi_i),
    }


def current_policy_irf(
    calibration: Calibration,
    coeffs: dict[str, float],
) -> dict[str, np.ndarray]:
    """Build IRFs for a persistent current monetary-policy wedge."""
    periods = np.arange(calibration.irf_horizon)
    shock = calibration.shock_size * calibration.rho_v**periods

    return {
        "periods": periods,
        "shock": shock,
        "output": coeffs["psi_x"] * shock,
        "inflation": coeffs["psi_pi"] * shock,
        "policy_rate": coeffs["psi_i"] * shock,
    }


def solve_one_period(
    calibration: Calibration,
    setting: AttentionSetting,
    next_output: float,
    next_inflation: float,
    policy_wedge: float,
) -> tuple[float, float]:
    """Solve date t output and inflation given date t+1 values."""
    c = calibration
    lhs = np.array(
        [
            [1.0 + c.sigma * c.phi_x, c.sigma * c.phi_pi],
            [-c.kappa, 1.0],
        ]
    )
    rhs = np.array(
        [
            setting.m * next_output + c.sigma * next_inflation - c.sigma * policy_wedge,
            c.beta * setting.m_f * next_inflation,
        ]
    )
    output, inflation = np.linalg.solve(lhs, rhs)
    return float(output), float(inflation)


def forward_guidance_path(
    calibration: Calibration,
    setting: AttentionSetting,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Solve a perfect-foresight one-period policy wedge at a future horizon."""
    output = np.zeros(horizon + 1)
    inflation = np.zeros(horizon + 1)
    policy_wedge = np.zeros(horizon + 1)
    policy_wedge[horizon] = calibration.shock_size

    next_output = 0.0
    next_inflation = 0.0
    for t in range(horizon, -1, -1):
        output[t], inflation[t] = solve_one_period(
            calibration,
            setting,
            next_output,
            next_inflation,
            policy_wedge[t],
        )
        next_output = output[t]
        next_inflation = inflation[t]

    policy_rate = calibration.phi_pi * inflation + calibration.phi_x * output + policy_wedge
    return {
        "output": output,
        "inflation": inflation,
        "policy_rate": policy_rate,
        "policy_wedge": policy_wedge,
    }


def forward_guidance_summary(
    calibration: Calibration,
    setting: AttentionSetting,
) -> pd.DataFrame:
    """Collect date-0 responses to announced future policy wedges."""
    rows = []
    for horizon in range(calibration.news_horizon + 1):
        path = forward_guidance_path(calibration, setting, horizon)
        rows.append(
            {
                "Horizon": horizon,
                "Setting": setting.label,
                "Output": path["output"][0],
                "Inflation": path["inflation"][0],
                "Policy rate": path["policy_rate"][0],
            }
        )
    return pd.DataFrame(rows)


def format_pp(value: float) -> float:
    """Convert a decimal response to percent or percentage points."""
    return round(100.0 * value, 3)


def main() -> None:
    calibration = Calibration()
    settings = [
        AttentionSetting("Rational NK", 1.0, 1.0, "#1f77b4"),
        AttentionSetting("Behavioral NK", 0.85, 0.85, "#b2182b"),
    ]

    coeffs = {setting.label: solve_policy_coefficients(calibration, setting) for setting in settings}
    irfs = {
        setting.label: current_policy_irf(calibration, coeffs[setting.label])
        for setting in settings
    }
    fg = pd.concat(
        [forward_guidance_summary(calibration, setting) for setting in settings],
        ignore_index=True,
    )

    print("Solved behavioral NK coefficient systems.")
    for setting in settings:
        c = coeffs[setting.label]
        print(
            f"  {setting.label}: psi_x={c['psi_x']:.4f}, "
            f"psi_pi={c['psi_pi']:.4f}, psi_i={c['psi_i']:.4f}"
        )

    setup_style()
    periods = np.arange(calibration.irf_horizon)

    fig1, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for setting in settings:
        paths = irfs[setting.label]
        axes[0].plot(
            periods,
            100.0 * paths["output"],
            color=setting.color,
            linewidth=2.5,
            label=setting.label,
        )
        axes[1].plot(
            periods,
            100.0 * paths["inflation"],
            color=setting.color,
            linewidth=2.5,
            label=setting.label,
        )

    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
    axes[0].set_title("Output gap")
    axes[1].set_title("Inflation")
    axes[0].set_xlabel("Quarters after shock")
    axes[1].set_xlabel("Quarters after shock")
    axes[0].set_ylabel("Percent")
    axes[1].set_ylabel("Percentage points")
    axes[1].legend(frameon=False, loc="lower right")
    fig1.suptitle("Current Monetary-Policy Wedge", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig1, "figures/irf-current-policy-shock.png", dpi=150)

    horizons = np.arange(calibration.news_horizon + 1)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for setting in settings:
        subset = fg[fg["Setting"] == setting.label].sort_values("Horizon")
        axes2[0].plot(
            horizons,
            100.0 * np.abs(subset["Output"].to_numpy()),
            color=setting.color,
            linewidth=2.5,
            marker="o",
            markersize=3.5,
            label=setting.label,
        )
        axes2[1].plot(
            horizons,
            100.0 * np.abs(subset["Inflation"].to_numpy()),
            color=setting.color,
            linewidth=2.5,
            marker="o",
            markersize=3.5,
            label=setting.label,
        )

    axes2[0].set_title("Date-0 output response")
    axes2[1].set_title("Date-0 inflation response")
    axes2[0].set_xlabel("Quarters until one-period wedge")
    axes2[1].set_xlabel("Quarters until one-period wedge")
    axes2[0].set_ylabel("Absolute response, percent")
    axes2[1].set_ylabel("Absolute response, percentage points")
    axes2[1].legend(frameon=False, loc="upper right")
    fig2.suptitle("Forward-Guidance Attenuation", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig2, "figures/forward-guidance-attenuation.png", dpi=150)

    table_rows = []
    for setting in settings:
        paths = irfs[setting.label]
        future_8 = forward_guidance_path(calibration, setting, 8)
        table_rows.append(
            {
                "Setting": setting.label,
                "M": setting.m,
                "M_f": setting.m_f,
                "Output impact": format_pp(paths["output"][0]),
                "Inflation impact": format_pp(paths["inflation"][0]),
                "Nominal-rate impact": format_pp(paths["policy_rate"][0]),
                "Cumulative output": format_pp(paths["output"].sum()),
                "Cumulative inflation": format_pp(paths["inflation"].sum()),
                "FG output H=8": format_pp(future_8["output"][0]),
                "FG inflation H=8": format_pp(future_8["inflation"][0]),
            }
        )

    summary_table = pd.DataFrame(table_rows)
    Path("tables").mkdir(parents=True, exist_ok=True)
    summary_table.to_csv("tables/policy-responses.csv", index=False)

    save_thumbnail("figures/forward-guidance-attenuation.png", "figures/thumb.png")
    print(
        f"Saved 2 figures and 1 table."
    )


if __name__ == "__main__":
    main()
