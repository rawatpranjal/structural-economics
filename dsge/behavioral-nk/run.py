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
from lib.output import ModelReport
from lib.plotting import save_thumbnail, setup_style


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
    report = ModelReport(
        "Cognitive Discounting in a Behavioral New Keynesian Model",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Central banks often talk about future rates. The rational New Keynesian "
        "model gives those announcements a strong effect today because households "
        "and firms look far ahead. Gabaix's behavioral model asks what changes "
        "when they put less weight on future conditions.\n\n"
        "This tutorial keeps the same linearized New Keynesian block in both "
        "models. The rational benchmark sets the attention parameters to one. "
        "The behavioral case sets them below one. Agents still respond to the "
        "future, but the future pulls less on today's output and inflation.\n\n"
        "The computation is small. Coefficient matching solves a current AR(1) "
        "policy wedge. A backward recursion solves forward guidance, where a "
        "one-quarter policy wedge arrives several quarters from now."
    )

    report.add_equations(
        rf"""
All variables are deviations from steady state. Let $x_t$ be the output gap,
$\pi_t$ inflation, $i_t$ the policy rate, $r^n_t$ the natural real rate, and
$v_t$ a policy-rate wedge. The behavioral New Keynesian block is

$$
x_t = M\mathbb{{E}}_t x_{{t+1}} - \sigma(i_t-\mathbb{{E}}_t\pi_{{t+1}}-r^n_t),
$$

$$
\pi_t = \beta M_f\mathbb{{E}}_t\pi_{{t+1}} + \kappa x_t + u_t,
$$

$$
i_t = \phi_\pi \pi_t + \phi_x x_t + v_t.
$$

Here $u_t$ is a cost-push shock, set to zero in both experiments run here.

The only change from the rational model is the pair $(M,M_f)$. The parameter
$M$ multiplies expected future output in the IS curve. The parameter $M_f$
multiplies expected future inflation in the Phillips curve. They do not change
the static Taylor rule. They change how strongly future variables enter current
private decisions.

The rational benchmark uses $M=M_f=1$. The behavioral benchmark uses
$M=M_f={settings[1].m:.2f}$.

For the current monetary-policy experiment,

$$
v_t=\rho_v v_{{t-1}}+\varepsilon^v_t.
$$

The forward-guidance experiment instead sets $v_H=\varepsilon^v_H$ for one future
quarter $H$ and sets all other policy wedges to zero.
"""
    )

    report.add_model_setup(
        "| Primitive | Value | Role |\n"
        "|---|---:|---|\n"
        f"| $\\sigma$ | {calibration.sigma:.3g} | Interest sensitivity in the IS curve |\n"
        f"| $\\beta$ | {calibration.beta:.3g} | Quarterly discount factor |\n"
        f"| $\\kappa$ | {calibration.kappa:.3g} | Slope of the Phillips curve |\n"
        f"| $\\phi_\\pi$ | {calibration.phi_pi:.3g} | Taylor-rule response to inflation |\n"
        f"| $\\phi_x$ | {calibration.phi_x:.3g} | Taylor-rule response to the output gap |\n"
        f"| $\\rho_v$ | {calibration.rho_v:.3g} | Persistence of the current policy wedge |\n"
        f"| Shock innovation | {calibration.shock_size:.3f} | One-percentage-point policy wedge |\n"
        f"| Rational attention | 1.000 | $M=M_f=1$ |\n"
        f"| Behavioral attention | {settings[1].m:.3f} | $M=M_f={settings[1].m:.2f}$ |\n"
        f"| IRF horizon | {calibration.irf_horizon} quarters | Length of the current-shock paths |\n"
        f"| News horizon | {calibration.news_horizon} quarters | Furthest date of the future policy wedge |"
    )

    report.add_solution_method(
        "Let the active current shock be $s_t=v_t$ with "
        "$\\mathbb{E}_t s_{t+1}=\\rho_v s_t$. Guess linear responses:\n\n"
        "$$x_t=\\psi_x s_t,\\qquad \\pi_t=\\psi_\\pi s_t,\\qquad i_t=\\psi_i s_t.$$\n\n"
        "Plug the guess into the IS curve and Phillips curve. The Taylor rule "
        "then gives this 2 by 2 system:\n\n"
        "$$\n"
        "\\begin{bmatrix}\n"
        "1-M\\rho_v+\\sigma\\phi_x & \\sigma(\\phi_\\pi-\\rho_v) \\\\\n"
        "-\\kappa & 1-\\beta M_f\\rho_v\n"
        "\\end{bmatrix}\n"
        "\\begin{bmatrix}\\psi_x \\\\ \\psi_\\pi\\end{bmatrix} =\n"
        "\\begin{bmatrix}-\\sigma \\\\ 0\\end{bmatrix}.\n"
        "$$\n\n"
        "After solving for $\\psi_x$ and $\\psi_\\pi$, compute "
        "$\\psi_i=\\phi_\\pi\\psi_\\pi+\\phi_x\\psi_x+1$.\n\n"
        "Forward guidance uses a different state. The state is the date of the "
        "announced wedge. Start from $x_{H+1}=\\pi_{H+1}=0$. Then step backward:\n\n"
        "```text\n"
        "Algorithm: current and future policy wedges\n"
        "Inputs: beta, sigma, kappa, phi_pi, phi_x, M, M_f, rho_v, shock size eps\n"
        "Outputs: output, inflation, and policy-rate responses\n\n"
        "1. For a current AR(1) wedge, solve the 2 by 2 coefficient system above.\n"
        "2. Iterate v_t=rho_v^t eps to draw the current-shock IRF.\n"
        "3. For a future wedge at H, set v_H=eps and v_t=0 otherwise.\n"
        "4. Given x_{t+1} and pi_{t+1}, solve the two date-t equations backward.\n"
        "5. Record x_0 and pi_0 for each H and compare rational with behavioral attention.\n"
        "```\n\n"
        "The forward-guidance recursion uses the same linear model. It treats the "
        "future policy wedge as a deterministic announcement. It does not use an "
        "AR(1) state."
    )

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

    report.add_results(
        "A current rate wedge lowers output today. Lower output then lowers "
        "inflation through the Phillips curve. In the behavioral model, households "
        "and firms put less weight on future output and future inflation. That "
        "weakens the feedback loop from future conditions back to today. The "
        "current-shock response is therefore slightly smaller in cumulative terms."
    )
    report.add_figure(
        "figures/irf-current-policy-shock.png",
        "Rational and behavioral impulse responses to a current policy-rate wedge",
        fig1,
    )

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

    report.add_results(
        "The forward-guidance figure reports absolute date-0 responses. A future "
        "rate wedge matters today only through expectations. The rational model "
        "lets that future wedge travel backward through the IS curve and Phillips "
        "curve. Cognitive discounting breaks part of that backward chain. The "
        "farther away the wedge is, the more the behavioral response shrinks. "
        "This is the main tutorial result."
    )
    report.add_figure(
        "figures/forward-guidance-attenuation.png",
        "Date-0 response magnitudes to policy wedges announced for future quarters",
        fig2,
    )

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
    report.add_results(
        "The table reports percent or percentage-point responses. The first columns "
        "use the persistent current wedge. The last two columns show the signed "
        "date-0 response to a one-quarter wedge announced eight quarters ahead."
    )
    report.add_table(
        "tables/policy-responses.csv",
        "Policy-Wedge Responses",
        summary_table,
    )

    report.add_takeaway(
        "Cognitive discounting changes the expectation channel. It does not change "
        "the static Taylor rule. When $M$ and $M_f$ fall below one, future output "
        "and inflation matter less for today's choices. Current monetary shocks "
        "have slightly smaller cumulative effects. Distant forward-guidance shocks "
        "lose much more of their current bite."
    )

    report.add_references(
        [
            "Gabaix, X. (2020). A Behavioral New Keynesian Model. *American Economic Review*, 110(8), 2271-2327. https://doi.org/10.1257/aer.20162005.",
            "Gabaix, X. (2016). A Behavioral New Keynesian Model. NBER Working Paper 22954. https://doi.org/10.3386/w22954.",
            "Gabaix, X. (2020). Replication Code for: A Behavioral New Keynesian Model. AEA Data and Code Repository. https://doi.org/10.3886/E117842V1.",
            "Gali, J. (2015). *Monetary Policy, Inflation, and the Business Cycle*. Princeton University Press, 2nd edition.",
            "Woodford, M. (2003). *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.",
        ]
    )

    report.write("README.md")
    save_thumbnail("figures/forward-guidance-attenuation.png", "figures/thumb.png")
    print(
        f"Generated README.md, {len(report._figures)} figures, "
        f"and {len(report._tables)} table."
    )


if __name__ == "__main__":
    main()
