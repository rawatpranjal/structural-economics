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
from lib.output import ModelReport
from lib.plotting import setup_style


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

    report = ModelReport(
        "Aggregate Adjustment Around a Steady State",
        "Taylor perturbations for nonlinear macro dynamics and impulse responses.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A macro model often asks what happens after an economy is pushed away from its "
        "steady state. Think of a productivity surprise, a demand shock, or a policy change "
        "that moves a state variable above its normal level. The researcher wants the path "
        "back to steady state and wants to know whether an equally sized negative shock would "
        "unwind in the same way.\n\n"
        "The hard part is that the full nonlinear law of motion may be expensive to solve or "
        "awkward to interpret. Perturbation replaces that law with a Taylor approximation near "
        "the steady state. First order gives the familiar linearized impulse response. Higher "
        "orders keep local curvature, so positive and negative shocks can produce different "
        "adjustment paths. This tutorial keeps the exact nonlinear transition in view, which "
        "lets us see what the local approximation gains and what it misses."
    )

    report.add_equations(
        r"""
Let $x_t$ be a scalar deviation from the deterministic steady state. The
steady state is normalized to $x=0$. The exact nonlinear transition is:

$$
F(x) = \rho x + \gamma x^2 - \eta x^3 + \kappa x^4.
$$

The coefficients are chosen to create persistent adjustment, local curvature,
and asymmetric responses. A Taylor perturbation of order $n$ around zero keeps
the derivatives through order $n$:

$$
F_n(x) = \sum_{j=1}^{n} \frac{F^{(j)}(0)}{j!} x^j.
$$

For the first three orders in this example:

$$
\begin{aligned}
F_1(x) &= \rho x, \\
F_2(x) &= \rho x + \gamma x^2, \\
F_3(x) &= \rho x + \gamma x^2 - \eta x^3.
\end{aligned}
$$

Impulse responses after a one-time shock $\epsilon$ are generated by iterating:

$$
x_{t+1} = F_n(x_t), \qquad x_0 = \epsilon.
$$
"""
    )

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        f"| Persistence $\\rho$ | {RHO:.2f}, so deviations decay gradually |\n"
        f"| Quadratic term $\\gamma$ | {GAMMA:.2f}, adding local curvature |\n"
        f"| Cubic term $\\eta$ | {ETA:.2f}, changing the speed of large responses |\n"
        f"| Fourth-order term $\\kappa$ | {KAPPA:.2f}, left out by third order |\n"
        f"| Shock size | {SHOCK_SIZE:.2f} in either direction |\n"
        f"| IRF periods | {periods} |"
    )

    report.add_solution_method(
        "The calculation treats the exact transition as a benchmark and asks how much of it a "
        "local expansion recovers. The map error answers a numerical question: how close is the "
        "approximated law of motion at nearby states? The impulse-response error answers the "
        "economic question the researcher would usually care about: does the computed shock "
        "path tell the same adjustment story as the nonlinear model?\n\n"
        "```text\n"
        "Algorithm: perturbation check for a shock response\n"
        "Input: nonlinear law F(x), steady state x_bar = 0, order n, shock epsilon\n"
        "Output: approximate law F_n, map errors, IRF errors, asymmetry statistic\n"
        "1. Differentiate F at x_bar and keep terms through order n\n"
        "2. Build the local law F_n(x) from those Taylor coefficients\n"
        "3. Compare F_n(x) with F(x) on a tight neighborhood and a wider interval\n"
        "4. Starting from x_0 = epsilon, iterate x_{t+1} = F_n(x_t)\n"
        "5. Repeat from x_0 = -epsilon and add the two response paths\n"
        "6. Read nonzero sums as nonlinear asymmetry, not as linear adjustment\n"
        "```\n\n"
        "Large DSGE systems apply the same idea to a vector of equilibrium conditions. This "
        "scalar example strips that machinery away, leaving the main discipline: a local "
        "solution is useful only for the shocks and states that stay near the expansion point."
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
    report.add_figure(
        "figures/local-approximations.png",
        "Taylor approximations around the steady state",
        fig1,
        description=(
            "The curves agree at the steady state because each approximation is built there. "
            "Away from zero, the missing curvature changes the next-period state."
        ),
    )

    fig2, ax2 = plt.subplots()
    for order in [1, 2, 3]:
        error = np.abs(perturbation_transition(x_grid, order) - exact)
        ax2.semilogy(np.abs(x_grid), np.maximum(error, 1e-14), label=f"order {order}")
    ax2.set_xlabel("Distance from steady state")
    ax2.set_ylabel("Absolute map error")
    ax2.set_title("Approximation Error by Distance")
    ax2.legend()
    report.add_figure(
        "figures/local-errors.png",
        "Approximation error by distance from the expansion point",
        fig2,
        description=(
            "Higher-order terms lower error near the steady state. Error still rises with "
            "distance, so the relevant accuracy check depends on the states reached by the "
            "shock experiment."
        ),
    )

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
    report.add_figure(
        "figures/impulse-responses.png",
        "Impulse responses under exact and approximated dynamics",
        fig3,
        description=(
            "The first-order path is symmetric by construction. Higher-order paths can let a "
            "positive shock and a negative shock unwind at different speeds."
        ),
    )

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
    report.add_figure(
        "figures/asymmetry.png",
        "Nonlinear asymmetry in positive and negative responses",
        fig4,
        description=(
            "In a linearized model, the positive and negative impulse responses cancel exactly. "
            "A nonzero sum shows how much nonlinear asymmetry the local solution preserves."
        ),
    )

    report.add_table(
        "tables/approximation-errors.csv",
        "Perturbation accuracy by order",
        format_table(table),
        description=(
            "Map errors compare transition rules. IRF errors compare the full adjustment path "
            "after the same shock."
        ),
    )

    report.add_results(
        "The first-order approximation works well in a tight neighborhood of the steady state, "
        "but it misses the curvature that shapes asymmetric adjustment. Second and third order "
        "terms reduce local map error and follow the nonlinear impulse responses more closely "
        "for this shock size. The asymmetry plot gives the most direct economic reading: the "
        "linearized model forces the positive and negative responses to cancel, while the "
        "nonlinear economy does not."
    )

    report.add_takeaway(
        "Linearization is often enough for small deviations, but its symmetry is an economic "
        "restriction. Higher-order perturbation adds curvature without solving the full global "
        "model. The practical check is simple: trace the simulated or impulse-response path and "
        "ask whether it remains in the neighborhood where the local approximation is accurate."
    )

    report.add_references(
        [
            "[Blanchard, O. J. and Kahn, C. M. (1980). The Solution of Linear Difference Models under Rational Expectations. *Econometrica*, 48(5), 1305-1311.](https://doi.org/10.2307/1912186)",
            "[Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press.](https://mitpress.mit.edu/9780262100717/numerical-methods-in-economics/)",
            "[Schmitt-Grohe, S. and Uribe, M. (2004). Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function. *Journal of Economic Dynamics and Control*, 28(4), 755-775.](https://doi.org/10.1016/S0165-1889(03)00043-5)",
        ]
    )

    report.write("README.md")


if __name__ == "__main__":
    main()
