#!/usr/bin/env python3
"""Perturbation and linearization for nonlinear dynamics.

The tutorial compares first-, second-, and third-order local approximations to a
nonlinear transition rule. It is intentionally small, so the local nature of
perturbation methods is visible in both approximation errors and impulse
responses.
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
        "Perturbation and Linearization",
        "Local Taylor approximations, nonlinear errors, and impulse responses.",
    )

    report.add_overview(
        "Perturbation methods solve nonlinear models by expanding decision rules around a known "
        "point, usually a deterministic steady state. The first-order version is linearization. "
        "Higher-order terms add curvature and asymmetry, but the approximation remains local.\n\n"
        "This tutorial uses one nonlinear transition rule so the method can be seen directly. "
        "The exact rule is known. First-, second-, and third-order approximations are compared "
        "as functions and as impulse responses after positive and negative shocks."
    )

    report.add_equations(
        r"""
Let $x=0$ be the steady state. The exact transition rule is:

$$
F(x) = \rho x + \gamma x^2 - \eta x^3 + \kappa x^4.
$$

The first three perturbation approximations around zero are:

$$
\begin{aligned}
F_1(x) &= \rho x, \\
F_2(x) &= \rho x + \gamma x^2, \\
F_3(x) &= \rho x + \gamma x^2 - \eta x^3.
\end{aligned}
$$

Impulse responses are generated by iterating:

$$
x_{t+1} = F_n(x_t), \qquad x_0 = \epsilon.
$$
"""
    )

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        f"| Persistence $\\rho$ | {RHO:.2f} |\n"
        f"| Quadratic term $\\gamma$ | {GAMMA:.2f} |\n"
        f"| Cubic term $\\eta$ | {ETA:.2f} |\n"
        f"| Fourth-order term $\\kappa$ | {KAPPA:.2f} |\n"
        f"| Shock size | {SHOCK_SIZE:.2f} |\n"
        f"| IRF periods | {periods} |"
    )

    report.add_solution_method(
        "The code evaluates the exact transition rule and its Taylor truncations around the "
        "steady state. Approximation errors are computed over a local interval and a wider "
        "interval. Impulse responses are generated by applying each transition rule recursively "
        "after a one-time positive or negative shock.\n\n"
        "This is the same logic used in larger DSGE applications: solve around a steady state, "
        "then ask whether the local approximation is good enough for the shocks and states the "
        "research question actually visits."
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
    ax1.set_title("Exact Nonlinear Map and Local Approximations")
    ax1.legend()
    report.add_figure(
        "figures/local-approximations.png",
        "Taylor approximations around the steady state",
        fig1,
        description=(
            "All approximations agree at the steady state. Differences grow as the state moves "
            "away from the expansion point."
        ),
    )

    fig2, ax2 = plt.subplots()
    for order in [1, 2, 3]:
        error = np.abs(perturbation_transition(x_grid, order) - exact)
        ax2.semilogy(np.abs(x_grid), np.maximum(error, 1e-14), label=f"order {order}")
    ax2.set_xlabel("Distance from steady state")
    ax2.set_ylabel("Absolute map error")
    ax2.set_title("Local Accuracy Decays with Distance")
    ax2.legend()
    report.add_figure(
        "figures/local-errors.png",
        "Approximation error by distance from the expansion point",
        fig2,
        description=(
            "Higher order terms reduce local error, but no finite Taylor expansion is a global "
            "solution. The relevant question is how far the model travels from the steady state."
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
            "The first-order approximation is symmetric: changing the sign of the shock changes "
            "only the sign of the response. Higher-order terms can capture asymmetric adjustment."
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
            "For a purely linear model, positive and negative impulse responses cancel exactly. "
            "A nonzero sum is a direct diagnostic for nonlinear asymmetry."
        ),
    )

    report.add_table(
        "tables/approximation-errors.csv",
        "Perturbation accuracy by order",
        format_table(table),
        description="Map errors are computed directly; IRF errors compare each approximation to the exact path.",
    )

    report.add_results(
        "The first-order approximation is accurate very close to the steady state, but it misses "
        "curvature and asymmetric responses. The second and third orders reduce local error and "
        "track the nonlinear impulse responses more closely for the shock size used here."
    )

    report.add_takeaway(
        "Linearization is fast and often enough for small shocks, but it imposes symmetry and "
        "removes higher-order risk effects. Perturbation adds curvature without solving the full "
        "global problem. The practical discipline is to check the state region visited by the "
        "simulation or impulse response before trusting a local approximation."
    )

    report.add_references(
        [
            "Chang, M. ECON 609 lecture slides: Linearization and Perturbation.",
            "[Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press.](https://mitpress.mit.edu/9780262100717/numerical-methods-in-economics/)",
            "[Schmitt-Grohe, S. and Uribe, M. (2004). Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function. *Journal of Economic Dynamics and Control*, 28(4), 755-775.](https://doi.org/10.1016/S0165-1889(03)00043-5)",
        ]
    )

    report.write("README.md")


if __name__ == "__main__":
    main()
