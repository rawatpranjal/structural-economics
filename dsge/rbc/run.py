#!/usr/bin/env python3
"""RBC TFP shocks and first-order perturbation."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.perturbation import solve_klein
from lib.plotting import setup_style


def parse_mod_file(mod_path: Path) -> str:
    """Read the model spec file. Used for documentation; not executed."""
    return mod_path.read_text()


def klein_qz_policy(
    alpha: float,
    beta: float,
    delta: float,
    rho: float,
    sigma: float,
    ss: dict[str, float],
) -> dict[str, float]:
    """Solve the same first-order RBC system by Klein (2000) generalized Schur.

    Used as a cross-check on ``solve_log_linear_policy``; for this small model
    Klein QZ and method of undetermined coefficients must agree to machine
    precision.
    """
    capital_output = ss["K_Y"]
    consumption_share = ss["C_Y"]
    consumption_capital = consumption_share / capital_output
    gross_marginal_product_share = beta * alpha / capital_output

    A = np.array(
        [
            [capital_output, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-(alpha - 1.0) * gross_marginal_product_share, -gross_marginal_product_share, sigma],
        ]
    )
    B = np.array(
        [
            [alpha + (1.0 - delta) * capital_output, 1.0, -consumption_share],
            [0.0, rho, 0.0],
            [0.0, 0.0, sigma],
        ]
    )
    sol = solve_klein(A, B, n_predetermined=2)
    return {
        "p": float(sol.F[0, 0]),
        "q": float(sol.F[0, 1]),
        "c_k": float(sol.P[0, 0]),
        "c_a": float(sol.P[0, 1]),
        "blanchard_kahn": sol.bk_message,
        "eigenvalues": sol.eigenvalues,
    }


def steady_state(
    alpha: float,
    beta: float,
    delta: float,
) -> dict[str, float]:
    """Return the deterministic steady state for the fixed-labor RBC model."""
    mpk = 1.0 / beta - 1.0 + delta
    capital = (alpha / mpk) ** (1.0 / (1.0 - alpha))
    output = capital**alpha
    investment = delta * capital
    consumption = output - investment

    return {
        "K": capital,
        "Y": output,
        "C": consumption,
        "I": investment,
        "K_Y": capital / output,
        "C_Y": consumption / output,
        "I_Y": investment / output,
        "mpk": mpk,
        "gross_return": 1.0 / beta,
    }


def solve_log_linear_policy(
    alpha: float,
    beta: float,
    delta: float,
    rho: float,
    sigma: float,
    ss: dict[str, float],
) -> dict[str, float]:
    """Solve the first-order RBC decision rule by matching coefficients.

    The rule is

        khat_t = p * khat_{t-1} + q * ahat_t,

    where hats are log deviations from steady state.
    """
    capital_output = ss["K_Y"]
    consumption_share = ss["C_Y"]
    gross_marginal_product_share = beta * alpha / capital_output
    resource_lag_weight = alpha + capital_output * (1.0 - delta)

    def consumption_coefficients(p: float, q: float) -> tuple[float, float]:
        c_k = (resource_lag_weight - capital_output * p) / consumption_share
        c_a = (1.0 - capital_output * q) / consumption_share
        return c_k, c_a

    def residual(coefficients: np.ndarray) -> np.ndarray:
        p, q = coefficients
        c_k, c_a = consumption_coefficients(p, q)

        euler_k = c_k - (
            c_k * p
            - (gross_marginal_product_share / sigma) * (alpha - 1.0) * p
        )
        euler_a = c_a - (
            c_k * q
            + c_a * rho
            - (gross_marginal_product_share / sigma)
            * (rho + (alpha - 1.0) * q)
        )
        return np.array([euler_k, euler_a])

    solution = root(residual, np.array([0.95, 0.08]))
    if not solution.success:
        raise RuntimeError(f"Could not solve linearized RBC policy: {solution.message}")

    p, q = solution.x
    c_k, c_a = consumption_coefficients(p, q)
    return {
        "p": float(p),
        "q": float(q),
        "c_k": float(c_k),
        "c_a": float(c_a),
        "max_residual": float(np.max(np.abs(residual(solution.x)))),
    }


def linear_irfs(
    alpha: float,
    delta: float,
    rho: float,
    shock: float,
    policy: dict[str, float],
    ss: dict[str, float],
    periods: int,
) -> dict[str, np.ndarray]:
    """Compute log-linear impulse responses to a TFP innovation."""
    a_hat = shock * rho ** np.arange(periods)
    k_hat = np.zeros(periods)
    y_hat = np.zeros(periods)
    i_hat = np.zeros(periods)
    c_hat = np.zeros(periods)

    for t in range(periods):
        k_lag = k_hat[t - 1] if t > 0 else 0.0
        k_hat[t] = policy["p"] * k_lag + policy["q"] * a_hat[t]
        y_hat[t] = a_hat[t] + alpha * k_lag
        i_hat[t] = (k_hat[t] - (1.0 - delta) * k_lag) / delta
        c_hat[t] = (y_hat[t] - ss["I_Y"] * i_hat[t]) / ss["C_Y"]

    return {
        "Output": y_hat,
        "Consumption": c_hat,
        "Investment": i_hat,
        "Capital": k_hat,
        "TFP": a_hat,
    }


def nonlinear_transition_irfs(
    alpha: float,
    beta: float,
    delta: float,
    rho: float,
    sigma: float,
    shock: float,
    ss: dict[str, float],
    initial_guess: np.ndarray,
    periods: int,
) -> dict[str, np.ndarray]:
    """Solve the exact nonlinear transition for a decaying TFP shock path.

    This is a deterministic perfect-foresight benchmark conditional on one
    innovation at date 0 and no later innovations.
    """
    K_ss = ss["K"]
    C_ss = ss["C"]
    Y_ss = ss["Y"]
    I_ss = ss["I"]

    a_path = shock * rho ** np.arange(periods + 1)
    A_path = np.exp(a_path)
    terminal_k = K_ss

    def residual(log_k_path: np.ndarray) -> np.ndarray:
        K_path = np.empty(periods + 1)
        K_path[:periods] = K_ss * np.exp(log_k_path)
        K_path[periods] = terminal_k

        errors = np.empty(periods)
        K_lag = K_ss
        for t in range(periods):
            C_t = A_path[t] * K_lag**alpha + (1.0 - delta) * K_lag - K_path[t]
            C_next = (
                A_path[t + 1] * K_path[t] ** alpha
                + (1.0 - delta) * K_path[t]
                - K_path[t + 1]
            )
            R_next = alpha * A_path[t + 1] * K_path[t] ** (alpha - 1.0) + (
                1.0 - delta
            )
            if C_t <= 0.0 or C_next <= 0.0 or R_next <= 0.0:
                return np.full(periods, 1e6)
            errors[t] = (
                -sigma * np.log(C_t)
                - np.log(beta)
                + sigma * np.log(C_next)
                - np.log(R_next)
            )
            K_lag = K_path[t]
        return errors

    solution = root(
        residual,
        initial_guess[:periods],
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 20000},
    )
    if not solution.success:
        raise RuntimeError(f"Could not solve nonlinear transition: {solution.message}")

    K_path = np.empty(periods + 1)
    K_path[:periods] = K_ss * np.exp(solution.x)
    K_path[periods] = terminal_k

    y_hat = np.zeros(periods)
    c_hat = np.zeros(periods)
    i_hat = np.zeros(periods)
    k_hat = np.log(K_path[:periods] / K_ss)

    K_lag = K_ss
    for t in range(periods):
        Y_t = A_path[t] * K_lag**alpha
        C_t = Y_t + (1.0 - delta) * K_lag - K_path[t]
        I_t = K_path[t] - (1.0 - delta) * K_lag

        y_hat[t] = np.log(Y_t / Y_ss)
        c_hat[t] = np.log(C_t / C_ss)
        i_hat[t] = np.log(I_t / I_ss)
        K_lag = K_path[t]

    return {
        "Output": y_hat,
        "Consumption": c_hat,
        "Investment": i_hat,
        "Capital": k_hat,
        "TFP": a_path[:periods],
        "max_residual": float(np.max(np.abs(residual(solution.x)))),
    }


def half_life_after_peak(series: np.ndarray) -> int | str:
    """Return periods after the absolute peak until the response halves."""
    abs_series = np.abs(series)
    peak_period = int(np.argmax(abs_series))
    peak = abs_series[peak_period]
    below_half = np.where(abs_series[peak_period:] <= peak / 2.0)[0]
    if len(below_half) == 0:
        return f">{len(series) - peak_period - 1}"
    return int(below_half[0])


def format_percent(value: float, digits: int = 3) -> str:
    """Format a log-deviation as a percentage response."""
    return f"{100.0 * value:.{digits}f}"


def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    mod_text = parse_mod_file(tutorial_dir / "model.mod")
    model_equation_count = mod_text.split("model;")[1].split("end;")[0].count(";")

    alpha = 0.33
    beta = 0.99
    delta = 0.025
    rho = 0.95
    sigma = 1.0
    sigma_e = 0.01
    shock = sigma_e
    periods_irf = 40
    periods_benchmark = 120

    print("Solving the RBC model around its deterministic steady state...")
    ss = steady_state(alpha, beta, delta)
    policy = solve_log_linear_policy(alpha, beta, delta, rho, sigma, ss)
    qz = klein_qz_policy(alpha, beta, delta, rho, sigma, ss)
    qz_diff = max(abs(policy["p"] - qz["p"]), abs(policy["q"] - qz["q"]))
    linear_long = linear_irfs(
        alpha,
        delta,
        rho,
        shock,
        policy,
        ss,
        periods_benchmark,
    )
    nonlinear_long = nonlinear_transition_irfs(
        alpha,
        beta,
        delta,
        rho,
        sigma,
        shock,
        ss,
        linear_long["Capital"],
        periods_benchmark,
    )
    print(f"  Parsed {model_equation_count} model equations from model.mod.")
    print(f"  Steady-state K/Y: {ss['K_Y']:.2f}")
    print(f"  Capital decision rule: k_t = {policy['p']:.4f} k_(t-1) + {policy['q']:.4f} a_t")
    print(f"  Klein QZ cross-check ({qz['blanchard_kahn']}): max abs diff = {qz_diff:.2e}")
    print(f"  Max Euler residual in nonlinear benchmark: {nonlinear_long['max_residual']:.2e}")

    variables = ["Output", "Consumption", "Investment", "Capital"]
    plot_variables = variables + ["TFP"]
    linear = {key: linear_long[key][:periods_irf] for key in plot_variables}
    nonlinear = {key: nonlinear_long[key][:periods_irf] for key in plot_variables}

    setup_style()
    report = ModelReport(
        "RBC TFP Shocks and Capital Propagation",
        "How a productivity shock moves output when capital adjusts slowly.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Imagine TFP rises by 1 percent this quarter. The existing capital stock "
        "can produce more right away, so output jumps. Capital itself was chosen "
        "last period, though. The household can only carry the shock forward by "
        "investing part of today's extra output.\n\n"
        "This tutorial follows that tradeoff. A persistent TFP shock changes the "
        "marginal product of capital. The Euler equation then tells the household "
        "how much consumption to postpone. Investment is the adjustment margin, and "
        "capital moves with a lag.\n\n"
        "To compute the path, we log-linearize the RBC model around its steady "
        "state and solve for the capital decision rule. In this fixed-labor example, "
        "coefficient matching gives the rule directly. Klein's generalized Schur "
        "(QZ) solver gives the same coefficients, which is useful because QZ scales "
        "to larger DSGE systems. The figure also includes the exact nonlinear "
        "transition for the same decaying TFP path. At this shock size, the local "
        "solution and nonlinear path nearly coincide."
    )

    report.add_equations(
        rf"""
This is a representative-agent RBC allocation after a one-time technology
innovation. Let $A_t$ denote total factor productivity,
$K_{{t-1}}$ the capital stock chosen last period, $C_t$ consumption,
$I_t$ investment, and $Y_t$ output. Production and goods-market clearing are

$$
Y_t = A_t K_{{t-1}}^\alpha,
\qquad
Y_t = C_t + I_t,
$$

$$
K_t = I_t + (1-\delta)K_{{t-1}},
$$

so investment is the only way to move the state. The representative household's
Euler equation is

$$
C_t^{{-\sigma}} =
\beta \mathbb{{E}}_t\left[
C_{{t+1}}^{{-\sigma}}
\left(\alpha A_{{t+1}}K_t^{{\alpha-1}}+1-\delta\right)
\right].
$$

Technology follows

$$
\log A_t = \rho \log A_{{t-1}} + \varepsilon_t,
\qquad
\varepsilon_t \sim N(0,\sigma_\varepsilon^2).
$$

The accompanying `model.mod` spec stores $y,c,i,k,a$ as logs for documentation,
so expressions such as `exp(y)` are level variables. Around the deterministic
steady state with $A=1$,

$$
\alpha K^{{\alpha-1}} = \frac{{1}}{{\beta}} - 1 + \delta,
\qquad
Y=K^\alpha,\qquad I=\delta K,\qquad C=Y-I.
$$

The calibration implies $K/Y={ss["K_Y"]:.2f}$ and $C/Y={ss["C_Y"]:.2f}$.
"""
    )

    report.add_model_setup(
        "| Primitive | Value | Role |\n"
        "|---|---:|---|\n"
        f"| $\\alpha$ | {alpha:.2f} | Capital share in production |\n"
        f"| $\\beta$ | {beta:.2f} | Quarterly discount factor |\n"
        f"| $\\delta$ | {delta:.3f} | Quarterly depreciation |\n"
        f"| $\\rho$ | {rho:.2f} | Persistence of log TFP |\n"
        f"| $\\sigma$ | {sigma:.1f} | CRRA coefficient; here log utility |\n"
        f"| $\\sigma_\\varepsilon$ | {sigma_e:.3f} | Innovation standard deviation in log TFP |\n"
        f"| Shock | {100 * shock:.1f}% | One-standard-deviation innovation at date 0 |\n"
        f"| IRF horizon | {periods_irf} quarters | Periods shown in the figure |\n\n"
        "| Steady-state object | Value |\n"
        "|---|---:|\n"
        f"| $K$ | {ss['K']:.3f} |\n"
        f"| $Y$ | {ss['Y']:.3f} |\n"
        f"| $C$ | {ss['C']:.3f} |\n"
        f"| $I$ | {ss['I']:.3f} |\n"
        f"| $K/Y$ | {ss['K_Y']:.3f} |\n"
        f"| $C/Y$ | {ss['C_Y']:.3f} |"
    )

    report.add_solution_method(
        "The computation needs a stable law of motion for capital. Write "
        "$\\hat k_t=\\log(K_t/K)$ and $\\hat a_t=\\log A_t$. Since capital is the "
        "only endogenous state, the decision rule is linear in last period's "
        "capital and current productivity:\n\n"
        "$$\n"
        f"\\hat k_t = {policy['p']:.4f}\\hat k_{{t-1}} + {policy['q']:.4f}\\hat a_t.\n"
        "$$\n\n"
        "Once we have this rule, production and the resource constraint give output, "
        "consumption, and investment. The stable capital root is below one. A "
        "temporary productivity shock can raise investment today, but capital still "
        "builds gradually because today's state inherits yesterday's choice.\n\n"
        "```text\n"
        "Algorithm: first-order RBC impulse response\n"
        "Inputs: alpha, beta, delta, rho, sigma, shock size eps_0, horizon T\n"
        "Outputs: paths for yhat_t, chat_t, ihat_t, khat_t\n\n"
        "1. Compute the deterministic steady state K, Y, C, I.\n"
        "2. Linearize the resource constraint and Euler equation in log deviations.\n"
        "3. Guess khat_t = p khat_{t-1} + q ahat_t.\n"
        "4. Substitute the guess into the linearized equations and match the\n"
        "   coefficients on khat_{t-1} and ahat_t.\n"
        "5. Select the stable solution for p and q.\n"
        "6. Set ahat_0 = eps_0 and ahat_t = rho ahat_{t-1}.\n"
        "7. Iterate the decision rule and recover yhat_t, ihat_t, and chat_t from\n"
        "   production, capital accumulation, and goods-market clearing.\n"
        "8. As a local accuracy check, solve the exact nonlinear perfect-foresight\n"
        "   transition for the same TFP path and compare the two IRFs.\n"
        "```\n\n"
        f"The coefficient-matching residual is {policy['max_residual']:.1e}. Klein's "
        "(2000) generalized Schur (QZ) decomposition solves the same linearized "
        "system and agrees to "
        f"{qz_diff:.1e}, machine precision for this problem. The stable eigenvalues "
        f"are {qz['eigenvalues'][0].real:.4f} and {qz['eigenvalues'][1].real:.4f}; "
        "they are the roots that govern capital and TFP propagation. The nonlinear "
        "benchmark is not a second stochastic model. It is the exact deterministic "
        "transition implied by the same one-time shock path."
    )

    periods = np.arange(periods_irf)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {
        "Output": "#1b6ca8",
        "Consumption": "#4b8f29",
        "Investment": "#b85c00",
        "Capital": "#6f4aa8",
    }

    for ax, variable in zip(axes.flat, variables):
        ax.plot(
            periods,
            100.0 * linear[variable],
            color=colors[variable],
            linewidth=2.3,
            label="First-order perturbation",
        )
        ax.plot(
            periods,
            100.0 * nonlinear[variable],
            color="black",
            linewidth=1.6,
            linestyle="--",
            label="Nonlinear transition",
        )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_title(variable)
        ax.set_xlabel("Quarters after shock")
        ax.set_ylabel("Percent log deviation")

    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Responses to a 1 Percent TFP Innovation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    report.add_results(
        "Output rises immediately because the same capital is more productive. "
        "Investment jumps more than output because the household wants more capital "
        "while productivity remains high. Consumption rises by less on impact and "
        "keeps drifting upward for several quarters as the Euler equation smooths "
        "marginal utility. The dashed nonlinear transition sits almost on top of the "
        "first-order solution at this shock size."
    )
    report.add_figure(
        "figures/irf-tfp-shock.png",
        "Impulse responses of output, consumption, investment, and capital to a 1 percent TFP shock",
        fig,
    )

    rows = []
    for variable in ["Output", "Consumption", "Investment", "Capital", "TFP"]:
        series = linear_long[variable]
        display_gap = float(
            np.max(
                np.abs(
                    100.0
                    * (
                        linear_long[variable][:periods_irf]
                        - nonlinear_long[variable][:periods_irf]
                    )
                )
            )
        )
        peak_period = int(np.argmax(np.abs(series)))
        rows.append(
            {
                "Variable": variable,
                "Impact (%)": format_percent(series[0]),
                "Peak (%)": format_percent(series[peak_period]),
                "Peak quarter": peak_period,
                "Half-life after peak": half_life_after_peak(series),
                "Max nonlinear gap (pp)": f"{display_gap:.3f}",
            }
        )

    summary = pd.DataFrame(rows)
    report.add_results(
        "The summary statistics separate impact effects from delayed peaks. Capital "
        "and consumption peak well after the shock because the state is slow-moving; "
        "investment peaks immediately because it is the margin that changes the state."
    )
    report.add_table(
        "tables/irf-summary.csv",
        "IRF Summary Statistics",
        summary,
    )

    report.add_takeaway(
        "In this RBC model, a productivity shock is both a level effect and an "
        "intertemporal price signal. Output rises on impact because firms are more "
        "productive. Investment responds strongly because the marginal product of "
        "capital is temporarily high. Consumption moves more smoothly, and capital "
        "accumulates only gradually. The first-order perturbation isolates that "
        "propagation mechanism by solving for the stable capital decision rule near "
        "steady state.\n\n"
        "This tutorial is the equilibrium counterpart to the "
        "[persistent-shock tutorial](../../time-series/ar-processes/): the AR(1) process supplies "
        "the shock's timing, while the Euler equation and capital law of motion decide "
        "how that timing shows up in macro quantities. For a global Bellman version of "
        "the same RBC mechanism, compare this local solution with the "
        "[dynamic-programming RBC tutorial](../../dynamic-programming/rbc/)."
    )

    report.add_references(
        [
            "Kydland, F. and Prescott, E. (1982). Time to Build and Aggregate Fluctuations. *Econometrica*, 50(6), 1345-1370.",
            "King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.",
            "Uhlig, H. (1999). A Toolkit for Analysing Nonlinear Dynamic Stochastic Models Easily. In *Computational Methods for the Study of Dynamic Economies*.",
            "Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.",
        ]
    )

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figure + {len(report._tables)} table")


if __name__ == "__main__":
    main()
