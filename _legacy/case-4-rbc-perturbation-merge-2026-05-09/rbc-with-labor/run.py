#!/usr/bin/env python3
"""RBC labor supply after a productivity shock, solved by Klein QZ."""
from __future__ import annotations

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


def steady_state(
    alpha: float,
    beta: float,
    delta: float,
    sigma: float,
    chi: float,
    n_target: float,
) -> dict[str, float]:
    """Steady state for RBC with endogenous labor.

    The labor disutility weight ``psi`` is calibrated to deliver ``n_target``
    hours in steady state, given the other primitives.
    """
    mpk = 1.0 / beta - 1.0 + delta
    k_over_n = (alpha / mpk) ** (1.0 / (1.0 - alpha))
    K = k_over_n * n_target
    Y = K**alpha * n_target ** (1.0 - alpha)
    I = delta * K
    C = Y - I
    KY = K / Y
    CY = C / Y
    IY = I / Y
    real_wage = (1.0 - alpha) * Y / n_target
    psi = real_wage * C ** (-sigma) / n_target**chi
    return {
        "K": K, "Y": Y, "C": C, "I": I, "N": n_target,
        "K_Y": KY, "C_Y": CY, "I_Y": IY,
        "K_over_N": k_over_n, "wage": real_wage, "psi": psi, "mpk": mpk,
    }


def klein_system(
    alpha: float, beta: float, delta: float, rho: float, sigma: float,
    chi: float, ss: dict[str, float],
):
    """Build the (A, B) Klein matrices for the linearized RBC-with-labor model.

    State ordering ``s_t = (k_lag, a, c, n)`` with two predetermined variables
    on top. Output is eliminated via the production function, investment via
    the resource constraint, so the system reduces to four equations in four
    variables. Equations: capital accumulation, TFP AR(1), intratemporal
    labor supply, intertemporal Euler.
    """
    KY = ss["K_Y"]
    CY = ss["C_Y"]
    CK = CY / KY
    mpk_share = 1.0 - beta * (1.0 - delta)

    A = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [mpk_share * (alpha - 1.0), mpk_share, -sigma, mpk_share * (1.0 - alpha)],
        ]
    )
    B = np.array(
        [
            [alpha / KY + (1.0 - delta), 1.0 / KY, -CK, (1.0 - alpha) / KY],
            [0.0, rho, 0.0, 0.0],
            [-alpha, -1.0, sigma, chi + alpha],
            [0.0, 0.0, -sigma, 0.0],
        ]
    )
    return A, B


def linear_irfs(
    F: np.ndarray, P: np.ndarray, ss: dict[str, float],
    alpha: float, delta: float, shock: float, periods: int,
) -> dict[str, np.ndarray]:
    """Iterate the QZ-derived policy functions to produce IRFs.

    State x = (k_lag, a). Jump variables y = (c, n) = P x. Output and
    investment recovered from production and capital accumulation.
    """
    x = np.zeros((periods + 1, 2))
    x[0] = np.array([0.0, shock])  # k_lag = 0, a_0 = shock
    for t in range(periods):
        x[t + 1] = F @ x[t]

    k_lag = x[:periods, 0]
    a = x[:periods, 1]
    k = x[1:periods + 1, 0]
    c = (P @ x[:periods].T)[0]
    n = (P @ x[:periods].T)[1]
    y = a + alpha * k_lag + (1.0 - alpha) * n
    inv = (k - (1.0 - delta) * k_lag) / delta

    return {
        "TFP": a, "Output": y, "Consumption": c, "Investment": inv,
        "Capital": k, "Labor": n,
    }


def nonlinear_perfect_foresight(
    alpha: float, beta: float, delta: float, rho: float, sigma: float,
    chi: float, ss: dict[str, float], shock: float, periods: int,
) -> dict[str, np.ndarray]:
    """Exact deterministic transition for the same one-time TFP path.

    Solves the perfect-foresight system with terminal condition K_T = K_ss.
    Used as a local-accuracy benchmark for the linear policy.
    """
    K_ss = ss["K"]; C_ss = ss["C"]; N_ss = ss["N"]; psi = ss["psi"]
    a_path = shock * rho ** np.arange(periods + 1)
    A_path = np.exp(a_path)

    def production(K_lag, A, N):
        return A * K_lag**alpha * N ** (1.0 - alpha)

    def labor_from_static(K_lag, A, C):
        # psi N^chi C^sigma = (1-alpha) Y / N
        # => N^(chi + 1) = (1-alpha) A K_lag^alpha N^(1-alpha) / (psi C^sigma)
        # => N^(chi + alpha) = (1-alpha) A K_lag^alpha / (psi C^sigma)
        rhs = (1.0 - alpha) * A * K_lag**alpha / (psi * C**sigma)
        return rhs ** (1.0 / (chi + alpha))

    def euler_residuals(log_K_path):
        K = np.empty(periods + 1)
        K[:periods] = K_ss * np.exp(log_K_path)
        K[periods] = K_ss
        # Solve C from resource each period given (K_lag, K, A, N)
        C = np.empty(periods)
        N = np.empty(periods)
        # Iterate: use static labor supply with previous C guess; simple fix point.
        C_guess = np.full(periods, C_ss)
        for _ in range(50):
            for t in range(periods):
                K_lag = K_ss if t == 0 else K[t - 1]
                N[t] = labor_from_static(K_lag, A_path[t], C_guess[t])
                Y_t = production(K_lag, A_path[t], N[t])
                C[t] = Y_t + (1.0 - delta) * K_lag - K[t]
                if C[t] <= 0:
                    return np.full(periods, 1e6)
            if np.max(np.abs(C - C_guess)) < 1e-12:
                break
            C_guess = C.copy()
        # Euler residuals
        errs = np.empty(periods)
        for t in range(periods):
            K_lag = K_ss if t == 0 else K[t - 1]
            N_next = labor_from_static(K[t], A_path[t + 1], C_ss if t == periods - 1 else C[t + 1])
            R_next = alpha * A_path[t + 1] * K[t] ** (alpha - 1.0) * N_next ** (1.0 - alpha) + (1.0 - delta)
            C_next = C_ss if t == periods - 1 else C[t + 1]
            errs[t] = -sigma * np.log(C[t]) - np.log(beta) + sigma * np.log(C_next) - np.log(R_next)
        return errs

    sol = root(euler_residuals, np.zeros(periods), method="hybr",
               options={"xtol": 1e-10, "maxfev": 30000})
    if not sol.success:
        raise RuntimeError(f"Nonlinear PF transition failed: {sol.message}")

    K = np.empty(periods + 1); K[:periods] = K_ss * np.exp(sol.x); K[periods] = K_ss
    C = np.empty(periods); N = np.empty(periods)
    for _ in range(50):
        for t in range(periods):
            K_lag = K_ss if t == 0 else K[t - 1]
            N[t] = labor_from_static(K_lag, A_path[t], C_ss if _ == 0 else C[t])
            Y_t = production(K_lag, A_path[t], N[t])
            C[t] = Y_t + (1.0 - delta) * K_lag - K[t]

    Y = np.array([production(K_ss if t == 0 else K[t - 1], A_path[t], N[t]) for t in range(periods)])
    INV = np.array([K[t] - (1.0 - delta) * (K_ss if t == 0 else K[t - 1]) for t in range(periods)])
    K_lagged = np.concatenate(([K_ss], K[:periods - 1]))

    return {
        "TFP": a_path[:periods],
        "Output": np.log(Y / ss["Y"]),
        "Consumption": np.log(C / C_ss),
        "Investment": np.log(INV / ss["I"]),
        "Capital": np.log(K[:periods] / K_ss),
        "Labor": np.log(N / N_ss),
    }


def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    alpha = 0.33
    beta = 0.99
    delta = 0.025
    rho = 0.95
    sigma = 1.0
    chi = 1.0          # inverse Frisch elasticity (= 1)
    n_target = 1.0 / 3.0
    sigma_e = 0.01
    periods_irf = 40

    print("Solving RBC labor response by Klein-style generalized Schur QZ...")
    ss = steady_state(alpha, beta, delta, sigma, chi, n_target)
    A, B = klein_system(alpha, beta, delta, rho, sigma, chi, ss)
    sol = solve_klein(A, B, n_predetermined=2)
    F, P = sol.F, sol.P
    print(f"  Steady-state K/Y={ss['K_Y']:.2f}, C/Y={ss['C_Y']:.2f}, N={ss['N']:.3f}")
    print(f"  Blanchard-Kahn: {sol.bk_message}")
    print(f"  Stable eigenvalues: {[f'{e.real:.4f}' for e in sol.eigenvalues if abs(e) < 1.0]}")
    print(f"  Capital decision rule: k_t = {F[0,0]:.4f} k_lag + {F[0,1]:.4f} a_t")
    print(f"  Consumption rule:      c_t = {P[0,0]:.4f} k_lag + {P[0,1]:.4f} a_t")
    print(f"  Labor rule:            n_t = {P[1,0]:.4f} k_lag + {P[1,1]:.4f} a_t")

    linear = linear_irfs(F, P, ss, alpha, delta, sigma_e, periods_irf)
    print("Computing nonlinear perfect-foresight benchmark...")
    nonlinear = nonlinear_perfect_foresight(alpha, beta, delta, rho, sigma, chi, ss, sigma_e, periods_irf)

    setup_style()
    report = ModelReport(
        "RBC Labor Supply and TFP Shocks",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A positive TFP shock raises the marginal product of inputs. Capital is mostly "
        "inherited, so hours carry much of the impact response.\n\n"
        "The object is the local impulse response of output, consumption, investment, "
        "capital, and labor. The household chooses consumption and hours while firms use "
        "capital and labor to produce output.\n\n"
        "Log-linearization gives a rational-expectations system with two states and two "
        "jump variables. Klein QZ selects the stable path and returns decision rules for "
        "consumption and labor."
    )

    report.add_equations(
        rf"""
The household chooses consumption $C_t$ and labor $N_t$ to maximize

$$\mathbb{{E}}_0\sum_{{t=0}}^{{\infty}}\beta^t \left[\frac{{C_t^{{1-\sigma}}}}{{1-\sigma}}-\psi\frac{{N_t^{{1+\chi}}}}{{1+\chi}}\right]$$

subject to $C_t+I_t=Y_t$, $K_t=I_t+(1-\delta)K_{{t-1}}$, and a Cobb-Douglas
production technology

$$Y_t=A_t K_{{t-1}}^\alpha N_t^{{1-\alpha}},\qquad \log A_t=\rho\log A_{{t-1}}+\varepsilon_t.$$

The shock $\varepsilon_t$ is i.i.d. with mean zero and standard deviation $\sigma_\varepsilon$.

The intratemporal labor-supply condition is

$$\psi N_t^\chi=(1-\alpha)\frac{{Y_t}}{{N_t}}\,C_t^{{-\sigma}},$$

and the Euler equation for capital is

$$C_t^{{-\sigma}}=\beta \mathbb{{E}}_t\left[C_{{t+1}}^{{-\sigma}}\left(\alpha A_{{t+1}}K_t^{{\alpha-1}}N_{{t+1}}^{{1-\alpha}}+1-\delta\right)\right].$$

Log-linearization turns the model into decision rules for consumption and labor.
The states are lagged capital and current TFP. Output and investment follow from
production and the resource constraint.
"""
    )

    report.add_model_setup(
        "| Primitive | Value | Role |\n"
        "|---|---:|---|\n"
        f"| $\\alpha$ | {alpha:.2f} | Capital share |\n"
        f"| $\\beta$ | {beta:.2f} | Quarterly discount factor |\n"
        f"| $\\delta$ | {delta:.3f} | Quarterly depreciation |\n"
        f"| $\\rho$ | {rho:.2f} | Persistence of log TFP |\n"
        f"| $\\sigma$ | {sigma:.1f} | CRRA coefficient (log utility) |\n"
        f"| $\\chi$ | {chi:.1f} | Inverse Frisch elasticity |\n"
        f"| $\\bar N$ | {n_target:.3f} | Steady-state hours target |\n"
        f"| $\\sigma_\\varepsilon$ | {sigma_e:.3f} | Innovation s.d. |\n"
        f"| Shock | {100*sigma_e:.1f}% | One-s.d. innovation at $t=0$ |\n\n"
        "| Steady-state object | Value |\n"
        "|---|---:|\n"
        f"| $K$ | {ss['K']:.3f} |\n"
        f"| $Y$ | {ss['Y']:.3f} |\n"
        f"| $C$ | {ss['C']:.3f} |\n"
        f"| $K/Y$ | {ss['K_Y']:.3f} |\n"
        f"| $C/Y$ | {ss['C_Y']:.3f} |\n"
        f"| Real wage | {ss['wage']:.3f} |\n"
        f"| Labor weight $\\psi$ | {ss['psi']:.3f} |"
    )

    stable_eigs = sorted([float(e.real) for e in sol.eigenvalues if abs(e) < 1.0])
    report.add_solution_method(
        "Stack the linearized equilibrium as\n\n"
        "$$A\\,\\mathbb{E}_t s_{t+1}=B\\,s_t.$$\n\n"
        "Use $s_t=(\\hat k_{t-1},\\hat a_t,\\hat c_t,\\hat n_t)'$, where a hat denotes the log-deviation from steady state. "
        "The first two entries are states. The last two entries are jump variables.\n\n"
        "The rows are capital accumulation, TFP, labor supply, and the Euler equation.\n\n"
        "The solution is a pair of matrices\n\n"
        "$$x_{t+1}=F x_t,\\qquad y_t=P x_t,$$\n\n"
        "where $x_t=(\\hat k_{t-1},\\hat a_t)'$ and $y_t=(\\hat c_t,\\hat n_t)'$. "
        "The matrix $P$ maps states into consumption and labor.\n\n"
        "Klein QZ orders the generalized eigenvalues. The stable block gives the "
        "non-explosive path for the two jump variables.\n\n"
        "```text\n"
        "Algorithm: Klein QZ for the linearized RBC system\n"
        "Inputs:  matrices A and B; number of state variables n_x = 2\n"
        "Outputs: state transition F, decision rule P, and impulse responses\n\n"
        "1. Compute the ordered QZ decomposition of (B, A), with stable roots first.\n"
        "2. Check Blanchard-Kahn: the number of stable roots must equal n_x.\n"
        "3. Partition the Schur vectors into state and jump blocks.\n"
        "4. Recover P from the jump block relative to the state block.\n"
        "5. Recover F from the stable triangular blocks.\n"
        "6. Start from x_0 = (0, sigma_e) and iterate x_{t+1} = F x_t, y_t = P x_t.\n"
        "```\n\n"
        f"The Blanchard-Kahn count matches: {sol.n_stable} stable roots for "
        f"{sol.n_predetermined} states. The stable roots are {stable_eigs[0]:.4f} "
        f"and {stable_eigs[1]:.4f}. This selects one local equilibrium path."
    )

    periods = np.arange(periods_irf)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panel_vars = ["Output", "Consumption", "Investment", "Labor", "Capital", "TFP"]
    colors = {
        "Output": "#1b6ca8", "Consumption": "#4b8f29", "Investment": "#b85c00",
        "Labor": "#a3007a", "Capital": "#6f4aa8", "TFP": "#444444",
    }
    for ax, var in zip(axes.flat, panel_vars):
        ax.plot(periods, 100.0 * linear[var], color=colors[var], linewidth=2.3,
                label="Klein QZ (linear)")
        ax.plot(periods, 100.0 * nonlinear[var], color="black", linewidth=1.4,
                linestyle="--", label="Nonlinear PF")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.6)
        ax.set_title(var)
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Percent")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Responses to a 1% TFP Innovation (RBC with Endogenous Labor)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    report.add_results(
        "The productivity innovation raises output on impact because TFP and hours rise "
        "together. Investment moves more than consumption because capital carries the "
        "shock forward. Consumption adjusts smoothly through the Euler equation.\n\n"
        "The dashed nonlinear path is a local check on the linear solution. It keeps the "
        "same ranking of margins. The largest gaps appear for investment and capital, "
        "which move through the resource constraint."
    )
    report.add_figure(
        "figures/irf-tfp-shock.png",
        "Linear Klein QZ vs nonlinear perfect-foresight responses to a 1% TFP shock",
        fig,
    )

    rows = []
    for var in panel_vars:
        s = linear[var]
        gap = float(np.max(np.abs(100.0 * (linear[var] - nonlinear[var]))))
        rows.append({
            "Variable": var,
            "Impact (%)": f"{100.0*s[0]:.3f}",
            "Peak (%)": f"{100.0*s[np.argmax(np.abs(s))]:.3f}",
            "Peak quarter": int(np.argmax(np.abs(s))),
            "Max linear-vs-PF gap (pp)": f"{gap:.3f}",
        })
    summary = pd.DataFrame(rows)
    report.add_table(
        "tables/irf-summary.csv",
        "IRF Summary",
        summary,
    )

    report.add_takeaway(
        "The TFP shock splits into an hours response and a capital response. The labor "
        "rule $\\hat n_t="
        + f"{P[1,0]:.4f}\\hat k_{{t-1}}+{P[1,1]:.4f}\\hat a_t$ rises with productivity "
        "and falls with inherited capital.\n\n"
        "Klein QZ delivers that rule from the stable subspace."
    )

    report.add_references([
        "King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.",
        "Hansen, G. (1985). Indivisible Labor and the Business Cycle. *Journal of Monetary Economics*, 16(3), 309-327.",
        "Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.",
        "Villemot, S. (2011). Solving Rational Expectations Models at First Order: What Dynare Does. *Dynare Working Paper 2*, CEPREMAP.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figure + {len(report._tables)} table")


if __name__ == "__main__":
    main()
