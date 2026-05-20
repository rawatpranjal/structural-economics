#!/usr/bin/env python3
"""Linearized RBC by perturbation and QZ, with and without endogenous labor.

Two cases run in sequence on the same primitives:

(a) Fixed-labor RBC. Three equations (capital accumulation, TFP, Euler) with
    one jump variable (consumption). Hand-derived undetermined coefficients
    give the capital decision rule. A Klein QZ generalized-Schur solve is the
    cross-check.

(b) Endogenous-labor RBC. Four equations (capital, TFP, intratemporal labor,
    Euler) with two jump variables (consumption, labor). Klein QZ is the
    primary solver because the algebra is harder to redo by hand.

Each case checks the linear policy against an exact nonlinear
perfect-foresight transition for the same TFP path.
"""
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


# =========================================================================
# Case A: Fixed-labor RBC (3x3 system)
# =========================================================================

def steady_state_fixed(alpha: float, beta: float, delta: float) -> dict[str, float]:
    """Deterministic steady state for the fixed-labor RBC model."""
    mpk = 1.0 / beta - 1.0 + delta
    capital = (alpha / mpk) ** (1.0 / (1.0 - alpha))
    output = capital ** alpha
    investment = delta * capital
    consumption = output - investment
    return {
        "K": capital, "Y": output, "C": consumption, "I": investment,
        "K_Y": capital / output, "C_Y": consumption / output,
        "I_Y": investment / output, "mpk": mpk, "gross_return": 1.0 / beta,
    }


def solve_log_linear_policy(
    alpha: float, beta: float, delta: float, rho: float, sigma: float,
    ss: dict[str, float],
) -> dict[str, float]:
    """Hand-derived undetermined-coefficients solve for k_hat = p k_lag + q a."""
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
            c_k * p - (gross_marginal_product_share / sigma) * (alpha - 1.0) * p
        )
        euler_a = c_a - (
            c_k * q + c_a * rho
            - (gross_marginal_product_share / sigma) * (rho + (alpha - 1.0) * q)
        )
        return np.array([euler_k, euler_a])

    solution = root(residual, np.array([0.95, 0.08]))
    if not solution.success:
        raise RuntimeError(f"Could not solve linearized RBC policy: {solution.message}")

    p, q = solution.x
    c_k, c_a = consumption_coefficients(p, q)
    return {
        "p": float(p), "q": float(q), "c_k": float(c_k), "c_a": float(c_a),
        "max_residual": float(np.max(np.abs(residual(solution.x)))),
    }


def klein_qz_policy_fixed(
    alpha: float, beta: float, delta: float, rho: float, sigma: float,
    ss: dict[str, float],
) -> dict[str, float]:
    """Klein QZ cross-check for the fixed-labor 3x3 system."""
    capital_output = ss["K_Y"]
    consumption_share = ss["C_Y"]
    consumption_capital = consumption_share / capital_output
    gross_marginal_product_share = beta * alpha / capital_output

    A = np.array(
        [
            [capital_output, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-(alpha - 1.0) * gross_marginal_product_share,
             -gross_marginal_product_share, sigma],
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
        "p": float(sol.F[0, 0]), "q": float(sol.F[0, 1]),
        "c_k": float(sol.P[0, 0]), "c_a": float(sol.P[0, 1]),
        "blanchard_kahn": sol.bk_message, "eigenvalues": sol.eigenvalues,
    }


def linear_irfs_fixed(
    alpha: float, delta: float, rho: float, shock: float,
    policy: dict[str, float], ss: dict[str, float], periods: int,
) -> dict[str, np.ndarray]:
    """Iterate the fixed-labor decision rule along a decaying TFP path."""
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
    return {"Output": y_hat, "Consumption": c_hat, "Investment": i_hat,
            "Capital": k_hat, "TFP": a_hat}


def nonlinear_irfs_fixed(
    alpha: float, beta: float, delta: float, rho: float, sigma: float,
    shock: float, ss: dict[str, float], initial_guess: np.ndarray, periods: int,
) -> dict[str, np.ndarray]:
    """Exact nonlinear perfect-foresight transition for the fixed-labor case."""
    K_ss, C_ss, Y_ss, I_ss = ss["K"], ss["C"], ss["Y"], ss["I"]
    a_path = shock * rho ** np.arange(periods + 1)
    A_path = np.exp(a_path)

    def residual(log_k_path: np.ndarray) -> np.ndarray:
        K_path = np.empty(periods + 1)
        K_path[:periods] = K_ss * np.exp(log_k_path)
        K_path[periods] = K_ss
        errors = np.empty(periods)
        K_lag = K_ss
        for t in range(periods):
            C_t = A_path[t] * K_lag ** alpha + (1.0 - delta) * K_lag - K_path[t]
            C_next = (A_path[t + 1] * K_path[t] ** alpha
                      + (1.0 - delta) * K_path[t] - K_path[t + 1])
            R_next = (alpha * A_path[t + 1] * K_path[t] ** (alpha - 1.0)
                      + (1.0 - delta))
            if C_t <= 0.0 or C_next <= 0.0 or R_next <= 0.0:
                return np.full(periods, 1e6)
            errors[t] = (-sigma * np.log(C_t) - np.log(beta)
                         + sigma * np.log(C_next) - np.log(R_next))
            K_lag = K_path[t]
        return errors

    sol = root(residual, initial_guess[:periods], method="hybr",
               options={"xtol": 1e-11, "maxfev": 20000})
    if not sol.success:
        raise RuntimeError(f"Nonlinear PF solve failed: {sol.message}")

    K_path = np.empty(periods + 1)
    K_path[:periods] = K_ss * np.exp(sol.x)
    K_path[periods] = K_ss

    y_hat = np.zeros(periods); c_hat = np.zeros(periods); i_hat = np.zeros(periods)
    k_hat = np.log(K_path[:periods] / K_ss)
    K_lag = K_ss
    for t in range(periods):
        Y_t = A_path[t] * K_lag ** alpha
        C_t = Y_t + (1.0 - delta) * K_lag - K_path[t]
        I_t = K_path[t] - (1.0 - delta) * K_lag
        y_hat[t] = np.log(Y_t / Y_ss)
        c_hat[t] = np.log(C_t / C_ss)
        i_hat[t] = np.log(I_t / I_ss)
        K_lag = K_path[t]
    return {"Output": y_hat, "Consumption": c_hat, "Investment": i_hat,
            "Capital": k_hat, "TFP": a_path[:periods],
            "max_residual": float(np.max(np.abs(residual(sol.x))))}


# =========================================================================
# Case B: Endogenous-labor RBC (4x4 system)
# =========================================================================

def steady_state_labor(
    alpha: float, beta: float, delta: float, sigma: float, chi: float,
    n_target: float,
) -> dict[str, float]:
    """Steady state for RBC with endogenous labor."""
    mpk = 1.0 / beta - 1.0 + delta
    k_over_n = (alpha / mpk) ** (1.0 / (1.0 - alpha))
    K = k_over_n * n_target
    Y = K ** alpha * n_target ** (1.0 - alpha)
    I = delta * K
    C = Y - I
    real_wage = (1.0 - alpha) * Y / n_target
    psi = real_wage * C ** (-sigma) / n_target ** chi
    return {"K": K, "Y": Y, "C": C, "I": I, "N": n_target,
            "K_Y": K / Y, "C_Y": C / Y, "I_Y": I / Y,
            "K_over_N": k_over_n, "wage": real_wage, "psi": psi, "mpk": mpk}


def klein_system_labor(
    alpha: float, beta: float, delta: float, rho: float, sigma: float,
    chi: float, ss: dict[str, float],
):
    """Build (A, B) Klein matrices for the 4x4 endogenous-labor system."""
    KY = ss["K_Y"]
    CY = ss["C_Y"]
    CK = CY / KY
    mpk_share = 1.0 - beta * (1.0 - delta)
    A = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [mpk_share * (alpha - 1.0), mpk_share, -sigma,
             mpk_share * (1.0 - alpha)],
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


def linear_irfs_labor(
    F: np.ndarray, P: np.ndarray, ss: dict[str, float],
    alpha: float, delta: float, shock: float, periods: int,
) -> dict[str, np.ndarray]:
    """Iterate the QZ-derived rules x_{t+1} = F x_t, y_t = P x_t."""
    x = np.zeros((periods + 1, 2))
    x[0] = np.array([0.0, shock])
    for t in range(periods):
        x[t + 1] = F @ x[t]
    k_lag = x[:periods, 0]
    a = x[:periods, 1]
    k = x[1:periods + 1, 0]
    c = (P @ x[:periods].T)[0]
    n = (P @ x[:periods].T)[1]
    y = a + alpha * k_lag + (1.0 - alpha) * n
    inv = (k - (1.0 - delta) * k_lag) / delta
    return {"TFP": a, "Output": y, "Consumption": c, "Investment": inv,
            "Capital": k, "Labor": n}


def nonlinear_irfs_labor(
    alpha: float, beta: float, delta: float, rho: float, sigma: float,
    chi: float, ss: dict[str, float], shock: float, periods: int,
) -> dict[str, np.ndarray]:
    """Exact nonlinear perfect-foresight transition for the endogenous-labor case."""
    K_ss, C_ss, N_ss, psi = ss["K"], ss["C"], ss["N"], ss["psi"]
    a_path = shock * rho ** np.arange(periods + 1)
    A_path = np.exp(a_path)

    def production(K_lag, A, N):
        return A * K_lag ** alpha * N ** (1.0 - alpha)

    def labor_static(K_lag, A, C):
        rhs = (1.0 - alpha) * A * K_lag ** alpha / (psi * C ** sigma)
        return rhs ** (1.0 / (chi + alpha))

    def euler_residuals(log_K_path):
        K = np.empty(periods + 1)
        K[:periods] = K_ss * np.exp(log_K_path)
        K[periods] = K_ss
        C = np.empty(periods); N = np.empty(periods)
        C_guess = np.full(periods, C_ss)
        for _ in range(50):
            for t in range(periods):
                K_lag = K_ss if t == 0 else K[t - 1]
                N[t] = labor_static(K_lag, A_path[t], C_guess[t])
                Y_t = production(K_lag, A_path[t], N[t])
                C[t] = Y_t + (1.0 - delta) * K_lag - K[t]
                if C[t] <= 0:
                    return np.full(periods, 1e6)
            if np.max(np.abs(C - C_guess)) < 1e-12:
                break
            C_guess = C.copy()
        errs = np.empty(periods)
        for t in range(periods):
            K_lag = K_ss if t == 0 else K[t - 1]
            N_next = labor_static(K[t], A_path[t + 1],
                                  C_ss if t == periods - 1 else C[t + 1])
            R_next = (alpha * A_path[t + 1] * K[t] ** (alpha - 1.0)
                      * N_next ** (1.0 - alpha) + (1.0 - delta))
            C_next = C_ss if t == periods - 1 else C[t + 1]
            errs[t] = (-sigma * np.log(C[t]) - np.log(beta)
                       + sigma * np.log(C_next) - np.log(R_next))
        return errs

    sol = root(euler_residuals, np.zeros(periods), method="hybr",
               options={"xtol": 1e-10, "maxfev": 30000})
    if not sol.success:
        raise RuntimeError(f"Nonlinear PF solve (labor) failed: {sol.message}")

    K = np.empty(periods + 1); K[:periods] = K_ss * np.exp(sol.x); K[periods] = K_ss
    C = np.empty(periods); N = np.empty(periods)
    for _ in range(50):
        for t in range(periods):
            K_lag = K_ss if t == 0 else K[t - 1]
            N[t] = labor_static(K_lag, A_path[t], C_ss if _ == 0 else C[t])
            Y_t = production(K_lag, A_path[t], N[t])
            C[t] = Y_t + (1.0 - delta) * K_lag - K[t]

    Y = np.array([production(K_ss if t == 0 else K[t - 1], A_path[t], N[t])
                  for t in range(periods)])
    INV = np.array([K[t] - (1.0 - delta) * (K_ss if t == 0 else K[t - 1])
                    for t in range(periods)])
    return {"TFP": a_path[:periods],
            "Output": np.log(Y / ss["Y"]),
            "Consumption": np.log(C / C_ss),
            "Investment": np.log(INV / ss["I"]),
            "Capital": np.log(K[:periods] / K_ss),
            "Labor": np.log(N / N_ss)}


# =========================================================================
# Helpers
# =========================================================================

def half_life_after_peak(series: np.ndarray) -> int | str:
    abs_series = np.abs(series)
    peak_period = int(np.argmax(abs_series))
    peak = abs_series[peak_period]
    below_half = np.where(abs_series[peak_period:] <= peak / 2.0)[0]
    if len(below_half) == 0:
        return f">{len(series) - peak_period - 1}"
    return int(below_half[0])


def format_percent(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f}"


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    # Common primitives
    alpha = 0.33
    beta = 0.99
    delta = 0.025
    rho = 0.95
    sigma = 1.0
    sigma_e = 0.01
    shock = sigma_e
    periods_irf = 40

    # Case B-only primitives
    chi = 1.0
    n_target = 1.0 / 3.0

    # --------------------------------------------------------------
    # Case A: fixed labor (3x3)
    # --------------------------------------------------------------
    print("=" * 72)
    print("Case A. Fixed-labor RBC (3x3 system)")
    print("=" * 72)
    ss_A = steady_state_fixed(alpha, beta, delta)
    policy_A = solve_log_linear_policy(alpha, beta, delta, rho, sigma, ss_A)
    qz_A = klein_qz_policy_fixed(alpha, beta, delta, rho, sigma, ss_A)
    qz_diff_A = max(abs(policy_A["p"] - qz_A["p"]),
                    abs(policy_A["q"] - qz_A["q"]))
    print(f"  K/Y = {ss_A['K_Y']:.2f}, C/Y = {ss_A['C_Y']:.2f}")
    print(f"  Capital rule: k_t = {policy_A['p']:.4f} k_lag + {policy_A['q']:.4f} a_t")
    print(f"  Klein QZ cross-check ({qz_A['blanchard_kahn']}): max abs diff = {qz_diff_A:.2e}")

    linear_A_long = linear_irfs_fixed(alpha, delta, rho, shock, policy_A, ss_A, 120)
    nonlinear_A_long = nonlinear_irfs_fixed(alpha, beta, delta, rho, sigma, shock,
                                            ss_A, linear_A_long["Capital"], 120)
    linear_A = {k: v[:periods_irf] for k, v in linear_A_long.items()}
    nonlinear_A = {k: v[:periods_irf] for k, v in nonlinear_A_long.items()
                   if k != "max_residual"}
    print(f"  Max Euler residual in nonlinear benchmark: "
          f"{nonlinear_A_long['max_residual']:.2e}")

    # --------------------------------------------------------------
    # Case B: endogenous labor (4x4)
    # --------------------------------------------------------------
    print()
    print("=" * 72)
    print("Case B. Endogenous-labor RBC (4x4 system)")
    print("=" * 72)
    ss_B = steady_state_labor(alpha, beta, delta, sigma, chi, n_target)
    A_mat, B_mat = klein_system_labor(alpha, beta, delta, rho, sigma, chi, ss_B)
    sol_B = solve_klein(A_mat, B_mat, n_predetermined=2)
    F, P = sol_B.F, sol_B.P
    print(f"  K/Y = {ss_B['K_Y']:.2f}, C/Y = {ss_B['C_Y']:.2f}, N = {ss_B['N']:.3f}")
    print(f"  Blanchard-Kahn: {sol_B.bk_message}")
    print(f"  Capital rule: k_t = {F[0, 0]:.4f} k_lag + {F[0, 1]:.4f} a_t")
    print(f"  Consumption rule: c_t = {P[0, 0]:.4f} k_lag + {P[0, 1]:.4f} a_t")
    print(f"  Labor rule: n_t = {P[1, 0]:.4f} k_lag + {P[1, 1]:.4f} a_t")

    linear_B = linear_irfs_labor(F, P, ss_B, alpha, delta, shock, periods_irf)
    nonlinear_B = nonlinear_irfs_labor(alpha, beta, delta, rho, sigma, chi, ss_B,
                                       shock, periods_irf)

    # =====================================================================
    # Build report
    # =====================================================================
    setup_style()
    report = ModelReport(
        "Linearized RBC by Perturbation and QZ (with and without endogenous labor)",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A productivity shock changes the marginal product of every input. "
        "Capital was chosen yesterday and cannot move on impact. So the household "
        "carries the shock forward through investment. If labor is endogenous, "
        "hours can move on impact too and share the response.\n\n"
        "The tutorial walks two cases on the same primitives. Case A keeps labor "
        "fixed. The system has three equations and one jump variable. Case B adds "
        "endogenous labor. The system grows to four equations and two jump "
        "variables.\n\n"
        "Two solvers run on the same model, one after the other. The fixed-labor "
        "case is small enough to solve "
        "by hand. We guess a linear capital decision rule and match coefficients on "
        "the linearized Euler equation. Klein QZ generalized-Schur cross-checks the "
        "answer to machine precision. The endogenous-labor case is too messy for "
        "hand algebra. Klein QZ is the primary solver there.\n\n"
        "Each linear solution is checked against the exact nonlinear "
        "perfect-foresight transition for the same TFP path."
    )

    report.add_equations(
        rf"""
The same primitives drive both cases. The only structural difference is
whether labor is fixed or chosen.

### A. Common setup

Let $A_t$ denote total factor productivity, $K_{{t-1}}$ predetermined capital,
$C_t$ consumption, $I_t$ investment, $Y_t$ output, and (when present) $N_t$
hours worked. Production is Cobb-Douglas and the resource constraint splits
output into consumption and investment:

$$Y_t = A_t K_{{t-1}}^\alpha N_t^{{1-\alpha}}, \qquad Y_t = C_t + I_t,$$

$$K_t = I_t + (1-\delta)K_{{t-1}}.$$

TFP follows an AR(1) in logs:

$$\log A_t = \rho \log A_{{t-1}} + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma_\varepsilon^2).$$

The household has CRRA utility over consumption. With endogenous labor it also
dislikes hours:

$$\mathbb{{E}}_0\sum_{{t=0}}^{{\infty}} \beta^t \left[\frac{{C_t^{{1-\sigma}}}}{{1-\sigma}} - \psi \frac{{N_t^{{1+\chi}}}}{{1+\chi}}\right].$$

The labor-disutility weight $\psi$ disappears in Case A because $N_t$ is fixed
at one. The consumption Euler equation is

$$C_t^{{-\sigma}} = \beta\,\mathbb{{E}}_t\left[ C_{{t+1}}^{{-\sigma}} \left(\alpha A_{{t+1}} K_t^{{\alpha-1}} N_{{t+1}}^{{1-\alpha}} + 1 - \delta\right)\right].$$

When labor is endogenous, the intratemporal labor-supply condition adds

$$\psi N_t^\chi = (1-\alpha)\frac{{Y_t}}{{N_t}}\,C_t^{{-\sigma}}.$$

### B. Steady state

At the deterministic steady state ($A=1$, $\varepsilon=0$),

$$\alpha (K/N)^{{\alpha-1}} = \frac{{1}}{{\beta}} - 1 + \delta, \qquad I = \delta K, \qquad C = Y - I.$$

Case A pins $N = 1$. Case B picks $N = \bar N$ as a calibration target and
recovers $\psi$ from the steady-state labor-supply condition. The Case A
calibration gives $K/Y = {ss_A['K_Y']:.2f}$, $C/Y = {ss_A['C_Y']:.2f}$. The
Case B calibration with $\bar N = {n_target:.3f}$ gives $K/Y = {ss_B['K_Y']:.2f}$,
$C/Y = {ss_B['C_Y']:.2f}$, and a labor weight $\psi = {ss_B['psi']:.3f}$.

### C. Linearized system in log deviations

Let a hat denote a log deviation from steady state, so $\hat x_t = \log(X_t/X)$.
Linearizing the equilibrium conditions gives a system of the form

$$A\,\mathbb{{E}}_t s_{{t+1}} = B\,s_t,$$

with $s_t$ stacking the predetermined and jump variables. Predetermined
variables enter at their lagged value. Jump variables can move freely on impact.

Case A. The state vector is $s_t = (\hat k_{{t-1}}, \hat a_t, \hat c_t)$. Two
predetermined entries (capital and TFP), one jump (consumption).

Case B. The state vector is $s_t = (\hat k_{{t-1}}, \hat a_t, \hat c_t, \hat n_t)$.
Two predetermined entries, two jumps. The fourth equation is the labor-supply
condition.

Both cases linearize the same primitives. Case B is the strict augmentation.
"""
    )

    report.add_model_setup(
        "Two cases share most primitives. Case B adds the labor-disutility "
        "parameters.\n\n"
        "**Common primitives.**\n\n"
        "| Primitive | Value | Role |\n"
        "|---|---:|---|\n"
        f"| $\\alpha$ | {alpha:.2f} | Capital share in production |\n"
        f"| $\\beta$ | {beta:.2f} | Quarterly discount factor |\n"
        f"| $\\delta$ | {delta:.3f} | Quarterly depreciation |\n"
        f"| $\\rho$ | {rho:.2f} | Persistence of log TFP |\n"
        f"| $\\sigma$ | {sigma:.1f} | CRRA coefficient (log utility) |\n"
        f"| $\\sigma_\\varepsilon$ | {sigma_e:.3f} | Innovation s.d. of log TFP |\n"
        f"| Shock | {100 * shock:.1f}% | One-s.d. innovation at $t = 0$ |\n"
        f"| IRF horizon | {periods_irf} quarters | Periods plotted |\n\n"
        "**Case B labor block.**\n\n"
        "| Primitive | Value | Role |\n"
        "|---|---:|---|\n"
        f"| $\\chi$ | {chi:.1f} | Inverse Frisch elasticity |\n"
        f"| $\\bar N$ | {n_target:.3f} | Steady-state hours target |\n"
        f"| $\\psi$ | {ss_B['psi']:.3f} | Labor-disutility weight (calibrated) |\n\n"
        "**Steady states.**\n\n"
        "| Object | Case A (fixed labor) | Case B (endogenous labor) |\n"
        "|---|---:|---:|\n"
        f"| $K$ | {ss_A['K']:.3f} | {ss_B['K']:.3f} |\n"
        f"| $Y$ | {ss_A['Y']:.3f} | {ss_B['Y']:.3f} |\n"
        f"| $C$ | {ss_A['C']:.3f} | {ss_B['C']:.3f} |\n"
        f"| $N$ | 1.000 | {ss_B['N']:.3f} |\n"
        f"| $K/Y$ | {ss_A['K_Y']:.3f} | {ss_B['K_Y']:.3f} |\n"
        f"| $C/Y$ | {ss_A['C_Y']:.3f} | {ss_B['C_Y']:.3f} |"
    )

    report.add_solution_method(
        "Two methods run in sequence. Both return a linear policy that maps "
        "states into jumps. The defining linearized equations live in the "
        "Equations section. The pseudocode here uses those symbols by name.\n\n"
        "### Method 1: Method of undetermined coefficients (fixed labor, 3x3)\n\n"
        "Capital is the only true state. Consumption is the one jump variable. "
        "We guess a linear capital decision rule. Then we substitute it into the "
        "linearized resource constraint and Euler equation. Coefficients on "
        "$\\hat k_{t-1}$ and $\\hat a_t$ have to match on both sides. That gives "
        "two equations in two unknowns. The match is exact algebra. Klein QZ on "
        "the same system reproduces $(p, q)$ to machine precision and confirms "
        "Blanchard-Kahn.\n\n"
        "```text\n"
        "Inputs:  alpha, beta, delta, rho, sigma; steady state K/Y, C/Y\n"
        "Outputs: capital decision rule k_t = p * k_lag + q * a_t,\n"
        "         consumption rule    c_t = c_k * k_lag + c_a * a_t\n"
        "\n"
        "1. Compute steady-state ratios K/Y, C/Y, mpk = 1/beta - 1 + delta.\n"
        "2. Linearize resource constraint:\n"
        "      C/Y * c_t + (K/Y) * k_t\n"
        "      = a_t + alpha * k_lag + (K/Y) * (1 - delta) * k_lag\n"
        "3. Linearize Euler equation:\n"
        "      sigma * (c_{t+1} - c_t) = (beta * alpha / (K/Y)) *\n"
        "                                 (a_{t+1} + (alpha - 1) * k_t)\n"
        "4. Guess k_t = p * k_lag + q * a_t, infer c_t = c_k * k_lag + c_a * a_t\n"
        "   from the resource constraint.\n"
        "5. Substitute into the linearized Euler equation.\n"
        "6. Match coefficients on k_lag and a_t -> two-equation root solve for (p, q).\n"
        "7. Cross-check: build (A, B) Klein matrices for the same 3x3 system and\n"
        "   solve via generalized Schur. Verify (p, q) match to ~1e-15.\n"
        "```\n\n"
        f"The undetermined-coefficients residual is {policy_A['max_residual']:.1e}. "
        f"Klein QZ agrees with the hand-derived (p, q) to {qz_diff_A:.1e}. Both "
        "methods isolate the same stable rule.\n\n"
        "### Method 2: Klein QZ on the augmented 4x4 system (endogenous labor)\n\n"
        "Adding labor pushes the system past comfortable hand algebra. The state "
        "vector becomes $s_t = (\\hat k_{t-1}, \\hat a_t, \\hat c_t, \\hat n_t)'$ "
        "with two predetermined entries on top and two jumps below. Klein QZ "
        "computes the ordered generalized Schur decomposition of $(B, A)$, places "
        "the stable roots first, and reads off the state transition $F$ and the "
        "jump rule $P$ from the Schur partition. Blanchard-Kahn determinacy holds "
        "when the number of stable roots equals the number of predetermined "
        "states.\n\n"
        "```text\n"
        "Inputs:  alpha, beta, delta, rho, sigma, chi; steady state with calibrated psi\n"
        "Outputs: state transition F (2x2), jump rule P (2x2)\n"
        "         x_{t+1} = F * x_t,    y_t = P * x_t\n"
        "         x_t = (k_lag, a_t)',  y_t = (c_t, n_t)'\n"
        "\n"
        "1. Build (A, B) for the 4x4 system. Rows: capital accumulation,\n"
        "   TFP AR(1), intratemporal labor supply, intertemporal Euler.\n"
        "   Order entries (k_lag, a, c, n) so the first two are predetermined.\n"
        "2. Compute the ordered generalized Schur decomposition QZ of (B, A),\n"
        "   placing stable roots (|lambda| < 1) first.\n"
        "3. Blanchard-Kahn check: # stable roots == # predetermined states (= 2).\n"
        "4. Partition the Schur vectors into [Z_xx Z_xy; Z_yx Z_yy].\n"
        "5. Recover P = Z_yx * Z_xx^{-1}                        # jump rule\n"
        "6. Recover F = Z_xx * T_xx^{-1} * S_xx * Z_xx^{-1}     # state transition\n"
        "   from the stable triangular blocks T_xx, S_xx.\n"
        "7. Initialize x_0 = (0, sigma_e). Iterate x_{t+1} = F x_t, y_t = P x_t.\n"
        "8. Recover output and investment from production and capital accumulation.\n"
        "```\n\n"
        f"Blanchard-Kahn passes: {sol_B.bk_message}. The capital rule is "
        f"$\\hat k_t = {F[0, 0]:.4f}\\hat k_{{t-1}} + {F[0, 1]:.4f}\\hat a_t$. "
        f"The labor rule is $\\hat n_t = {P[1, 0]:.4f}\\hat k_{{t-1}} + "
        f"{P[1, 1]:.4f}\\hat a_t$. Hours rise with productivity and fall with "
        "inherited capital. Each linear solution is then checked against the exact "
        "nonlinear perfect-foresight transition for the same shock path."
    )

    # =====================================================================
    # Figures and Results
    # =====================================================================

    report.add_results(
        "### Part 1: Fixed-labor case\n\n"
        "Output rises immediately because the same capital is more productive. "
        "Investment jumps more than output because the household wants more capital "
        "while productivity is high. Consumption rises by less on impact and keeps "
        "drifting upward as the Euler equation smooths marginal utility. The dashed "
        "nonlinear transition sits almost on top of the first-order solution at "
        "this shock size."
    )

    variables_A = ["Output", "Consumption", "Investment", "Capital"]
    periods = np.arange(periods_irf)
    fig_A, axes_A = plt.subplots(2, 2, figsize=(11, 8))
    colors_A = {"Output": "#1b6ca8", "Consumption": "#4b8f29",
                "Investment": "#b85c00", "Capital": "#6f4aa8"}
    for ax, var in zip(axes_A.flat, variables_A):
        ax.plot(periods, 100.0 * linear_A[var], color=colors_A[var],
                linewidth=2.3, label="First-order perturbation")
        ax.plot(periods, 100.0 * nonlinear_A[var], color="black",
                linewidth=1.6, linestyle="--", label="Nonlinear transition")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_title(var)
        ax.set_xlabel("Quarters after shock")
        ax.set_ylabel("Percent log deviation")
    axes_A[0, 0].legend(frameon=False, loc="upper right")
    fig_A.suptitle("Case A. Fixed-labor RBC: response to a 1% TFP innovation",
                   fontsize=14, fontweight="bold")
    fig_A.tight_layout(rect=[0, 0, 1, 0.96])
    report.add_figure(
        "figures/irf-fixed-labor.png",
        "Fixed-labor IRFs to a 1% TFP shock",
        fig_A,
        description="The four panels track output, consumption, investment, and "
        "capital. The solid line is the first-order perturbation solution. The "
        "dashed line is the exact nonlinear transition for the same shock path. "
        "At a 1% shock the two lines almost coincide, so the linear solution is "
        "locally accurate.",
    )

    rows_A = []
    for var in ["Output", "Consumption", "Investment", "Capital", "TFP"]:
        series = linear_A_long[var]
        gap = float(np.max(np.abs(
            100.0 * (linear_A_long[var][:periods_irf]
                     - nonlinear_A_long[var][:periods_irf]))))
        peak = int(np.argmax(np.abs(series)))
        rows_A.append({"Variable": var,
                       "Impact (%)": format_percent(series[0]),
                       "Peak (%)": format_percent(series[peak]),
                       "Peak quarter": peak,
                       "Half-life after peak": half_life_after_peak(series),
                       "Max nonlinear gap (pp)": f"{gap:.3f}"})
    df_A = pd.DataFrame(rows_A)
    report.add_table(
        "tables/irf-summary-fixed-labor.csv",
        "Case A: Fixed-Labor IRF Summary",
        df_A,
        description="Capital and consumption peak well after the shock because the "
        "state moves slowly. Investment peaks immediately because it is what moves "
        "the state. The last column is the maximum local-approximation gap to the "
        "nonlinear transition.",
    )

    report.add_results(
        "### Part 2: Endogenous-labor case\n\n"
        "Hours move with productivity. So output rises by more on impact than in "
        "the fixed-labor case. Investment still moves more than consumption because "
        "the marginal product of capital is temporarily high. Consumption is "
        "smoother because the household uses labor to bear part of the shock. The "
        "nonlinear path stays close to the linear one but with slightly larger "
        "gaps for investment and capital, where the resource constraint amplifies "
        "small linearization errors."
    )

    variables_B = ["Output", "Consumption", "Investment", "Labor", "Capital", "TFP"]
    fig_B, axes_B = plt.subplots(2, 3, figsize=(13, 8))
    colors_B = {"Output": "#1b6ca8", "Consumption": "#4b8f29",
                "Investment": "#b85c00", "Labor": "#a3007a",
                "Capital": "#6f4aa8", "TFP": "#444444"}
    for ax, var in zip(axes_B.flat, variables_B):
        ax.plot(periods, 100.0 * linear_B[var], color=colors_B[var],
                linewidth=2.3, label="Klein QZ (linear)")
        ax.plot(periods, 100.0 * nonlinear_B[var], color="black",
                linewidth=1.4, linestyle="--", label="Nonlinear PF")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.6)
        ax.set_title(var)
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Percent")
    axes_B[0, 0].legend(frameon=False, loc="upper right")
    fig_B.suptitle("Case B. Endogenous-labor RBC: response to a 1% TFP innovation",
                   fontsize=14, fontweight="bold")
    fig_B.tight_layout(rect=[0, 0, 1, 0.96])
    report.add_figure(
        "figures/irf-endogenous-labor.png",
        "Endogenous-labor IRFs to a 1% TFP shock",
        fig_B,
        description="Six panels track the same set as Case A plus labor. The solid "
        "line is the Klein QZ linear solution. The dashed line is the nonlinear "
        "perfect-foresight transition. Hours rise with TFP and fall with inherited "
        "capital, in line with the labor-supply rule estimated above.",
    )

    rows_B = []
    for var in variables_B:
        s = linear_B[var]
        gap = float(np.max(np.abs(100.0 * (linear_B[var] - nonlinear_B[var]))))
        rows_B.append({"Variable": var,
                       "Impact (%)": f"{100.0 * s[0]:.3f}",
                       "Peak (%)": f"{100.0 * s[np.argmax(np.abs(s))]:.3f}",
                       "Peak quarter": int(np.argmax(np.abs(s))),
                       "Max linear-vs-PF gap (pp)": f"{gap:.3f}"})
    df_B = pd.DataFrame(rows_B)
    report.add_table(
        "tables/irf-summary-endogenous-labor.csv",
        "Case B: Endogenous-Labor IRF Summary",
        df_B,
        description="The peak quarter for capital is later than the peak for "
        "labor or output. Labor responds on impact, capital builds up over time. "
        "Investment carries most of the savings response.",
    )

    report.add_results(
        "### Part 3: How adding labor changes impact responses\n\n"
        "The clearest comparison is the impact response. Both cases face the same "
        "1% TFP shock and the same primitives apart from the labor block. The bar "
        "chart below puts the impact responses side by side."
    )

    fig_C, ax_C = plt.subplots(figsize=(10, 5))
    common_vars = ["Output", "Consumption", "Investment", "Capital"]
    impact_A_vals = [100.0 * linear_A[v][0] for v in common_vars]
    impact_B_vals = [100.0 * linear_B[v][0] for v in common_vars]
    x_C = np.arange(len(common_vars))
    bar_w_C = 0.35
    ax_C.bar(x_C - bar_w_C / 2, impact_A_vals, bar_w_C,
             label="Case A: fixed labor", color="#1b6ca8")
    ax_C.bar(x_C + bar_w_C / 2, impact_B_vals, bar_w_C,
             label="Case B: endogenous labor", color="#b85c00")
    for i, (a_val, b_val) in enumerate(zip(impact_A_vals, impact_B_vals)):
        ax_C.text(i - bar_w_C / 2, a_val + 0.02, f"{a_val:.2f}",
                  ha="center", va="bottom", fontsize=9)
        ax_C.text(i + bar_w_C / 2, b_val + 0.02, f"{b_val:.2f}",
                  ha="center", va="bottom", fontsize=9)
    ax_C.axhline(0.0, color="black", linewidth=0.6)
    ax_C.set_xticks(x_C)
    ax_C.set_xticklabels(common_vars)
    ax_C.set_ylabel("Impact response (%, log deviation)")
    ax_C.set_title("Impact responses to a 1% TFP shock: with vs without endogenous labor")
    ax_C.legend(frameon=False, loc="upper right")
    fig_C.tight_layout()
    report.add_figure(
        "figures/impact-comparison.png",
        "Impact responses with and without endogenous labor",
        fig_C,
        description="Adding endogenous labor lifts the impact response of output "
        "because hours co-move with productivity. Investment also rises on impact "
        "in both cases, but the gap is larger when labor moves. Capital is "
        "unchanged on impact in both cases because it is predetermined. "
        "Consumption is nearly the same because the household smooths it through "
        "the Euler equation in either setup.",
    )

    # =====================================================================
    # Takeaway
    # =====================================================================
    report.add_takeaway(
        "Fixing labor or letting it move changes how a TFP shock propagates. With "
        "fixed labor, output rises only because TFP and inherited capital matter. "
        "Investment carries the entire savings response. With endogenous labor, "
        "hours rise too, output rises more on impact, and the household uses labor "
        "to share part of the burden with consumption smoothing.\n\n"
        "Two solvers see the same model. Hand-derived undetermined coefficients are "
        "transparent enough to be auditable for the 3x3 fixed-labor system. The "
        "Klein QZ generalized-Schur decomposition handles the 4x4 endogenous-labor "
        "system without writing the algebra by hand. Both solvers agree on the "
        "fixed-labor coefficients to machine precision.\n\n"
        "First-order perturbation is locally accurate at a 1% shock. The dashed "
        "nonlinear paths sit close to the linear ones in both cases. Larger shocks "
        "and stronger nonlinearities would widen the gap. The check is cheap and "
        "should be a habit rather than an afterthought."
    )

    report.add_references([
        "Kydland, F. and Prescott, E. (1982). Time to Build and Aggregate Fluctuations. *Econometrica*, 50(6), 1345-1370.",
        "King, R., Plosser, C., and Rebelo, S. (1988). Production, Growth and Business Cycles: I. The Basic Neoclassical Model. *Journal of Monetary Economics*, 21(2-3), 195-232.",
        "Hansen, G. (1985). Indivisible Labor and the Business Cycle. *Journal of Monetary Economics*, 16(3), 309-327.",
        "Uhlig, H. (1999). A Toolkit for Analysing Nonlinear Dynamic Stochastic Models Easily. In *Computational Methods for the Study of Dynamic Economies*.",
        "Klein, P. (2000). Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model. *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.",
        "Villemot, S. (2011). Solving Rational Expectations Models at First Order: What Dynare Does. *Dynare Working Paper 2*, CEPREMAP.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
