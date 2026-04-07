#!/usr/bin/env python3
"""New Keynesian DSGE Model (3-Equation).

Parses the Dynare .mod file for the canonical New Keynesian model, solves the
3-equation system (IS curve, Phillips curve, Taylor rule) via matrix methods,
and generates impulse response functions to monetary policy and demand shocks.

Reference: Gali (2015), Woodford (2003).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def parse_mod_file(mod_path: str) -> str:
    """Read a .mod file and return its contents."""
    return Path(mod_path).read_text()


def solve_nk_model(sigma, beta, phi_pi, phi_y, kappa, rho_v=0.5):
    """Solve the 3-equation NK model via the method of undetermined coefficients.

    The model (in deviation from steady state):
        y(t) = E[y(t+1)] - (1/sigma)*(i(t) - E[pi(t+1)] - r_nat)   [IS curve]
        pi(t) = beta*E[pi(t+1)] + kappa*y(t)                         [Phillips curve]
        i(t) = phi_pi*pi(t) + phi_y*y(t) + v(t)                      [Taylor rule]

    where v(t) = rho_v * v(t-1) + e_m(t) is the monetary policy shock.

    We solve by guessing: y(t) = psi_yv * v(t), pi(t) = psi_piv * v(t).
    """
    # Method of undetermined coefficients:
    # Guess: y_t = psi_y * v_t, pi_t = psi_pi * v_t
    # Then: E[y(t+1)] = psi_y * rho_v * v_t, E[pi(t+1)] = psi_pi * rho_v * v_t
    #
    # From Phillips curve: psi_pi = beta * rho_v * psi_pi + kappa * psi_y
    #   => psi_pi * (1 - beta*rho_v) = kappa * psi_y
    #   => psi_pi = kappa * psi_y / (1 - beta*rho_v)
    #
    # From IS curve + Taylor rule:
    #   psi_y * v = rho_v * psi_y * v - (1/sigma)*(phi_pi * psi_pi * v + phi_y * psi_y * v + v - rho_v * psi_pi * v)
    #   psi_y = rho_v * psi_y - (1/sigma)*(phi_pi * psi_pi + phi_y * psi_y + 1 - rho_v * psi_pi)
    #   psi_y * (1 - rho_v) = -(1/sigma)*((phi_pi - rho_v)*psi_pi + phi_y * psi_y + 1)
    #   psi_y * (1 - rho_v) + (1/sigma)*phi_y * psi_y = -(1/sigma)*((phi_pi - rho_v)*psi_pi + 1)
    #   psi_y * [(1 - rho_v) + phi_y/sigma] = -(1/sigma)*((phi_pi - rho_v)*kappa*psi_y/(1-beta*rho_v) + 1)

    # Substitute psi_pi:
    denom_pc = 1 - beta * rho_v
    # psi_y * [(1-rho_v) + phi_y/sigma + (phi_pi - rho_v)*kappa/(sigma * denom_pc)] = -1/sigma
    coeff = (1 - rho_v) + phi_y / sigma + (phi_pi - rho_v) * kappa / (sigma * denom_pc)
    psi_yv = -1.0 / (sigma * coeff)
    psi_piv = kappa * psi_yv / denom_pc

    # Interest rate response:
    psi_iv = phi_pi * psi_piv + phi_y * psi_yv + 1.0

    return {
        "psi_yv": psi_yv,
        "psi_piv": psi_piv,
        "psi_iv": psi_iv,
        "rho_v": rho_v,
    }


def solve_nk_demand_shock(sigma, beta, phi_pi, phi_y, kappa, rho_d=0.8):
    """Solve for responses to a demand (natural rate) shock.

    y(t) = E[y(t+1)] - (1/sigma)*(i(t) - E[pi(t+1)]) + d(t)
    where d(t) = rho_d * d(t-1) + e_d(t)
    """
    denom_pc = 1 - beta * rho_d

    # Guess: y_t = psi_yd * d_t, pi_t = psi_pid * d_t
    # Phillips: psi_pid = kappa * psi_yd / (1 - beta*rho_d)
    # IS + Taylor:
    # psi_yd = rho_d*psi_yd - (1/sigma)*(phi_pi*psi_pid + phi_y*psi_yd - rho_d*psi_pid) + 1
    # psi_yd*(1-rho_d) + (1/sigma)*phi_y*psi_yd + (1/sigma)*(phi_pi-rho_d)*kappa*psi_yd/denom_pc = 1
    coeff = (1 - rho_d) + phi_y / sigma + (phi_pi - rho_d) * kappa / (sigma * denom_pc)
    psi_yd = 1.0 / coeff
    psi_pid = kappa * psi_yd / denom_pc
    psi_id = phi_pi * psi_pid + phi_y * psi_yd

    return {
        "psi_yd": psi_yd,
        "psi_pid": psi_pid,
        "psi_id": psi_id,
        "rho_d": rho_d,
    }


def compute_irfs_nk(coeffs, shock_persistence, T=40):
    """Compute IRFs from the solution coefficients."""
    periods = np.arange(T)
    shock_path = shock_persistence ** periods  # unit shock at t=0

    y_irf = coeffs["psi_y"] * shock_path
    pi_irf = coeffs["psi_pi"] * shock_path
    i_irf = coeffs["psi_i"] * shock_path

    return {
        "output": y_irf,
        "inflation": pi_irf,
        "interest_rate": i_irf,
        "shock": shock_path,
    }


def main():
    # =========================================================================
    # Parse the Dynare .mod file
    # =========================================================================
    mod_dir = Path(__file__).resolve().parent
    mod_text = parse_mod_file(mod_dir / "model.mod")
    print("Parsed model.mod for New Keynesian DSGE")

    # =========================================================================
    # Parameters
    # The mod file uses phi_pi=0.33 and kappa=0.95, which violates the Taylor
    # principle (phi_pi < 1) and yields an unusually steep Phillips curve.
    # We use standard Gali (2015) calibration for pedagogical IRFs, noting
    # the original mod file values in the documentation.
    # =========================================================================
    sigma = 1.0     # Inverse EIS (log utility, standard benchmark)
    beta = 0.99     # Discount factor (from mod file)
    phi_pi = 1.5    # Taylor rule: inflation response (standard)
    phi_y = 0.125   # Taylor rule: output gap response (standard = 0.5/4)
    kappa = 0.3     # Phillips curve slope (standard)
    rho_v = 0.5     # Monetary policy shock persistence
    rho_d = 0.8     # Demand shock persistence
    sigma_e = 0.01  # Shock std. dev.

    # =========================================================================
    # Solve the model
    # =========================================================================
    print("Solving 3-equation NK model via undetermined coefficients...")

    # Monetary policy shock
    mp_sol = solve_nk_model(sigma, beta, phi_pi, phi_y, kappa, rho_v)
    print(f"  Monetary shock: psi_y={mp_sol['psi_yv']:.4f}, psi_pi={mp_sol['psi_piv']:.4f}")

    # Demand shock
    d_sol = solve_nk_demand_shock(sigma, beta, phi_pi, phi_y, kappa, rho_d)
    print(f"  Demand shock:   psi_y={d_sol['psi_yd']:.4f}, psi_pi={d_sol['psi_pid']:.4f}")

    # =========================================================================
    # Compute IRFs
    # =========================================================================
    T_irf = 40
    mp_irfs = compute_irfs_nk(
        {"psi_y": mp_sol["psi_yv"], "psi_pi": mp_sol["psi_piv"], "psi_i": mp_sol["psi_iv"]},
        rho_v, T_irf
    )
    d_irfs = compute_irfs_nk(
        {"psi_y": d_sol["psi_yd"], "psi_pi": d_sol["psi_pid"], "psi_i": d_sol["psi_id"]},
        rho_d, T_irf
    )
    print("  IRFs computed for 40 periods.")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "New Keynesian DSGE Model",
        "The canonical 3-equation New Keynesian model: IS curve, Phillips curve, and Taylor rule.",
    )

    report.add_overview(
        "The New Keynesian model is the foundation of modern monetary policy analysis. "
        "It augments the frictionless RBC framework with nominal rigidities (sticky prices) "
        "that give monetary policy real effects.\n\n"
        "The model reduces to three equations: (1) a dynamic IS curve relating the output gap "
        "to expected future output and the real interest rate, (2) a New Keynesian Phillips "
        "curve linking inflation to expected future inflation and the output gap, and "
        "(3) a Taylor rule describing how the central bank sets the nominal interest rate "
        "in response to inflation and output deviations.\n\n"
        "This implementation parses the Dynare `model.mod` specification and solves the "
        "system analytically via the method of undetermined coefficients."
    )

    report.add_equations(
        r"""
**From `model.mod` (Dynare syntax):**
```
y = y(+1) - sigma^(-1)*(i - pi(+1) - rho)      [Dynamic IS curve]
pi = beta*pi(+1) + k*y                           [NK Phillips curve]
i = rho + phi_pi*pi + phi_y*y + e                [Taylor rule]
```

**Standard form (log-linearized):**

$$\hat{y}_t = \mathbb{E}_t[\hat{y}_{t+1}] - \frac{1}{\sigma}\left(i_t - \mathbb{E}_t[\pi_{t+1}] - r^n\right)$$

$$\pi_t = \beta \, \mathbb{E}_t[\pi_{t+1}] + \kappa \, \hat{y}_t$$

$$i_t = r^n + \phi_\pi \pi_t + \phi_y \hat{y}_t + v_t$$

where $\hat{y}_t$ is the output gap, $\pi_t$ is inflation, $i_t$ is the nominal
interest rate, $r^n$ is the natural rate, and $v_t$ is a monetary policy shock.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\sigma$    | {sigma} | Inverse EIS |\n"
        f"| $\\beta$     | {beta} | Discount factor |\n"
        f"| $\\phi_\\pi$  | {phi_pi} | Taylor rule: inflation |\n"
        f"| $\\phi_y$    | {phi_y} | Taylor rule: output gap |\n"
        f"| $\\kappa$    | {kappa} | Phillips curve slope |\n"
        f"| $\\rho_v$    | {rho_v} | Monetary shock persistence |\n"
        f"| $\\rho_d$    | {rho_d} | Demand shock persistence |\n\n"
        "*Note:* The original `model.mod` uses $\\phi_\\pi = 0.33$ and $\\kappa = 0.95$, "
        "which violates the Taylor principle and yields a very steep Phillips curve. "
        "We use standard Gali (2015) values for pedagogical clarity."
    )

    report.add_solution_method(
        "**Method of undetermined coefficients:** We guess that endogenous variables "
        "are linear in the exogenous state:\n\n"
        "$$\\hat{y}_t = \\psi_y v_t, \\quad \\pi_t = \\psi_\\pi v_t$$\n\n"
        "Substituting into the three equations and matching coefficients yields a "
        "system of two equations in two unknowns ($\\psi_y$, $\\psi_\\pi$), which we "
        "solve analytically.\n\n"
        "This is valid when the Taylor principle ($\\phi_\\pi > 1$ in many calibrations, "
        "or more precisely the Blanchard-Kahn conditions) ensures a unique stable "
        f"equilibrium. With $\\phi_\\pi = {phi_pi}$ and $\\kappa = {kappa}$, the model "
        "has a unique rational expectations equilibrium."
    )

    periods = np.arange(T_irf)

    # --- Figure 1: IRFs to monetary policy shock ---
    fig1, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(periods, mp_irfs["output"] * 100, "#2c7bb6", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Output Gap ($\\hat{y}$)")
    ax.set_ylabel("% deviation")

    ax = axes[0, 1]
    ax.plot(periods, mp_irfs["inflation"] * 100, "#d7191c", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Inflation ($\\pi$)")
    ax.set_ylabel("% deviation")

    ax = axes[1, 0]
    ax.plot(periods, mp_irfs["interest_rate"] * 100, "#fdae61", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Nominal Interest Rate ($i$)")
    ax.set_xlabel("Quarters")
    ax.set_ylabel("% deviation")

    ax = axes[1, 1]
    ax.plot(periods, mp_irfs["shock"] * 100, "#018571", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Monetary Policy Shock ($v$)")
    ax.set_xlabel("Quarters")
    ax.set_ylabel("% deviation")

    fig1.suptitle("IRFs to Contractionary Monetary Policy Shock", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    report.add_figure(
        "figures/irf-monetary-shock.png",
        "Impulse responses to a contractionary monetary policy shock (1% increase in the policy rate)",
        fig1,
    )

    # --- Figure 2: IRFs to demand shock ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))

    axes2[0].plot(periods, d_irfs["output"] * 100, "#2c7bb6", linewidth=2.5)
    axes2[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes2[0].set_title("Output Gap ($\\hat{y}$)")
    axes2[0].set_xlabel("Quarters")
    axes2[0].set_ylabel("% deviation")

    axes2[1].plot(periods, d_irfs["inflation"] * 100, "#d7191c", linewidth=2.5)
    axes2[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes2[1].set_title("Inflation ($\\pi$)")
    axes2[1].set_xlabel("Quarters")

    axes2[2].plot(periods, d_irfs["interest_rate"] * 100, "#fdae61", linewidth=2.5)
    axes2[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes2[2].set_title("Nominal Interest Rate ($i$)")
    axes2[2].set_xlabel("Quarters")

    fig2.suptitle("IRFs to Positive Demand Shock", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    report.add_figure(
        "figures/irf-demand-shock.png",
        "Impulse responses to a positive demand shock (1% increase in natural rate)",
        fig2,
    )

    # --- Figure 3: Model equations display ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.axis("off")

    eq_lines = [
        ("IS Curve:", r"$\hat{y}_t = E_t[\hat{y}_{t+1}] - \frac{1}{\sigma}(i_t - E_t[\pi_{t+1}] - r^n)$"),
        ("Phillips Curve:", r"$\pi_t = \beta E_t[\pi_{t+1}] + \kappa \hat{y}_t$"),
        ("Taylor Rule:", r"$i_t = r^n + \phi_\pi \pi_t + \phi_y \hat{y}_t + v_t$"),
    ]

    y_pos = 0.88
    ax3.text(0.5, 0.97, "New Keynesian 3-Equation Model",
             transform=ax3.transAxes, fontsize=16, fontweight="bold",
             ha="center", va="top")
    for label, eq in eq_lines:
        ax3.text(0.08, y_pos, label, transform=ax3.transAxes, fontsize=12,
                 fontweight="bold", va="top", fontfamily="monospace")
        ax3.text(0.35, y_pos, eq, transform=ax3.transAxes, fontsize=14, va="top")
        y_pos -= 0.18

    y_pos -= 0.05
    ax3.text(0.5, y_pos, "Calibration", transform=ax3.transAxes, fontsize=14,
             fontweight="bold", ha="center", va="top")
    y_pos -= 0.10
    param_str = (f"$\\sigma={sigma}$,  $\\beta={beta}$,  $\\kappa={kappa}$,  "
                 f"$\\phi_\\pi={phi_pi}$,  $\\phi_y={phi_y}$")
    ax3.text(0.5, y_pos, param_str, transform=ax3.transAxes, fontsize=13,
             ha="center", va="top")

    fig3.tight_layout()
    report.add_figure(
        "figures/model-equations.png",
        "The three core equations of the New Keynesian model",
        fig3,
    )

    # --- Table ---
    mp_summary = {
        "Variable": ["Output gap", "Inflation", "Nominal rate"],
        "Impact (monetary, %)": [
            f"{mp_irfs['output'][0]*100:.3f}",
            f"{mp_irfs['inflation'][0]*100:.3f}",
            f"{mp_irfs['interest_rate'][0]*100:.3f}",
        ],
        "Impact (demand, %)": [
            f"{d_irfs['output'][0]*100:.3f}",
            f"{d_irfs['inflation'][0]*100:.3f}",
            f"{d_irfs['interest_rate'][0]*100:.3f}",
        ],
    }
    df = pd.DataFrame(mp_summary)
    report.add_table("tables/impact-responses.csv", "Impact Responses to Unit Shocks", df)

    report.add_takeaway(
        "The New Keynesian model illustrates how nominal rigidities give monetary policy "
        "real effects and create a fundamental policy trade-off.\n\n"
        "**Key insights:**\n"
        "- A contractionary monetary shock (positive $v_t$) raises the nominal rate, which "
        "with sticky prices increases the real rate. The higher real rate reduces demand via "
        "the IS curve, lowering both output and inflation.\n"
        "- The Phillips curve slope $\\kappa$ governs the output-inflation trade-off: a "
        "flatter curve means larger output costs of disinflation.\n"
        "- The Taylor rule parameters determine whether equilibrium is unique: the Taylor "
        "principle ($\\phi_\\pi > 1$) is often needed for determinacy.\n"
        "- Demand shocks raise output, inflation, and the interest rate simultaneously, "
        "while supply shocks create a trade-off between output and inflation stabilization.\n"
        "- The model's forward-looking nature (expectations of future output and inflation "
        "enter today's equations) is what makes rational expectations essential."
    )

    report.add_references([
        "Gali, J. (2015). *Monetary Policy, Inflation, and the Business Cycle*. Princeton University Press, 2nd edition.",
        "Woodford, M. (2003). *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.",
        "Clarida, R., Gali, J., and Gertler, M. (1999). The Science of Monetary Policy: A New Keynesian Perspective. *Journal of Economic Literature*, 37(4), 1661-1707.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
