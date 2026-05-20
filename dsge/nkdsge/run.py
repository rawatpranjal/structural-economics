#!/usr/bin/env python3
"""Sticky-price monetary transmission in a New Keynesian DSGE.

The tutorial keeps the three-equation New Keynesian block visible and solves
its rational-expectations impulse responses by undetermined coefficients.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style
from lib.perturbation import solve_klein


def klein_qz_nk(sigma, beta, phi_pi, phi_y, kappa, rho_shock, shock_kind):
    """Solve the same 3-equation NK system by Klein-style generalized Schur.

    State ordering s = (shock, y, pi) with shock predetermined and (y, pi)
    forward-looking. ``shock_kind`` is "monetary" (Taylor-rule shock) or
    "demand" (natural-rate shock); the only difference is the sign and
    placement of the shock loading in the IS curve.
    """
    if shock_kind == "monetary":
        v_in_is = 1.0 / sigma
    elif shock_kind == "demand":
        v_in_is = -1.0
    else:
        raise ValueError(shock_kind)
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0 / sigma],
            [0.0, 0.0, beta],
        ]
    )
    B = np.array(
        [
            [rho_shock, 0.0, 0.0],
            [v_in_is, 1.0 + phi_y / sigma, phi_pi / sigma],
            [0.0, -kappa, 1.0],
        ]
    )
    sol = solve_klein(A, B, n_predetermined=1)
    return {
        "psi_y": float(sol.P[0, 0]),
        "psi_pi": float(sol.P[1, 0]),
        "blanchard_kahn": sol.bk_message,
        "eigenvalues": sol.eigenvalues,
    }


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


def compute_irfs_nk(coeffs, shock_persistence, shock_size, T=40):
    """Compute IRFs from the solution coefficients."""
    periods = np.arange(T)
    shock_path = shock_size * shock_persistence ** periods

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
    # Read the model spec file (DSL syntax; not executed)
    # =========================================================================
    mod_dir = Path(__file__).resolve().parent
    mod_text = parse_mod_file(mod_dir / "model.mod")
    print("Read model.mod for New Keynesian DSGE (textbook spec; not executed by run.py)")
    stale_equation_figure = mod_dir / "figures" / "model-equations.png"
    if stale_equation_figure.exists():
        stale_equation_figure.unlink()

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
    sigma_e = 0.01  # One-percentage-point innovation

    # =========================================================================
    # Solve the model
    # =========================================================================
    print("Solving 3-equation NK model via undetermined coefficients...")

    # Monetary policy shock
    mp_sol = solve_nk_model(sigma, beta, phi_pi, phi_y, kappa, rho_v)
    mp_qz = klein_qz_nk(sigma, beta, phi_pi, phi_y, kappa, rho_v, "monetary")
    mp_qz_diff = max(
        abs(mp_sol["psi_yv"] - mp_qz["psi_y"]),
        abs(mp_sol["psi_piv"] - mp_qz["psi_pi"]),
    )
    print(f"  Monetary shock: psi_y={mp_sol['psi_yv']:.4f}, psi_pi={mp_sol['psi_piv']:.4f}")
    print(f"    Klein QZ ({mp_qz['blanchard_kahn']}): max abs diff = {mp_qz_diff:.2e}")

    # Demand shock
    d_sol = solve_nk_demand_shock(sigma, beta, phi_pi, phi_y, kappa, rho_d)
    d_qz = klein_qz_nk(sigma, beta, phi_pi, phi_y, kappa, rho_d, "demand")
    d_qz_diff = max(
        abs(d_sol["psi_yd"] - d_qz["psi_y"]),
        abs(d_sol["psi_pid"] - d_qz["psi_pi"]),
    )
    print(f"  Demand shock:   psi_y={d_sol['psi_yd']:.4f}, psi_pi={d_sol['psi_pid']:.4f}")
    print(f"    Klein QZ ({d_qz['blanchard_kahn']}): max abs diff = {d_qz_diff:.2e}")

    # =========================================================================
    # Compute IRFs
    # =========================================================================
    T_irf = 40
    mp_irfs = compute_irfs_nk(
        {"psi_y": mp_sol["psi_yv"], "psi_pi": mp_sol["psi_piv"], "psi_i": mp_sol["psi_iv"]},
        rho_v, sigma_e, T_irf
    )
    d_irfs = compute_irfs_nk(
        {"psi_y": d_sol["psi_yd"], "psi_pi": d_sol["psi_pid"], "psi_i": d_sol["psi_id"]},
        rho_d, sigma_e, T_irf
    )
    print("  IRFs computed for 40 periods.")

    # =========================================================================
    # Figures
    # =========================================================================
    setup_style()

    periods = np.arange(T_irf)

    # --- Figure 1: IRFs to monetary policy shock ---
    fig1, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(periods, mp_irfs["output"] * 100, "#2c7bb6", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Output gap ($y_t$)")
    ax.set_ylabel("Percent or pp")

    ax = axes[0, 1]
    ax.plot(periods, mp_irfs["inflation"] * 100, "#d7191c", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Inflation ($\\pi_t$)")
    ax.set_ylabel("Percent or pp")

    ax = axes[1, 0]
    ax.plot(periods, mp_irfs["interest_rate"] * 100, "#fdae61", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Policy rate ($i_t$)")
    ax.set_xlabel("Quarters")
    ax.set_ylabel("Percent or pp")

    ax = axes[1, 1]
    ax.plot(periods, mp_irfs["shock"] * 100, "#018571", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Policy wedge ($v_t$)")
    ax.set_xlabel("Quarters")
    ax.set_ylabel("Percent or pp")

    fig1.suptitle("One-Percentage-Point Monetary Tightening", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig1, "figures/irf-monetary-shock.png", dpi=150)

    # --- Figure 2: IRFs to demand shock ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))

    axes2[0].plot(periods, d_irfs["output"] * 100, "#2c7bb6", linewidth=2.5)
    axes2[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes2[0].set_title("Output gap ($y_t$)")
    axes2[0].set_xlabel("Quarters")
    axes2[0].set_ylabel("Percent or pp")

    axes2[1].plot(periods, d_irfs["inflation"] * 100, "#d7191c", linewidth=2.5)
    axes2[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes2[1].set_title("Inflation ($\\pi_t$)")
    axes2[1].set_xlabel("Quarters")

    axes2[2].plot(periods, d_irfs["interest_rate"] * 100, "#fdae61", linewidth=2.5)
    axes2[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes2[2].set_title("Policy rate ($i_t$)")
    axes2[2].set_xlabel("Quarters")

    fig2.suptitle("One-Percentage-Point Demand Shock", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig2, "figures/irf-demand-shock.png", dpi=150)

    # --- Table ---
    mp_summary = {
        "Variable": ["Output gap", "Inflation", "Nominal rate"],
        "Monetary shock impact": [
            f"{mp_irfs['output'][0]*100:.3f}",
            f"{mp_irfs['inflation'][0]*100:.3f}",
            f"{mp_irfs['interest_rate'][0]*100:.3f}",
        ],
        "Demand shock impact": [
            f"{d_irfs['output'][0]*100:.3f}",
            f"{d_irfs['inflation'][0]*100:.3f}",
            f"{d_irfs['interest_rate'][0]*100:.3f}",
        ],
    }
    df = pd.DataFrame(mp_summary)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/impact-responses.csv", index=False)

    # Thumbnail from first figure
    save_thumbnail("figures/irf-monetary-shock.png", "figures/thumb.png")

    print(f"\nSaved: 2 figures + 1 table")


if __name__ == "__main__":
    main()
