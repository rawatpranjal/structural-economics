#!/usr/bin/env python3
"""New Keynesian monetary shocks and determinacy.

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
from lib.plotting import setup_style
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
    # Parse the Dynare .mod file
    # =========================================================================
    mod_dir = Path(__file__).resolve().parent
    mod_text = parse_mod_file(mod_dir / "model.mod")
    print("Parsed model.mod for New Keynesian DSGE")
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
        rho_v, sigma_e, T_irf
    )
    d_irfs = compute_irfs_nk(
        {"psi_y": d_sol["psi_yd"], "psi_pi": d_sol["psi_pid"], "psi_i": d_sol["psi_id"]},
        rho_d, sigma_e, T_irf
    )
    print("  IRFs computed for 40 periods.")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "New Keynesian Monetary Shocks and Determinacy",
        "Sticky prices, the IS curve, Phillips curve, and Taylor rule in a three-equation DSGE model.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The question is why a central bank can move real activity when households and firms "
        "understand the policy rule. In this small New Keynesian model the answer is sticky "
        "prices. A surprise increase in the nominal interest rate raises the real rate before "
        "prices fully adjust, so demand falls. The Phillips curve then translates the weaker "
        "output gap into lower inflation.\n\n"
        "The model is deliberately small: an output gap $y_t$, inflation $\\pi_t$, a nominal "
        "policy rate $i_t$, and one persistent shock at a time. The `model.mod` file gives the "
        "Dynare-style three-equation block. The Python code solves the same log-linear system "
        "directly, which makes the expectations algebra and the Taylor-rule determinacy "
        "condition easy to inspect.\n\n"
        "Compared with the [RBC Dynare tutorial](../rbc/), propagation here does not come from "
        "slow capital accumulation. It comes from the interaction between forward-looking "
        "demand, sticky-price inflation, and a policy rule that leans against inflation."
    )

    report.add_equations(
        r"""
All variables are deviations from the zero-inflation steady state. Let $y_t$ be
the output gap, $\pi_t$ inflation, $i_t$ the nominal policy rate, and $r^n_t$ the
natural real rate. The three equations are

$$
y_t =
\mathbb{E}_t y_{t+1} - \frac{1}{\sigma}
\left(i_t-\mathbb{E}_t\pi_{t+1}-r^n_t\right),
$$

$$
\pi_t = \beta \mathbb{E}_t \pi_{t+1}+\kappa y_t,
$$

$$
i_t = \phi_\pi \pi_t+\phi_y y_t+v_t.
$$

The monetary-policy disturbance follows

$$
v_t=\rho_v v_{t-1}+\varepsilon^v_t,
$$

and the demand experiment treats the natural-rate term as

$$
r^n_t=d_t,\qquad d_t=\rho_d d_{t-1}+\varepsilon^d_t.
$$

The Dynare file writes the same core block as

```text
y = y(+1) - sigma^(-1)*(i - pi(+1) - rho)
pi = beta*pi(+1) + k*y
i = rho + phi_pi*pi + phi_y*y + e
```

The Python report uses $v_t$ for the Taylor-rule shock and $d_t$ for the
natural-rate shifter so the two impulse responses can be read separately.
"""
    )

    report.add_model_setup(
        f"| Primitive | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| $\\sigma$ | {sigma:.3g} | Inverse EIS in the IS curve |\n"
        f"| $\\beta$ | {beta:.3g} | Quarterly discount factor |\n"
        f"| $\\kappa$ | {kappa:.3g} | Slope of the New Keynesian Phillips curve |\n"
        f"| $\\phi_\\pi$ | {phi_pi:.3g} | Taylor-rule response to inflation |\n"
        f"| $\\phi_y$ | {phi_y:.3g} | Taylor-rule response to the output gap |\n"
        f"| $\\rho_v$ | {rho_v:.3g} | Persistence of the policy shock |\n"
        f"| $\\rho_d$ | {rho_d:.3g} | Persistence of the demand shock |\n"
        f"| Shock innovation | {sigma_e:.3f} | One-percentage-point innovation at date 0 |\n"
        f"| IRF horizon | {T_irf} quarters | Periods shown in each impulse response |\n\n"
        "The source `model.mod` uses $\\phi_\\pi=0.33$ and $\\kappa=0.95$. This report "
        "uses a standard determinate calibration, $\\phi_\\pi=1.5$ and $\\kappa=0.3$, "
        "because the economic point is monetary transmission under a stable Taylor "
        "rule. The contrast matters: when policy fails to lean hard enough against "
        "inflation, the forward-looking system no longer selects a unique stable path."
    )

    report.add_solution_method(
        "For either shock, write the scalar state as $s_t=\\rho_s s_{t-1}+\\varepsilon_t$. "
        "Because the model is already log-linear, the rational-expectations solution is "
        "linear in that state:\n\n"
        "$$y_t=\\psi_y s_t,\\qquad \\pi_t=\\psi_\\pi s_t. $$\n\n"
        "The Phillips curve gives\n\n"
        "$$\\psi_\\pi=\\frac{\\kappa\\psi_y}{1-\\beta\\rho_s}. $$\n\n"
        "The IS curve and Taylor rule then pin down $\\psi_y$. For a monetary-policy "
        "shock, the right-hand side is negative because $v_t$ raises the policy rate. "
        "For a demand shock, the right-hand side is positive because $d_t$ raises the "
        "natural rate:\n\n"
        "$$\\psi_y\\left[(1-\\rho_s)+\\frac{\\phi_y}{\\sigma}"
        "+\\frac{(\\phi_\\pi-\\rho_s)\\kappa}{\\sigma(1-\\beta\\rho_s)}\\right]"
        "= b_s,$$\n\n"
        "where $b_s=-1/\\sigma$ for $s_t=v_t$ and $b_s=1$ for $s_t=d_t$.\n\n"
        "```text\n"
        "Algorithm: New Keynesian impulse responses\n"
        "Inputs: beta, sigma, kappa, phi_pi, phi_y, rho_s, shock eps_0, horizon T\n"
        "Outputs: paths for y_t, pi_t, i_t, and the shock state s_t\n\n"
        "1. Pick the shock experiment: monetary policy v_t or natural-rate demand d_t.\n"
        "2. Guess y_t = psi_y s_t and pi_t = psi_pi s_t.\n"
        "3. Use the Phillips curve to express psi_pi in terms of psi_y.\n"
        "4. Substitute both coefficients into the IS curve and Taylor rule.\n"
        "5. Match coefficients on s_t to solve for psi_y, then recover psi_pi.\n"
        "6. Recover the policy-rate coefficient psi_i from the Taylor rule.\n"
        "7. Set s_0 = eps_0 and iterate s_t = rho_s s_{t-1} for t = 1,...,T.\n"
        "8. Plot y_t = psi_y s_t, pi_t = psi_pi s_t, and i_t = psi_i s_t.\n"
        "```\n\n"
        "There is no finer-grid benchmark to add here. Within this tutorial's "
        "log-linear model, coefficient matching is the exact solution. Approximation "
        "error would enter only if we replaced the three-equation block with a nonlinear "
        "price-setting model and then compared a local perturbation to a global or "
        "perfect-foresight solution."
    )

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
    report.add_figure(
        "figures/irf-monetary-shock.png",
        "Impulse responses to a one-percentage-point contractionary monetary-policy shock",
        fig1,
        description="The monetary shock is a wedge in the Taylor rule, not the total policy-rate response. "
        "On impact the wedge is one percentage point, but the systematic part of the rule partly offsets "
        "it because expected output and inflation fall. The real rate still rises, demand contracts, and "
        "inflation falls with the output gap. Persistence in $v_t$ controls how slowly the economy returns "
        "to steady state.",
    )

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
    report.add_figure(
        "figures/irf-demand-shock.png",
        "Impulse responses to a one-percentage-point natural-rate demand shock",
        fig2,
        description="A positive natural-rate shock pushes current demand up at the same nominal rate. "
        "Output and inflation therefore rise together. The Taylor rule raises the policy rate in response, "
        "which dampens but does not eliminate the expansion because the shock is persistent and agents "
        "expect demand pressure to continue.",
    )

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
    report.add_table(
        "tables/impact-responses.csv",
        "Impact Responses to One-Percentage-Point Shocks",
        df,
        description="The impact table gives the signs and scale without asking the reader to read them "
        "off the figure. Output is in percent deviations; inflation and the policy rate are in quarterly "
        "percentage points. Monetary and demand shocks move output and inflation in opposite directions "
        "across experiments because the shocks enter different equations.",
    )

    report.add_takeaway(
        "The three-equation New Keynesian model is compact, but it already separates two "
        "central ideas. First, sticky prices let a nominal policy surprise move the real "
        "rate and therefore current demand. Second, determinacy is not a numerical detail: "
        "with forward-looking inflation, the Taylor rule has to make expected inflation "
        "costly enough for the model to select one stable path.\n\n"
        "The policy-shock and demand-shock experiments use the same solution method but "
        "differ in their economics. A policy wedge contracts demand and inflation. A "
        "natural-rate shock expands both, with the central bank leaning back through the "
        "Taylor rule. For a supply or cost-push shock, the same block would show the sharper "
        "output-inflation stabilization trade-off."
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
