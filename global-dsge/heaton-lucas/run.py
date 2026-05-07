#!/usr/bin/env python3
"""Heaton-Lucas (1996): incomplete markets with two agents via STPFI.

Two agents trade equity shares and bonds with short-sale and borrowing
constraints. The global solution follows the HL1996.gmod example from
github.com/gdsge/gdsge and uses Simultaneous Transition and Policy Function
Iterations with JAX autodiff Jacobians.

Reference: Heaton & Lucas (1996, JPE), Cao, Luo & Nie (2023, RED).
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import jacfwd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root

jax.config.update("jax_platform_name", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport
from lib.stpfi import solve_stpfi
from lib.simulate import simulate_markov, compute_ergodic_histogram


def main():
    # =========================================================================
    # Parameters: exact from HL1996.gmod lines 1-5
    # =========================================================================
    beta = 0.95
    gamma = 1.5
    Kb = -0.05

    # =========================================================================
    # Exogenous shocks: HL1996.gmod lines 7-23
    # =========================================================================
    shock_num = 8
    g = jnp.array([.9904, 1.0470, .9904, 1.0470, .9904, 1.0470, .9904, 1.0470])
    d = jnp.array([.1402, .1437, .1561, .1599, .1402, .1437, .1561, .1599])
    eta1 = jnp.array([.3772, .3772, .3772, .3772, .6228, .6228, .6228, .6228])

    shock_trans = np.array([
        [0.3932, 0.2245, 0.0793, 0.0453, 0.1365, 0.0779, 0.0275, 0.0157],
        [0.3044, 0.3470, 0.0425, 0.0484, 0.1057, 0.1205, 0.0147, 0.0168],
        [0.0484, 0.0425, 0.3470, 0.3044, 0.0168, 0.0147, 0.1205, 0.1057],
        [0.0453, 0.0793, 0.2245, 0.3932, 0.0157, 0.0275, 0.0779, 0.1365],
        [0.1365, 0.0779, 0.0275, 0.0157, 0.3932, 0.2245, 0.0793, 0.0453],
        [0.1057, 0.1205, 0.0147, 0.0168, 0.3044, 0.3470, 0.0425, 0.0484],
        [0.0168, 0.0147, 0.1205, 0.1057, 0.0484, 0.0425, 0.3470, 0.3044],
        [0.0157, 0.0275, 0.0779, 0.1365, 0.0453, 0.0793, 0.2245, 0.3932],
    ])
    shock_trans = shock_trans / shock_trans.sum(axis=1, keepdims=True)
    shock_trans_jnp = jnp.array(shock_trans)

    # =========================================================================
    # State grid: HL1996.gmod line 26, w1 = linspace(-0.05,1.05,201)
    # =========================================================================
    n_w = 201
    w_grid = jnp.linspace(-0.05, 1.05, n_w)
    w_grid_np = np.asarray(w_grid)

    # =========================================================================
    # 19 unknowns (gmod line 28):
    #   c1, c2, s1p, nb1p, nb2p, ms1, ms2, mb1, mb2, ps, pb, w1n[8]
    # 19 equations (gmod lines 75-88):
    #   4 Euler + 4 complementarity + 1 bond clearing + 2 budget + 8 consistency
    # =========================================================================
    n_eq = 19

    # =========================================================================
    # JAX residual: direct translation of gmod model block (lines 56-88)
    # =========================================================================
    def make_residual_fn(iz_val):
        @jax.jit
        def res_fn(x, w1, ps_interp, c1_interp, c2_interp):
            c1   = jnp.maximum(x[0], 1e-10)
            c2   = jnp.maximum(x[1], 1e-10)
            s1p  = x[2]
            nb1p = x[3]
            nb2p = x[4]
            ms1  = x[5]
            ms2  = x[6]
            mb1  = x[7]
            mb2  = x[8]
            ps   = jnp.maximum(x[9], 1e-10)
            pb   = jnp.maximum(x[10], 1e-10)
            w1n  = x[11:19]

            b1p = nb1p + Kb          # gmod line 65
            b2p = nb2p + Kb          # gmod line 66
            s2p = 1.0 - s1p          # gmod line 67

            # Interpolate future values (gmod line 58)
            psn  = jnp.array([jnp.interp(w1n[j], w_grid, ps_interp[j])
                              for j in range(shock_num)])
            c1n  = jnp.array([jnp.maximum(
                              jnp.interp(w1n[j], w_grid, c1_interp[j]), 1e-10)
                              for j in range(shock_num)])
            c2n  = jnp.array([jnp.maximum(
                              jnp.interp(w1n[j], w_grid, c2_interp[j]), 1e-10)
                              for j in range(shock_num)])

            # Euler expectations (gmod lines 60-63)
            es1, es2, eb1, eb2 = 0.0, 0.0, 0.0, 0.0
            for j in range(shock_num):
                prob = shock_trans_jnp[iz_val, j]
                r1 = (c1n[j] / c1) ** (-gamma)
                r2 = (c2n[j] / c2) ** (-gamma)
                geq = g[j] ** (1.0 - gamma)
                gbo = g[j] ** (-gamma)
                ret = (psn[j] + d[j]) / ps
                es1 += prob * geq * r1 * ret
                es2 += prob * geq * r2 * ret
                eb1 += prob * gbo * r1 / pb
                eb2 += prob * gbo * r2 / pb

            # Budget (gmod lines 69-70)
            budget_1 = w1 * (ps + d[iz_val]) + eta1[iz_val] \
                       - c1 - ps * s1p - pb * b1p
            budget_2 = (1.0 - w1) * (ps + d[iz_val]) + (1.0 - eta1[iz_val]) \
                       - c2 - ps * s2p - pb * b2p

            # Consistency (gmod line 72)
            w1c = jnp.array([
                (s1p * (psn[j] + d[j]) + b1p / g[j]) / (psn[j] + d[j]) - w1n[j]
                for j in range(shock_num)])

            # 19 equations (gmod lines 76-88)
            return jnp.concatenate([jnp.array([
                -1.0 + beta * es1 + ms1,   # 76
                -1.0 + beta * es2 + ms2,   # 77
                -1.0 + beta * eb1 + mb1,   # 78
                -1.0 + beta * eb2 + mb2,   # 79
                ms1 * s1p,                  # 80
                ms2 * s2p,                  # 81
                mb1 * nb1p,                 # 82
                mb2 * nb2p,                 # 83
                b1p + b2p,                  # 84
                budget_1,                   # 85
                budget_2,                   # 86
            ]), w1c])                       # 87 (8 eqs)

        @jax.jit
        def jac_fn(x, w1, ps_i, c1_i, c2_i):
            return jacfwd(lambda xx: res_fn(xx, w1, ps_i, c1_i, c2_i))(x)

        return res_fn, jac_fn

    # =========================================================================
    # Compile JAX functions
    # =========================================================================
    print("Compiling JAX residual + Jacobian (8 states)...")
    res_fns, jac_fns = [], []
    dummy = jnp.zeros((shock_num, n_w))
    x0_dummy = jnp.zeros(n_eq)
    for iz in range(shock_num):
        rf, jf = make_residual_fn(iz)
        _ = rf(x0_dummy, 0.5, dummy, dummy + 0.5, dummy + 0.5)
        _ = jf(x0_dummy, 0.5, dummy, dummy + 0.5, dummy + 0.5)
        res_fns.append(rf)
        jac_fns.append(jf)
        print(f"  iz={iz} done")
    print("  Compiled.\n")

    # =========================================================================
    # Initial guesses: gmod lines 44-52
    # =========================================================================
    ps_pol = np.ones((shock_num, n_w))
    c1_pol = np.zeros((shock_num, n_w))
    c2_pol = np.zeros((shock_num, n_w))
    for iz in range(shock_num):
        c1_pol[iz] = np.asarray(w_grid * d[iz] + eta1[iz])
        c2_pol[iz] = np.asarray((1.0 - w_grid) * d[iz] + 1.0 - eta1[iz])
    c1_pol = np.maximum(c1_pol, 1e-6)
    c2_pol = np.maximum(c2_pol, 1e-6)

    all_x = np.zeros((shock_num, n_w, n_eq))
    for iz in range(shock_num):
        for io in range(n_w):
            w1 = w_grid_np[io]
            all_x[iz, io, 0] = c1_pol[iz, io]
            all_x[iz, io, 1] = c2_pol[iz, io]
            all_x[iz, io, 2] = np.clip(w1, 0.01, 0.99)  # s1p
            all_x[iz, io, 3] = 0.05                       # nb1p
            all_x[iz, io, 4] = 0.05                       # nb2p
            all_x[iz, io, 5:9] = 0.0                      # multipliers
            all_x[iz, io, 9] = 1.0                         # ps
            all_x[iz, io, 10] = beta                       # pb
            all_x[iz, io, 11:19] = w1                      # w1n

    policy_init = {"x": all_x, "ps": ps_pol, "c1": c1_pol, "c2": c2_pol}
    trans_init = {"w1n": all_x[:, :, 11:19].copy()}

    # =========================================================================
    # STPFI step
    # =========================================================================
    def stpfi_step(policy, trans_funcs):
        x_old = policy["x"]
        ps_j = jnp.array(policy["ps"])
        c1_j = jnp.array(policy["c1"])
        c2_j = jnp.array(policy["c2"])

        x_new = np.zeros_like(x_old)
        max_residual = 0.0

        for iz in range(shock_num):
            for io in range(n_w):
                w1 = float(w_grid_np[io])
                x0 = x_old[iz, io].copy()

                def res_np(x, _iz=iz, _w1=w1):
                    return np.asarray(
                        res_fns[_iz](jnp.array(x), _w1, ps_j, c1_j, c2_j))

                def jac_np(x, _iz=iz, _w1=w1):
                    J = np.asarray(
                        jac_fns[_iz](jnp.array(x), _w1, ps_j, c1_j, c2_j))
                    return J if not np.any(np.isnan(J)) else None

                sol = root(res_np, x0, jac=jac_np, method='hybr',
                           options={'maxfev': 5000})
                if np.max(np.abs(sol.fun)) > 1e-3:
                    sol2 = root(res_np, x0, method='lm',
                                options={'maxiter': 5000})
                    if np.max(np.abs(sol2.fun)) < np.max(np.abs(sol.fun)):
                        sol = sol2

                max_residual = max(max_residual, np.max(np.abs(sol.fun)))
                x_new[iz, io] = sol.x

        ps_new = np.maximum(x_new[:, :, 9], 1e-10)
        c1_new = np.maximum(x_new[:, :, 0], 1e-10)
        c2_new = np.maximum(x_new[:, :, 1], 1e-10)

        policy_new = {"x": x_new, "ps": ps_new, "c1": c1_new, "c2": c2_new}
        trans_new = {"w1n": x_new[:, :, 11:19].copy()}
        return policy_new, trans_new, max_residual

    # =========================================================================
    # Solve
    # =========================================================================
    print(f"Solving Heaton-Lucas (1996) via STPFI "
          f"({shock_num}x{n_w} = {shock_num * n_w} points)...")
    policy, trans_result, info = solve_stpfi(
        stpfi_step, policy_init, trans_init,
        tol=5e-4, max_iter=80, dampen=0.5, verbose=True,
    )

    # =========================================================================
    # Extract solution
    # =========================================================================
    x_sol = policy["x"]
    c1_sol  = np.maximum(x_sol[:, :, 0], 1e-10)
    c2_sol  = np.maximum(x_sol[:, :, 1], 1e-10)
    s1p_sol = x_sol[:, :, 2]
    ms1_sol = x_sol[:, :, 5]
    mb1_sol = x_sol[:, :, 7]
    ps_sol  = np.maximum(x_sol[:, :, 9], 1e-10)
    pb_sol  = np.maximum(x_sol[:, :, 10], 1e-10)
    w1n_sol = x_sol[:, :, 11:19]

    # Equity premium (gmod line 74)
    eq_prem = np.zeros((shock_num, n_w))
    for iz in range(shock_num):
        for io in range(n_w):
            ep = 0.0
            for j in range(shock_num):
                psp = float(jnp.interp(
                    w1n_sol[iz, io, j], w_grid, jnp.array(ps_sol[j])))
                ep += shock_trans[iz, j] * (psp + float(d[j])) \
                      / ps_sol[iz, io] * float(g[j])
            eq_prem[iz, io] = ep - 1.0 / pb_sol[iz, io]

    # =========================================================================
    # Simulation (gmod lines 91-98)
    # =========================================================================
    print("\nSimulating...")
    n_paths, n_per = 24, 10000
    omega_all = []
    for p in range(n_paths):
        zs = simulate_markov(shock_trans, n_per, 0, seed=42 + p)
        ws = np.zeros(n_per); ws[0] = 0.5
        for t in range(n_per - 1):
            ws[t+1] = np.clip(float(jnp.interp(
                ws[t], w_grid, jnp.array(w1n_sol[zs[t], :, zs[t+1]]))),
                -0.05, 1.05)
        omega_all.append(ws[1000:])
    omega_all = np.concatenate(omega_all)

    # Euler errors
    print("Euler errors...")
    ee_s, ee_b = [], []
    for p in range(min(n_paths, 4)):
        zs = simulate_markov(shock_trans, n_per, 0, seed=42 + p)
        ws = np.zeros(n_per); ws[0] = 0.5
        for t in range(n_per - 1):
            ws[t+1] = np.clip(float(jnp.interp(
                ws[t], w_grid, jnp.array(w1n_sol[zs[t], :, zs[t+1]]))),
                -0.05, 1.05)
        for t in range(1000, n_per - 1):
            iz, w = zs[t], ws[t]
            c1v = max(float(jnp.interp(w, w_grid, jnp.array(c1_sol[iz]))), 1e-10)
            psv = max(float(jnp.interp(w, w_grid, jnp.array(ps_sol[iz]))), 1e-10)
            pbv = max(float(jnp.interp(w, w_grid, jnp.array(pb_sol[iz]))), 1e-10)
            s, b = 0.0, 0.0
            for j in range(shock_num):
                wp = np.clip(float(jnp.interp(
                    w, w_grid, jnp.array(w1n_sol[iz, :, j]))), -0.05, 1.05)
                c1p = max(float(jnp.interp(wp, w_grid, jnp.array(c1_sol[j]))), 1e-10)
                psp = max(float(jnp.interp(wp, w_grid, jnp.array(ps_sol[j]))), 1e-10)
                r1 = (c1p / c1v) ** (-gamma)
                s += shock_trans[iz, j] * float(g[j])**(1.-gamma) * r1 * (psp + float(d[j])) / psv
                b += shock_trans[iz, j] * float(g[j])**(-gamma) * r1 / pbv
            ee_s.append(abs(beta * s - 1.0))
            ee_b.append(abs(beta * b - 1.0))
    ee_s = np.array(ee_s) if ee_s else np.array([0.])
    ee_b = np.array(ee_b) if ee_b else np.array([0.])

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()
    interior_mask = (w_grid_np >= 0.0) & (w_grid_np <= 1.0)
    eq_prem_pct = eq_prem[:, interior_mask] * 100.0
    eq_min = float(np.nanmin(eq_prem_pct))
    eq_max = float(np.nanmax(eq_prem_pct))
    no_short_share = float(np.mean(ms1_sol[:, interior_mask] > 1e-5) * 100.0)
    borrow_share = float(np.mean(mb1_sol[:, interior_mask] > 1e-5) * 100.0)
    omega_mean = float(np.mean(omega_all))
    omega_p10, omega_p90 = np.percentile(omega_all, [10, 90])
    status_phrase = "converged" if info["converged"] else "stopped at the iteration cap"

    report = ModelReport(
        "Heaton-Lucas Risk Sharing and Equity Premia",
        "Incomplete risk sharing makes the wealth distribution part of the "
        "asset-pricing state.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Households do not all price risk in the same way when trade is limited. "
        "In Heaton and Lucas (1996), two CRRA agents receive different endowment "
        "shares, trade a claim to aggregate dividends, and trade a one-period "
        "bond. Short-sale and borrowing limits prevent full insurance, so the "
        "household that holds wealth after a shock can matter for the stochastic "
        "discount factor.\n\n"
        "The state variable is agent 1's wealth share, $\\omega_1$. In a "
        "complete-markets Lucas tree, a fixed Pareto weight would summarize "
        "risk sharing. With portfolio constraints, wealth shares move after "
        "shocks and the price of aggregate risk moves with them. That is why "
        "the tutorial needs a global computation: prices, portfolios, and the "
        "next wealth share have to be solved together at each point in the "
        "state space. The example connects the "
        "[Lucas-tree pricing](../../dynamic-programming/asset-pricing/) "
        "environment with the incomplete-markets logic in the "
        "[Huggett bond-market](../../heterogeneous-agents/huggett-incomplete-markets/) "
        "tutorial."
    )

    report.add_equations(r"""
Let $z_t\in\{1,\ldots,8\}$ be the Markov shock. In state $z$, aggregate growth
is $g_z$, the equity dividend is $d_z$, and agent 1 receives endowment share
$\eta_{1z}$ with $\eta_{2z}=1-\eta_{1z}$. Agent $i$ has CRRA utility with
risk aversion $\gamma$ and chooses consumption $c_i$, next-period equity
holdings $s_i'$, and next-period bond holdings $b_i'$. The equity and bond
prices are $p_s$ and $p_b$.

For $i=1,2$, the budget constraint is

$$c_i+p_s s_i' + p_b b_i'
=\omega_i(p_s+d_z)+\eta_{iz},\qquad \omega_2=1-\omega_1.$$

Asset markets clear through

$$s_1'+s_2'=1,\qquad b_1'+b_2'=0,$$

with constraints

$$s_i'\geq 0,\qquad b_i'\geq \bar K^b.$$

The Kuhn-Tucker conditions for equity and bond positions are

$$1=\beta E_z\left[
g_{z'}^{1-\gamma}\left(\frac{c_i'}{c_i}\right)^{-\gamma}
\frac{p_s(z',\omega_1')+d_{z'}}{p_s}\right]+\mu_i^s,$$

$$1=\beta E_z\left[
g_{z'}^{-\gamma}\left(\frac{c_i'}{c_i}\right)^{-\gamma}
\frac{1}{p_b}\right]+\mu_i^b,$$

$$\mu_i^s\geq0,\quad \mu_i^s s_i'=0,\qquad
\mu_i^b\geq0,\quad \mu_i^b(b_i'-\bar K^b)=0.$$

The future wealth share is not an exogenous Markov transition. It must be
consistent with today's portfolio choice and tomorrow's asset prices:

$$\omega_1'(z')=
\frac{s_1'[p_s(z',\omega_1'(z'))+d_{z'}]+b_1'/g_{z'}}
{p_s(z',\omega_1'(z'))+d_{z'}}.$$
""")

    report.add_model_setup(
        f"| Object | Value | Why it matters |\n"
        f"|---|---:|---|\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $\\gamma$ | {gamma} | CRRA risk aversion |\n"
        f"| $\\bar{{K}}^b$ | {Kb} | Lower bound on each agent's bond position |\n"
        f"| Shock states | {shock_num} | Joint Markov chain for growth, dividends, and endowment shares |\n"
        f"| Wealth-share grid | {n_w} points on $[-0.05,1.05]$ | "
        f"Collocation grid for $\\omega_1$, with a small buffer around $[0,1]$ |\n"
        f"| Unknowns per collocation point | {n_eq} | "
        f"Consumption, portfolios, prices, multipliers, and eight shock-contingent wealth shares |\n"
        f"| Simulation | 24 paths x 10,000 periods | Used to approximate the ergodic wealth-share distribution |"
    )

    report.add_solution_method(
        "The transition for $\\omega_1$ is implicit. Tomorrow's wealth share "
        "depends on tomorrow's equity price, and tomorrow's equity price is "
        "itself a function of that wealth share. Simultaneous Transition and "
        "Policy Function Iteration (STPFI) treats the policy rules and the "
        "transition rule as one fixed point. At each current shock and wealth "
        "share, the nonlinear system solves consumption, portfolios, prices, "
        "Kuhn-Tucker multipliers, and all eight possible next wealth shares at "
        "the same time.\n\n"
        "```text\n"
        "Algorithm: STPFI for the Heaton-Lucas wealth-share economy\n"
        "Input: grid Omega, shock transition P, primitives beta, gamma, Kb\n"
        "Output: policies c_i(z,omega), s_i'(z,omega), b_i'(z,omega), prices p_s, p_b, and transition omega'(z')\n"
        "Initialize c_1^0 and c_2^0 from endowment resources, and set p_s^0=1\n"
        "repeat:\n"
        "    for each current shock z and wealth grid point omega:\n"
        "        evaluate current guesses for future c_i^n(z',omega') and p_s^n(z',omega')\n"
        "        solve for y=(c_1,c_2,s_1',b_1',b_2',mu^s,mu^b,p_s,p_b,{omega'(z')})\n"
        "        impose Euler equations, complementary slackness, market clearing, budgets, and consistency\n"
        "    damp the policy and transition updates\n"
        "until the sup-norm policy change is below epsilon or the iteration cap is reached\n"
        "simulate the Markov chain and the implied omega transition to read the ergodic distribution\n"
        "```\n\n"
        f"This run {status_phrase} after **{info['iterations']}** STPFI iterations "
        f"with final policy change {info['error']:.2e} and maximum pointwise "
        f"equation residual {info['residual']:.2e}. The nonlinear systems use "
        "`scipy.optimize.root`; JAX supplies the 19-by-19 Jacobian at each "
        "collocation point."
    )

    report.add_results(
        f"The computed equity premium changes across wealth shares, ranging "
        f"from {eq_min:.2f}% to {eq_max:.2f}% on the displayed interior grid. "
        "Those movements do more than relabel shocks. They reflect "
        "which agent is close to a portfolio constraint and whose marginal "
        "utility receives more weight in pricing aggregate dividends."
    )

    # Figures
    fig1, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    cols = plt.cm.viridis(np.linspace(.15, .95, shock_num))
    m = (w_grid_np >= 0) & (w_grid_np <= 1)
    for iz in range(shock_num):
        a1.plot(w_grid_np[m], eq_prem[iz, m]*100, color=cols[iz], lw=1.2, alpha=.8)
    a1.set_xlabel("$\\omega_1$"); a1.set_ylabel("%"); a1.set_title("Equity Premium")
    a1.axhline(0, color='k', lw=.5, alpha=.3)

    ct, dn = compute_ergodic_histogram(omega_all, 60, 0)
    a2.bar(ct, dn, width=ct[1]-ct[0], alpha=.7, color='steelblue')
    a2.set_xlabel("$\\omega_1$"); a2.set_ylabel("Density")
    a2.set_title("Ergodic Distribution")
    fig1.tight_layout()
    report.add_figure(
        "figures/equity-premium-and-distribution.png",
        "Equity premium and ergodic distribution of wealth share.",
        fig1,
        description=(
            "The first panel reads equity premia against the distributional state. "
            "The second panel shows where the simulated economy spends its time: "
            f"the mean wealth share is {omega_mean:.3f}, with the 10th and 90th "
            f"percentiles at {omega_p10:.3f} and {omega_p90:.3f}. The solution "
            "still prices assets over the whole state space, including states "
            "that the simulation visits less often."
        ),
    )

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
    for iz in [0, 3, 4, 7]:
        ax2[0].plot(w_grid_np[m], ms1_sol[iz, m], color=cols[iz], lw=1.5)
        ax2[1].plot(w_grid_np[m], mb1_sol[iz, m], color=cols[iz], lw=1.5)
        ax2[2].plot(w_grid_np[m], eq_prem[iz, m]*100, color=cols[iz], lw=1.5)
    ax2[0].set_title("$\\mu_1^s$ (no-short)"); ax2[1].set_title("$\\mu_1^b$ (borrowing)")
    ax2[2].set_title("Equity Premium (%)")
    for a in ax2: a.set_xlabel("$\\omega_1$")
    fig2.tight_layout()
    report.add_figure(
        "figures/policy-functions.png",
        "Multipliers and equity premium where constraints bind.",
        fig2,
        description=(
            "The multiplier panels show where constraints bind for agent 1. "
            f"The no-short-sale multiplier is positive at {no_short_share:.1f}% "
            f"of interior collocation points, and the borrowing multiplier is "
            f"positive at {borrow_share:.1f}%. The equity-premium panel puts "
            "the pricing object on the same states, so the constraint regions "
            "can be read against risk compensation."
        ),
    )

    df = pd.DataFrame({
        "Metric": [
            "Mean simulated residual",
            "Median simulated residual",
            "Max simulated residual",
            "GDSGE benchmark mean",
            "GDSGE benchmark max",
        ],
        "Equity EE": [
            f"{ee_s.mean():.2e}",
            f"{np.median(ee_s):.2e}",
            f"{ee_s.max():.2e}",
            "2.08e-05",
            "3.40e-03",
        ],
        "Bond EE": [
            f"{ee_b.mean():.2e}",
            f"{np.median(ee_b):.2e}",
            f"{ee_b.max():.2e}",
            "not reported",
            "not reported",
        ],
        "Interpretation": [
            "Average Euler-equation miss on simulated states",
            "Typical miss away from the worst simulated states",
            "Worst simulated miss in this coarse Python run",
            "Original GDSGE C++ scale reported for the model",
            "Original GDSGE C++ scale reported for the model",
        ],
    })
    report.add_table(
        "tables/euler-errors.csv",
        "Euler Residuals and Benchmark Scale",
        df,
        description=(
            "The table reports numerical accuracy, not a new economic moment. "
            "This Python/JAX version keeps the model close to the original "
            "GDSGE file; the original C++ benchmark gives the tighter reference "
            "scale for the equity Euler equation."
        ),
    )

    report.add_results(
        "The diagnostic table limits how far to push the quantitative numbers. "
        "The plotted policies recover state-dependent pricing and constraint "
        "patterns, but this coarse pedagogical run is looser than the optimized "
        "GDSGE benchmark. A production exercise would raise the iteration cap, "
        "tune damping, or run a denser grid in the original compiled "
        "implementation before treating the Euler errors as benchmark accuracy."
    )

    report.add_takeaway(
        "Limited asset trade turns the wealth distribution into an asset-pricing "
        "state. With moderate risk aversion, risk premia move because constrained "
        "households cannot freely trade away bad marginal-utility states. STPFI "
        "fits this problem because it solves the implicit wealth-share transition "
        "and the occasionally binding portfolio constraints inside one global "
        "fixed point."
    )

    report.add_references([
        "Heaton, J. & Lucas, D. (1996). *JPE* 104(3), 443-487.",
        "Cao, D., Luo, W. & Nie, G. (2023). *RED* 51, 199-225."])

    report.write("README.md")
    print(f"\nDone: README.md + {len(report._figures)} figs + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
