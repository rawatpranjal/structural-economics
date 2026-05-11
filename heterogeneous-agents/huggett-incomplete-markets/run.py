#!/usr/bin/env python3
"""Huggett (1993) Incomplete Markets Model in Continuous Time.

Solves the Huggett economy using the HJB-KFE approach of Achdou et al. (2022).
Households face two-state Poisson income risk and trade a single bond subject
to a hard borrowing limit. The interest rate clears the zero-net-supply bond
market in general equilibrium. A finer reference grid is solved alongside the
working grid as a discretisation audit.

References:
    Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-
        insurance economies." JEDC 17(5-6), 953-969.
    Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022).
        "Income and Wealth Distribution in Macroeconomics: A Continuous-Time
        Approach." REStud 89(1), 45-86.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


# =============================================================================
# Core solver functions
# =============================================================================

def solve_hjb(r, params):
    """Solve the HJB equation for given interest rate r via implicit method.

    Returns:
        V: (I, 2) value function on asset grid x income states
        c: (I, 2) consumption policy
        s: (I, 2) savings policy s = z + r*a - c
        A: (2I, 2I) sparse transition matrix for the KFE
        info: dict with convergence information
    """
    rho = params["rho"]
    sigma = params["sigma"]
    z = params["z"]
    la = params["la"]
    a = params["a"]
    I = params["I"]
    da = params["da"]
    Delta = params["Delta"]
    maxit = params["maxit"]
    crit = params["crit"]

    # Broadcast grids: aa is (I, 2), zz is (I, 2)
    aa = np.column_stack([a, a])
    zz = np.ones((I, 1)) * z[np.newaxis, :]

    # Income switching matrix (2I x 2I)
    Aswitch = sparse.bmat([
        [-sparse.eye(I) * la[0],  sparse.eye(I) * la[0]],
        [ sparse.eye(I) * la[1], -sparse.eye(I) * la[1]],
    ], format="csc")

    # Initial guess: consume everything (steady-state guess)
    income = zz + r * aa
    income_pos = np.maximum(income, 1e-10)
    V = income_pos ** (1 - sigma) / (1 - sigma) / rho

    dVf = np.zeros((I, 2))
    dVb = np.zeros((I, 2))

    for n in range(1, maxit + 1):
        # Forward difference
        dVf[:I-1, :] = (V[1:I, :] - V[:I-1, :]) / da
        dVf[I-1, :] = np.maximum(z + r * a[-1], 1e-10) ** (-sigma)  # state constraint

        # Backward difference
        dVb[1:I, :] = (V[1:I, :] - V[:I-1, :]) / da
        dVb[0, :] = np.maximum(z + r * a[0], 1e-10) ** (-sigma)  # state constraint at borrowing limit

        # Consumption and savings from forward difference
        cf = np.maximum(dVf, 1e-10) ** (-1.0 / sigma)
        ssf = zz + r * aa - cf

        # Consumption and savings from backward difference
        cb = np.maximum(dVb, 1e-10) ** (-1.0 / sigma)
        ssb = zz + r * aa - cb

        # Consumption at steady state (zero savings)
        c0 = zz + r * aa

        # Upwind scheme: choose forward/backward/zero based on drift sign
        If = (ssf > 0).astype(float)   # positive drift -> forward
        Ib = (ssb < 0).astype(float)   # negative drift -> backward
        I0 = 1.0 - If - Ib             # at steady state

        c = cf * If + cb * Ib + c0 * I0
        u = c ** (1 - sigma) / (1 - sigma)

        # Construct the transition matrix A (upwind scheme)
        X = -np.minimum(ssb, 0) / da  # sub-diagonal
        Y = -np.maximum(ssf, 0) / da + np.minimum(ssb, 0) / da  # main diagonal
        Z = np.maximum(ssf, 0) / da   # super-diagonal

        # Build A1 (state z1) and A2 (state z2) as tridiagonal matrices
        A1 = (sparse.diags(Y[:, 0], 0, shape=(I, I))
              + sparse.diags(X[1:I, 0], -1, shape=(I, I))
              + sparse.diags(Z[:I-1, 0], 1, shape=(I, I)))
        A2 = (sparse.diags(Y[:, 1], 0, shape=(I, I))
              + sparse.diags(X[1:I, 1], -1, shape=(I, I))
              + sparse.diags(Z[:I-1, 1], 1, shape=(I, I)))

        A = sparse.bmat([[A1, None], [None, A2]], format="csc") + Aswitch

        # Implicit update: (1/Delta + rho)*I - A) * V_new = u + V_old/Delta
        B = (1.0 / Delta + rho) * sparse.eye(2 * I, format="csc") - A

        u_stacked = np.concatenate([u[:, 0], u[:, 1]])
        V_stacked = np.concatenate([V[:, 0], V[:, 1]])

        b = u_stacked + V_stacked / Delta
        V_new_stacked = spsolve(B, b)

        V_new = np.column_stack([V_new_stacked[:I], V_new_stacked[I:2*I]])

        change = np.max(np.abs(V_new - V))
        V = V_new

        if change < crit:
            break

    # Recompute final policy at converged V
    dVf[:I-1, :] = (V[1:I, :] - V[:I-1, :]) / da
    dVf[I-1, :] = np.maximum(z + r * a[-1], 1e-10) ** (-sigma)
    dVb[1:I, :] = (V[1:I, :] - V[:I-1, :]) / da
    dVb[0, :] = np.maximum(z + r * a[0], 1e-10) ** (-sigma)

    cf = np.maximum(dVf, 1e-10) ** (-1.0 / sigma)
    ssf = zz + r * aa - cf
    cb = np.maximum(dVb, 1e-10) ** (-1.0 / sigma)
    ssb = zz + r * aa - cb
    c0 = zz + r * aa

    If = (ssf > 0).astype(float)
    Ib = (ssb < 0).astype(float)
    I0 = 1.0 - If - Ib

    c = cf * If + cb * Ib + c0 * I0
    s = zz + r * aa - c

    info = {"iterations": n, "converged": change < crit, "error": change}
    return V, c, s, A, info


def solve_kfe(A, params):
    """Solve the Kolmogorov Forward Equation for stationary distribution.

    Solves A' * g = 0 with integral(g) = 1.

    Returns:
        g: (I, 2) stationary density on (a, z) grid
    """
    I = params["I"]
    da = params["da"]

    AT = A.T.tocsc()

    # Pin one equation to fix the scale (otherwise the system is singular)
    b = np.zeros(2 * I)
    i_fix = 0
    b[i_fix] = 0.1
    AT = AT.tolil()
    AT[i_fix, :] = 0
    AT[i_fix, i_fix] = 1.0
    AT = AT.tocsc()

    gg = spsolve(AT, b)

    # Normalise so the density integrates to one
    g_sum = np.sum(gg) * da
    gg = gg / g_sum

    g = np.column_stack([gg[:I], gg[I:2*I]])
    return g


def excess_demand(r, params):
    """Aggregate bond demand S(r) = integral a g(a, z) da dz at rate r."""
    V, c, s, A, info = solve_hjb(r, params)
    g = solve_kfe(A, params)
    a = params["a"]
    da = params["da"]

    S = (g[:, 0] @ a) * da + (g[:, 1] @ a) * da
    return S, V, c, s, g, info


def find_equilibrium(params, r_min=0.001, r_max=0.04, tol=1e-5, max_iter=40, label=""):
    """Bisection on r until the bond market clears."""
    r_lo, r_hi = r_min, r_max
    r_history = []
    S_history = []

    r_eq = 0.5 * (r_lo + r_hi)
    V = c = s = g = info = None

    for it in range(1, max_iter + 1):
        r_mid = 0.5 * (r_lo + r_hi)
        S, V, c, s, g, info = excess_demand(r_mid, params)

        r_history.append(r_mid)
        S_history.append(S)

        print(f"  {label}Bisection {it:2d}: r = {r_mid:.6f}, S(r) = {S:+.6f}")

        if abs(S) < tol:
            r_eq = r_mid
            print(f"  {label}Equilibrium: r* = {r_eq:.6f}")
            break
        elif S > 0:
            # Aggregate desired assets are positive; lower r to depress saving.
            r_hi = r_mid
        else:
            # Aggregate desired assets are negative; raise r to depress borrowing.
            r_lo = r_mid
        r_eq = r_mid

    return r_eq, V, c, s, g, info, np.array(r_history), np.array(S_history)


def compute_supply_curve(params, r_grid):
    """Compute S(r) on a grid of interest rates for plotting."""
    S_vals = np.zeros(len(r_grid))
    for i, r in enumerate(r_grid):
        S, *_ = excess_demand(r, params)
        S_vals[i] = S
        print(f"  Supply curve: r = {r:+.4f}, S(r) = {S:+.6f}")
    return S_vals


def build_params(I, a_min, a_max, base):
    """Construct a params dict for a given grid size on [a_min, a_max]."""
    a = np.linspace(a_min, a_max, I)
    da = (a_max - a_min) / (I - 1)
    out = dict(base)
    out["a"] = a
    out["I"] = I
    out["da"] = da
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    # =========================================================================
    # Calibration
    # =========================================================================
    rho = 0.05      # Rate of time preference (continuous-time discount rate)
    sigma = 2.0     # CRRA coefficient
    z = np.array([0.1, 0.2])       # Income endowments in low and high state
    la = np.array([1.2, 1.2])      # Poisson switching intensities
    a_min = -0.15   # Hard borrowing limit
    a_max = 5.0     # Upper bound on the asset grid
    I = 2000        # Working grid size
    I_ref = 6000    # Reference grid for the discretisation audit

    base = {
        "rho": rho, "sigma": sigma, "z": z, "la": la,
        "Delta": 1000, "maxit": 100, "crit": 1e-6,
    }
    params = build_params(I, a_min, a_max, base)
    params_ref = build_params(I_ref, a_min, a_max, base)

    # Stationary income probabilities (analytic, from balanced-flow lambda_L p_L = lambda_H p_H)
    p_low = la[1] / (la[0] + la[1])
    p_high = la[0] / (la[0] + la[1])
    mean_z_analytic = p_low * z[0] + p_high * z[1]

    # =========================================================================
    # Equilibrium on the working grid
    # =========================================================================
    print(f"Working grid (I={I}): bisecting on r for bond market clearing")
    r_eq, V, c, s, g, info, r_hist, S_hist = find_equilibrium(
        params, r_min=0.001, r_max=0.045, tol=1e-5, max_iter=40, label=f"[I={I}] ",
    )
    a = params["a"]
    da = params["da"]

    # Aggregate moments on the working grid
    mean_wealth = (g[:, 0] @ a) * da + (g[:, 1] @ a) * da
    market_residual = abs(mean_wealth)
    mean_wealth_display = 0.0 if market_residual < 5e-5 else mean_wealth
    mean_cons = (g[:, 0] @ c[:, 0]) * da + (g[:, 1] @ c[:, 1]) * da
    # Mass within a fixed asset-range window of the borrowing limit
    constraint_window = 0.02
    a_window_mask = a <= a_min + constraint_window
    mass_at_constraint = float(np.sum(g[a_window_mask, :]) * da)
    prob_z_low = float(np.sum(g[:, 0]) * da)
    prob_z_high = float(np.sum(g[:, 1]) * da)
    wedge = rho - r_eq
    wedge_pct = 100 * wedge / rho
    p_balance_err = abs(prob_z_low - 0.5)

    # =========================================================================
    # Reference grid: same equilibrium computation, finer asset discretisation
    # =========================================================================
    print(f"\nReference grid (I={I_ref}): bisecting on r for the audit")
    r_eq_ref, V_ref, c_ref, s_ref, g_ref, info_ref, _, _ = find_equilibrium(
        params_ref, r_min=0.001, r_max=0.045, tol=1e-5, max_iter=40, label=f"[I={I_ref}] ",
    )
    a_ref = params_ref["a"]
    da_ref = params_ref["da"]
    mean_wealth_ref = (g_ref[:, 0] @ a_ref) * da_ref + (g_ref[:, 1] @ a_ref) * da_ref

    # Discretisation diagnostics: gap in equilibrium price and policy on the
    # active range. Compare on a common subset for the policy norms.
    r_gap = abs(r_eq - r_eq_ref)
    a_lo, a_hi = a_min, 1.0
    mask = (a >= a_lo) & (a <= a_hi)
    s_ref_on_a = np.column_stack([
        np.interp(a, a_ref, s_ref[:, 0]),
        np.interp(a, a_ref, s_ref[:, 1]),
    ])
    s_gap = float(np.max(np.abs(s[mask, :] - s_ref_on_a[mask, :])))
    V_ref_on_a = np.column_stack([
        np.interp(a, a_ref, V_ref[:, 0]),
        np.interp(a, a_ref, V_ref[:, 1]),
    ])
    V_gap = float(np.max(np.abs(V[mask, :] - V_ref_on_a[mask, :])))
    V_scale = float(np.max(np.abs(V[mask, :])))
    V_gap_rel = V_gap / V_scale

    # =========================================================================
    # Bond-market supply curve for the equilibrium plot (working grid)
    # =========================================================================
    print("\nWorking grid: tracing the bond-market supply curve")
    r_grid_plot = np.linspace(-0.02, 0.049, 20)
    S_grid_plot = compute_supply_curve(params, r_grid_plot)

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Huggett Equilibrium and the Risk-Free Rate",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        r"""
Households receive stochastic income and trade one risk-free bond. They can borrow only
down to $a \geq \underline a$. Since the bond is in zero net supply, aggregate asset
demand must equal zero.

The object is the equilibrium return $r^{\ast}$. It is the rate that makes stationary
bond demand $S(r^{\ast})$ equal zero. With incomplete insurance, households want buffer
wealth at $r = \rho$. Market clearing therefore requires $r^{\ast} < \rho$.

The computation links household policy to the cross section. The HJB gives consumption
and savings drift at a candidate $r$. The KFE turns that drift into a stationary density.
Bisection updates $r$ until aggregate bond demand clears.
"""
    )

    report.add_equations(
        r"""
A household in income state $i \in \lbrace L, H \rbrace$ receives endowment
$z_i$ as a flow per unit time. Income jumps from state $i$ to the other state
$j$ at Poisson intensity $\lambda_i$, so the expected duration in state $i$ is
$1/\lambda_i$. Assets $a$ accumulate continuously between jumps according to

$$
\dot a \,=\, s_i(a) \,=\, z_i + r\, a - c_i(a),
\qquad a \,\geq\, \underline a ,
$$

where $r$ is the equilibrium return on the bond, $c_i(a)$ is the consumption
policy in state $i$, and $\underline a$ is the borrowing limit. Period utility
is CRRA: $u(c) = c^{1-\sigma}/(1-\sigma)$ for $\sigma \ne 1$.

### Deriving the HJB with Poisson income switching

Let $V_i(a)$ be the household's value when assets are $a$ and income state is
$i$. Over a small interval of length $\Delta t$, with probability
$1 - \lambda_i\, \Delta t + o(\Delta t)$ the income state stays at $i$ and
assets drift by $\dot a\, \Delta t$. With probability
$\lambda_i\, \Delta t + o(\Delta t)$ the income state jumps to $j$ at the start
of the interval and the household enters next period with the same assets but
the new value function $V_j$. The discrete-time Bellman is

$$
V_i(a) \,=\, \max_{c \,\geq\, 0} \Big\lbrace
u(c)\, \Delta t \,+\, e^{-\rho\, \Delta t}\,
\big[(1 - \lambda_i\, \Delta t)\, V_i(a + \dot a\, \Delta t)
\,+\, \lambda_i\, \Delta t\, V_j(a)\big] \Big\rbrace
\,+\, o(\Delta t) .
$$

Expand $e^{-\rho \Delta t} = 1 - \rho\, \Delta t + o(\Delta t)$ and
$V_i(a + \dot a\, \Delta t) = V_i(a) + V_i'(a)\, \dot a\, \Delta t +
o(\Delta t)$. Subtract $V_i(a)$, divide by $\Delta t$, and take
$\Delta t \to 0$. The cross terms $\rho\, \Delta t \cdot \lambda_i$ and
$\rho\, \Delta t \cdot V_i'(a)\, \dot a$ are $o(\Delta t)$ and drop out. The
result is the **HJB equation with Poisson switching**

$$
\rho\, V_i(a) \,=\, \max_{c \,>\, 0}\, \Big\lbrace
\underbrace{u(c)}_{\text{flow utility}}
\,+\, \underbrace{V_i'(a)\, (z_i + r\, a - c)}_{\text{drift in } a}
\,+\, \underbrace{\lambda_i\, (V_j(a) - V_i(a))}_{\text{income jump}}
\Big\rbrace .
$$

This is one continuous-state HJB per income state, coupled by the jump term.
The first two pieces are exactly the Ramsey-style HJB (flow utility plus
shadow value times drift). The third piece is new: it is the rate
$\lambda_i$ of leaving the current income state times the value gain
$V_j(a) - V_i(a)$ from arriving in the other one.

### First-order condition

The maximand depends on $c$ through $u(c) - V_i'(a)\, c$, so the interior
first-order condition equates marginal utility to the marginal value of
assets,

$$
u'(c_i(a)) \,=\, V_i'(a)
\quad\Longrightarrow\quad
c_i(a) \,=\, [V_i'(a)]^{-1/\sigma} ,
$$

and the implied savings drift is

$$
s_i(a) \,=\, z_i + r\, a - c_i(a) .
$$

The marginal value $V_i'(a)$ is the **shadow price** of one extra unit of
assets in state $i$ and serves the same role here as $V'(k)$ in the Ramsey
HJB.

### The borrowing limit as a state constraint

The borrowing limit $a \geq \underline a$ is a state constraint, not a budget
constraint. It binds whenever the unconstrained drift would push assets
through the floor. The Kuhn-Tucker condition is

$$
s_i(\underline a) \,\geq\, 0
\quad\Longleftrightarrow\quad
V_i'(\underline a) \,\geq\, u'(z_i + r\, \underline a) ,
$$

with equality when the constraint is slack and strict inequality (a kink in
$V_i'$) when the household would prefer to dissave further. The numerical
scheme enforces this by computing the implied unconstrained drift at
$a = \underline a$ and clipping consumption to $z_i + r\, \underline a$ when
the drift would be negative.

### The Kolmogorov forward equation

Once the household policy is known, the cross-section of households evolves
as a deterministic transport along the drift, plus stochastic switching
between income states. Let $g_i(a, t)$ be the time-$t$ density of households
in state $i$ at asset level $a$. Mass conservation requires the density to
satisfy a continuity equation: the rate of change of mass in any region
equals the inflow at the left boundary minus the outflow at the right
boundary, plus the income-switching gain or loss. In differential form,

$$
\frac{\partial g_i}{\partial t}(a, t)
\,=\,
-\frac{\partial}{\partial a}\big[s_i(a)\, g_i(a, t)\big]
\,-\, \lambda_i\, g_i(a, t)
\,+\, \lambda_j\, g_j(a, t) .
$$

The first term is the divergence of the deterministic flux $s_i\, g_i$ along
the asset axis. The second term removes mass from state $i$ at the leaving
rate $\lambda_i$. The third term adds mass arriving from state $j$ at rate
$\lambda_j$. The stationary density satisfies

$$
0 \,=\, -\frac{\partial}{\partial a}\big[s_i(a)\, g_i(a)\big]
\,-\, \lambda_i\, g_i(a) \,+\, \lambda_j\, g_j(a),
\qquad \int_{\underline a}^{\bar a} \big[g_L(a) + g_H(a)\big]\, da \,=\, 1 .
$$

Discretised on the same asset grid as the HJB, this becomes
$\mathbf{A}^{\top} g \,=\, 0$, where $\mathbf{A}$ is the upwind generator
used to solve the HJB and $g$ is the joint density across grid points and
income states. The HJB and KFE are dual under one transposition: the same
matrix encodes both the operator that propagates values backward and the
operator that propagates densities forward.

### Equilibrium return

The single bond is in zero net supply. The bond market clears when the
average asset holding integrates to zero,

$$
S(r) \,\equiv\, \int_{\underline a}^{\bar a} a\, [g_L(a) + g_H(a)]\, da \,=\, 0 .
$$

The equilibrium return $r^{\ast}$ is the root of $S(r)$. With incomplete
insurance, households want a buffer at $r = \rho$ (so $S(\rho) > 0$), and
they want to borrow at very low $r$ (so $S(r) < 0$ for small $r$). The
bisection on $r$ finds the wedge $r^{\ast} < \rho$ that closes the bond
market. In this run,
""" + f"$r^{{\\ast}} = {r_eq:.5f}$ " + r"""and the residual is """
        + f"${market_residual:.2e}$." + r"""
"""
    )

    report.add_model_setup(
        f"""
The calibration keeps only the ingredients needed for Huggett pricing. There are two
income states, symmetric switching, one bond, and a borrowing limit.

| Object | Value | Role |
|---|---:|---|
| Discount rate $\\rho$ | {rho} | Continuous-time time preference; complete-markets benchmark for $r$ |
| CRRA $\\sigma$ | {sigma} | Curvature; sets the precautionary motive and Euler curvature |
| Income endowments $(z_L, z_H)$ | ({z[0]}, {z[1]}) | Two-state Poisson chain with stationary mean $\\bar z = {mean_z_analytic:.4f}$ |
| Switching intensities $(\\lambda_L, \\lambda_H)$ | ({la[0]}, {la[1]}) | Symmetric jumps; expected duration in each state $1/\\lambda_i \\approx {1/la[0]:.2f}$ |
| Borrowing limit $\\underline a$ | {a_min} | Hard lower bound; chosen so $z_L + r\\underline a > 0$ at the equilibrium $r$ |
| Upper bound $\\bar a$ | {a_max} | Set wide enough that the right tail of $g_i$ is numerically zero |
| Working asset grid | {I} pts | Uniform on $[\\underline a, \\bar a]$; HJB upwind scheme |
| Reference asset grid | {I_ref} pts | Audit solve at the same calibration; defines the discretisation gap |
| Implicit step $\\Delta$ | {base['Delta']} | Large step keeps the implicit HJB update close to a Newton step on $V$ |
| HJB tolerance | {base['crit']:.0e} | Sup-norm on successive value functions |
| Bisection tolerance | $10^{{-5}}$ | On the bond-market residual $\\lvert S(r)\\rvert$ |

The symmetric income chain implies $p_L = p_H = 0.5$. Expected income is
$\\bar z = {mean_z_analytic:.4f}$. The KFE solution recovers
$|p_L - 0.5| = {p_balance_err:.2e}$.
"""
    )

    report.add_solution_method(
        r"""
The equilibrium is found by three nested loops. The outer loop bisects on the
return $r$. For each candidate $r$, the middle loop solves the HJB by an
implicit upwind iteration, then solves the KFE for the stationary density,
then evaluates the bond-market residual $S(r)$. Each piece below is the
Achdou-Han-Lasry-Lions-Moll method specialised to two income states.

### Upwind discretisation of the HJB

Place a uniform grid $a_1 < a_2 < \cdots < a_I$ on $[\underline a, \bar a]$
with spacing $\Delta a$. At each pair $(a_k, i)$ the solver computes the
forward and backward asset slopes

$$
D^{+}_{k, i} V \,=\, \frac{V_i(a_{k+1}) - V_i(a_k)}{\Delta a},
\qquad
D^{-}_{k, i} V \,=\, \frac{V_i(a_k) - V_i(a_{k-1})}{\Delta a},
$$

and the candidate consumptions $c^{+}_{k, i} = (D^{+}_{k, i} V)^{-1/\sigma}$
and $c^{-}_{k, i} = (D^{-}_{k, i} V)^{-1/\sigma}$. The implied drifts are
$s^{+}_{k, i} = z_i + r\, a_k - c^{+}_{k, i}$ and $s^{-}_{k, i}$ analogously.
The upwind rule keeps the side whose drift points away from the grid point:
forward when $s^{+}_{k, i} > 0$, backward when $s^{-}_{k, i} < 0$, and the
zero-drift consumption $c^{0}_{k, i} = z_i + r\, a_k$ otherwise. A central
difference would mix the two sides with equal weight and produce oscillating
iterates because information in the HJB flows in the direction of the drift.

At the borrowing limit $a_1 = \underline a$ the backward difference is
undefined, so the algorithm uses the forward difference and additionally
enforces the state constraint by clipping consumption to
$z_i + r\, \underline a$ when the implied forward drift is negative. At the
upper end $a_I = \bar a$ the forward difference is undefined, so the
algorithm uses the backward difference; the upper bound is set wide enough
that no probability mass sits there in equilibrium.

### The upwind generator

Once the upwind drifts are picked at every grid point, define $s^{+}_{k, i}
\equiv \max(s_{k, i}, 0)$ and $s^{-}_{k, i} \equiv \min(s_{k, i}, 0)$. The
asset block of the upwind generator is tridiagonal: at row $(k, i)$ the
super-diagonal entry is $s^{+}_{k, i}/\Delta a$, the sub-diagonal entry is
$-s^{-}_{k, i}/\Delta a$, and the diagonal entry is the negative of their sum.
Stacking both income states gives a $2I \times 2I$ block-tridiagonal generator
$\mathbf A^{n}$,

$$
\mathbf A^{n} \,=\, \mathrm{diag}(A_{L}^{n},\, A_{H}^{n})
\,+\,
\begin{pmatrix} -\lambda_L\, \mathbf I & \lambda_L\, \mathbf I \\
\lambda_H\, \mathbf I & -\lambda_H\, \mathbf I \end{pmatrix} ,
$$

where $A_i^{n}$ is the upwind asset-drift block for income state $i$ at the
current consumption policy and the off-block matrices encode income
switching. The matrix has zero row sums (it is a CTMC generator) and
non-positive diagonal.

### Implicit pseudo-time step

An explicit update $V^{n+1} = V^n + \Delta\, (u(c^n) + \mathbf A^{n} V^n -
\rho V^n)$ is unstable for moderate $\Delta$ because the upwind transition
rates $|s_{k, i}|/\Delta a$ can be large on fine grids. The implicit version
replaces $\mathbf A^{n} V^n$ with $\mathbf A^{n} V^{n+1}$ and rearranges to

$$
[(1/\Delta + \rho)\, \mathbf{I} - \mathbf A^{n}]\, V^{n+1}
\,=\, u(c^{n}) + V^{n} / \Delta .
$$

The matrix on the left is strictly diagonally dominant with positive
diagonal (because $\mathbf A^{n}$ has zero row sums and non-positive
diagonal), so the system is unconditionally invertible regardless of
$\Delta$. Taking $\Delta \to \infty$ recovers a Newton step on
$\rho V - u(c) - \mathbf A V = 0$ with the policy frozen, which is why the
HJB inner loop converges in a handful of iterations rather than the hundreds
that an explicit value iteration would need.

### KFE by transposing the same generator

When the HJB inner loop converges, the upwind generator $\mathbf A^{\ast}$ at
the equilibrium policy is exactly the operator whose transpose pushes the
density forward in time:

$$
\frac{\partial g}{\partial t} \,=\, \mathbf A^{\top}\, g .
$$

The stationary density solves $\mathbf A^{\top} g = 0$, a singular system
because $\mathbf A$ has zero row sums (so $\mathbf A^{\top}$ has a zero
right-singular vector). The code pins down the scale by replacing one row
with a normalisation constraint, solving the resulting non-singular system
by sparse LU, and rescaling so that $\int (g_L + g_H)\, da = 1$. The same
matrix that solved the HJB therefore solves the KFE; this **operator
duality** is the elegance of the continuous-time framework.

### Outer bisection on $r$

Bond demand $S(r) \,=\, \int a\, (g_L + g_H)\, da$ is monotone increasing in
$r$ at this calibration: a higher return makes saving more attractive
($s_i(a)$ rises by $a$ for all $a > 0$) and borrowing more painful, so the
density shifts rightward on the asset axis. Since $S(\rho) > 0$ (the
precautionary motive at $r = \rho$ pulls households toward positive assets)
and $S(r) < 0$ for $r$ small enough that low-income borrowers would max out
at $\underline a$, bisection finds a unique $r^{\ast} \in (0, \rho)$ with
$S(r^{\ast}) = 0$.

```text
Algorithm: Huggett equilibrium by HJB-KFE bisection
Inputs    asset grid {a_k}, income states (z_L, z_H), Poisson rates (lambda_L, lambda_H),
          primitives (rho, sigma, a_min), bisection bracket [r_lo, r_hi]
Output    equilibrium r*, value V_i(a), policies c_i(a), s_i(a), density g_i(a)

repeat (outer bisection)
    r = 0.5 * (r_lo + r_hi)

    # Inner HJB by implicit upwind finite differences
    initialise V_i(a) = u(z_i + r a) / rho                         # myopic guess
    repeat
        for each (a_k, i):
            dVf = (V_i(a_{k+1}) - V_i(a_k)) / da                   # forward
            dVb = (V_i(a_k) - V_i(a_{k-1})) / da                   # backward
            cf  = (dVf)^(-1/sigma);   sf = z_i + r a_k - cf
            cb  = (dVb)^(-1/sigma);   sb = z_i + r a_k - cb
            if   sf > 0: c_i(a_k) = cf;            drift = sf      # upwind forward
            elif sb < 0: c_i(a_k) = cb;            drift = sb      # upwind backward
            else        : c_i(a_k) = z_i + r a_k;  drift = 0       # local steady state

        build A from upwind drifts plus income-switching block
        solve [(1/Delta + rho) I - A] V_new = u(c) + V / Delta     # implicit step
        if max|V_new - V| < eps_HJB: break
        V <- V_new

    # KFE for the stationary distribution
    fix one row of A^T to pin scale; solve A^T g = e_fix; renormalise so int g = 1

    # Bond-market excess demand
    S(r) = sum_k a_k * (g_L(a_k) + g_H(a_k)) * da
    if |S(r)| < eps_S: return r, V, c, s, g
    if S(r) > 0: r_hi = r              # too much saving; lower r
    else       : r_lo = r              # too much borrowing; raise r
```

**Working solve.** The HJB inner loop converged in **""" + f"{info['iterations']} iterations**. "
        + f"The final sup-norm change was ${info['error']:.2e}$. Bisection found "
        + f"$r^{{\\ast}} = {r_eq:.5f}$ on the {I}-point grid. The bond-market residual is "
        + f"${market_residual:.2e}$."
        + r"""

**Reference solve.** The reference grid repeats the solve with """
        + f"$I_{{\\rm ref}} = {I_ref}$ points. It gives "
        + f"$r^{{\\ast}}_{{\\rm ref}} = {r_eq_ref:.5f}$. The interest-rate gap is "
        + f"${r_gap:.2e}$. On $a \\in [\\underline a, 1]$, the savings-policy gap is "
        + f"${s_gap:.2e}$ in sup norm."
        + r"""
"""
    )

    # =========================================================================
    # Figures
    # =========================================================================

    # --- Figure 1: Value Function (with reference overlay) ---
    fig1, ax1 = plt.subplots()
    ax1.plot(a, V[:, 0], "b-", linewidth=2, label=f"$V_L(a)$, $z_L = {z[0]}$")
    ax1.plot(a, V[:, 1], "r-", linewidth=2, label=f"$V_H(a)$, $z_H = {z[1]}$")
    ax1.plot(a_ref, V_ref[:, 0], "b--", linewidth=1.0, alpha=0.7,
             label=f"reference ({I_ref} pts)")
    ax1.plot(a_ref, V_ref[:, 1], "r--", linewidth=1.0, alpha=0.7)
    ax1.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V_i(a)$")
    ax1.set_title("Value Function at the Equilibrium $r^{\\ast}$")
    ax1.legend(loc="lower right")
    report.add_figure(
        "figures/value-function.png",
        "Value functions by income state at r*",
        fig1,
        description="The value functions are increasing and concave in assets. "
        "$V_H(a)$ lies above $V_L(a)$ because high income raises cash on hand. "
        "Both curves steepen near the borrowing limit. "
        f"The relative value gap against the reference grid is ${100 * V_gap_rel:.2f}\\%$.",
    )

    # --- Figure 2: Savings Policy (with reference overlay) ---
    fig2, ax2 = plt.subplots()
    ax2.plot(a, s[:, 0], "b-", linewidth=2, label=f"$s_L(a)$, $z_L = {z[0]}$")
    ax2.plot(a, s[:, 1], "r-", linewidth=2, label=f"$s_H(a)$, $z_H = {z[1]}$")
    ax2.plot(a_ref, s_ref[:, 0], "b--", linewidth=1.0, alpha=0.7,
             label=f"reference ({I_ref} pts)")
    ax2.plot(a_ref, s_ref[:, 1], "r--", linewidth=1.0, alpha=0.7)
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax2.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Savings drift $s_i(a) = z_i + r^{\\ast} a - c_i(a)$")
    ax2.set_title("Savings Policy by Income State")
    ax2.set_xlim([a_min - 0.03, 1.0])
    ax2.legend(loc="upper right")
    report.add_figure(
        "figures/savings-policy.png",
        "Savings drift by income state at r*",
        fig2,
        description="The savings policy is the asset drift at $r^{\\ast}$. "
        "Low-income households decumulate above the borrowing limit. "
        "High-income households save near the limit to rebuild buffer wealth. "
        "Income switching moves households between the two drift fields. "
        f"The reference gap is ${s_gap:.2e}$ in sup norm on $[\\underline a, 1]$.",
    )

    # --- Figure 3: Stationary Wealth Distribution (with reference overlay) ---
    fig3, ax3 = plt.subplots()
    ax3.plot(a, g[:, 0], "b-", linewidth=2, label=f"$g_L(a)$, $z_L = {z[0]}$")
    ax3.plot(a, g[:, 1], "r-", linewidth=2, label=f"$g_H(a)$, $z_H = {z[1]}$")
    ax3.plot(a_ref, g_ref[:, 0], "b--", linewidth=1.0, alpha=0.7,
             label=f"reference ({I_ref} pts)")
    ax3.plot(a_ref, g_ref[:, 1], "r--", linewidth=1.0, alpha=0.7)
    ax3.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Density $g_i(a)$")
    ax3.set_title("Stationary Wealth Distribution by Income State")
    ax3.set_xlim([a_min - 0.03, 1.0])
    ax3.legend(loc="upper right")
    report.add_figure(
        "figures/wealth-distribution.png",
        "Stationary asset densities by income state at r*",
        fig3,
        description="The KFE turns drift into a cross-sectional density. "
        "Low-income mass piles near the borrowing limit because negative drift pushes left. "
        "High-income density is flatter because positive drift moves households right. "
        f"The population share within ${constraint_window}$ of the limit is "
        f"${100 * mass_at_constraint:.1f}\\%$.",
    )

    # --- Figure 4: Bond-market clearing ---
    fig4, ax4 = plt.subplots()
    ax4.plot(S_grid_plot, r_grid_plot, "b-", linewidth=2, label="$S(r)$ on working grid")
    ax4.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax4.axhline(rho, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.plot(0, r_eq, "ro", markersize=8, zorder=5,
             label=f"$r^{{\\ast}} = {r_eq:.4f}$")
    ax4.plot(0, r_eq_ref, "kx", markersize=8, zorder=5,
             label=f"$r^{{\\ast}}_{{\\rm ref}} = {r_eq_ref:.4f}$")
    ax4.set_xlabel("Aggregate asset demand $S(r) = \\int a\\,(g_L + g_H)\\,da$")
    ax4.set_ylabel("Interest rate $r$")
    ax4.set_title("Bond Market Clearing")
    ax4.set_xlim([-0.12, 0.12])
    ax4.text(0.04, rho + 0.001, "$r = \\rho$ (complete markets)", fontsize=9, color="gray")
    ax4.legend(loc="lower right")
    report.add_figure(
        "figures/bond-market.png",
        "Aggregate asset demand against the interest rate",
        fig4,
        description="The demand curve plots aggregate asset demand against $r$. "
        "Higher returns raise saving and reduce borrowing, so $S(r)$ rises with $r$. "
        "The complete-markets benchmark is $r = \\rho$. "
        f"The Huggett equilibrium is lower, at $r^{{\\ast}} = {r_eq:.4f}$. "
        f"The precautionary wedge is $\\rho - r^{{\\ast}} = {wedge:.4f}$.",
    )

    # =========================================================================
    # Equilibrium summary table
    # =========================================================================
    table_data = {
        "Statistic": [
            "Discount rate rho",
            "Equilibrium r* (working grid)",
            "Equilibrium r* (reference grid)",
            "Precautionary wedge rho - r*",
            "Mean wealth E[a]",
            "Mean income E[z]",
            "Mean consumption E[c]",
            f"Mass within {constraint_window} of borrowing limit",
            "Prob(z = z_low)",
            "Prob(z = z_high)",
            "Bond-market residual abs(S(r*))",
            "r* gap, working vs reference",
            "Sup-norm savings gap, a in [a_min, 1]",
            "Sup-norm value gap, a in [a_min, 1]",
            "Relative value gap (% of value scale)",
            "HJB iterations (working)",
            "HJB sup-norm change (working)",
        ],
        "Value": [
            f"{rho:.4f}",
            f"{r_eq:.5f}",
            f"{r_eq_ref:.5f}",
            f"{wedge:.5f}",
            f"{mean_wealth_display:.5f}",
            f"{(g[:, 0] @ (np.ones(I) * z[0])) * da + (g[:, 1] @ (np.ones(I) * z[1])) * da:.4f}",
            f"{mean_cons:.4f}",
            f"{mass_at_constraint:.4f}",
            f"{prob_z_low:.4f}",
            f"{prob_z_high:.4f}",
            f"{market_residual:.2e}",
            f"{r_gap:.2e}",
            f"{s_gap:.2e}",
            f"{V_gap:.2e}",
            f"{100 * V_gap_rel:.3f}%",
            f"{info['iterations']}",
            f"{info['error']:.2e}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/equilibrium.csv",
        "Equilibrium and Discretisation Summary",
        df,
        description="The table reports prices, cross-sectional moments, and discretisation "
        "diagnostics. Mean assets are zero because bisection chose $r^{\\ast}$ to satisfy "
        "$S(r^{\\ast}) = 0$.",
    )

    report.add_takeaway(
        r"""
The Huggett price is a market-clearing return. Income risk and the borrowing limit make
households want buffer wealth at $r = \rho$. The bond market clears only at a lower
return. In this run the wedge is """ + f"$\\rho - r^{{\\ast}} = {wedge:.4f}$" + r""".

The HJB/KFE loop ties the household policy to the stationary cross section. The upwind
HJB respects the borrowing limit. The KFE then measures aggregate asset demand. Bisection
on $r$ closes the zero-net-supply bond market.
"""
    )

    report.add_references([
        "Huggett, M. (1993). \"The risk-free rate in heterogeneous-agent incomplete-insurance economies.\" *Journal of Economic Dynamics and Control* 17(5-6), 953-969.",
        "Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). \"Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach.\" *Review of Economic Studies* 89(1), 45-86.",
        "Moll, B. \"Lecture notes on continuous-time heterogeneous-agent models.\" https://benjaminmoll.com/lectures/",
        "**See also.** The continuous-time Aiyagari extension in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/) reuses the upwind HJB and KFE solver developed here, then adds production, mean-field game equilibrium, and a discrete-time benchmark.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
