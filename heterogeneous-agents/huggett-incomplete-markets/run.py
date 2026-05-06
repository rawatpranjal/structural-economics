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
        "A continuous-time pure-exchange economy where idiosyncratic income "
        "risk and a hard borrowing limit pin down the equilibrium bond return below "
        r"the rate of time preference, $r^{\ast} < \rho$.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        r"""
The Huggett (1993) economy is the canonical pure-exchange model of an incomplete-markets
risk-free rate. A continuum of households face idiosyncratic income risk, can lend or
borrow in a single non-state-contingent bond at return $r$, and are bounded below by a
hard borrowing limit $a \geq \underline a$. The bond is in zero net supply: every unit lent
matches a unit borrowed, so the cross-sectional asset demand has to integrate to zero.
The price that makes that integral vanish is the equilibrium return $r^{\ast}$.

The interesting comparison is the complete-markets benchmark. Without idiosyncratic risk,
the household Euler equation pins $r$ at exactly the rate of time preference $\rho$.
Once insurance is incomplete, the marginal value of buffer wealth is strictly positive
at any household with little cash on hand, and aggregate desired bond holdings are
positive at $r = \rho$. Markets only clear at a strictly lower bond return; the size of
the wedge $\rho - r^{\ast}$ is the price the economy puts on the missing insurance
contract. In this calibration the wedge is """ + f"$\\rho - r^{{\\ast}} = {wedge:.4f}$ "
        + f"(about ${wedge_pct:.0f}\\%$ of $\\rho$) — a quantitatively non-trivial "
        + r"""precautionary discount on the risk-free rate even with only two income states.

The continuous-time HJB/KFE representation of [Achdou et al. (2022)](https://benjaminmoll.com/lectures/)
is what `run.py` here implements. It carries two equilibrium objects in parallel: the
asset drift implied by household consumption decisions, $s_i(a) = z_i + ra - c_i(a)$, and
the stationary distribution that drift induces, $g_i(a)$. Bisection on $r$ closes the
loop. The neighbouring [Aiyagari tutorial](../../dynamic-programming/aiyagari/) runs
the same equilibrium logic in discrete time, but closes the model with a representative
firm so the asset in fixed supply is physical capital $K$ rather than a bond — a useful
distinction when comparing $r^{\ast}$ to $1/\beta - 1$ (the discrete-time mirror of $\rho$).
The household block in this tutorial is solved by an HJB upwind finite-difference scheme;
the Euler-equation analogues for discrete time, [EGP](../endogenous-grid-points/) and
[EEI](../envelope-equation-iteration/), live in the same section and target the same
buffer-stock policy.
"""
    )

    report.add_equations(
        r"""
A continuum of households is indexed by current income state $i \in \{L, H\}$ with
endowment $z_i$ and Poisson switching intensity $\lambda_i$ into the other state $j$. A
household holds assets $a$, earns the bond return $r$, and consumes $c_i(a)$; the asset
position evolves deterministically between income jumps according to the drift

$$\dot a \;=\; s_i(a) \;=\; z_i + r\,a - c_i(a), \qquad a \geq \underline a.$$

The state space is the half-line $a \in [\underline a, \infty)$ together with the
two-point chain over $i$. With CRRA flow utility $u(c) = c^{1-\sigma}/(1-\sigma)$ and
discount rate $\rho > 0$, the Hamilton-Jacobi-Bellman equation is

$$\rho\,V_i(a) \;=\; \max_{c > 0}\,
\Bigl[\,u(c) \;+\; V_i'(a)\,(z_i + r\,a - c) \;+\; \lambda_i\,\bigl(V_j(a) - V_i(a)\bigr)\,\Bigr].$$

The first two terms are the certainty-equivalent piece — current utility plus the
deterministic continuation value implied by $\dot a$ — and the last term is the
expected jump in the value function when income switches. The first-order condition
delivers the Euler/envelope identity

$$c_i(a) \;=\; \bigl[V_i'(a)\bigr]^{-1/\sigma}, \qquad
s_i(a) \;=\; z_i + r\,a - c_i(a).$$

The borrowing constraint enters as a *state constraint*: at $a = \underline a$ the asset
drift cannot point further left, so

$$s_i(\underline a) \;\geq\; 0
\quad\Longleftrightarrow\quad
V_i'(\underline a) \;\geq\; u'(z_i + r\,\underline a),$$

with equality whenever the constraint is slack and inequality when it binds. This is the
continuous-time counterpart to the Kuhn-Tucker margin in [EGP](../endogenous-grid-points/).

The cross-sectional density $g_i(a)$ on $(\underline a, \infty)$ satisfies the Kolmogorov
Forward Equation

$$0 \;=\; -\frac{\partial}{\partial a}\bigl[s_i(a)\,g_i(a)\bigr]
\;-\; \lambda_i\,g_i(a) \;+\; \lambda_j\,g_j(a),
\qquad \int g_L + g_H \;=\; 1,$$

with a delta-mass component at $\underline a$ for income states whose drift hits the
constraint with positive probability. Equilibrium in the bond market is the
zero-net-supply condition

$$S(r) \;\equiv\; \int_{\underline a}^{\bar a} a\,\bigl[g_L(a) + g_H(a)\bigr]\,da \;=\; 0.$$

In the deterministic mirror of the model, the household Euler equation reduces to
$\dot c / c = (r - \rho)/\sigma$, and a non-degenerate stationary equilibrium exists
only at $r = \rho$. Incomplete markets break that result: $S(\rho) > 0$ because households
want positive precautionary asset holdings, so equilibrium requires $r^{\ast} < \rho$.
Quantifying that wedge is the headline output here.
"""
    )

    report.add_model_setup(
        f"""
The calibration is intentionally compact — two income states, symmetric switching, a
one-dimensional asset grid — so that the precautionary-saving mechanism is the only
source of action. Larger income chains (Tauchen or Rouwenhorst) plug into the same
solver but obscure the Huggett wedge with calibration noise.

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

The two switching intensities are equal so the income chain has symmetric stationary
probabilities $p_L = p_H = 0.5$, and expected income is $\\bar z = {mean_z_analytic:.4f}$.
At the working solution the cross-sectional probabilities recover this prediction to
$|p_L - 0.5| = {p_balance_err:.2e}$, a basic sanity check on the KFE solve.
"""
    )

    report.add_solution_method(
        r"""
The household block at a candidate $r$ is solved by an implicit upwind
finite-difference scheme on the asset grid. The two delicate pieces are the choice of
derivative for $V_i'(a)$ at each grid point and the construction of the discrete generator
$A$ that approximates $\partial/\partial a[s_i(a)\,\cdot] + \lambda$-switching.

**Upwind derivative.** At each $(a_k, i)$ the algorithm computes both the forward and
backward finite-difference approximations of $V_i'(a_k)$, converts each into a candidate
consumption via $c = (V_i')^{-1/\sigma}$, and then picks the one whose implied drift
$s_i(a_k) = z_i + r a_k - c$ points *into* the grid (forward when $s>0$, backward when $s<0$).
This upwinding is what makes the discrete generator a sub-stochastic Markov matrix and
keeps the borrowing limit from being crossed numerically. Centred differences would
break that property and admit unphysical reflections off $\underline a$.

**Implicit step.** Stack $V$ over income states into a vector of length $2I$. The
implicit HJB update is

$$\bigl[(\Delta^{-1} + \rho)\,\mathbf I - A^{n}\bigr]\,V^{n+1} \;=\; u(c^{n}) + \Delta^{-1} V^{n},$$

where $A^{n}$ is the upwind transition generator built from the current drift and the
income-switching intensities $\lambda_i$. With a large step $\Delta = 1000$ this update
behaves like a Newton step on the steady-state HJB $\rho V = u(c) + AV$, so convergence
is essentially quadratic; in this calibration the inner loop terminates in single-digit
iterations.

**KFE.** Once $V$ converges, the same generator delivers the stationary distribution as
the left null space of $A$: solve $A^{\top} g = 0$ subject to $\int g = 1$. The system
is singular (the generator has a zero eigenvalue), so the algorithm pins one row of
$A^{\top}$ and rescales the solution to integrate to one.

**Equilibrium.** The bond-market excess demand $S(r) = \int a\,(g_L + g_H)\,da$ is
strictly increasing in $r$ on the relevant range — higher returns make saving more
attractive and discourage borrowing — so a single bisection on $[r_{\min}, r_{\max}]$
locates $r^{\ast}$ to any desired tolerance.

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

**Working solve.** The HJB inner loop converged in **""" + f"{info['iterations']} iterations** "
        + f"(sup-norm change ${info['error']:.2e}$), and the outer bisection located "
        + f"$r^{{\\ast}} = {r_eq:.5f}$ with bond-market residual ${market_residual:.2e}$ on "
        + f"the {I}-point asset grid."
        + r"""

**Reference solve and audit.** The same equilibrium is recomputed on a """
        + f"$I_{{\\rm ref}} = {I_ref}$-point reference grid as a discretisation audit. The "
        + f"reference equilibrium is $r^{{\\ast}}_{{\\rm ref}} = {r_eq_ref:.5f}$, so the "
        + f"interest-rate gap is $|r^{{\\ast}}_{{{I}}} - r^{{\\ast}}_{{{I_ref}}}| = {r_gap:.2e}$. "
        + r"""On the active asset range $a \in [\underline a, 1]$ the working savings policy
$s_i(a)$ lies within """ + f"${s_gap:.2e}$ " + r"""of the interpolated reference policy in
sup norm; the value function gap is """ + f"${V_gap:.2e}$, or about ${100 * V_gap_rel:.2f}\\%$ "
        + r"""relative to the value scale. The grid convergence of $r^{\ast}$ in this
calibration is genuinely slow — uniform-grid HJB is first-order accurate at the borrowing
limit, where the policy has a kink — so refining beyond $I = """ + f"{I_ref}" + r"""$ would
shift $r^{\ast}$ further toward $\rho$ at a rate that scales like $1/I$. A non-uniform
asset grid concentrated near $\underline a$ (cf. Achdou et al. 2022, App. C) tightens this
quickly. For the qualitative wedge $\rho - r^{\ast} > 0$ and the cross-sectional shapes
the working grid is more than enough.
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
        description="The first figure plots $V_L(a)$ and $V_H(a)$ at $r^{\\ast}$ on the "
        "working grid (solid) and the reference equilibrium values on the finer grid (dashed). "
        "Both curves are increasing and concave in $a$, with $V_H > V_L$ uniformly because "
        "income enters cash on hand linearly. Near the borrowing limit $\\underline a$ both "
        "curves steepen sharply: a marginal dollar of wealth there relaxes the state "
        "constraint $s_i(\\underline a) \\geq 0$ and buys insurance against staying in the "
        "low-income state. The reference and working curves are visually indistinguishable on "
        f"the active range — the relative gap in $V$ is about ${100 * V_gap_rel:.2f}\\%$ — "
        "while the small vertical level shift at the right edge reflects the discretisation "
        f"in the equilibrium price ($|r^{{\\ast}}_{{{I}}} - r^{{\\ast}}_{{{I_ref}}}| = "
        f"{r_gap:.2e}$).",
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
        description="The savings policy is the asset drift $\\dot a = s_i(a)$ at the "
        "equilibrium price. The low-income household decumulates ($s_L < 0$) almost everywhere "
        "above the borrowing limit and is pushed onto the constraint by the state-constraint "
        "boundary condition; this is the visible kink of $s_L$ at $\\underline a$. The high-"
        "income household saves ($s_H > 0$) at small $a$ to rebuild buffer wealth and crosses "
        "zero at the income-state-specific asset target where $z_H + r^{\\ast} a = c_H(a)$. "
        "Income switching keeps the cross section moving across the two drift fields — a "
        "household never stays on a single curve. The "
        f"reference overlay agrees to ${s_gap:.2e}$ in sup norm on $[\\underline a, 1]$.",
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
        description="The KFE turns the drift fields above into a cross-sectional density. The "
        "low-income $g_L$ piles up at the borrowing limit because $s_L < 0$ pushes households "
        "toward $\\underline a$ and the state constraint stops them there; the kink in the "
        "low-income drift translates into a sharp spike that becomes more concentrated as the "
        "grid is refined. The high-income $g_H$ is flatter and supported on a wider range "
        "because $s_H > 0$ near $\\underline a$ moves households to the right. Together the "
        f"two densities place ${100 * mass_at_constraint:.1f}\\%$ of the population within "
        f"${constraint_window}$ of the borrowing limit, the visible signature of incomplete "
        "insurance. The reference density (dashed) shows a slightly taller and narrower spike "
        "at $\\underline a$ — finer discretisation resolves the constraint mass more sharply — "
        "but the away-from-constraint shape is unchanged.",
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
        description="The supply curve $S(r)$ is the equilibrium-pricing argument made visible. "
        "At any $r$, every household solves the HJB with that bond return as a primitive, and "
        "$S(r)$ aggregates their stationary asset positions. The curve is monotone in $r$ "
        "because higher returns simultaneously raise desired saving and discourage borrowing. "
        "The dashed horizontal line at $r = \\rho$ is the complete-markets benchmark — the "
        "rate at which a representative household with no insurance demand would price the "
        "bond. The Huggett equilibrium sits strictly below it: even at $r$ as low as "
        f"$r^{{\\ast}} = {r_eq:.4f}$ (red dot), aggregate asset demand only just clears zero, "
        "and the precautionary wedge is "
        f"$\\rho - r^{{\\ast}} = {wedge:.4f}$. The reference equilibrium "
        f"$r^{{\\ast}}_{{\\rm ref}} = {r_eq_ref:.4f}$ (black cross) lies on top of the working "
        "solution at this resolution.",
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
        description="The table separates economic outputs from numerical diagnostics. The top "
        "block is the equilibrium price and its precautionary wedge; the middle block reports "
        "cross-sectional moments; the bottom block bounds the discretisation by comparing the "
        "working grid against the finer reference grid. Mean assets are numerically zero by "
        "construction — the bisection chose $r^{\\ast}$ to enforce $S(r^{\\ast}) = 0$ to the "
        "tolerance shown in the residual row.",
    )

    report.add_takeaway(
        r"""
The Huggett pricing mechanism is the lesson. With incomplete insurance and a hard
borrowing limit, every household's problem assigns strictly positive marginal value to
buffer wealth, so aggregate desired bond holdings at the deterministic-benchmark rate
$r = \rho$ are positive. Markets clear only at a strictly lower bond return, and the
wedge $\rho - r^{\ast}$ — """
        + f"$\\,{wedge:.4f}$ on the working grid, about ${wedge_pct:.0f}\\%$ of $\\rho$ — "
        + r"""is the price the economy charges for the missing state-contingent insurance
contract. That number tightens by about """ + f"${r_gap:.1e}$ " + r"""when the grid is
refined from """ + f"$I = {I}$ to $I = {I_ref}$" + r""", but the qualitative wedge does not.

The continuous-time HJB/KFE machinery is what keeps the household decision and the
induced cross section in the same equilibrium loop. The state-constraint boundary
condition $V_i'(\underline a) \geq u'(z_i + r\underline a)$ is the natural analogue of the
discrete-time Kuhn-Tucker margin in [EGP](../endogenous-grid-points/) and
[EEI](../envelope-equation-iteration/), and the upwind finite-difference scheme is what
makes that boundary condition hold without spurious mass leakage across $\underline a$.
The discrete-time mirror is the [Aiyagari tutorial](../../dynamic-programming/aiyagari/),
where the asset in fixed supply is physical capital and the wedge becomes
$1/\beta - 1 - r^{\ast}$. The Euler-based household solvers from EGP and EEI would slot
directly into the inner step of either equilibrium computation, with the bisection on $r$
unchanged.
"""
    )

    report.add_references([
        "Huggett, M. (1993). \"The risk-free rate in heterogeneous-agent incomplete-insurance economies.\" *Journal of Economic Dynamics and Control* 17(5-6), 953-969.",
        "Aiyagari, S. R. (1994). \"Uninsured Idiosyncratic Risk and Aggregate Saving.\" *Quarterly Journal of Economics* 109(3), 659-684.",
        "Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). \"Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach.\" *Review of Economic Studies* 89(1), 45-86.",
        "Moll, B. \"Lecture notes on continuous-time heterogeneous-agent models.\" https://benjaminmoll.com/lectures/",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
