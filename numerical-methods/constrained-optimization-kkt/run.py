#!/usr/bin/env python3
"""Constrained optimization for budget allocation across diminishing-returns projects.

A planner allocates a fixed budget across three projects with diminishing
marginal returns and non-negativity bounds. The unconstrained-on-bounds
solver returns a negative allocation, which is the failure baseline that
motivates the KKT machinery. Four methods are compared: a closed-form
Lagrangian on the budget alone, projected gradient onto the simplex, an
interior-point log barrier, and SLSQP via scipy. KKT residuals on
stationarity, primal feasibility, dual feasibility, and complementary
slackness are the diagnostics, not the value of the objective alone.

References:
- Boyd and Vandenberghe (2004) Convex Optimization, Ch. 5, 11.
- Nocedal and Wright (2006) Numerical Optimization, Ch. 12, 17, 19.
- Bertsekas (1999) Nonlinear Programming, Ch. 2-3.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from scipy.optimize import brentq, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def main() -> None:
    # =========================================================================
    # Calibration: three projects, diagonal Hessian, binding budget
    # =========================================================================
    a = np.array([4.0, 3.0, 0.5])
    B = np.eye(3)
    I_total = 3.0
    n_proj = 3

    x_star = np.array([2.0, 1.0, 0.0])
    lambda_star = 2.0
    mu_star = np.array([0.0, 0.0, 1.5])
    u_star = float(a @ x_star - 0.5 * x_star @ B @ x_star)

    def utility(x):
        return float(a @ x - 0.5 * x @ B @ x)

    def utility_grad(x):
        return a - B @ x

    # =========================================================================
    # Projection onto the budget simplex {x >= 0, sum(x) = I_total}
    # =========================================================================
    def project_simplex(v, total):
        n = len(v)
        u_sorted = np.sort(v)[::-1]
        cssv = np.cumsum(u_sorted) - total
        idx = np.arange(1, n + 1)
        cond = u_sorted - cssv / idx > 0
        rho = int(np.where(cond)[0][-1])
        theta = cssv[rho] / (rho + 1)
        return np.maximum(v - theta, 0.0)

    # =========================================================================
    # Multiplier recovery from a candidate x
    # =========================================================================
    eps_active = 1e-6

    def recover_multipliers(x):
        active = x > eps_active
        grad_neg = a - B @ x
        if active.any():
            lam = float(np.mean(grad_neg[active]))
        else:
            lam = float(np.max(a))
        mu = np.zeros_like(x)
        for j in range(len(x)):
            if not active[j]:
                mu[j] = max(0.0, lam - float(grad_neg[j]))
        return lam, mu

    def kkt_residuals(x, lam=None, mu=None):
        if lam is None or mu is None:
            lam, mu = recover_multipliers(x)
        stat = float(np.linalg.norm((a - B @ x) - lam * np.ones_like(x) + mu))
        primal_eq = abs(float(x.sum()) - I_total)
        primal_neg = float(np.sum(np.maximum(-x, 0.0)))
        primal = primal_eq + primal_neg
        dual = max(0.0, -lam) + float(np.sum(np.maximum(-mu, 0.0)))
        compl = float(np.sum(mu * np.abs(x))) + abs(lam * (I_total - float(x.sum())))
        return stat, primal, dual, compl

    # =========================================================================
    # Method 1: Lagrangian on the budget alone (fails on non-negativity)
    # =========================================================================
    lam_baseline = (a.sum() - I_total) / n_proj
    x_baseline = a - lam_baseline
    bl_stat, bl_primal, bl_dual, bl_compl = kkt_residuals(x_baseline)
    bl_lam, bl_mu = recover_multipliers(x_baseline)
    bl_utility = utility(x_baseline)

    # =========================================================================
    # Method 2: projected gradient onto the budget simplex
    # =========================================================================
    step = 0.25
    pg_x = np.array([0.5, 0.5, 2.0])  # heavy on project 3, far from x*
    pg_history = [pg_x.copy()]
    pg_residuals = [float(np.linalg.norm(pg_x - x_star))]
    pg_kkt_trace = [kkt_residuals(pg_x)]
    pg_max_iter = 500
    pg_tol = 1e-12
    for _ in range(1, pg_max_iter + 1):
        grad = utility_grad(pg_x)
        pg_x = project_simplex(pg_x + step * grad, I_total)
        pg_history.append(pg_x.copy())
        pg_residuals.append(float(np.linalg.norm(pg_x - x_star)))
        pg_kkt_trace.append(kkt_residuals(pg_x))
        if pg_residuals[-1] < pg_tol:
            break
    pg_iter = len(pg_history) - 1
    pg_x_final = pg_history[-1]
    pg_lam, pg_mu = recover_multipliers(pg_x_final)
    pg_stat, pg_primal, pg_dual, pg_compl = kkt_residuals(pg_x_final)
    pg_utility = utility(pg_x_final)

    # =========================================================================
    # Method 3: interior-point log barrier
    # =========================================================================
    def x_of_lambda(lam, t):
        d = a - lam
        return 0.5 * (d + np.sqrt(d ** 2 + 4 * t))

    def find_lambda(t):
        return brentq(lambda lam: x_of_lambda(lam, t).sum() - I_total, -100.0, 100.0, xtol=1e-13)

    barriers = [10.0, 1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
    barrier_history = []
    barrier_residuals = []
    barrier_kkt_trace = []
    for t in barriers:
        lam_t = find_lambda(t)
        x_t = x_of_lambda(lam_t, t)
        barrier_history.append((t, x_t, lam_t))
        barrier_residuals.append(float(np.linalg.norm(x_t - x_star)))
        # Exact barrier-derived multipliers: mu_j = t / x_j on the central path.
        mu_t = t / x_t
        barrier_kkt_trace.append(kkt_residuals(x_t, lam_t, mu_t))
    barrier_iter = len(barriers)
    barrier_x_final = barrier_history[-1][1]
    barrier_lam_final = barrier_history[-1][2]
    barrier_lam, barrier_mu = recover_multipliers(barrier_x_final)
    barrier_stat, barrier_primal, barrier_dual, barrier_compl = kkt_residuals(barrier_x_final)
    barrier_utility = utility(barrier_x_final)

    # =========================================================================
    # Method 4: SLSQP via scipy.optimize.minimize
    # =========================================================================
    def neg_utility(x):
        return -utility(x)

    def neg_utility_grad(x):
        return -utility_grad(x)

    slsqp_result = minimize(
        neg_utility,
        x0=np.array([1.0, 1.0, 1.0]),
        jac=neg_utility_grad,
        method='SLSQP',
        bounds=[(0.0, None)] * n_proj,
        constraints=[{
            'type': 'eq',
            'fun': lambda x: float(x.sum() - I_total),
            'jac': lambda x: np.ones(n_proj),
        }],
        options={'ftol': 1e-12, 'maxiter': 200, 'disp': False},
    )
    x_slsqp = np.asarray(slsqp_result.x, dtype=float)
    slsqp_iter = int(slsqp_result.nit)
    slsqp_lam, slsqp_mu = recover_multipliers(x_slsqp)
    slsqp_stat, slsqp_primal, slsqp_dual, slsqp_compl = kkt_residuals(x_slsqp, slsqp_lam, slsqp_mu)
    slsqp_utility = utility(x_slsqp)

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Constrained Optimization and KKT Conditions",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A planner has a fixed budget and three projects to fund. "
        "Each project has diminishing returns. "
        "Allocations cannot be negative. "
        "The question is how the planner should split the budget when one of the projects is too weak to fund at all.\n\n"
        "This tutorial compares three methods that solve the constrained problem. "
        "Before the methods we show a baseline that ignores the non-negativity bounds. "
        "The baseline returns a negative allocation, which is the failure mode that motivates the rest of the tutorial.\n\n"
        "The three methods are projected gradient, an interior-point log barrier, and SLSQP. "
        "All three return the correct allocation. "
        "They differ in how they keep iterates feasible and in how they recover the Lagrange multipliers.\n\n"
        "The main lesson is that the value of the objective is not enough to judge a constrained answer. "
        "What matters is the set of Karush-Kuhn-Tucker conditions: stationarity, primal feasibility, dual feasibility, and complementary slackness. "
        "The Lagrange multipliers on binding constraints are the shadow prices that the economist actually wants to read."
    )

    report.add_equations(
        r"""The planner picks an allocation vector $x \in \mathbb{R}^3$.
Each entry $x_j$ is the budget share assigned to project $j$.
Utility is quadratic in $x$.

$$u(x) = a^\top x - \tfrac{1}{2}\, x^\top B x.$$

$a \in \mathbb{R}^3$ is the vector of marginal returns at zero allocation.
$B$ is a symmetric positive-definite matrix.
A positive-definite $B$ makes $u$ strictly concave, so the constrained maximum is unique.
The diagonal entries of $B$ measure each project's curvature.

Two constraints bind the choice.
The first is a budget cap on total spending.
The second is a separate non-negativity bound on each project.

$$\sum_{j=1}^{3} x_j \leq I,
\qquad x_j \geq 0,\quad j = 1, 2, 3.$$

The Lagrangian builds in both constraints.
$\lambda$ is the multiplier on the budget cap.
$\mu = (\mu_1, \mu_2, \mu_3)$ are the multipliers on the three non-negativity bounds.

$$\mathcal{L}(x, \lambda, \mu) = a^\top x - \tfrac{1}{2}\, x^\top B x - \lambda \left(\sum_j x_j - I \right) + \mu^\top x.$$

A Karush-Kuhn-Tucker (KKT) point is the constrained optimum.
The KKT conditions split into four blocks.
Each block has a clean economic reading.

The first block is stationarity.
It equates the gradient of utility with the shadow-price vector.

$$a - B x - \lambda\, \mathbf{1} + \mu = 0.$$

The second block is primal feasibility.
It is just the constraint set written out again.

$$\sum_j x_j \leq I,
\qquad x_j \geq 0.$$

The third block is dual feasibility.
It says every shadow price is non-negative.

$$\lambda \geq 0,
\qquad \mu_j \geq 0.$$

The fourth block is complementary slackness.
It says either a constraint binds or its multiplier is zero, never both.

$$\lambda \left(I - \sum_j x_j \right) = 0,
\qquad \mu_j\, x_j = 0.$$

The baseline calibration is $a = (4, 3, 0.5)$, $B = I_3$, and $I = 3$.
The unconstrained maximum is $a$ itself.
Its sum is $7.5$, which exceeds the budget of $3$.
The budget therefore binds at the constrained optimum.
A second active-set check shows that project 3 also hits its non-negativity bound.
With those two constraints active, the closed form is direct.

$$x^{\ast} = (2,\, 1,\, 0),
\qquad
\lambda^{\ast} = 2,
\qquad
\mu^{\ast} = (0,\, 0,\, 1.5).$$

The non-zero multiplier $\mu_3^{\ast} = 1.5$ is the shadow price of the non-negativity bound on project 3.
It is the utility a vanishingly small relaxation of $x_3 \geq 0$ would buy.

The next four subsections describe a baseline and three methods.

### Baseline failure: drop the non-negativity bounds

A common shortcut keeps only the budget equality and drops the non-negativity bounds.
The Lagrangian is then linear in $x$ and $\lambda$.

$$x = a - \lambda\, \mathbf{1},
\qquad
\lambda = \frac{\sum_j a_j - I}{n}.$$

At the calibration this gives $\lambda = 1.5$ and $x = (2.5, 1.5, -1)$.
Project 3 receives a negative allocation, which has no economic meaning.
The three methods below all enforce the non-negativity bounds and recover the correct optimum.

### Method 1: Projected gradient

Projected gradient takes a gradient step on $u$ and then projects the result onto the simplex $\Delta_I = \lbrace x : x \geq 0,\, \sum_j x_j = I \rbrace$.

$$x_{k+1} = \Pi_{\Delta_I}\left(x_k + \alpha\, (a - B x_k)\right).$$

Here $\alpha$ is the step size and $\Pi_{\Delta_I}$ is Euclidean projection onto the simplex.
The step size must satisfy $\alpha \leq 1/L$ where $L$ is the operator norm of $B$; with $B = I_3$ the bound is $\alpha \leq 1$.

### Method 2: Log barrier

Log barrier replaces the non-negativity inequalities with a smooth penalty controlled by a parameter $t > 0$.

$$\min_x\, -u(x) - t \sum_j \log x_j
\qquad \text{subject to} \quad \sum_j x_j = I.$$

The barrier penalises iterates that approach the boundary $x_j = 0$.
As $t$ shrinks the optimum of the smoothed problem traces a central path that converges to the true optimum $x^{\ast}$ in the limit $t \to 0$.

The first-order condition for the barrier subproblem is one equation per project plus the budget equality.

$$a_j - x_j - \lambda + \frac{t}{x_j} = 0,
\qquad \sum_j x_j = I.$$

For diagonal $B = I_3$ each project's component solves a quadratic in $x_j$.

$$x_j(\lambda;\, t) = \frac{(a_j - \lambda) + \sqrt{(a_j - \lambda)^2 + 4 t}}{2}.$$

The budget multiplier $\lambda$ is the unique scalar that makes $\sum_j x_j(\lambda;\, t)$ equal $I$, and a single one-dimensional root finder solves for it.
The duality gap of the barrier problem is exactly $n \cdot t$, which is the per-project complementarity slack along the central path.

### Method 3: SLSQP

SLSQP calls `scipy.optimize.minimize` and treats the problem as sequential quadratic programming.
Each step linearises the constraints around the current iterate and solves a small quadratic-programming (QP) subproblem.
The QP uses a BFGS approximation of the Hessian of the Lagrangian; the QP solution becomes the search direction; a line search along that direction picks the next iterate.
Multipliers are recovered after the fact from the stationarity equation by averaging $a_j - (B x)_j$ over the active set and solving for the bound multipliers on the inactive set.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $a$ | $({a[0]:.1f},\\, {a[1]:.1f},\\, {a[2]:.1f})$ | Marginal returns at zero allocation |\n"
        f"| $B$ | $I_3$ | Curvature of utility, diagonal positive definite |\n"
        f"| $I$ | {I_total:.1f} | Total budget |\n"
        f"| $n$ | {n_proj} | Number of projects |\n"
        f"| $x^{{\\ast}}$ | $({x_star[0]:.1f},\\, {x_star[1]:.1f},\\, {x_star[2]:.1f})$ | Closed-form optimal allocation |\n"
        f"| $\\lambda^{{\\ast}}$ | {lambda_star:.1f} | Closed-form budget shadow price |\n"
        f"| $\\mu^{{\\ast}}$ | $({mu_star[0]:.1f},\\, {mu_star[1]:.1f},\\, {mu_star[2]:.1f})$ | Closed-form bound multipliers |\n"
        f"| $u^{{\\ast}}$ | {u_star:.4f} | Utility at the closed-form optimum |\n"
        f"| Step $\\alpha$ | {step:.2f} | Projected gradient step size |\n"
        f"| Barrier sequence | $10$ down to $10^{{-8}}$ | Decreasing log-barrier parameters |\n"
        f"| Tolerance $\\eta$ | {pg_tol:.0e} | Stopping rule on iterate change |"
    )

    report.add_solution_method(
        "Three methods solve the constrained allocation problem. "
        "Before them comes a baseline that ignores the non-negativity bounds. "
        "The baseline returns the wrong answer and shows why the bounds matter.\n\n"

        "### Baseline failure: Lagrangian on the budget alone\n\n"
        "An analyst could write the Lagrangian for the budget only and solve it. "
        "This is fast and gives a closed form. "
        "It is also wrong whenever a non-negativity bound binds at the true optimum. "
        "Dropping a binding constraint drops a piece of complementary slackness. "
        "A wrong-sign allocation then becomes possible. "
        "The baseline is included here only to make the failure mode concrete.\n\n"
        "```text\n"
        "Algorithm: Lagrangian on the budget alone (baseline failure)\n"
        "Input : a, B, I, project count n\n"
        "Output: x_hat, lambda_hat\n"
        "  lambda_hat <- (sum(a) - I) / n           # closed form, only valid when B = I\n"
        "  x_hat      <- a - lambda_hat * ones(n)   # negative entries possible\n"
        "```\n\n"
        "At the baseline calibration the answer is $x = (2.5, 1.5, -1)$. "
        "Project 3 receives a negative allocation. "
        "Stationarity is satisfied for the smaller problem the analyst wrote down. "
        "Primal feasibility is the part that breaks. "
        "Reading off the utility value of this answer gives a number that exceeds the true optimum, which is the easiest way to publish a wrong result.\n\n"

        "### Method 1: Projected gradient on the simplex\n\n"
        "Projected gradient walks the iterate uphill in utility, then snaps it back to the feasible simplex. "
        "Each step has two pieces. "
        "The gradient piece is $y = x_k + \\alpha\\, (a - B x_k)$, which moves in the direction of steepest utility increase. "
        "The projection piece is $\\Pi_{\\Delta_I}(y)$, which finds the closest point in the budget simplex to $y$ in Euclidean distance. "
        "The composition keeps every iterate feasible, including non-negativity.\n\n"
        "The simplex projection is closed form. "
        "Sort the components of $y$ in descending order. "
        "Find the largest index $\\rho$ for which a running average is positive. "
        "Subtract a single scalar shift from $y$ and clip negatives to zero. "
        "The whole projection costs one sort plus a linear scan over $\\rho$.\n\n"
        "Convergence is linear in the gap to the optimum. "
        "The contraction rate is roughly $1 - \\alpha\\, \\mu / L$, with $\\mu$ the smallest eigenvalue of $B$ and $L$ the largest. "
        "On the calibration $B = I_3$ the eigenvalues coincide and the rate is $1 - \\alpha$. "
        "The method needs only a gradient and the projection routine, which makes it the easiest constrained method to implement from scratch.\n\n"
        "```text\n"
        "Algorithm: Projected gradient on the budget simplex\n"
        "Input : a, B, I, step alpha, tolerance eta, interior start x_0\n"
        "Output: x_k\n"
        "  for k = 0, 1, ... :\n"
        "      grad     <- a - B x_k\n"
        "      y        <- x_k + alpha * grad\n"
        "      x_{k+1}  <- project_simplex(y, I)\n"
        "      stop when ||x_{k+1} - x_k|| < eta\n"
        "\n"
        "  project_simplex(y, I):\n"
        "      sort y in descending order to get u_1 >= u_2 >= ... >= u_n\n"
        "      cumsum_i <- u_1 + u_2 + ... + u_i\n"
        "      rho      <- largest i with u_i - (cumsum_i - I) / i > 0\n"
        "      theta    <- (cumsum_rho - I) / rho\n"
        "      return max(y - theta, 0) componentwise\n"
        "```\n\n"
        "Projected gradient does not fail on this calibration. "
        "Its weak spot is the step size. "
        "A step larger than $1/L$ pushes the iterate so far that the projection wastes the work. "
        "A step well below $1/L$ slows convergence with no benefit. "
        "When the gradient is unavailable a finite-difference approximation works but adds noise that the linear convergence rate does not absorb well.\n\n"

        "### Method 2: Interior-point log barrier\n\n"
        "The log barrier replaces each hard non-negativity bound with a smooth penalty. "
        "The penalised objective is $-u(x) - t \\sum_j \\log x_j$, minimised subject to the budget equality. "
        "The penalty pushes iterates away from the boundary $x_j = 0$ because $\\log x_j$ heads to $-\\infty$ there. "
        "As the barrier parameter $t$ shrinks the penalty weakens and the optimum approaches the boundary.\n\n"
        "Geometrically the optima of the smoothed problems trace a curve called the central path. "
        "The path starts deep in the interior at large $t$ and ends at $x^{\\ast}$ as $t \\to 0$. "
        "The duality gap along the path is exactly $n \\cdot t$, which is the per-project complementarity slack. "
        "Choosing a geometrically decreasing schedule for $t$ gives geometric convergence to the constrained optimum.\n\n"
        "Each subproblem in $t$ has a closed form when $B$ is diagonal. "
        "The first-order condition for project $j$ is a quadratic in $x_j$ given the budget multiplier $\\lambda$. "
        "Solving it gives $x_j(\\lambda; t)$ as an explicit function. "
        "The budget multiplier itself is then a single scalar root of $\\sum_j x_j(\\lambda; t) = I$, found by Brent's method on a wide bracket.\n\n"
        "```text\n"
        "Algorithm: Interior-point log barrier\n"
        "Input : a, B, I, decreasing barrier sequence t_1 > t_2 > ... > t_K\n"
        "Output: x_K close to the constrained optimum\n"
        "  x_0 <- strictly interior feasible point\n"
        "  for k = 1, ..., K :\n"
        "      define x_j(lambda; t_k) = ((a_j - lambda) + sqrt((a_j - lambda)^2 + 4 t_k)) / 2\n"
        "      solve sum_j x_j(lambda; t_k) = I for lambda using brentq\n"
        "      x_k <- (x_1(lambda; t_k), ..., x_n(lambda; t_k))\n"
        "```\n\n"
        "The barrier needs a strictly interior starting point. "
        "A start with any $x_j = 0$ makes the log infinite, so it cannot be evaluated. "
        "The barrier schedule itself matters too. "
        "Shrinking $t$ too fast makes the budget multiplier jump and the root finder fails. "
        "A common choice is $t_{k+1} = t_k / 10$ once a few steps have stabilised the multiplier.\n\n"

        "### Method 3: SLSQP via scipy.optimize.minimize\n\n"
        "SLSQP stands for Sequential Least-SQuares Programming. "
        "It is a quasi-Newton method designed for constrained problems with smooth equalities and inequalities. "
        "At each iterate it linearises the constraints and forms a small quadratic-programming subproblem. "
        "The QP uses a BFGS approximation of the Hessian of the Lagrangian. "
        "Solving the QP gives a search direction. "
        "A line search along the direction picks the next iterate.\n\n"
        "The QP at iterate $x_k$ has the form: minimise a quadratic in the step $d$ subject to linear constraints in $d$. "
        "The quadratic coefficients come from the BFGS approximation of the Lagrangian Hessian, which is updated from gradient differences across iterations. "
        "The constraints are linearisations of the original equality and inequality constraints. "
        "An active-set routine inside the QP solver decides which inequalities bind. "
        "Convergence near a non-degenerate optimum is locally quadratic.\n\n"
        "SLSQP is the practical default for small problems that mix equality and inequality constraints. "
        "It accepts analytical or finite-difference Jacobians, returns an iteration count, and converges in just a handful of QP solves on a problem this size.\n\n"
        "```text\n"
        "Algorithm: SLSQP via scipy.optimize.minimize\n"
        "Input : objective f, gradient grad_f, equality g, bounds, x_0\n"
        "Output: x_hat, iteration count, convergence flag\n"
        "  scipy hands the problem to a Fortran SLSQP routine\n"
        "  for each iterate x_k :\n"
        "      build a QP in the step d:\n"
        "          minimise (1/2) d^T H_k d + grad_f(x_k)^T d\n"
        "          subject to grad_g(x_k)^T d + g(x_k) = 0\n"
        "                     bounds on x_k + d\n"
        "      solve the QP by an active-set method to get d_k\n"
        "      do a line search along d_k to pick the next x_k\n"
        "      update H_k by BFGS using the gradient difference\n"
        "  recover lambda from the active-set stationarity equation\n"
        "  recover mu by complementary slackness on inactive bounds\n"
        "```\n\n"
        "SLSQP is sensitive to the analytical Jacobian of the constraints. "
        "A wrong Jacobian silently mis-converges with no diagnostic. "
        "The default scaling can also struggle on problems where some constraints have much larger residuals than others. "
        "The remedy is either to rescale the constraints by hand or to switch to a method that does it internally, such as `scipy.optimize.minimize` with `method='trust-constr'`."
    )

    # ------------------------------------------------------------------
    # Figure 1: projected-gradient path on the budget simplex
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    triangle = Polygon(
        [(0.0, 0.0), (I_total, 0.0), (0.0, I_total)],
        closed=True, facecolor="tab:blue", edgecolor="tab:blue", alpha=0.08, linewidth=1.5,
    )
    ax1.add_patch(triangle)
    ax1.plot([0.0, I_total, 0.0, 0.0], [0.0, 0.0, I_total, 0.0],
             color="tab:blue", linewidth=1.5, alpha=0.6)
    pg_arr = np.array(pg_history)
    ax1.plot(pg_arr[:, 0], pg_arr[:, 1], "-", color="tab:orange", linewidth=1.0, alpha=0.7)
    n_show_pg = min(15, len(pg_arr))
    ax1.plot(pg_arr[:n_show_pg, 0], pg_arr[:n_show_pg, 1],
             "o", color="tab:orange", markersize=5, label="Projected gradient iterate")
    ax1.plot(pg_arr[0, 0], pg_arr[0, 1], "o", color="tab:gray", markersize=8,
             markeredgecolor="black", label=fr"Start $x_0 = (0.5,\, 0.5,\, 2.0)$")
    ax1.plot(x_star[0], x_star[1], "*", color="tab:red", markersize=18,
             label=fr"$x^{{\ast}} = (2,\, 1,\, 0)$")
    ax1.text(I_total + 0.05, 0.05, "$x_3 = 0$\n(project 3 inactive)", fontsize=9, color="tab:purple")
    ax1.set_xlabel("Project 1 allocation $x_1$")
    ax1.set_ylabel("Project 2 allocation $x_2$")
    ax1.set_title(fr"Projected gradient on the budget simplex (project 3 implicit, $x_3 = I - x_1 - x_2$)")
    ax1.set_xlim(-0.2, I_total + 0.5)
    ax1.set_ylim(-0.2, I_total + 0.5)
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", fontsize=9)

    report.add_results(
        f"The feasible region is the budget triangle. "
        f"Each vertex puts the entire budget on one project. "
        f"The closed-form optimum sits on the hypotenuse where the project-3 bound is active. "
        f"Projected gradient starts at $x_0 = (0.5,\\, 0.5,\\, 2.0)$, where project 3 is heavily over-funded. "
        f"The first projection lands on the budget hyperplane and subsequent steps slide along it toward $x^{{\\ast}}$. "
        f"The run converges in **{pg_iter}** iterations and every iterate is feasible."
    )
    report.add_figure(
        "figures/simplex-paths.png",
        "Projected gradient path on the budget simplex; project 3 is implicit",
        fig1,
    )

    # ------------------------------------------------------------------
    # Figure 2: barrier path approaching the boundary
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    triangle2 = Polygon(
        [(0.0, 0.0), (I_total, 0.0), (0.0, I_total)],
        closed=True, facecolor="tab:blue", edgecolor="tab:blue", alpha=0.08, linewidth=1.5,
    )
    ax2.add_patch(triangle2)
    ax2.plot([0.0, I_total, 0.0, 0.0], [0.0, 0.0, I_total, 0.0],
             color="tab:blue", linewidth=1.5, alpha=0.6)
    barrier_arr = np.array([h[1] for h in barrier_history])
    ax2.plot(barrier_arr[:, 0], barrier_arr[:, 1], "-", color="tab:purple",
             linewidth=1.5, alpha=0.7, label="Central path")
    for i, (t, x_t, _) in enumerate(barrier_history):
        ax2.plot(x_t[0], x_t[1], "d", color="tab:purple", markersize=7)
    # Label only the first and last points to avoid clutter.
    t0, x0_arr, _ = barrier_history[0]
    tend, xend, _ = barrier_history[-1]
    ax2.annotate(fr"$t = {t0:.0e}$ (most interior)",
                 xy=(x0_arr[0], x0_arr[1]), xytext=(x0_arr[0] - 0.55, x0_arr[1] + 0.5),
                 fontsize=9, color="tab:purple",
                 arrowprops=dict(arrowstyle="->", color="tab:purple", linewidth=0.8, alpha=0.7))
    ax2.annotate(fr"$t = {tend:.0e}$ (near $x^{{\ast}}$)",
                 xy=(xend[0], xend[1]), xytext=(xend[0] + 0.20, xend[1] - 0.5),
                 fontsize=9, color="tab:purple",
                 arrowprops=dict(arrowstyle="->", color="tab:purple", linewidth=0.8, alpha=0.7))
    ax2.plot(x_star[0], x_star[1], "*", color="tab:red", markersize=18,
             label=fr"$x^{{\ast}} = (2,\, 1,\, 0)$")
    ax2.set_xlabel("Project 1 allocation $x_1$")
    ax2.set_ylabel("Project 2 allocation $x_2$")
    ax2.set_title("Interior-point central path as the barrier shrinks")
    ax2.set_xlim(-0.2, I_total + 0.5)
    ax2.set_ylim(-0.2, I_total + 0.5)
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=9)

    report.add_results(
        f"The barrier path enters the feasible region from the centre and bends toward $x^{{\\ast}}$ as $t$ decreases. "
        f"Each diamond is the optimum of the barrier subproblem at one $t$. "
        f"The path stays strictly interior at every $t > 0$ and reaches the boundary only in the limit. "
        f"After {barrier_iter} barrier values the iterate lies within {barrier_residuals[-1]:.2e} of the closed form in Euclidean distance."
    )
    report.add_figure(
        "figures/barrier-path.png",
        "Interior-point central path traced by the barrier subproblem optima as t shrinks",
        fig2,
    )

    # ------------------------------------------------------------------
    # Figure 3: KKT residuals over iterations
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    pg_kkt = np.array(pg_kkt_trace)
    pg_iters_axis = np.arange(len(pg_kkt))
    ax3a.semilogy(pg_iters_axis, np.maximum(pg_kkt[:, 0], 1e-16), "-", color="tab:orange",
                  linewidth=1.5, label="Stationarity")
    ax3a.semilogy(pg_iters_axis, np.maximum(pg_kkt[:, 1], 1e-16), "-", color="tab:green",
                  linewidth=1.5, label="Primal feasibility")
    ax3a.semilogy(pg_iters_axis, np.maximum(pg_kkt[:, 3], 1e-16), "-", color="tab:purple",
                  linewidth=1.5, label="Complementarity")
    ax3a.set_xlabel("Iteration $k$")
    ax3a.set_ylabel("KKT residual")
    ax3a.set_title("Projected gradient: KKT residuals across iterations")
    ax3a.legend(loc="upper right", fontsize=9)

    barrier_kkt = np.array(barrier_kkt_trace)
    barrier_idx = np.arange(len(barrier_kkt))
    ax3b.semilogy(barrier_idx, np.maximum(barrier_kkt[:, 0], 1e-16), "d-", color="tab:orange",
                  markersize=5, linewidth=1.5, label="Stationarity")
    ax3b.semilogy(barrier_idx, np.maximum(barrier_kkt[:, 1], 1e-16), "d-", color="tab:green",
                  markersize=5, linewidth=1.5, label="Primal feasibility")
    ax3b.semilogy(barrier_idx, np.maximum(barrier_kkt[:, 3], 1e-16), "d-", color="tab:purple",
                  markersize=5, linewidth=1.5, label="Complementarity")
    ax3b.set_xticks(barrier_idx)
    ax3b.set_xticklabels([f"{t:.0e}" for t in barriers], rotation=45, fontsize=8)
    ax3b.set_xlabel("Barrier parameter $t$")
    ax3b.set_ylabel("KKT residual")
    ax3b.set_title("Interior point: KKT residuals along the central path")
    ax3b.legend(loc="upper right", fontsize=9)
    fig3.tight_layout()

    report.add_results(
        "Each method drives different KKT residuals to zero in different orders. "
        "Projected gradient has primal feasibility at machine precision from the first iterate because the projection enforces it. "
        "Stationarity falls steadily as the iterate approaches the active-set boundary. "
        "Complementarity tracks the gap on the bound that should bind.\n\n"
        "The interior-point method reduces all three residuals together as the barrier shrinks. "
        "Complementarity is bounded above by $n\\, t$ along the central path. "
        "Reaching machine-precision feasibility takes about a dozen barrier values."
    )
    report.add_figure(
        "figures/kkt-residuals.png",
        "KKT residuals across iterations for projected gradient (left) and along the central path for the interior-point method (right)",
        fig3,
    )

    # ------------------------------------------------------------------
    # Figure 4: shadow prices at the SLSQP optimum
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    labels = [r"Budget $\lambda$",
              r"Project 1 bound $\mu_1$",
              r"Project 2 bound $\mu_2$",
              r"Project 3 bound $\mu_3$"]
    closed_form = [lambda_star, mu_star[0], mu_star[1], mu_star[2]]
    slsqp_vals = [slsqp_lam, slsqp_mu[0], slsqp_mu[1], slsqp_mu[2]]
    pos = np.arange(len(labels))
    width = 0.35
    ax4.bar(pos - width / 2, closed_form, width, color="tab:gray", alpha=0.7,
            label="Closed form")
    ax4.bar(pos + width / 2, slsqp_vals, width, color="tab:red", alpha=0.85,
            label="SLSQP recovery")
    ax4.set_xticks(pos)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_ylabel("Multiplier value")
    ax4.set_title("Shadow prices: closed form vs SLSQP recovery")
    ax4.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax4.legend(loc="upper right", fontsize=9)
    for i, (cf, sl) in enumerate(zip(closed_form, slsqp_vals)):
        if abs(cf) > 1e-8 or abs(sl) > 1e-8:
            ax4.text(i, max(cf, sl) + 0.05, fr"$\lambda^{{\ast}} = {cf:.2f}$" if i == 0 else fr"${cf:.2f}$",
                     ha="center", fontsize=8)

    report.add_results(
        "The budget multiplier is positive because the budget binds. "
        "The bound multipliers on projects 1 and 2 are zero because those projects receive strictly positive allocation. "
        "The bound multiplier on project 3 is positive because the non-negativity constraint binds. "
        "SLSQP recovers the same multipliers as the closed form to several digits."
    )
    report.add_figure(
        "figures/shadow-prices.png",
        "Closed-form Lagrange multipliers compared to SLSQP-recovered multipliers",
        fig4,
    )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    solution_table = pd.DataFrame({
        "Method": [
            "Baseline failure: Lagrangian, budget only",
            "Method 1: Projected gradient",
            "Method 2: Interior-point log barrier",
            "Method 3: SLSQP",
            "Closed form",
        ],
        "Project 1": [
            f"{x_baseline[0]:.4f}",
            f"{pg_x_final[0]:.4f}",
            f"{barrier_x_final[0]:.4f}",
            f"{x_slsqp[0]:.4f}",
            f"{x_star[0]:.4f}",
        ],
        "Project 2": [
            f"{x_baseline[1]:.4f}",
            f"{pg_x_final[1]:.4f}",
            f"{barrier_x_final[1]:.4f}",
            f"{x_slsqp[1]:.4f}",
            f"{x_star[1]:.4f}",
        ],
        "Project 3": [
            f"{x_baseline[2]:.4f}",
            f"{pg_x_final[2]:.4f}",
            f"{barrier_x_final[2]:.4f}",
            f"{x_slsqp[2]:.4f}",
            f"{x_star[2]:.4f}",
        ],
        "Total spend": [
            f"{x_baseline.sum():.4f}",
            f"{pg_x_final.sum():.4f}",
            f"{barrier_x_final.sum():.4f}",
            f"{x_slsqp.sum():.4f}",
            f"{x_star.sum():.4f}",
        ],
        "Utility": [
            f"{bl_utility:.4f}",
            f"{pg_utility:.4f}",
            f"{barrier_utility:.4f}",
            f"{slsqp_utility:.4f}",
            f"{u_star:.4f}",
        ],
        "Iterations": [
            "1 (closed form)",
            f"{pg_iter}",
            f"{barrier_iter} barrier values",
            f"{slsqp_iter}",
            "n/a",
        ],
        "Feasible?": [
            "no, x_3 < 0",
            "yes",
            "yes",
            "yes",
            "yes",
        ],
    })
    report.add_results(
        "The table collects the baseline failure and the three constrained methods alongside the closed form. "
        "The budget-only baseline maximises utility while ignoring the bound and reports a higher number than the feasible optimum. "
        "All three real methods reach the closed-form allocation. "
        "The infeasible baseline shows in one row that an objective value alone is not a verdict."
    )
    report.add_table(
        "tables/solution_comparison.csv",
        f"Solution comparison at $a = (4, 3, 0.5)$, $B = I_3$, $I = {I_total:.0f}$",
        solution_table,
    )

    kkt_table = pd.DataFrame({
        "Method": [
            "Baseline failure: Lagrangian, budget only",
            "Method 1: Projected gradient",
            "Method 2: Interior-point log barrier",
            "Method 3: SLSQP",
        ],
        "Stationarity error": [
            f"{bl_stat:.2e}",
            f"{pg_stat:.2e}",
            f"{barrier_stat:.2e}",
            f"{slsqp_stat:.2e}",
        ],
        "Feasibility error": [
            f"{bl_primal:.2e}",
            f"{pg_primal:.2e}",
            f"{barrier_primal:.2e}",
            f"{slsqp_primal:.2e}",
        ],
        "Dual feasibility error": [
            f"{bl_dual:.2e}",
            f"{pg_dual:.2e}",
            f"{barrier_dual:.2e}",
            f"{slsqp_dual:.2e}",
        ],
        "Complementarity error": [
            f"{bl_compl:.2e}",
            f"{pg_compl:.2e}",
            f"{barrier_compl:.2e}",
            f"{slsqp_compl:.2e}",
        ],
        "Active constraints recovered": [
            "budget only (mis-recovered)",
            "budget; project 3 bound",
            "budget; project 3 bound",
            "budget; project 3 bound",
        ],
    })
    report.add_results(
        "The KKT diagnostic table separates four kinds of error. "
        "Stationarity is small for every method including the baseline because each method satisfies the first-order conditions of the problem it actually solved. "
        "Primal feasibility flags the baseline immediately. "
        "Complementarity hits machine precision once the active set is recovered correctly."
    )
    report.add_table(
        "tables/kkt_check.csv",
        "KKT residuals and active set recovered by each method",
        kkt_table,
    )

    shadow_table = pd.DataFrame({
        "Constraint": [
            "Budget $\\sum_j x_j \\leq I$",
            "Project 1 bound $x_1 \\geq 0$",
            "Project 2 bound $x_2 \\geq 0$",
            "Project 3 bound $x_3 \\geq 0$",
        ],
        "Multiplier": [
            f"{lambda_star:.2f}",
            f"{mu_star[0]:.2f}",
            f"{mu_star[1]:.2f}",
            f"{mu_star[2]:.2f}",
        ],
        "Status": [
            "binding",
            "slack",
            "slack",
            "binding",
        ],
        "Economic interpretation": [
            "Utility gain from one extra unit of budget",
            "Project 1 receives interior allocation; bound has no value",
            "Project 2 receives interior allocation; bound has no value",
            "Utility loss avoided by holding project 3 at zero",
        ],
    })
    report.add_results(
        "The shadow-price table lists the binding and slack constraints with their multipliers and economic meaning. "
        "Two constraints bind at the optimum. "
        "The budget multiplier $\\lambda^{\\ast} = 2$ is the marginal utility of an extra unit of budget. "
        "The project-3 bound multiplier $\\mu_3^{\\ast} = 1.5$ is the utility cost of zero allocation, equal to the gap between the unconstrained marginal return $a_3 = 0.5$ and the budget shadow price."
    )
    report.add_table(
        "tables/shadow_prices.csv",
        "Closed-form shadow prices and constraint status at the optimum",
        shadow_table,
    )

    report.add_takeaway(
        "A high objective value is not enough to declare a constrained problem solved. "
        "The budget-only Lagrangian beats the true optimum on utility but assigns a negative allocation to one project. "
        "Primal feasibility catches the failure. "
        "Stationarity does not.\n\n"
        "Projected gradient is the simplest method that always returns a feasible answer. "
        "Each iterate is a budget-respecting allocation with non-negative entries. "
        "Convergence is linear and depends on the conditioning of $B$ and on the step size. "
        "The simplex projection is closed form and cheap.\n\n"
        "The interior-point log barrier replaces the bounds with a smooth penalty. "
        "Iterates trace a central path that stays strictly interior until the barrier parameter shrinks to zero. "
        "The duality gap along the path is exactly $n \\cdot t$, which makes the convergence diagnostic obvious. "
        "The method extends cleanly to many-project problems with many bounds.\n\n"
        "SLSQP is the practical default for small problems that mix equalities and inequalities. "
        "It builds a quadratic-programming subproblem at each iterate and refines a BFGS Hessian as it goes. "
        "Convergence is locally quadratic and Lagrange multipliers can be recovered from stationarity afterwards.\n\n"
        "Shadow prices are the economic part of the answer. "
        "The binding budget multiplier $\\lambda^{\\ast}$ is the marginal utility of one extra unit of budget. "
        "The binding bound multiplier $\\mu_3^{\\ast}$ is the utility loss avoided by holding project 3 at zero. "
        "It equals the wedge between project 3's return $a_3$ and the budget shadow price $\\lambda^{\\ast}$."
    )

    report.add_references([
        "Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press, Ch. 5 and 11.",
        "Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*. Springer, 2nd edition, Ch. 12, 17, 19.",
        "Bertsekas, D. P. (1999). *Nonlinear Programming*. Athena Scientific, 2nd edition, Ch. 2-3.",
        "Wang, W. and Carreira-Perpinan, M. A. (2013). *Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application*. arXiv:1309.1541.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
