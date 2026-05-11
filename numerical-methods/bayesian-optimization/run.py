#!/usr/bin/env python3
"""Bayesian optimization with a Gaussian-process surrogate on a nonconcave pricing objective.

A monopolist sells to two consumer segments. The mixture profit function has
two local maxima. Earlier we showed that multi-start L-BFGS-B, random search,
and simulated annealing all recover the global peak but spend hundreds to
thousands of function evaluations doing so. Bayesian optimization instead
fits a probabilistic surrogate to the evaluations it already has, then
chooses each next evaluation to maximize an acquisition function that
balances exploration and exploitation. Thirty evaluations are enough.

The surrogate is a Gaussian process with a squared-exponential kernel.
The acquisition rule is Expected Improvement. The comparison is run on the
same objective and bracket as `numerical-methods/global-search-multistart/`
so the per-evaluation budget is directly comparable.

References:
- Mockus, J., Tiesis, V., and Zilinskas, A. (1978). The application of Bayesian methods for seeking the extremum.
- Jones, D. R., Schonlau, M., and Welch, W. J. (1998). Efficient Global Optimization of Expensive Black-Box Functions.
- Snoek, J., Larochelle, H., and Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms.
- Frazier, P. I. (2018). A Tutorial on Bayesian Optimization.
- Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian Processes for Machine Learning, Ch. 2 and 5.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from scipy.optimize import minimize, dual_annealing

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


# =============================================================================
# Gaussian process surrogate with squared-exponential kernel
# =============================================================================
def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float, sigma_f: float) -> np.ndarray:
    """Squared-exponential kernel for 1-D inputs."""
    x1 = np.atleast_1d(x1).astype(float)
    x2 = np.atleast_1d(x2).astype(float)
    diff = x1[:, None] - x2[None, :]
    return (sigma_f ** 2) * np.exp(-0.5 * (diff / length_scale) ** 2)


class GaussianProcess:
    """Gaussian-process regressor with a squared-exponential kernel.

    The mean function is the training-target mean. The kernel signal variance
    and length scale are user-supplied; the noise variance is small to
    represent the (essentially deterministic) profit evaluations.
    """

    def __init__(self, length_scale: float = 1.0, sigma_f: float = 2.0, sigma_n: float = 1e-3):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.X = None
        self.y = None
        self.y_mean = 0.0
        self._cho = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.X = np.atleast_1d(X).astype(float)
        self.y = np.atleast_1d(y).astype(float)
        self.y_mean = float(self.y.mean())
        y_centered = self.y - self.y_mean
        K = rbf_kernel(self.X, self.X, self.length_scale, self.sigma_f)
        K += (self.sigma_n ** 2) * np.eye(len(self.X))
        self._cho = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._cho, y_centered)
        return self

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_star = np.atleast_1d(X_star).astype(float)
        K_s = rbf_kernel(self.X, X_star, self.length_scale, self.sigma_f)
        mu = self.y_mean + K_s.T @ self._alpha
        v = cho_solve(self._cho, K_s)
        var = (self.sigma_f ** 2) - np.sum(K_s * v, axis=0)
        var = np.clip(var, 1e-10, None)
        return mu, np.sqrt(var)

    def log_marginal_likelihood(self) -> float:
        L = self._cho[0]
        y_centered = self.y - self.y_mean
        n = len(self.y)
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        return float(-0.5 * y_centered @ self._alpha - 0.5 * log_det_K - 0.5 * n * np.log(2.0 * np.pi))


def fit_length_scale(X: np.ndarray, y: np.ndarray, sigma_f: float, sigma_n: float,
                     length_grid: np.ndarray) -> float:
    """Pick the length scale that maximizes the GP log marginal likelihood."""
    best_ll, best_ell = -np.inf, float(length_grid[0])
    for ell in length_grid:
        gp = GaussianProcess(length_scale=float(ell), sigma_f=sigma_f, sigma_n=sigma_n).fit(X, y)
        ll = gp.log_marginal_likelihood()
        if ll > best_ll:
            best_ll, best_ell = ll, float(ell)
    return best_ell


# =============================================================================
# Acquisition functions
# =============================================================================
def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.0) -> np.ndarray:
    """Expected Improvement for maximization.

    EI(x) = (mu - f_best - xi) * Phi(z) + sigma * phi(z),
    z = (mu - f_best - xi) / sigma when sigma > 0, and zero otherwise.
    """
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - f_best - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    # -------------------------------------------------------------------------
    # Calibration: same two-segment monopoly as global-search-multistart
    # -------------------------------------------------------------------------
    A_L, b_L = 10.0, 5.0
    A_H, b_H = 8.0, 1.0
    c = 0.5
    lam = 0.6

    p_low_peak = 10.9 / 6.8
    profit_low_peak = (p_low_peak - c) * (lam * (A_L - b_L * p_low_peak) + (1 - lam) * (A_H - b_H * p_low_peak))
    p_high_peak = 4.25
    profit_high_peak = (p_high_peak - c) * (1 - lam) * (A_H - b_H * p_high_peak)
    p_kink = A_L / b_L

    p_lo, p_hi = c + 1e-3, 8.0
    p_global, profit_global = (p_high_peak, profit_high_peak) if profit_high_peak >= profit_low_peak else (p_low_peak, profit_low_peak)

    def D_L(p):
        return np.maximum(0.0, A_L - b_L * p)

    def D_H(p):
        return np.maximum(0.0, A_H - b_H * p)

    def profit(p):
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = (p_arr - c) * (lam * D_L(p_arr) + (1 - lam) * D_H(p_arr))
        return float(out.item()) if out.size == 1 else out

    def neg_profit_scalar(p):
        if np.ndim(p) > 0:
            return -float(profit(float(p[0])))
        return -float(profit(float(p)))

    # -------------------------------------------------------------------------
    # Bayesian-optimization configuration
    # -------------------------------------------------------------------------
    n_initial = 5
    n_iter_bo = 25
    n_total = n_initial + n_iter_bo
    seed_bo = 0
    sigma_f = 2.0
    sigma_n = 1e-3
    length_grid = np.linspace(0.3, 2.5, 12)
    acq_grid = np.linspace(p_lo, p_hi, 1001)
    xi_ei = 0.0

    # -------------------------------------------------------------------------
    # BO with Expected Improvement (main run, snapshots saved for plotting)
    # -------------------------------------------------------------------------
    snapshot_iters = [n_initial, 10, 20, n_total]

    def take_snapshot(X_list, y_list):
        X_arr = np.array(X_list)
        y_arr = np.array(y_list)
        f_best = float(np.max(y_arr))
        ell = fit_length_scale(X_arr, y_arr, sigma_f, sigma_n, length_grid)
        gp = GaussianProcess(length_scale=ell, sigma_f=sigma_f, sigma_n=sigma_n).fit(X_arr, y_arr)
        mu, sd = gp.predict(acq_grid)
        ei = expected_improvement(mu, sd, f_best, xi=xi_ei)
        return {
            "X": X_arr.copy(),
            "y": y_arr.copy(),
            "mu": mu.copy(),
            "sd": sd.copy(),
            "acq": ei.copy(),
            "ell": ell,
            "next_x": float(acq_grid[int(np.argmax(ei))]),
        }

    rng = np.random.default_rng(seed_bo)
    X_ei = list(rng.uniform(p_lo, p_hi, n_initial))
    y_ei = [float(profit(x)) for x in X_ei]
    bo_log = []
    snapshots = {}

    # Log the initial design.
    for i in range(n_initial):
        bo_log.append({
            "iteration": i + 1,
            "phase": "initial",
            "x": X_ei[i],
            "f": y_ei[i],
            "best_so_far": float(np.max(y_ei[: i + 1])),
        })
    if n_initial in snapshot_iters:
        snapshots[n_initial] = take_snapshot(X_ei, y_ei)

    # EI-guided iterations.
    for t in range(n_initial + 1, n_total + 1):
        X_arr = np.array(X_ei)
        y_arr = np.array(y_ei)
        f_best = float(np.max(y_arr))
        ell = fit_length_scale(X_arr, y_arr, sigma_f, sigma_n, length_grid)
        gp = GaussianProcess(length_scale=ell, sigma_f=sigma_f, sigma_n=sigma_n).fit(X_arr, y_arr)
        mu, sd = gp.predict(acq_grid)
        ei = expected_improvement(mu, sd, f_best, xi=xi_ei)
        x_next = float(acq_grid[int(np.argmax(ei))])
        y_next = float(profit(x_next))
        X_ei.append(x_next)
        y_ei.append(y_next)
        bo_log.append({
            "iteration": t,
            "phase": "EI-guided",
            "x": x_next,
            "f": y_next,
            "best_so_far": float(np.max(y_ei)),
        })
        if t in snapshot_iters:
            snapshots[t] = take_snapshot(X_ei, y_ei)

    bo_log_df = pd.DataFrame(bo_log)
    best_bo_ei = float(np.max(y_ei))
    p_bo_ei = float(X_ei[int(np.argmax(y_ei))])
    eval_to_global_ei = next((row["iteration"] for row in bo_log if row["best_so_far"] >= profit_global - 1e-3), None)

    # -------------------------------------------------------------------------
    # Comparison baselines on the same objective
    # -------------------------------------------------------------------------
    def best_so_far(values: list[float]) -> np.ndarray:
        return np.maximum.accumulate(np.asarray(values, dtype=float))

    # Random search
    n_random = 500
    rng_rs = np.random.default_rng(seed_bo + 1)
    rs_draws = rng_rs.uniform(p_lo, p_hi, n_random)
    rs_values = profit(rs_draws)
    rs_curve = best_so_far(list(rs_values))
    eval_to_global_rs = int(np.argmax(rs_curve >= profit_global - 1e-3)) + 1 if np.any(rs_curve >= profit_global - 1e-3) else None
    p_rs = float(rs_draws[int(np.argmax(rs_values))])
    profit_rs = float(np.max(rs_values))

    # Multi-start L-BFGS-B with calls instrumented
    multi_calls: list[float] = []

    def profit_logged(p):
        x = float(np.atleast_1d(p)[0])
        v = float(profit(x))
        multi_calls.append(v)
        return -v

    rng_ms = np.random.default_rng(seed_bo + 2)
    n_starts = 50
    starts = rng_ms.uniform(p_lo, p_hi, n_starts)
    best_multi = -np.inf
    p_multi = float("nan")
    for p0 in starts:
        res = minimize(profit_logged, x0=np.array([p0]), method="L-BFGS-B", bounds=[(p_lo, p_hi)])
        pf = float(res.x[0])
        v = float(profit(pf))
        if v > best_multi:
            best_multi = v
            p_multi = pf
    multi_curve = best_so_far(multi_calls)
    eval_to_global_ms = int(np.argmax(multi_curve >= profit_global - 1e-3)) + 1 if np.any(multi_curve >= profit_global - 1e-3) else None

    # Simulated annealing
    sa_calls: list[float] = []

    def profit_logged_sa(p):
        v = float(profit(float(p[0])))
        sa_calls.append(v)
        return -v

    res_sa = dual_annealing(profit_logged_sa, bounds=[(p_lo, p_hi)], seed=seed_bo + 3, maxiter=500)
    sa_curve = best_so_far(sa_calls)
    p_sa = float(res_sa.x[0])
    profit_sa = float(profit(p_sa))
    eval_to_global_sa = int(np.argmax(sa_curve >= profit_global - 1e-3)) + 1 if np.any(sa_curve >= profit_global - 1e-3) else None

    bo_curve = best_so_far(y_ei)

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    setup_style()
    report = ModelReport(
        "Bayesian Optimization with a Gaussian-Process Surrogate",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Many structural objectives are expensive. "
        "A particle-filtered likelihood, a simulated-method-of-moments criterion, or a nested fixed-point estimator can take seconds to minutes per evaluation. "
        "Multi-start L-BFGS-B and simulated annealing recover the global optimum on such surfaces, "
        "but they pay hundreds to thousands of evaluations to do so.\n\n"
        "Bayesian optimization is a sample-efficient alternative for that regime. "
        "It places a probabilistic prior on the unknown objective and updates that prior to a posterior conditional on the evaluations collected so far. "
        "It then chooses the next evaluation by maximizing an acquisition function that trades off exploration of uncertain regions against exploitation of high-mean regions. "
        "On problems with a few tens of dimensions and expensive black-box evaluations, this loop typically finds the global optimum in tens of evaluations rather than thousands.\n\n"
        "Bayesian optimization is the gradient-free member of a family. "
        "When the objective is differentiable and the goal is to sample its posterior rather than maximize it, Hamiltonian Monte Carlo in [`computational-methods/hamiltonian-monte-carlo/`](../../computational-methods/hamiltonian-monte-carlo/) is the gradient-based analogue. "
        "Both methods share the same motivation, sample efficiency under expensive evaluations, and the same probabilistic framing, but they apply at opposite ends of the gradient-availability spectrum.\n\n"
        "The objective here is the same two-segment monopoly profit used in [`numerical-methods/global-search-multistart/`](../../numerical-methods/global-search-multistart/). "
        "It is cheap to evaluate, which makes it a poor production target for Bayesian optimization. "
        "It is a good teaching target. "
        "The two local peaks are well separated, the global is known analytically, and the head-to-head budget is directly comparable to multi-start, random search, and simulated annealing on the same problem."
    )

    report.add_equations(
        r"""A monopolist faces a population of consumers split between two segments.
Segment $L$ has linear demand with intercept $A_L > 0$ and slope $b_L > 0$.
Segment $H$ has linear demand with intercept $A_H > 0$ and slope $b_H > 0$.

$$D_L(p) = \max\lbrace 0,\, A_L - b_L\, p \rbrace,
\qquad
D_H(p) = \max\lbrace 0,\, A_H - b_H\, p \rbrace.$$

With low-segment share $\lambda \in (0, 1)$ and constant marginal cost $c \ge 0$, the mixture profit is

$$\pi(p) = (p - c) \left[\lambda\, D_L(p) + (1 - \lambda)\, D_H(p)\right].$$

The objective is piecewise quadratic in $p$ with a strict local maximum at the both-segments peak $p_L^{\ast}$ and a global maximum at the high-only peak $p_H^{\ast}$.
On the calibration used here, $p_L^{\ast} \approx 1.603$ with $\pi \approx 4.14$, and $p_H^{\ast} = 4.25$ with $\pi \approx 5.625$.

Bayesian optimization treats $\pi$ as an unknown function on a bracket $\mathcal{X} = [p_{\mathrm{lo}}, p_{\mathrm{hi}}]$.
It places a probabilistic prior on $\pi$, updates that prior to a posterior conditional on the evaluations collected so far, and selects the next evaluation by maximizing an acquisition function on the posterior.
This is the same Bayesian update that produces a Beta posterior from a Beta-Binomial conjugate model in [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/); here the prior is over an unknown function rather than a scalar probability, and conjugacy is replaced by the closed form for conditioning a joint Gaussian.

### Method 1: Gaussian-process surrogate

A Gaussian process $\mathcal{GP}(m, k)$ is a distribution over functions $f : \mathcal{X} \to \mathbb{R}$ such that for any finite set of inputs $X = (x_1, \ldots, x_n) \in \mathcal{X}^n$ the vector of function values $f(X) = (f(x_1), \ldots, f(x_n)) \in \mathbb{R}^n$ is jointly Gaussian with mean $m(X) = (m(x_1), \ldots, m(x_n))$ and covariance matrix $K(X, X) \in \mathbb{R}^{n \times n}$ with entries $K_{ij} = k(x_i, x_j)$.
The process is fully specified by its mean function $m : \mathcal{X} \to \mathbb{R}$ and its covariance kernel $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$.
We use a zero-mean prior, $f \sim \mathcal{GP}(0, k)$, with the squared-exponential kernel

$$k(x, x') = \sigma_f^2 \exp\left(-\tfrac{(x - x')^2}{2\, \ell^2}\right).$$

Here $\sigma_f > 0$ is the prior signal standard deviation and $\ell > 0$ is the length scale, which controls how quickly the kernel decays with distance.
A small $\ell$ gives a wiggly prior; a large $\ell$ gives a smooth prior.

Suppose we have observed evaluations $y_i = f(x_i) + \varepsilon_i$ for $i = 1, \ldots, n$, where the observation noise $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$ is independent and $\sigma_n > 0$ is the noise standard deviation.
Stack the targets into $y = (y_1, \ldots, y_n)^{\top} \in \mathbb{R}^n$.
Because the joint distribution of $(y, f(x_{\ast}))$ at any new input $x_{\ast} \in \mathcal{X}$ is Gaussian by construction, the conditional distribution $f(x_{\ast}) \mid (X, y)$ is also Gaussian, with closed-form posterior mean $\mu(x_{\ast})$ and variance $\sigma^2(x_{\ast})$:

$$\mu(x_{\ast}) = \underbrace{k(x_{\ast}, X)}_{\text{similarity to training inputs}} \underbrace{\left[K(X, X) + \sigma_n^2 I\right]^{-1} y}_{\text{noise-corrected training residual}},$$

$$\sigma^2(x_{\ast}) = \underbrace{k(x_{\ast}, x_{\ast})}_{\text{prior variance at } x_{\ast}} - \underbrace{k(x_{\ast}, X) \left[K(X, X) + \sigma_n^2 I\right]^{-1} k(X, x_{\ast})}_{\text{variance explained by the data}}.$$

The vector $k(x_{\ast}, X) \in \mathbb{R}^n$ collects the kernel values $(k(x_{\ast}, x_1), \ldots, k(x_{\ast}, x_n))$ and $I$ is the $n \times n$ identity matrix.
Read the posterior mean as a kernel-weighted regression: the row vector $k(x_{\ast}, X)$ gives the similarity of the candidate to each evaluated point, and the precision-weighted residual $[K + \sigma_n^2 I]^{-1} y$ tells the formula how to combine those similarities.
Read the posterior variance as "prior variance minus what the data already explain", which is the GP analogue of the Bayesian shrinkage identity $\mathrm{Var}(\theta) = \mathrm{Var}(\mathbb{E}[\theta \mid D]) + \mathbb{E}[\mathrm{Var}(\theta \mid D)]$.
The subtracted term cannot exceed the prior, so the posterior variance is always nonnegative and shrinks toward zero as the candidate moves close to an evaluated point.
The variance collapsing at evaluated points is what makes Expected Improvement avoid re-querying the same input, and it is the reason posterior variance is the right signal for "where would another evaluation be informative".

### Method 2: Expected Improvement acquisition

Let $f^{\ast} = \max_{i \le n} y_i$ denote the best observed value so far.
Expected Improvement scores a candidate $x \in \mathcal{X}$ by the expected positive gain over $f^{\ast}$, with expectation taken under the GP posterior at $x$:

$$\mathrm{EI}(x) = \mathbb{E}\left[\max\lbrace f(x) - f^{\ast} - \xi,\, 0 \rbrace \mid X, y \right].$$

The parameter $\xi \ge 0$ is an exploration tilt, in units of the objective: it requires a posterior improvement of at least $\xi$ before contributing to the score.
Since $f(x) \mid X, y \sim \mathcal{N}(\mu(x), \sigma^2(x))$, the expectation is a truncated-Gaussian integral with the closed form

$$\mathrm{EI}(x) = \underbrace{(\mu(x) - f^{\ast} - \xi)\, \Phi(z)}_{\text{exploitation: bet on posterior mean}} + \underbrace{\sigma(x)\, \phi(z)}_{\text{exploration: bet on posterior spread}},
\qquad
z = \frac{\mu(x) - f^{\ast} - \xi}{\sigma(x)},$$

valid whenever $\sigma(x) > 0$.
Here $\Phi$ and $\phi$ denote the cumulative distribution function and probability density function of the standard normal distribution $\mathcal{N}(0, 1)$.

**Worked example.** Suppose after a handful of evaluations the GP at a candidate $x$ has posterior mean $\mu(x) = 5.2$ and standard deviation $\sigma(x) = 0.5$, and the best observation so far is $f^{\ast} = 4.5$. With $\xi = 0$, the standardized improvement is $z = (5.2 - 4.5)/0.5 = 1.4$. The closed form gives $\mathrm{EI}(x) = 0.7 \cdot \Phi(1.4) + 0.5 \cdot \phi(1.4) \approx 0.7 \cdot 0.919 + 0.5 \cdot 0.150 \approx 0.72$. The exploitation term dominates because the posterior mean already sits well above $f^{\ast}$; the candidate is mostly an exploit pick.
The split into exploitation plus exploration is why Expected Improvement works without a hand-tuned trade-off.
The first term is large where the posterior mean already exceeds the best observation, so it pulls the search toward known promising regions.
The second term is large where the posterior standard deviation is high, which only happens away from evaluated points, so it pulls the search toward unexplored regions.
Expected Improvement vanishes at evaluated points because $\sigma(x_i) = 0$ there, so the loop never re-evaluates the same input.
The Bayesian-optimization loop alternates between fitting the GP and maximizing $\mathrm{EI}$ to pick the next evaluation, repeating until the evaluation budget is exhausted.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $A_L$, $b_L$ | {A_L:.1f}, {b_L:.1f} | Low-valuation linear demand |\n"
        f"| $A_H$, $b_H$ | {A_H:.1f}, {b_H:.1f} | High-valuation linear demand |\n"
        f"| $c$ | {c:.1f} | Marginal cost |\n"
        f"| $\\lambda$ | {lam:.1f} | Share of low-valuation consumers |\n"
        f"| Search bracket | $[{p_lo:.3f},\\, {p_hi:.1f}]$ | Outer bounds for every method |\n"
        f"| Low peak | $p_L^{{\\ast}} = {p_low_peak:.4f}$, $\\pi = {profit_low_peak:.4f}$ | Local maximum |\n"
        f"| High peak | $p_H^{{\\ast}} = {p_high_peak:.4f}$, $\\pi = {profit_high_peak:.4f}$ | Global maximum |\n"
        f"| Initial design | {n_initial} uniform draws, seed {seed_bo} | Seed observations for the GP |\n"
        f"| BO iterations | {n_iter_bo} | Acquisition-driven evaluations |\n"
        f"| Total BO budget | {n_total} | Per acquisition rule |\n"
        f"| Kernel signal std $\\sigma_f$ | {sigma_f:.2f} | Squared-exponential kernel |\n"
        f"| Kernel noise std $\\sigma_n$ | {sigma_n:.0e} | Almost-deterministic profit |\n"
        f"| Length-scale grid | $[{length_grid.min():.2f},\\, {length_grid.max():.2f}]$, {len(length_grid)} points | Tuned by log marginal likelihood |\n"
        f"| EI exploration $\\xi$ | {xi_ei:.2f} | Posterior-improvement tilt |"
    )

    report.add_solution_method(
        "Bayesian optimization is a single loop. "
        "Fit a Gaussian-process surrogate to the evaluations collected so far, maximize an acquisition function on the surrogate to pick the next point, evaluate the true objective there, and repeat. "
        "The three components below define each part of the loop.\n\n"

        "### Method 1: Gaussian-process surrogate\n\n"
        "The Gaussian process places a prior on the unknown profit function. "
        "After $n$ evaluations $(X, y)$ the posterior at any candidate price $x_{\\ast}$ is Gaussian with closed-form mean and variance. "
        "The closed form requires one Cholesky factor of the $n \\times n$ kernel matrix, so the cost is $O(n^3)$ in evaluations and $O(n^2)$ per prediction. "
        "For budgets of tens to hundreds of evaluations this is negligible.\n\n"
        "```text\n"
        "Algorithm: GP posterior at candidates X_star\n"
        "Input : training inputs X, targets y, kernel k, noise sigma_n\n"
        "Output: posterior mean mu(X_star), posterior std sigma(X_star)\n"
        "  K   = k(X, X) + sigma_n^2 * I\n"
        "  L   = cholesky(K)\n"
        "  a   = solve(L^T, solve(L, y - mean(y)))\n"
        "  k_s = k(X, X_star)\n"
        "  mu        = mean(y) + k_s^T @ a\n"
        "  v         = solve(L, k_s)\n"
        "  variance  = k(X_star, X_star) - sum(v^2, axis=0)\n"
        "```\n\n"
        "The length scale $\\ell$ is the key hyperparameter. "
        "A small $\\ell$ produces a wiggly surrogate that fits each observation tightly but extrapolates poorly. "
        "A large $\\ell$ produces a smooth surrogate that may miss narrow basins. "
        "We refit $\\ell$ at each step by maximizing the log marginal likelihood over a coarse grid. "
        "This is the cleanest empirical-Bayes choice and avoids the optimizer-inside-optimizer problem of joint hyperparameter and acquisition maximization.\n\n"

        "### Method 2: Expected Improvement acquisition\n\n"
        "Expected Improvement is the canonical Bayesian-optimization acquisition. "
        "It is the expected value, under the GP posterior, of the gain over the best evaluation seen so far. "
        "It collapses to zero at evaluated points because their posterior variance is zero, which prevents the loop from re-evaluating the same input. "
        "It rewards both posterior mean and posterior standard deviation, so the trade-off between exploitation and exploration is automatic.\n\n"
        "```text\n"
        "Algorithm: One step of Bayesian optimization with Expected Improvement\n"
        "Input : evaluated inputs X, targets y, kernel hyperparameters, candidate grid X_star\n"
        "Output: next evaluation x_new\n"
        "  fit GP on (X, y)\n"
        "  predict (mu, sigma) on X_star\n"
        "  f_best = max(y)\n"
        "  z   = (mu - f_best - xi) / sigma\n"
        "  EI  = (mu - f_best - xi) * Phi(z) + sigma * phi(z)\n"
        "  x_new = X_star[argmax(EI)]\n"
        "```\n\n"
        "Expected Improvement fails when the posterior is badly miscalibrated. "
        "If the length scale is too large the surrogate underestimates the local curvature near a peak and Expected Improvement under-explores. "
        "If the length scale is too small the surrogate over-credits noise and Expected Improvement over-explores. "
        "The diagnostic is to plot the posterior mean and the one-sigma band against the true objective at the snapshot iterations, which we do below.\n\n"
        "The full Bayesian-optimization loop is the surrogate, the acquisition, and a small initial design glued together. "
        "The initial design is needed because the GP needs a few observations before its posterior is informative; "
        "five uniform draws is enough on this problem.\n\n"
        "```text\n"
        "Algorithm: Bayesian optimization with Expected Improvement\n"
        "Input : objective f, bounds, kernel, initial size n0, total budget T\n"
        "Output: argmax of evaluated points\n"
        "  draw n0 uniform points X = (x_1, ..., x_n0) from bounds, evaluate y = f(X)\n"
        "  for t = n0 + 1, ..., T:\n"
        "      refit GP hyperparameters by log marginal likelihood maximization\n"
        "      predict (mu, sigma) on a dense candidate grid\n"
        "      x_t = argmax of EI(x) on the grid\n"
        "      y_t = f(x_t)\n"
        "      X, y <- append (x_t, y_t)\n"
        "  return (x_k, y_k) with k = argmax of y\n"
        "```\n\n"
        "Bayesian optimization is not magic. "
        "It pays for sample efficiency with stronger assumptions on the objective and with model fitting in the inner loop. "
        "When evaluations are cheap, simulated annealing or multi-start L-BFGS-B is faster end to end. "
        "When evaluations are expensive, the inner-loop cost is dominated by a single objective call and Bayesian optimization wins by orders of magnitude."
    )

    # ------------------------------------------------------------------
    # Figure 1: profit surface with the two peaks
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    p_grid = np.linspace(p_lo, p_hi, 600)
    ax1.plot(p_grid, profit(p_grid), color="tab:blue", linewidth=2, label=r"$\pi(p)$")
    ax1.axvline(p_kink, color="tab:gray", linestyle=":", linewidth=1.0,
                label=fr"Low-segment exit $p_L^{{\max}} = {p_kink:.2f}$")
    ax1.plot(p_low_peak, profit_low_peak, "o", color="tab:orange", markersize=10,
             label=fr"Low peak $p_L^{{\ast}} = {p_low_peak:.3f}$, $\pi = {profit_low_peak:.3f}$")
    ax1.plot(p_high_peak, profit_high_peak, "*", color="tab:red", markersize=18,
             label=fr"High peak $p_H^{{\ast}} = {p_high_peak:.3f}$, $\pi = {profit_high_peak:.3f}$ (global)")
    ax1.set_xlabel("Price $p$")
    ax1.set_ylabel(r"Profit $\pi(p)$")
    ax1.set_title("Two-segment monopoly profit and its two local peaks")
    ax1.legend(loc="upper right", fontsize=9)
    report.add_results(
        f"The profit surface is reproduced from [`numerical-methods/global-search-multistart/`](../../numerical-methods/global-search-multistart/). "
        f"It has a local peak at $p_L^{{\\ast}} = {p_low_peak:.3f}$ with profit $\\pi = {profit_low_peak:.3f}$. "
        f"Above the kink at $p_L^{{\\max}} = {p_kink:.2f}$ only the high-valuation segment is active. "
        f"The high-only regime has its own peak at $p_H^{{\\ast}} = {p_high_peak:.2f}$ with profit $\\pi = {profit_high_peak:.3f}$, which is the global maximum on this calibration."
    )
    report.add_figure("figures/profit-surface.png",
                     "Two-segment monopoly profit with low-price and high-price peaks marked", fig1)

    # ------------------------------------------------------------------
    # Figure 2: BO iteration snapshots (GP posterior + EI)
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panel_order = sorted(snapshots.keys())
    f_true_grid = profit(acq_grid)
    for ax, k in zip(axes2.flat, panel_order):
        snap = snapshots[k]
        ax.plot(acq_grid, f_true_grid, color="tab:gray", linestyle="--", linewidth=1.2,
                label=r"True $\pi(p)$")
        ax.plot(acq_grid, snap["mu"], color="tab:blue", linewidth=1.8, label="GP posterior mean")
        ax.fill_between(acq_grid, snap["mu"] - 2 * snap["sd"], snap["mu"] + 2 * snap["sd"],
                        color="tab:blue", alpha=0.15, label="GP posterior $\\pm 2\\sigma$")
        ax.plot(snap["X"], snap["y"], "o", color="tab:orange", markersize=7, label="Evaluations")
        if snap["next_x"] is not None and k < n_total:
            ax.axvline(snap["next_x"], color="tab:green", linestyle="-", linewidth=1.0, alpha=0.7,
                       label=f"Next pick: $p = {snap['next_x']:.2f}$")
        ax.plot(p_high_peak, profit_high_peak, "*", color="tab:red", markersize=14,
                label="Global peak")
        ax.set_title(fr"After {k} evaluations ($\ell = {snap['ell']:.2f}$)")
        ax.set_ylabel(r"$\pi(p)$")
        ax.set_ylim(-1.5, 7.5)
        if k == panel_order[0]:
            ax.legend(loc="lower right", fontsize=8)
    for ax in axes2[-1, :]:
        ax.set_xlabel("Price $p$")
    fig2.suptitle("Gaussian-process posterior and next acquisition pick across BO iterations", y=1.00)
    fig2.tight_layout()
    report.add_results(
        f"The four panels show the Gaussian-process posterior at {panel_order[0]}, {panel_order[1]}, {panel_order[2]}, and {panel_order[3]} evaluations. "
        f"With {n_initial} uniform draws the posterior mean is flat between observations and the uncertainty band is wide. "
        f"Expected Improvement immediately probes regions of high mean and high variance, "
        f"which on this surface means evaluating points near the high-price peak. "
        f"By iteration 20 the posterior mean tracks the true profit closely in both basins, "
        f"and by iteration {n_total} Expected Improvement has localized around $p_H^{{\\ast}} = {p_high_peak:.2f}$ with very small posterior variance."
    )
    report.add_figure("figures/bo-iterations.png",
                     "GP posterior, evaluated points, and EI-chosen next pick at four iteration snapshots", fig2)

    # ------------------------------------------------------------------
    # Figure 3: convergence comparison vs other global methods
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    ax3.plot(np.arange(1, len(bo_curve) + 1), bo_curve, "o-", color="tab:purple",
             linewidth=2, markersize=4, label=f"Bayesian optimization (EI), best at eval {eval_to_global_ei}")
    ax3.plot(np.arange(1, len(multi_curve) + 1), multi_curve, color="tab:blue",
             linewidth=1.5, alpha=0.8,
             label=f"Multi-start L-BFGS-B, global at eval {eval_to_global_ms}")
    ax3.plot(np.arange(1, len(rs_curve) + 1), rs_curve, color="tab:green",
             linewidth=1.5, alpha=0.8,
             label=f"Random search, global at eval {eval_to_global_rs}")
    ax3.plot(np.arange(1, len(sa_curve) + 1), sa_curve, color="tab:orange",
             linewidth=1.2, alpha=0.85,
             label=f"Simulated annealing, global at eval {eval_to_global_sa}")
    ax3.axhline(profit_global, color="tab:red", linestyle="--", linewidth=1.5,
                label=fr"Global $\pi^{{\ast}} = {profit_global:.3f}$")
    ax3.axhline(profit_low_peak, color="tab:gray", linestyle=":", linewidth=1.0,
                label=fr"Local-only $\pi = {profit_low_peak:.3f}$")
    ax3.set_xscale("log")
    ax3.set_xlabel("Number of objective evaluations (log scale)")
    ax3.set_ylabel("Best profit found so far")
    ax3.set_title("Best-so-far profit by evaluation count, BO versus three baselines")
    ax3.legend(loc="lower right", fontsize=8)
    fig3.tight_layout()
    report.add_results(
        f"The convergence plot is the head-to-head against the same three baselines as [`numerical-methods/global-search-multistart/`](../../numerical-methods/global-search-multistart/). "
        f"Bayesian optimization with Expected Improvement finds the global at evaluation {eval_to_global_ei} and converges sharply within its budget of {n_total} evaluations. "
        f"The baselines also recover the global on this seed, but they spend much larger budgets to do so. "
        f"Random search needs {eval_to_global_rs} draws before luck delivers an above-global point, and runs through all {n_random} draws because it has no stopping rule. "
        f"Multi-start L-BFGS-B happens to seed its first start in the high basin and converges there in {eval_to_global_ms} L-BFGS-B calls. "
        f"Over {n_starts} starts it still spends {len(multi_curve)} L-BFGS-B calls in total. "
        f"Simulated annealing also locates the global early in this run but burns roughly {len(sa_curve)} evaluations on its cooling schedule. "
        f"The right comparison is total budget, not first discovery: Bayesian optimization uses {n_total} evaluations end to end, multi-start uses {len(multi_curve)}, random search uses {n_random}, and simulated annealing uses about {len(sa_curve)}."
    )
    report.add_figure("figures/convergence-comparison.png",
                     "Best-so-far profit versus evaluation count for BO, multi-start, random search, and simulated annealing", fig3)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    method_table = pd.DataFrame({
        "Method": [
            "Bayesian optimization (EI)",
            "Multi-start L-BFGS-B",
            "Random search",
            "Simulated annealing",
        ],
        "Setting": [
            f"{n_initial} initial + {n_iter_bo} EI steps, seed {seed_bo}",
            f"{n_starts} starts, seed {seed_bo + 2}",
            f"{n_random} draws, seed {seed_bo + 1}",
            f"max iterations 500, seed {seed_bo + 3}",
        ],
        "Estimated optimum": [
            f"{p_bo_ei:.4f}",
            f"{p_multi:.4f}",
            f"{p_rs:.4f}",
            f"{p_sa:.4f}",
        ],
        "Profit": [
            f"{best_bo_ei:.4f}",
            f"{best_multi:.4f}",
            f"{profit_rs:.4f}",
            f"{profit_sa:.4f}",
        ],
        "Function evaluations": [
            f"{n_total}",
            f"{len(multi_curve)}",
            f"{n_random}",
            f"{len(sa_curve)}",
        ],
        "Evaluations to global": [
            f"{eval_to_global_ei}" if eval_to_global_ei else "not reached",
            f"{eval_to_global_ms}" if eval_to_global_ms else "not reached",
            f"{eval_to_global_rs}" if eval_to_global_rs else "not reached",
            f"{eval_to_global_sa}" if eval_to_global_sa else "not reached",
        ],
    })
    report.add_results(
        "The comparison table is normalized on the same objective and bracket. "
        "All four methods recover the global peak. "
        "The Bayesian-optimization budget is two orders of magnitude smaller than simulated annealing and one order smaller than random search or multi-start."
    )
    report.add_table(
        "tables/method_comparison.csv",
        f"Method comparison at $\\lambda = {lam}$, $c = {c}$, segment intercepts $({A_L:.0f}, {A_H:.0f})$",
        method_table,
    )

    bo_print = bo_log_df.copy()
    bo_print["x"] = bo_print["x"].map(lambda v: f"{v:.4f}")
    bo_print["f"] = bo_print["f"].map(lambda v: f"{v:.4f}")
    bo_print["best_so_far"] = bo_print["best_so_far"].map(lambda v: f"{v:.4f}")
    bo_print = bo_print.rename(columns={
        "iteration": "Iteration",
        "phase": "Phase",
        "x": "Price evaluated",
        "f": "Profit observed",
        "best_so_far": "Best profit so far",
    })
    report.add_results(
        f"The iteration log records every Bayesian-optimization evaluation with Expected Improvement. "
        f"The first {n_initial} rows are the initial uniform design. "
        f"The remaining {n_iter_bo} rows are EI-chosen evaluations. "
        f"The best-so-far column converges to the global peak well before the {n_total}-evaluation budget is exhausted."
    )
    report.add_table(
        "tables/bo_iteration_log.csv",
        "Per-iteration log of the EI-driven Bayesian-optimization run",
        bo_print,
    )

    report.add_takeaway(
        "Bayesian optimization is the right tool when evaluations are expensive. "
        "On the two-segment monopoly profit it recovers the global peak in roughly thirty evaluations, "
        "where simulated annealing needs over a thousand and random search several hundred. "
        "Sample efficiency is the entire pitch.\n\n"
        "Bayesian optimization is the wrong tool when evaluations are cheap. "
        "The Gaussian-process posterior costs $O(n^3)$ in evaluations because of the kernel-matrix Cholesky factor. "
        "On a problem where one evaluation takes milliseconds, multi-start L-BFGS-B or simulated annealing dominates Bayesian optimization on wall-clock time even though it uses far more evaluations.\n\n"
        "Bayesian optimization is fragile in high dimensions and on non-stationary surfaces. "
        "The squared-exponential kernel assumes a single length scale across the whole input space. "
        "Many structural objectives have one length scale near a flat plateau and a much shorter one near a sharp peak. "
        "Beyond about twenty dimensions the curse of dimensionality erodes the sample-efficiency gain, and the right tool is usually a structured surrogate or a trust-region method.\n\n"
        "The Bayesian framing of the surrogate matters. "
        "Each acquisition decision is a tractable inference on the posterior of the unknown profit. "
        "The same framing returns in [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/) for posterior sampling of structural parameters, "
        "and the natural use case for Bayesian optimization in structural work is the outer search over a small set of parameters whose likelihood is itself estimated by an expensive inner routine."
    )

    report.add_references([
        "Mockus, J., Tiesis, V., and Zilinskas, A. (1978). The application of Bayesian methods for seeking the extremum. In *Towards Global Optimization*, vol. 2, North-Holland, 117-129.",
        "Jones, D. R., Schonlau, M., and Welch, W. J. (1998). *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization, 13, 455-492.",
        "Snoek, J., Larochelle, H., and Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NIPS.",
        "Srinivas, N., Krause, A., Kakade, S., and Seeger, M. (2010). *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*. ICML.",
        "Frazier, P. I. (2018). *A Tutorial on Bayesian Optimization*. arXiv:1807.02811.",
        "Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press, Ch. 2 and 5.",
        "**See also.** The same two-segment monopoly profit is optimized by single-start and multi-start L-BFGS-B, random search, Nelder-Mead, and simulated annealing in [`numerical-methods/global-search-multistart/`](../../numerical-methods/global-search-multistart/). That tutorial documents the reporting discipline for global search; the present one documents a sample-efficient alternative for expensive evaluations. The Bayesian update behind the Gaussian-process posterior is the same one used in conjugate form in [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/), and the gradient-based sampling analogue for expensive *differentiable* posteriors is [`computational-methods/hamiltonian-monte-carlo/`](../../computational-methods/hamiltonian-monte-carlo/). Together the four tutorials cover the global-search, surrogate-optimization, posterior-sampling, and gradient-sampling corners of expensive-objective inference.",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
