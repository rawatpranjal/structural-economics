#!/usr/bin/env python3
"""Simulation-based estimation of a search acceptance rule."""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def simulate_search_panel(theta: np.ndarray, draws: dict[str, np.ndarray], choice_scale: float) -> dict[str, np.ndarray]:
    """Simulate wage offers and stochastic acceptance decisions."""
    mu, sigma, reservation = theta
    log_wage = mu + sigma * draws["z"]
    accept_prob = expit((log_wage - reservation) / choice_scale)
    accept = draws["u"] < accept_prob
    return {"log_wage": log_wage, "accept": accept, "accept_prob": accept_prob}


def economic_moments(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Moments used by the direct MSM estimator."""
    log_wage = sample["log_wage"]
    accept = sample["accept"]
    accepted_wages = log_wage[accept]
    return np.array(
        [
            accept.mean(),
            log_wage.mean(),
            log_wage.std(ddof=0),
            accepted_wages.mean(),
            accepted_wages.std(ddof=0),
        ]
    )


def auxiliary_statistics(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Auxiliary linear probability model coefficients plus wage distribution moments."""
    log_wage = sample["log_wage"]
    accept = sample["accept"].astype(float)
    X = np.column_stack([np.ones_like(log_wage), log_wage])
    intercept, slope = np.linalg.lstsq(X, accept, rcond=None)[0]
    accepted_wages = log_wage[sample["accept"]]
    return np.array(
        [
            intercept,
            slope,
            log_wage.mean(),
            log_wage.std(ddof=0),
            accept.mean(),
            accepted_wages.mean(),
        ]
    )


def criterion(
    theta: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
    statistic_fn,
) -> float:
    """Scaled quadratic simulation criterion."""
    if theta[0] < 2.4 or theta[0] > 3.6 or theta[1] < 0.2 or theta[1] > 0.8 or theta[2] < 2.5 or theta[2] > 3.8:
        return 1e6
    sample = simulate_search_panel(theta, draws, choice_scale)
    diff = (statistic_fn(sample) - target) / scale
    return float(diff @ diff)


def estimate_by_simulation(
    target: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
    statistic_fn,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate parameters by minimizing a simulation criterion."""
    scale = np.maximum(np.abs(target), 0.1)
    objective = lambda theta: criterion(theta, target, scale, draws, choice_scale, statistic_fn)
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 350, "xatol": 1e-4, "fatol": 1e-6, "disp": False},
    )
    theta_hat = np.asarray(result.x, dtype=float)
    sample_hat = simulate_search_panel(theta_hat, draws, choice_scale)
    simulated_stats = statistic_fn(sample_hat)
    residual = (simulated_stats - target) / scale
    return {
        "theta": theta_hat,
        "criterion": float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
        "simulated_stats": simulated_stats,
        "residual": residual,
    }


def make_draws(seed: int, n: int) -> dict[str, np.ndarray]:
    """Common random numbers for simulation objectives."""
    rng = np.random.default_rng(seed)
    return {"z": rng.normal(size=n), "u": rng.uniform(size=n)}


def criterion_grid(
    mu_grid: np.ndarray,
    reservation_grid: np.ndarray,
    sigma_fixed: float,
    target: np.ndarray,
    scale: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
    statistic_fn,
) -> np.ndarray:
    """Evaluate a two-dimensional criterion surface."""
    values = np.zeros((len(reservation_grid), len(mu_grid)))
    for i, reservation in enumerate(reservation_grid):
        for j, mu in enumerate(mu_grid):
            theta = np.array([mu, sigma_fixed, reservation])
            values[i, j] = criterion(theta, target, scale, draws, choice_scale, statistic_fn)
    return values


def sample_prior(prior_bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n particles uniformly from the rectangle defined by prior_bounds (d, 2)."""
    lo = prior_bounds[:, 0]
    hi = prior_bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, lo.size))


def in_bounds(theta: np.ndarray, prior_bounds: np.ndarray) -> bool:
    """Check that every coordinate of theta lies inside prior_bounds."""
    return bool(np.all(theta >= prior_bounds[:, 0]) and np.all(theta <= prior_bounds[:, 1]))


def perturb(
    parent: np.ndarray,
    cov: np.ndarray,
    prior_bounds: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> np.ndarray:
    """Gaussian random-walk perturbation; resample if the proposal exits the prior box."""
    for _ in range(max_tries):
        proposal = rng.multivariate_normal(parent, cov)
        if in_bounds(proposal, prior_bounds):
            return proposal
    return parent


def simulated_distance(
    theta: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
) -> float:
    """Scaled Euclidean distance between simulated and observed economic moments."""
    sample = simulate_search_panel(theta, draws, choice_scale)
    diff = (economic_moments(sample) - target) / scale
    return float(np.sqrt(diff @ diff))


def smc_weights(
    particles: np.ndarray,
    prev_particles: np.ndarray,
    prev_weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Importance weights for a uniform prior: w_i propto 1 / sum_j w_{t-1,j} K(theta_i | theta_{t-1,j})."""
    n, d = particles.shape
    inv_cov = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    log_norm = -0.5 * (logdet + d * np.log(2 * np.pi))
    log_prev = np.log(np.maximum(prev_weights, 1e-300))
    log_w = np.zeros(n)
    for i in range(n):
        diff = particles[i] - prev_particles
        quad = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        log_k = log_norm - 0.5 * quad
        terms = log_prev + log_k
        m = terms.max()
        log_w[i] = -(m + np.log(np.exp(terms - m).sum()))
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def posterior_summary(particles: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted posterior mean and 5/95 percentile credible interval per parameter."""
    mean = (weights[:, None] * particles).sum(axis=0)
    d = particles.shape[1]
    ci = np.zeros((d, 2))
    for j in range(d):
        idx = np.argsort(particles[:, j])
        sorted_vals = particles[idx, j]
        sorted_w = weights[idx]
        cum = np.cumsum(sorted_w) / sorted_w.sum()
        ci[j, 0] = sorted_vals[np.searchsorted(cum, 0.05)]
        ci[j, 1] = sorted_vals[np.searchsorted(cum, 0.95)]
    return mean, ci


def abc_smc(
    target: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
    prior_bounds: np.ndarray,
    n_particles: int = 1000,
    n_rounds: int = 6,
    alpha: float = 0.5,
    seed: int = 789,
) -> dict[str, object]:
    """Run ABC-SMC with adaptive tolerance schedule on the economic moments."""
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(target), 0.1)
    d = prior_bounds.shape[0]
    n_round0 = int(np.ceil(n_particles / alpha))

    particles_hist = np.zeros((n_rounds, n_particles, d))
    weights_hist = np.zeros((n_rounds, n_particles))
    distances_hist = np.zeros((n_rounds, n_particles))
    tolerances = np.zeros(n_rounds)
    accept_rates = np.zeros(n_rounds)
    ess = np.zeros(n_rounds)
    proposals = np.zeros(n_rounds, dtype=int)

    raw = sample_prior(prior_bounds, n_round0, rng)
    raw_d = np.array([simulated_distance(p, target, scale, draws, choice_scale) for p in raw])
    order = np.argsort(raw_d)[:n_particles]
    particles_hist[0] = raw[order]
    distances_hist[0] = raw_d[order]
    weights_hist[0] = 1.0 / n_particles
    tolerances[0] = float(raw_d[order].max())
    proposals[0] = n_round0
    accept_rates[0] = n_particles / n_round0
    ess[0] = n_particles

    for t in range(1, n_rounds):
        eps_t = float(np.quantile(distances_hist[t - 1], alpha))
        tolerances[t] = eps_t

        w_prev = weights_hist[t - 1]
        mean_prev = (w_prev[:, None] * particles_hist[t - 1]).sum(axis=0)
        centered = particles_hist[t - 1] - mean_prev
        cov_prev = (w_prev[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(axis=0)
        cov_t = 2.0 * cov_prev + 1e-10 * np.eye(d)

        cum_weights = np.cumsum(w_prev)
        cum_weights /= cum_weights[-1]

        new_particles = np.zeros((n_particles, d))
        new_distances = np.zeros(n_particles)
        n_proposals_t = 0

        for i in range(n_particles):
            while True:
                u = rng.uniform()
                parent_idx = int(np.searchsorted(cum_weights, u))
                parent = particles_hist[t - 1, parent_idx]
                proposal = perturb(parent, cov_t, prior_bounds, rng)
                n_proposals_t += 1
                d_proposal = simulated_distance(proposal, target, scale, draws, choice_scale)
                if d_proposal <= eps_t:
                    new_particles[i] = proposal
                    new_distances[i] = d_proposal
                    break

        new_weights = smc_weights(new_particles, particles_hist[t - 1], w_prev, cov_t)
        ess_t = 1.0 / float((new_weights ** 2).sum())
        if ess_t < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=new_weights)
            new_particles = new_particles[idx]
            new_distances = new_distances[idx]
            new_weights = np.ones(n_particles) / n_particles
            ess_t = float(n_particles)

        particles_hist[t] = new_particles
        weights_hist[t] = new_weights
        distances_hist[t] = new_distances
        proposals[t] = n_proposals_t
        accept_rates[t] = n_particles / n_proposals_t
        ess[t] = ess_t

    post_mean, post_ci = posterior_summary(particles_hist[-1], weights_hist[-1])
    criterion_at_mean = float(simulated_distance(post_mean, target, scale, draws, choice_scale) ** 2)
    return {
        "particles": particles_hist,
        "weights": weights_hist,
        "distances": distances_hist,
        "tolerances": tolerances,
        "accept_rates": accept_rates,
        "ess": ess,
        "proposals": proposals,
        "theta_post_mean": post_mean,
        "theta_post_ci": post_ci,
        "theta": post_mean,
        "criterion": criterion_at_mean,
        "final_tolerance": float(tolerances[-1]),
        "total_proposals": int(proposals.sum()),
    }


def parameter_table(
    theta_true: np.ndarray,
    msm: dict[str, object],
    ii: dict[str, object],
    abc: dict[str, object],
) -> pd.DataFrame:
    """Parameter recovery table including the ABC posterior mean and 5/95 credible interval."""
    abc_mean = np.asarray(abc["theta_post_mean"], dtype=float)
    abc_ci = np.asarray(abc["theta_post_ci"], dtype=float)
    return pd.DataFrame(
        {
            "Parameter": ["Offer mean mu", "Offer sd sigma", "Reservation log wage"],
            "True": theta_true,
            "MSM estimate": np.asarray(msm["theta"], dtype=float),
            "MSM error": np.asarray(msm["theta"], dtype=float) - theta_true,
            "Indirect inference estimate": np.asarray(ii["theta"], dtype=float),
            "Indirect inference error": np.asarray(ii["theta"], dtype=float) - theta_true,
            "ABC posterior mean": abc_mean,
            "ABC error": abc_mean - theta_true,
            "ABC 5%": abc_ci[:, 0],
            "ABC 95%": abc_ci[:, 1],
        }
    )


def method_comparison_table(
    msm: dict[str, object],
    ii: dict[str, object],
    abc: dict[str, object],
    times: dict[str, float],
) -> pd.DataFrame:
    """One-row-per-method comparison of estimates, loss, work, and wall time."""
    msm_theta = np.asarray(msm["theta"], dtype=float)
    ii_theta = np.asarray(ii["theta"], dtype=float)
    abc_theta = np.asarray(abc["theta_post_mean"], dtype=float)
    return pd.DataFrame(
        {
            "Method": ["MSM", "Indirect inference", "ABC-SMC"],
            "Offer mean mu": [msm_theta[0], ii_theta[0], abc_theta[0]],
            "Offer sd sigma": [msm_theta[1], ii_theta[1], abc_theta[1]],
            "Reservation log wage": [msm_theta[2], ii_theta[2], abc_theta[2]],
            "Criterion at point estimate or posterior mean": [
                float(msm["criterion"]),
                float(ii["criterion"]),
                float(abc["criterion"]),
            ],
            "Iterations or proposals": [
                int(msm["iterations"]),
                int(ii["iterations"]),
                int(abc["total_proposals"]),
            ],
            "Wall time (s)": [times["msm"], times["ii"], times["abc"]],
        }
    )


def residual_table(names: list[str], target: np.ndarray, msm: dict[str, object], ii: dict[str, object]) -> pd.DataFrame:
    """Stack target and simulated residual diagnostics."""
    rows = []
    for estimator, result in [("MSM", msm), ("Indirect inference", ii)]:
        simulated = np.asarray(result["simulated_stats"], dtype=float)
        residual = np.asarray(result["residual"], dtype=float)
        for name, target_value, simulated_value, scaled_error in zip(names, target, simulated, residual):
            rows.append(
                {
                    "Estimator": estimator,
                    "Statistic": name,
                    "Observed target": target_value,
                    "Simulated at estimate": simulated_value,
                    "Scaled residual": scaled_error,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    theta_true = np.array([3.00, 0.45, 3.15])
    choice_scale = 0.18
    n_observed = 5_000
    n_simulated = 30_000
    observed_draws = make_draws(123, n_observed)
    simulation_draws = make_draws(456, n_simulated)

    observed = simulate_search_panel(theta_true, observed_draws, choice_scale)
    target_moments = economic_moments(observed)
    target_aux = auxiliary_statistics(observed)
    start = np.array([2.85, 0.36, 3.00])

    t0 = time.perf_counter()
    msm = estimate_by_simulation(target_moments, simulation_draws, choice_scale, economic_moments, start)
    t_msm = time.perf_counter() - t0

    t0 = time.perf_counter()
    ii = estimate_by_simulation(target_aux, simulation_draws, choice_scale, auxiliary_statistics, start)
    t_ii = time.perf_counter() - t0

    prior_bounds = np.array([[2.4, 3.6], [0.2, 0.8], [2.5, 3.8]])
    n_particles = 1000
    n_rounds = 6
    alpha_quantile = 0.5
    t0 = time.perf_counter()
    abc = abc_smc(
        target_moments,
        simulation_draws,
        choice_scale,
        prior_bounds,
        n_particles=n_particles,
        n_rounds=n_rounds,
        alpha=alpha_quantile,
        seed=789,
    )
    t_abc = time.perf_counter() - t0
    times = {"msm": t_msm, "ii": t_ii, "abc": t_abc}

    msm_scale = np.maximum(np.abs(target_moments), 0.1)
    ii_scale = np.maximum(np.abs(target_aux), 0.1)
    mu_grid = np.linspace(2.82, 3.18, 32)
    reservation_grid = np.linspace(2.95, 3.35, 32)
    msm_surface = criterion_grid(
        mu_grid,
        reservation_grid,
        theta_true[1],
        target_moments,
        msm_scale,
        simulation_draws,
        choice_scale,
        economic_moments,
    )
    ii_surface = criterion_grid(
        mu_grid,
        reservation_grid,
        theta_true[1],
        target_aux,
        ii_scale,
        simulation_draws,
        choice_scale,
        auxiliary_statistics,
    )

    print("Simulation-based estimation tutorial")
    print(f"  True theta: {theta_true}")
    print(f"  MSM theta: {np.asarray(msm['theta'])}")
    print(f"  Indirect inference theta: {np.asarray(ii['theta'])}")
    print(f"  ABC posterior mean: {np.asarray(abc['theta_post_mean'])}")
    print(f"  MSM criterion: {float(msm['criterion']):.4e}")
    print(f"  II criterion: {float(ii['criterion']):.4e}")
    print(f"  ABC criterion at posterior mean: {float(abc['criterion']):.4e}")
    print(f"  ABC final tolerance: {float(abc['final_tolerance']):.4e}")
    print(f"  Wall time (s): MSM {t_msm:.2f}, II {t_ii:.2f}, ABC {t_abc:.2f}")

    setup_style()
    report = ModelReport(
        "Estimating a Search Acceptance Rule by Simulation",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A researcher observes wage offers and whether workers accept them. The "
        "reservation wage is hidden.\n\n"
        "The object is a rule that maps offers into acceptance probabilities. It "
        "depends on offer mean, offer dispersion, and reservation log wage.\n\n"
        "The model is easy to simulate for any parameter vector. Simulation-based "
        "estimation searches for parameters whose simulated data match observed "
        "summaries."
    )

    report.add_equations(
        r"""
Worker $i$ receives a log wage offer,

$$
\log w_i = \mu + \sigma z_i,\qquad z_i \sim N(0,1),
$$

and accepts with probability

$$
\Pr(d_i = 1 \mid w_i; \theta) =
\frac{1}{1 + \exp[-(\log w_i - r)/s]}.
$$

The structural parameter vector is $\theta = (\mu, \sigma, r)$. Offer mean is $\mu$, offer dispersion is $\sigma$, reservation log wage is $r$. The scale $s$ fixes how sharply acceptance changes near $r$.

All three estimators share the same simulator $S(\theta, \varepsilon)$ driven by a fixed vector of common shocks $\varepsilon_{sim}$. They differ in which summary statistic the simulator must reproduce, and in whether the answer is a point estimate or a distribution over $\theta$.

### Method 1: Method of Simulated Moments

Let $m_{obs} \in \mathbb{R}^{5}$ collect five economic moments computed on the observed sample,

$$
m_{obs} =
(
\underbrace{\Pr(d = 1)}_{\text{acceptance rate}},\
\underbrace{\mathbb{E}[\log w]}_{\text{offer mean}},\
\underbrace{\mathrm{SD}[\log w]}_{\text{offer sd}},\
\underbrace{\mathbb{E}[\log w \mid d = 1]}_{\text{accepted mean}},\
\underbrace{\mathrm{SD}[\log w \mid d = 1]}_{\text{accepted sd}}
).
$$

Write $m_{sim}(\theta) = m(S(\theta, \varepsilon_{sim}))$ for the same moments computed on simulated data. MSM minimizes the scaled quadratic criterion

$$
\hat\theta_{MSM} = \arg\min_\theta\,
\underbrace{[m_{sim}(\theta) - m_{obs}]^{\prime}}_{\text{moment gap, simulated vs observed}}\
\underbrace{W_m}_{\text{scale matrix}}\
\underbrace{[m_{sim}(\theta) - m_{obs}]}_{\text{moment gap}}
\equiv Q_{MSM}(\theta),
$$

with the diagonal weight $W_m = \mathrm{diag}(1 / \max(|m_{obs}|, 0.1))^{2}$.
The criterion is a weighted sum of squared moment gaps, so $W_m$ is what makes a 1% gap in the acceptance rate comparable to a 0.01 gap in the offer mean.
Scaling each gap by the magnitude of the observed moment is the simplest such normalization; in production code one would replace it by the inverse of the moment covariance estimated by bootstrap.

### Method 2: Indirect Inference

Let $b(\cdot)$ be a vector of auxiliary statistics: the OLS coefficients of the linear probability regression $d_i = b_0 + b_1 \log w_i$, augmented with offer-distribution moments and the acceptance rate. Write $b_{obs} = b(\text{observed sample})$ and $b_{sim}(\theta) = b(S(\theta, \varepsilon_{sim}))$. Indirect inference minimizes

$$
\hat\theta_{II} = \arg\min_\theta\,
\underbrace{
[b_{sim}(\theta) - b_{obs}]^{\prime}\, W_b\,
[b_{sim}(\theta) - b_{obs}]
}_{Q_{II}(\theta)},
$$

with the same scaling form, $W_b = \mathrm{diag}(1 / \max(|b_{obs}|, 0.1))^{2}$. The auxiliary model is misspecified by design; its coefficients are just summary statistics.

### Method 3: Approximate Bayesian Computation (ABC-SMC)

ABC replaces the likelihood by a tolerance ball around the observed moments. Define the scaled Euclidean distance

$$
\rho(\theta) = \sqrt{Q_{MSM}(\theta)},
$$

so the MSM criterion is $\rho^2$. Place a uniform prior on the same rectangle that bounds MSM and II,

$$
\pi(\theta) = U(2.4, 3.6) \times U(0.2, 0.8) \times U(2.5, 3.8).
$$

For a tolerance $\varepsilon > 0$ the ABC posterior is

$$
\pi_\varepsilon(\theta \mid m_{obs}) \propto \underbrace{\pi(\theta)}_{\text{prior}}\, \underbrace{\Pr[\rho(\theta) \le \varepsilon]}_{\text{ABC pseudo-likelihood}}.
$$

The pseudo-likelihood replaces the unknown true likelihood by the probability that a fresh simulation lands within $\varepsilon$ of the observed moments.
That trade is the entire point of ABC: any model that can be simulated has a usable Bayesian update, even when its density is not available.
As $\varepsilon \to 0$ the pseudo-likelihood concentrates on parameters whose simulator matches $m_{obs}$ exactly, so the posterior concentrates on $\arg\min_\theta \rho^2 = \hat\theta_{MSM}$ and ABC and MSM target the same point in the noise-free limit.
ABC adds the spread around that point that MSM's point estimate alone cannot report.

ABC-SMC approaches $\pi_0$ through a sequence $\varepsilon_0 > \varepsilon_1 > \cdots > \varepsilon_{T-1}$ of shrinking tolerances. Round $t$ maintains $N$ weighted particles $\lbrace (\theta_t^{(i)}, w_t^{(i)}) \rbrace_{i=1}^{N}$ that approximate $\pi_{\varepsilon_t}$. The schedule is adaptive: $\varepsilon_t$ is the $\alpha$-quantile of the distances at round $t-1$, with $\alpha = 0.5$.

Particles in round $t \ge 1$ are drawn by sampling a parent $\theta_{t-1}^{(j)}$ with probability $w_{t-1}^{(j)}$, perturbing it with a Gaussian kernel

$$
K_t(\theta \mid \theta^{\prime}) = \mathcal{N}(\theta^{\prime},\, 2\, \widehat{\mathrm{Cov}}_{t-1}),
$$

and keeping the proposal only if $\rho(\theta) \le \varepsilon_t$. The factor two in the covariance is the Beaumont-Cornuet-Marin-Robert (2009) twice-empirical-covariance rule. The importance weight corrects for the proposal,

$$
w_t^{(i)} \propto \frac{\pi(\theta_t^{(i)})}{\sum_{j=1}^{N} w_{t-1}^{(j)}\, K_t(\theta_t^{(i)} \mid \theta_{t-1}^{(j)})}.
$$

Under the uniform prior $\pi$ is constant on the support, so the numerator drops out and the weight is just the inverse of the kernel-mixture density evaluated at $\theta_t^{(i)}$.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| True $\\mu$ | {theta_true[0]:.2f} | Mean of the latent log offer distribution |\n"
        f"| True $\\sigma$ | {theta_true[1]:.2f} | Dispersion of latent log offers |\n"
        f"| True $r$ | {theta_true[2]:.2f} | Latent reservation log wage |\n"
        f"| Choice scale $s$ | {choice_scale:.2f} | Smoothness of acceptance rule |\n"
        f"| Observed sample | {n_observed:,} | Synthetic data generated once from the model |\n"
        f"| Simulation draws | {n_simulated:,} | Common random numbers used in all three criteria |\n"
        f"| MSM targets | {len(target_moments)} | Acceptance rate and offer-wage moments |\n"
        f"| II targets | {len(target_aux)} | Auxiliary acceptance coefficients and moments |\n"
        f"| ABC particles $N$ | {n_particles:,} | Particles maintained at each ABC-SMC round |\n"
        f"| ABC rounds $T$ | {n_rounds} | Number of shrinking-tolerance rounds |\n"
        f"| ABC quantile $\\alpha$ | {alpha_quantile:.2f} | Adaptive tolerance is the $\\alpha$-quantile of previous distances |\n"
        f"| ABC prior $\\pi$ | $U(2.4, 3.6) \\times U(0.2, 0.8) \\times U(2.5, 3.8)$ | Uniform on $(\\mu, \\sigma, r)$ |"
    )

    report.add_solution_method(
        "The three estimators share the simulator $S(\\theta, \\varepsilon)$ and the same fixed shocks $\\varepsilon_{sim}$. Common random numbers keep the criterion from changing because of fresh Monte Carlo noise. The three differ in what summary the simulator must reproduce and in whether the answer is a point or a distribution.\n\n"
        "### Method 1: Method of Simulated Moments\n\n"
        "Pick $\\theta$ so that the simulator reproduces the five economic moments. The criterion scales each residual by the magnitude of the matching observed moment, so each moment contributes on a comparable order. Nelder-Mead minimizes the scaled quadratic distance from a fixed starting point.\n\n"
        "```text\n"
        "Algorithm: MSM\n"
        "Input : observed moments m_obs, simulator S(theta, eps), fixed shocks eps_sim\n"
        "For each candidate theta:\n"
        "  m_sim <- moments(S(theta, eps_sim))\n"
        "  Q     <- sum_k ((m_sim_k - m_obs_k) / scale_k)^2\n"
        "Return theta minimizing Q via Nelder-Mead.\n"
        "```\n\n"
        "Failure mode: identification depends on the moments. If they are not informative about a parameter the criterion has a flat direction and the optimizer wanders.\n\n"
        "### Method 2: Indirect Inference\n\n"
        "Pick $\\theta$ so that the simulator reproduces the fitted coefficients of an auxiliary regression of acceptance on log wages. The auxiliary regression is not the structural model; its coefficients are summary statistics. The slope captures threshold variation that pins down the reservation wage.\n\n"
        "```text\n"
        "Algorithm: Indirect Inference\n"
        "Input : observed auxiliary stats b_obs, simulator S, fixed shocks eps_sim\n"
        "For each candidate theta:\n"
        "  b_sim <- aux_stats(S(theta, eps_sim))\n"
        "  Q     <- sum_k ((b_sim_k - b_obs_k) / scale_k)^2\n"
        "Return theta minimizing Q via Nelder-Mead.\n"
        "```\n\n"
        "Failure mode: a weak auxiliary model gives weak identification. Drop the linear-probability slope and the criterion flattens in the same direction MSM does when its moments miss the threshold.\n\n"
        "### Method 3: Approximate Bayesian Computation (ABC-SMC)\n\n"
        "Sample $\\theta$ from the prior, keep draws whose simulated moments are close to the observed moments, then iteratively tighten the closeness threshold and reweight the survivors. The output is a posterior over $\\theta$, not a single point. Tolerance shrinks adaptively as the $\\alpha$-quantile of the previous round's distances, with $\\alpha = 0.5$.\n\n"
        "```text\n"
        "Algorithm: ABC-SMC (adaptive tolerance schedule)\n"
        "Input : observed moments m_obs, prior pi, simulator S, eps_sim,\n"
        "        N particles, T rounds, quantile alpha\n"
        "Round t = 0:\n"
        "  Sample ceil(N / alpha) thetas from pi.\n"
        "  Compute d_i = scaled_euclidean(moments(S(theta_i)), m_obs) for each.\n"
        "  Keep the N smallest. Set epsilon_0 to the largest kept distance.\n"
        "  Initialize weights w_0 uniform on the kept particles.\n"
        "For round t = 1, ..., T - 1:\n"
        "  Set epsilon_t to the alpha-quantile of distances at round t - 1.\n"
        "  Compute Cov_t = 2 * weighted_covariance(round t - 1 particles).\n"
        "  For i = 1, ..., N:\n"
        "    Repeat:\n"
        "      Sample parent index j proportional to w_{t-1}.\n"
        "      Propose theta_i = parent_j + N(0, Cov_t); resample if outside prior box.\n"
        "      d_i <- scaled_euclidean(moments(S(theta_i)), m_obs).\n"
        "    Until d_i <= epsilon_t.\n"
        "  Compute importance weights w_t^(i) (uniform-prior simplification).\n"
        "  If effective sample size ESS_t < N / 2, multinomial-resample.\n"
        "Output: weighted particles approximating the ABC posterior.\n"
        "```\n\n"
        "Failure mode: the perturbation covariance shrinks faster than the tolerance, so the kernel cannot reach the next level set. Acceptance rates collapse and the posterior degenerates onto a few particles. Diagnose by tracking ESS and per-round acceptance rate."
    )

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    extent = [mu_grid.min(), mu_grid.max(), reservation_grid.min(), reservation_grid.max()]
    im0 = axes[0].imshow(msm_surface, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0].scatter(theta_true[0], theta_true[2], color="white", edgecolor="black", label="True")
    axes[0].scatter(np.asarray(msm["theta"])[0], np.asarray(msm["theta"])[2], color="tab:red", label="Estimate")
    axes[0].set_title("MSM Criterion")
    axes[0].set_xlabel("Offer mean mu")
    axes[0].set_ylabel("Reservation log wage")
    axes[0].legend(loc="upper left")
    fig1.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(ii_surface, origin="lower", extent=extent, aspect="auto", cmap="magma")
    axes[1].scatter(theta_true[0], theta_true[2], color="white", edgecolor="black", label="True")
    axes[1].scatter(np.asarray(ii["theta"])[0], np.asarray(ii["theta"])[2], color="tab:cyan", edgecolor="black", label="Estimate")
    axes[1].set_title("Indirect-Inference Criterion")
    axes[1].set_xlabel("Offer mean mu")
    axes[1].legend(loc="upper left")
    fig1.colorbar(im1, ax=axes[1], fraction=0.046)
    report.add_results(
        "The criterion surfaces show how each estimator trades off offer mean and "
        "reservation wage. Both plots fix offer dispersion at its true value. The valley "
        "tilts because a higher mean can offset a higher reservation wage."
    )
    report.add_figure(
        "figures/criterion-surfaces.png",
        "Criterion surfaces for MSM and indirect inference",
        fig1,
    )

    observed_accept = observed["accept"]
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
    bins = np.linspace(1.5, 4.7, 34)
    ax2.hist(observed["log_wage"], bins=bins, alpha=0.45, label="All offers", density=True)
    ax2.hist(observed["log_wage"][observed_accept], bins=bins, alpha=0.65, label="Accepted offers", density=True)
    ax2.axvline(theta_true[2], color="black", linestyle="--", linewidth=1.3, label="True reservation")
    ax2.axvline(np.asarray(msm["theta"])[2], color="tab:red", linestyle=":", linewidth=2, label="MSM estimate")
    ax2.axvline(np.asarray(ii["theta"])[2], color="tab:blue", linestyle="-.", linewidth=2, label="II estimate")
    ax2.axvline(np.asarray(abc["theta_post_mean"])[2], color="tab:green", linestyle="-", linewidth=2, alpha=0.7, label="ABC posterior mean")
    ax2.set_xlabel("Log wage")
    ax2.set_ylabel("Density")
    ax2.set_title("Observed Search Data")
    ax2.legend()
    report.add_results(
        f"The observed acceptance rate is **{observed_accept.mean():.3f}**. Accepted "
        "wages mostly come from the upper tail of offers. Stochastic choice leaves "
        "overlap near the reservation wage. That overlap helps locate the latent "
        "threshold."
    )
    report.add_figure(
        "figures/search-data.png",
        "Offer and accepted-wage distributions",
        fig2,
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    wage_grid = np.linspace(1.8, 4.4, 200)
    true_prob = expit((wage_grid - theta_true[2]) / choice_scale)
    msm_prob = expit((wage_grid - np.asarray(msm["theta"])[2]) / choice_scale)
    ii_prob = expit((wage_grid - np.asarray(ii["theta"])[2]) / choice_scale)
    abc_prob = expit((wage_grid - np.asarray(abc["theta_post_mean"])[2]) / choice_scale)
    ax3.plot(wage_grid, true_prob, label="True")
    ax3.plot(wage_grid, msm_prob, "--", label="MSM")
    ax3.plot(wage_grid, ii_prob, ":", label="Indirect inference")
    ax3.plot(wage_grid, abc_prob, "-.", label="ABC posterior mean")
    ax3.set_xlabel("Log wage offer")
    ax3.set_ylabel("Acceptance probability")
    ax3.set_title("Recovered Acceptance Rule")
    ax3.legend()
    report.add_results(
        "All three estimators recover the acceptance curve closely. The small gaps reflect "
        "the observed sample and finite simulation. They do not come from different "
        "search models."
    )
    report.add_figure(
        "figures/acceptance-rule.png",
        "True and estimated acceptance probabilities",
        fig3,
    )

    final_particles = np.asarray(abc["particles"])[-1]
    final_weights = np.asarray(abc["weights"])[-1]
    abc_post_mean = np.asarray(abc["theta_post_mean"])
    abc_post_ci = np.asarray(abc["theta_post_ci"])
    fig4, axes4 = plt.subplots(1, 3, figsize=(13.5, 4.2))
    param_names = [r"Offer mean $\mu$", r"Offer sd $\sigma$", r"Reservation $r$"]
    grids = [
        np.linspace(prior_bounds[j, 0], prior_bounds[j, 1], 400) for j in range(3)
    ]
    for j, ax in enumerate(axes4):
        kde = gaussian_kde(final_particles[:, j], weights=final_weights, bw_method=0.35)
        density = kde(grids[j])
        ax.plot(grids[j], density, color="tab:green", linewidth=2, label="ABC posterior")
        mask = (grids[j] >= abc_post_ci[j, 0]) & (grids[j] <= abc_post_ci[j, 1])
        ax.fill_between(grids[j], 0, density, where=mask, color="tab:green", alpha=0.18, label="90% CI")
        ax.axvline(theta_true[j], color="black", linestyle="-", linewidth=1.5, label="True")
        ax.axvline(np.asarray(msm["theta"])[j], color="tab:red", linestyle="--", linewidth=1.4, label="MSM")
        ax.axvline(np.asarray(ii["theta"])[j], color="tab:blue", linestyle=":", linewidth=1.6, label="II")
        ax.axvline(abc_post_mean[j], color="tab:green", linestyle="-.", linewidth=1.4, label="ABC mean")
        ax.set_xlabel(param_names[j])
        ax.set_ylabel("Posterior density")
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig4.suptitle("ABC-SMC Posterior Marginals")
    fig4.tight_layout()
    report.add_results(
        "The ABC-SMC posterior summarizes how the five economic moments restrict each "
        "parameter. The marginal for the reservation log wage is tightest because "
        "acceptance variation near the threshold is highly informative. Offer mean is "
        "next most informed; offer dispersion is least constrained by these moments and "
        "shows the widest posterior. The MSM and indirect-inference point estimates sit "
        "near the posterior modes in every panel, which is the visual statement that "
        "all three methods are minimizing the same scaled distance under different "
        "aggregation rules."
    )
    report.add_figure(
        "figures/abc-posteriors.png",
        "ABC-SMC posterior marginals with MSM and II point estimates overlaid",
        fig4,
    )

    abc_tolerances = np.asarray(abc["tolerances"])
    abc_accept = np.asarray(abc["accept_rates"])
    rounds_axis = np.arange(len(abc_tolerances))
    fig5, ax5 = plt.subplots(figsize=(7, 4.2))
    color_tol = "tab:purple"
    ax5.plot(rounds_axis, abc_tolerances, color=color_tol, marker="o", label=r"Tolerance $\varepsilon_t$")
    ax5.set_yscale("log")
    ax5.set_xlabel("Round $t$")
    ax5.set_ylabel(r"Tolerance $\varepsilon_t$ (log scale)", color=color_tol)
    ax5.tick_params(axis="y", labelcolor=color_tol)
    ax5b = ax5.twinx()
    color_acc = "tab:orange"
    ax5b.plot(rounds_axis, abc_accept, color=color_acc, marker="s", linestyle="--", label="Acceptance rate")
    ax5b.set_ylabel("Acceptance rate", color=color_acc)
    ax5b.tick_params(axis="y", labelcolor=color_acc)
    ax5b.set_ylim(0, 1.05)
    ax5.set_title("ABC-SMC Tolerance Schedule")
    fig5.tight_layout()
    report.add_results(
        "Each round shrinks the tolerance and accepts a declining fraction of proposals "
        "as the level sets tighten. The first transition is the cheapest because the "
        "prior already overlaps the high-density region. Later rounds spend more "
        "simulator calls per accepted particle because the perturbation kernel keeps "
        "the same Beaumont 2009 scale while the posterior concentrates. Effective "
        "sample size stays well above the resampling threshold, so weight degeneracy "
        "is not the binding cost here."
    )
    report.add_figure(
        "figures/abc-tolerance-schedule.png",
        "Per-round ABC-SMC tolerance and acceptance rate",
        fig5,
    )

    report.add_results(
        "Parameter estimates and residuals give a compact diagnostic. MSM and indirect "
        "inference return point estimates; ABC-SMC returns a posterior whose mean is "
        "reported alongside a 90% credible interval. Small scaled residuals show that "
        "each target vector is matched closely."
    )

    report.add_table(
        "tables/parameter-recovery.csv",
        "Known-truth parameter recovery",
        parameter_table(theta_true, msm, ii, abc).round(5),
    )

    report.add_table(
        "tables/msm-residuals.csv",
        "MSM moment residuals",
        residual_table(
            ["Acceptance rate", "Mean log wage", "SD log wage", "Mean accepted log wage", "SD accepted log wage"],
            target_moments,
            msm,
            msm,
        ).query("Estimator == 'MSM'").drop(columns=["Estimator"]).round(5),
    )

    report.add_table(
        "tables/indirect-inference-residuals.csv",
        "Indirect-inference auxiliary residuals",
        residual_table(
            ["LPM intercept", "LPM slope", "Mean log wage", "SD log wage", "Acceptance rate", "Mean accepted log wage"],
            target_aux,
            ii,
            ii,
        ).query("Estimator == 'MSM'").drop(columns=["Estimator"]).round(5),
    )

    report.add_results(
        "The method-comparison table puts parameter recoveries, loss values, work, and "
        "wall times on the same row. MSM and indirect inference report Nelder-Mead "
        "iterations and the criterion value at the argmin. ABC-SMC reports the total "
        "number of simulator calls across all rounds and the same criterion evaluated "
        "at the posterior mean, which is on the same scale as the MSM and II numbers."
    )

    report.add_table(
        "tables/method-comparison.csv",
        "Estimates, loss, work, and wall time across the three methods",
        method_comparison_table(msm, ii, abc, times).round(5),
    )

    abc_summary = pd.DataFrame(
        {
            "Round": np.arange(len(abc["tolerances"])),
            "Tolerance": np.asarray(abc["tolerances"], dtype=float),
            "Proposals": np.asarray(abc["proposals"], dtype=int),
            "Acceptance rate": np.asarray(abc["accept_rates"], dtype=float),
            "ESS": np.asarray(abc["ess"], dtype=float),
            "Mean distance": np.asarray(abc["distances"]).mean(axis=1),
        }
    ).round({"Tolerance": 5, "Acceptance rate": 4, "ESS": 2, "Mean distance": 5})
    report.add_table(
        "tables/abc-summary.csv",
        "Per-round ABC-SMC diagnostics",
        abc_summary,
    )

    report.add_takeaway(
        "Simulation-based estimation is useful when the structural model is easier to "
        "simulate than to evaluate by likelihood. MSM matches economic moments chosen "
        "by the researcher. Indirect inference matches fitted statistics from an "
        "auxiliary acceptance model. Approximate Bayesian computation samples from "
        "level sets of the same scaled distance and reports the spread of acceptable "
        "parameters, not just the argmin.\n\n"
        "The three estimators are one family. All three pick a summary statistic, "
        "simulate, evaluate the distance between simulated and observed summaries, and "
        "search over $\\theta$. MSM and indirect inference return the point that "
        "minimizes the distance. ABC samples from level sets of the same distance with "
        "a tolerance that shrinks toward zero.\n\n"
        "The split is not really frequentist versus Bayesian. ABC quantifies the "
        "curvature of the criterion around its minimum, which is the question classical "
        "standard errors answer with a Hessian approximation. When the simulator is "
        "cheap and the prior is honest, ABC gives the most informative answer of the "
        "three."
    )

    report.add_references(
        [
            "[McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. *Econometrica*, 57(5), 995-1026.](https://doi.org/10.2307/1913621)",
            "[Gourieroux, C., Monfort, A., and Renault, E. (1993). Indirect Inference. *Journal of Applied Econometrics*, 8(S1), S85-S118.](https://doi.org/10.1002/jae.3950080507)",
            "[Sisson, S. A., Fan, Y., and Tanaka, M. M. (2007). Sequential Monte Carlo without likelihoods. *PNAS*, 104(6), 1760-1765.](https://doi.org/10.1073/pnas.0607208104)",
            "[Beaumont, M. A., Cornuet, J.-M., Marin, J.-M., and Robert, C. P. (2009). Adaptive approximate Bayesian computation. *Biometrika*, 96(4), 983-990.](https://doi.org/10.1093/biomet/asp052)",
            "[Toni, T., Welch, D., Strelkowa, N., Ipsen, A., and Stumpf, M. P. H. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. *Journal of the Royal Society Interface*, 6(31), 187-202.](https://doi.org/10.1098/rsif.2008.0172)",
            "[Drovandi, C. C. and Pettitt, A. N. (2011). Estimation of parameters for macroparasite population evolution using approximate Bayesian computation. *Biometrics*, 67(1), 225-233.](https://doi.org/10.1111/j.1541-0420.2010.01410.x)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
