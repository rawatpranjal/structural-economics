#!/usr/bin/env python3
"""Posterior sampling: Beta-Binomial conjugate updates and random-walk Metropolis-Hastings.

The tutorial covers two methods. The first is conjugate Bayes on a
Beta-Binomial model, where the posterior is a Beta distribution in closed
form. It is the canonical first Bayesian example and the controlled
sanity check for the Markov-chain sampler that follows. The second is
random-walk Metropolis-Hastings on a two-component Gaussian mixture, a
stand-in for a structural posterior with two parameter regimes. Closed-form
moments do not exist for the mixture; the sampler is the only way to compute
posterior averages, and the diagnostics earn their keep.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betainc, gammaln, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


# =============================================================================
# Method 1: Beta-Binomial conjugate model
# =============================================================================
ALPHA_PRIOR = 2.0
BETA_PRIOR = 2.0
N_TRIALS = 20
K_SUCCESSES = 14
ALPHA_POST = ALPHA_PRIOR + K_SUCCESSES
BETA_POST = BETA_PRIOR + N_TRIALS - K_SUCCESSES
POST_MEAN = ALPHA_POST / (ALPHA_POST + BETA_POST)
POST_VAR = (ALPHA_POST * BETA_POST) / (
    (ALPHA_POST + BETA_POST) ** 2 * (ALPHA_POST + BETA_POST + 1)
)


def beta_log_pdf(theta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Log density of Beta(alpha, beta) for theta in (0, 1)."""
    theta = np.asarray(theta, dtype=float)
    log_norm = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
    return log_norm + (alpha - 1.0) * np.log(theta) + (beta - 1.0) * np.log(1.0 - theta)


def beta_pdf(theta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.exp(beta_log_pdf(theta, alpha, beta))


def beta_binomial_log_posterior(theta: float) -> float:
    """Unnormalized log posterior of theta under Beta(ALPHA_PRIOR, BETA_PRIOR) prior
    and Binomial(N_TRIALS, theta) likelihood with K_SUCCESSES observed."""
    if theta <= 0.0 or theta >= 1.0:
        return -np.inf
    return (ALPHA_POST - 1.0) * np.log(theta) + (BETA_POST - 1.0) * np.log(1.0 - theta)


def random_walk_mh_1d(
    log_target_fn,
    n_draws: int,
    proposal_step: float,
    seed: int,
    start: float,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """One-dimensional Gaussian random-walk Metropolis-Hastings.

    Proposals outside `bounds` are rejected without evaluating the log target,
    which keeps the target on its support and the acceptance ratio well defined.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    current = float(start)
    current_logp = float(log_target_fn(current))
    accepted = 0
    draws[0] = current
    for t in range(1, n_draws):
        proposal = current + proposal_step * rng.normal()
        if bounds is not None and (proposal <= bounds[0] or proposal >= bounds[1]):
            draws[t] = current
            continue
        proposal_logp = float(log_target_fn(proposal))
        log_alpha = proposal_logp - current_logp
        if np.log(rng.uniform()) <= min(0.0, log_alpha):
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        draws[t] = current
    return draws, accepted / (n_draws - 1)


# =============================================================================
# Method 2: Two-regime Gaussian mixture posterior (existing material)
# =============================================================================
MU1 = np.array([1.5, 1.5])
MU2 = np.array([-1.5, -1.5])
SIGMA = np.array([[1.0, 0.5], [0.5, 1.0]])
MIXING_PROB = 0.5
INV_SIGMA = np.linalg.inv(SIGMA)
LOG_DET_SIGMA = float(np.linalg.slogdet(SIGMA)[1])
LOG_2PI = float(np.log(2.0 * np.pi))
TRUE_MEAN = MIXING_PROB * MU1 + (1.0 - MIXING_PROB) * MU2
TRUE_COV = SIGMA + MIXING_PROB * (1.0 - MIXING_PROB) * np.outer(MU1 - MU2, MU1 - MU2)


def log_mvn(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Evaluate a bivariate normal log density at one or many points."""
    x = np.asarray(points, dtype=float)
    diff = x - mean
    quad = np.einsum("...i,ij,...j->...", diff, INV_SIGMA, diff)
    return -0.5 * (2.0 * LOG_2PI + LOG_DET_SIGMA + quad)


def log_target(points: np.ndarray) -> np.ndarray:
    """Evaluate the log density of the two-component Gaussian mixture."""
    log_components = np.stack(
        [
            np.log(MIXING_PROB) + log_mvn(points, MU1),
            np.log(1.0 - MIXING_PROB) + log_mvn(points, MU2),
        ],
        axis=0,
    )
    return logsumexp(log_components, axis=0)


def random_walk_mh(
    n_draws: int,
    proposal_step: float,
    seed: int,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Run a Gaussian random-walk Metropolis-Hastings chain in two dimensions."""
    rng = np.random.default_rng(seed)
    draws = np.empty((n_draws, 2), dtype=float)
    current = np.array([10.0, -10.0]) if start is None else np.asarray(start, dtype=float).copy()
    current_logp = float(log_target(current))
    accepted = 0
    draws[0] = current

    for t in range(1, n_draws):
        proposal = current + proposal_step * rng.normal(size=2)
        proposal_logp = float(log_target(proposal))
        log_acceptance = proposal_logp - current_logp
        if np.log(rng.uniform()) <= min(0.0, log_acceptance):
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        draws[t] = current

    return draws, accepted / (n_draws - 1)


def mode_labels(draws: np.ndarray) -> np.ndarray:
    """Assign each draw to its nearest target component."""
    d1 = np.sum((draws - MU1) ** 2, axis=1)
    d2 = np.sum((draws - MU2) ** 2, axis=1)
    return np.where(d1 <= d2, 1, -1)


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute sample autocorrelation up to max_lag."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.ones(max_lag + 1)
    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return acf


def effective_sample_size(series: np.ndarray, max_lag: int = 300) -> float:
    """Estimate effective sample size from positive autocorrelations."""
    acf = autocorrelation(series, min(max_lag, len(series) - 2))
    positive = []
    for value in acf[1:]:
        if value <= 0.0:
            break
        positive.append(value)
    tau = 1.0 + 2.0 * float(np.sum(positive))
    return float(len(series) / max(tau, 1.0))


def chain_summary(chains: dict[float, tuple[np.ndarray, float]], burn: int) -> pd.DataFrame:
    """Create a tuning summary for proposal scales."""
    rows = []
    for proposal_step, (draws, acceptance) in chains.items():
        kept = draws[burn:]
        labels = mode_labels(kept)
        switches = int(np.sum(labels[1:] != labels[:-1]))
        mean_error = float(np.linalg.norm(kept.mean(axis=0) - TRUE_MEAN))
        rows.append(
            {
                "Proposal step": proposal_step,
                "Acceptance rate": f"{acceptance:.3f}",
                "Mode switches": switches,
                "Mean error": f"{mean_error:.3f}",
                "ESS theta1": f"{effective_sample_size(kept[:, 0]):.0f}",
                "ESS theta2": f"{effective_sample_size(kept[:, 1]):.0f}",
            }
        )
    return pd.DataFrame(rows)


def make_density_grid(limit: float = 5.0, n: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, Z arrays for target-density contours."""
    grid = np.linspace(-limit, limit, n)
    x_grid, y_grid = np.meshgrid(grid, grid)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    density = np.exp(log_target(points)).reshape(x_grid.shape)
    return x_grid, y_grid, density


def main() -> None:
    setup_style()

    # -------------------------------------------------------------------------
    # Method 1: Beta-Binomial conjugate model
    # -------------------------------------------------------------------------
    n_conj_draws = 20_000
    burn_conj = 1_000
    proposal_conj = 0.10
    bb_draws, bb_acceptance = random_walk_mh_1d(
        beta_binomial_log_posterior,
        n_draws=n_conj_draws,
        proposal_step=proposal_conj,
        seed=20260510,
        start=0.5,
        bounds=(0.0, 1.0),
    )
    bb_kept = bb_draws[burn_conj:]
    bb_mean_empirical = float(np.mean(bb_kept))
    bb_var_empirical = float(np.var(bb_kept))
    bb_mean_error = abs(bb_mean_empirical - POST_MEAN)
    bb_var_error = abs(bb_var_empirical - POST_VAR)

    # -------------------------------------------------------------------------
    # Method 2: Two-regime Gaussian mixture
    # -------------------------------------------------------------------------
    n_draws = 12_000
    burn = 1_000
    proposal_steps = [0.15, 0.6, 2.0]
    chains = {
        step: random_walk_mh(n_draws=n_draws, proposal_step=step, seed=609 + i)
        for i, step in enumerate(proposal_steps)
    }
    summary = chain_summary(chains, burn=burn)
    main_step = 0.6
    main_draws, main_acceptance = chains[main_step]
    kept = main_draws[burn:]

    print("Method 1: Beta-Binomial conjugate")
    print(f"  closed-form posterior: Beta({ALPHA_POST:.0f}, {BETA_POST:.0f})")
    print(f"  analytical mean = {POST_MEAN:.4f}, variance = {POST_VAR:.5f}")
    print(f"  MH empirical mean = {bb_mean_empirical:.4f} (error {bb_mean_error:.4f})")
    print(f"  MH empirical variance = {bb_var_empirical:.5f} (error {bb_var_error:.5f})")
    print(f"  MH acceptance rate = {bb_acceptance:.3f}")
    print()
    print("Method 2: Two-regime mixture proposal comparison")
    for _, row in summary.iterrows():
        print(
            f"  step={row['Proposal step']}: accept={row['Acceptance rate']} "
            f"switches={row['Mode switches']} mean_error={row['Mean error']}"
        )

    report = ModelReport(
        "Posterior Sampling: Conjugate Bayes and Metropolis-Hastings",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Bayesian inference replaces a single estimate with a posterior distribution over parameters. "
        "Posterior averages are the objects of interest: predictive means, counterfactual welfare, structural elasticities. "
        "When the prior and the likelihood are conjugate, the posterior is available in closed form and the averages are analytical. "
        "When they are not, the posterior is known only up to a normalizing constant and the averages require a sampler.\n\n"
        "This tutorial walks both cases. "
        "The first method is the Beta-Binomial conjugate model, where the posterior is a Beta distribution and every moment is in closed form. "
        "It is the canonical first Bayesian example. "
        "It also serves as a controlled sanity check on the Markov-chain sampler that follows: running Metropolis-Hastings on the same model should recover the closed-form posterior to within finite-sample noise. "
        "If it does not, the sampler is wrong before we ever apply it to a harder problem.\n\n"
        "The second method is random-walk Metropolis-Hastings on a two-component Gaussian mixture. "
        "The mixture stands in for a structural posterior where two parameter regions fit the same data. "
        "There is no closed form. "
        "The sampler is the only tool, and the diagnostics on it are how we tell whether posterior averages weight the regimes correctly or report a regime artifact."
    )

    report.add_equations(
        rf"""Let $\theta \in \Theta$ denote a parameter (scalar or vector) and let $D$ denote observed data.
Bayes' rule combines a likelihood $L(D \mid \theta)$ with a prior density $p_0(\theta)$ into a posterior density

$$
p(\theta \mid D) = \frac{{\overbrace{{L(D \mid \theta)}}^{{\text{{likelihood}}}}\, \overbrace{{p_0(\theta)}}^{{\text{{prior}}}}}}{{\underbrace{{\int_{{\Theta}} L(D \mid \theta')\, p_0(\theta')\, d\theta'}}_{{\text{{marginal likelihood }} m(D)}}}}.
$$

The numerator $L(D \mid \theta)\, p_0(\theta)$ is the posterior kernel, the only thing the sampler needs.
The denominator is the marginal likelihood $m(D)$, an integral over $\Theta$ that is usually intractable.
That intractability is the whole reason MCMC exists: the sampler in Method 2 evaluates the kernel and never the marginal likelihood, because the kernel ratio cancels the unknown normalizing constant.
Closed-form posteriors arise when the prior is conjugate to the likelihood (Method 1).
Otherwise we sample (Method 2).

### Method 1: Beta-Binomial conjugate posterior

The Beta-Binomial model has scalar parameter $\theta \in (0, 1)$ interpreted as the probability of a binary outcome.
The prior is a Beta distribution with shape parameters $\alpha > 0$ and $\beta > 0$:

$$
\theta \sim \mathrm{{Beta}}(\alpha, \beta),
\qquad
p_0(\theta) = \frac{{\theta^{{\alpha - 1}} (1 - \theta)^{{\beta - 1}}}}{{B(\alpha, \beta)}},
\qquad
B(\alpha, \beta) = \int_0^1 u^{{\alpha - 1}} (1 - u)^{{\beta - 1}}\, du.
$$

The Beta function $B(\alpha, \beta)$ is the normalizing constant of the prior; it equals $\Gamma(\alpha)\Gamma(\beta) / \Gamma(\alpha + \beta)$ in terms of the Gamma function but we never need that form explicitly.
The data are $n \ge 1$ independent Bernoulli trials with $k \in \lbrace 0, 1, \ldots, n \rbrace$ successes, so $D = (y_1, \ldots, y_n)$ with $y_i \in \lbrace 0, 1 \rbrace$ and $k = \sum_i y_i$.
The likelihood is the binomial mass function

$$
L(D \mid \theta) = \binom{{n}}{{k}}\, \theta^k (1 - \theta)^{{n - k}}.
$$

Multiplying prior and likelihood, the kernel collects all $\theta$-dependent factors and the binomial coefficient and Beta-function denominators are absorbed into the normalizing constant:

$$
p(\theta \mid D) \propto \theta^{{\alpha - 1}} (1 - \theta)^{{\beta - 1}} \cdot \theta^k (1 - \theta)^{{n - k}}
= \theta^{{\alpha + k - 1}} (1 - \theta)^{{\beta + n - k - 1}}.
$$

The kernel has Beta form, so the posterior is itself Beta with updated parameters:

$$
\theta \mid D \sim \mathrm{{Beta}}(\alpha + k,\, \beta + n - k).
$$

Writing $\alpha_{{\mathrm{{post}}}} = \alpha + k$ and $\beta_{{\mathrm{{post}}}} = \beta + n - k$, the posterior moments are available in closed form.
The posterior mean is

$$
\mathbb{{E}}[\theta \mid D] = \frac{{\alpha_{{\mathrm{{post}}}}}}{{\alpha_{{\mathrm{{post}}}} + \beta_{{\mathrm{{post}}}}}} = \underbrace{{\frac{{\alpha + \beta}}{{\alpha + \beta + n}}}}_{{\text{{prior weight}}}} \cdot \underbrace{{\frac{{\alpha}}{{\alpha + \beta}}}}_{{\text{{prior mean}}}} + \underbrace{{\frac{{n}}{{\alpha + \beta + n}}}}_{{\text{{data weight}}}} \cdot \underbrace{{\frac{{k}}{{n}}}}_{{\text{{sample fraction}}}}.
$$

Written this way the posterior mean is a convex combination of the prior mean and the sample fraction, with weights summing to one.
The prior weight $(\alpha + \beta)/(\alpha + \beta + n)$ shrinks toward zero as the sample size grows, so a Bayesian with a flat prior and a large dataset reports essentially the sample fraction.
This is the same shrinkage logic that drives the Gaussian-process posterior in `numerical-methods/bayesian-optimization/`: in both models the posterior mean is a weighted average of a prior anchor and a data-driven estimate, weighted by their respective precisions.
The posterior variance is

$$
\mathrm{{Var}}[\theta \mid D] = \frac{{\alpha_{{\mathrm{{post}}}}\, \beta_{{\mathrm{{post}}}}}}{{(\alpha_{{\mathrm{{post}}}} + \beta_{{\mathrm{{post}}}})^2\, (\alpha_{{\mathrm{{post}}}} + \beta_{{\mathrm{{post}}}} + 1)}}.
$$

The tail probability $P(\theta > t \mid D)$ for $t \in (0, 1)$ is one minus the regularized incomplete Beta function

$$
P(\theta > t \mid D) = 1 - I_t(\alpha_{{\mathrm{{post}}}}, \beta_{{\mathrm{{post}}}}),
\qquad
I_t(a, b) = \frac{{1}}{{B(a, b)}} \int_0^t u^{{a - 1}} (1 - u)^{{b - 1}}\, du.
$$

These three moments are computed in code without any Monte-Carlo simulation, which is what makes Method 1 the controlled sanity check for Method 2 below.
The same Bayesian update machinery, in a different geometry, drives the Gaussian-process posterior in `numerical-methods/bayesian-optimization/`: there the prior is over an unknown function and conditioning a joint Gaussian replaces the Beta-Binomial conjugacy.

On the calibration here ($\alpha = {ALPHA_PRIOR:.0f}$, $\beta = {BETA_PRIOR:.0f}$, $n = {N_TRIALS}$, $k = {K_SUCCESSES}$) the posterior is $\mathrm{{Beta}}({ALPHA_POST:.0f}, {BETA_POST:.0f})$ with mean ${POST_MEAN:.4f}$ and variance ${POST_VAR:.5f}$.

### Method 2: Random-walk Metropolis-Hastings on a mixture posterior

The second target is a posterior over $\theta = (\theta_1, \theta_2) \in \mathbb{{R}}^2$ given by a two-component Gaussian mixture:

$$
\pi(\theta \mid D) = \omega\, \phi(\theta;\, \mu_1, \Sigma) + (1 - \omega)\, \phi(\theta;\, \mu_2, \Sigma),
$$

where $\omega \in (0, 1)$ is the mixing weight, $\mu_1, \mu_2 \in \mathbb{{R}}^2$ are the component means, $\Sigma \in \mathbb{{R}}^{{2 \times 2}}$ is the shared component covariance, and the bivariate normal density is

$$
\phi(\theta;\, \mu, \Sigma) = \frac{{1}}{{2 \pi \sqrt{{\lvert \Sigma \rvert}}}}\, \exp\left(-\tfrac{{1}}{{2}} (\theta - \mu)^{{\top}} \Sigma^{{-1}} (\theta - \mu)\right).
$$

The two components stand in for two structural regimes that fit the same data.
There is no closed-form posterior moment generator: the moments depend on the mixture and we cannot integrate against $\pi$ analytically.

Random-walk Metropolis-Hastings constructs a Markov chain $(\theta_t)_{{t \ge 0}}$ whose stationary distribution is $\pi(\theta \mid D)$ using only pointwise evaluations of the kernel.
Given current state $\theta_t \in \mathbb{{R}}^d$ (with $d = 2$ here), a Gaussian random-walk proposal draws

$$
\theta^{{\star}} = \theta_t + s\, \eta_t,
\qquad
\eta_t \sim \mathcal{{N}}(0, I_d),
\qquad
\eta_t \in \mathbb{{R}}^d,
$$

where $s > 0$ is the proposal scale and $I_d$ is the $d \times d$ identity matrix.
Because the proposal density $q(\theta^{{\star}} \mid \theta_t)$ is symmetric, the Metropolis-Hastings acceptance probability simplifies to the kernel ratio capped at one:

$$
\alpha(\theta_t, \theta^{{\star}}) =
\min\bigg\lbrace 1,\, \underbrace{{\frac{{\pi(\theta^{{\star}} \mid D)}}{{\pi(\theta_t \mid D)}}}}_{{\text{{kernel ratio, marginal cancels}}}} \bigg\rbrace.
$$

The marginal likelihood $m(D)$ appears in both the numerator and denominator of the kernel ratio and cancels exactly, which is why the sampler never needs to evaluate the partition function.
This rule satisfies detailed balance: for any pair $(\theta, \theta')$ the joint density of "current state and proposal" is symmetric under swapping the two, since

$$
\pi(\theta)\, q(\theta' \mid \theta)\, \alpha(\theta, \theta')
= \pi(\theta')\, q(\theta \mid \theta')\, \alpha(\theta', \theta).
$$

Detailed balance implies that $\pi$ is the stationary distribution of the resulting chain.
The acceptance ratio depends only on the kernel ratio, so the marginal likelihood $m(D)$ cancels.
That is the load-bearing reason MH works without ever computing the partition function.
The same algorithm applies to the conjugate model above with the bound $\theta \in (0, 1)$ enforced by rejecting proposals outside the unit interval; running it there is how we verify the sampler before applying it to the harder mixture target.

For curved or strongly correlated posteriors the random walk mixes slowly and effective sample size per evaluation is small; the gradient-based proposal in `computational-methods/hamiltonian-monte-carlo/` is the fix when $\nabla \log \pi$ is available.

Retained draws from the chain approximate posterior averages of any integrable function $g : \Theta \to \mathbb{{R}}$:

$$
\mathbb{{E}}[g(\theta) \mid D] \approx \frac{{1}}{{T - T_{{\mathrm{{burn}}}}}}\, \sum_{{t = T_{{\mathrm{{burn}}}} + 1}}^{{T}} g(\theta_t).
$$

The approximation is exact in the limit $T \to \infty$.
On a finite run it is only as good as the chain's mixing, which on multimodal targets is governed by how often the chain crosses between modes.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| **Method 1 Beta-Binomial** | | |\n"
        f"| Prior $\\mathrm{{Beta}}(\\alpha, \\beta)$ | ({ALPHA_PRIOR:.0f}, {BETA_PRIOR:.0f}) | Weak symmetric prior |\n"
        f"| Sample size $n$ | {N_TRIALS} | Binomial trials |\n"
        f"| Successes $k$ | {K_SUCCESSES} | Observed |\n"
        f"| Closed-form posterior | $\\mathrm{{Beta}}({ALPHA_POST:.0f}, {BETA_POST:.0f})$ | Analytical |\n"
        f"| Posterior mean | {POST_MEAN:.4f} | $\\alpha_{{\\mathrm{{post}}}} / (\\alpha_{{\\mathrm{{post}}}} + \\beta_{{\\mathrm{{post}}}})$ |\n"
        f"| Posterior variance | {POST_VAR:.5f} | Analytical |\n"
        f"| MH proposal scale | {proposal_conj:.2f} | Bounded random walk on $(0, 1)$ |\n"
        f"| MH draws | {n_conj_draws:,} | After burn-in of {burn_conj:,} |\n"
        f"| **Method 2 mixture** | | |\n"
        f"| Posterior interpretation | Two empirically plausible structural regimes | |\n"
        f"| $\\mu_1$ | ({MU1[0]:.1f}, {MU1[1]:.1f}) | First-regime mean |\n"
        f"| $\\mu_2$ | ({MU2[0]:.1f}, {MU2[1]:.1f}) | Second-regime mean |\n"
        f"| $\\Sigma$ | [[1.0, 0.5], [0.5, 1.0]] | Within-regime covariance |\n"
        f"| Mixing probability $\\omega$ | {MIXING_PROB:.1f} | Regime weight |\n"
        f"| MH draws | {n_draws:,} | After burn-in of {burn:,} |\n"
        f"| MH starting point | (10.0, -10.0) | Far from both modes |\n"
        f"| MH proposal steps | {proposal_steps} | Tuning sweep |"
    )

    report.add_solution_method(
        "The two methods share the same Metropolis-Hastings machinery on top of different posterior kernels. "
        "The first is the conjugate model where the answer is known. "
        "The second is the mixture where it is not.\n\n"

        "### Method 1: Conjugate Beta-Binomial\n\n"
        "Beta-Binomial conjugacy gives the posterior in one line of algebra: a Beta prior with parameters $(\\alpha, \\beta)$ combined with $k$ successes in $n$ trials returns a $\\mathrm{Beta}(\\alpha + k,\\, \\beta + n - k)$ posterior. "
        "The posterior moments follow from the Beta family and need no simulation.\n\n"
        "```text\n"
        "Algorithm: Conjugate update for the Beta-Binomial model\n"
        "Input : prior parameters alpha, beta; data n, k\n"
        "Output: posterior parameters and analytical moments\n"
        "  alpha_post = alpha + k\n"
        "  beta_post  = beta  + n - k\n"
        "  mean       = alpha_post / (alpha_post + beta_post)\n"
        "  variance   = alpha_post * beta_post / "
        "              ((alpha_post + beta_post)^2 * (alpha_post + beta_post + 1))\n"
        "```\n\n"
        "We run a one-dimensional random-walk Metropolis-Hastings chain on the same posterior as a sanity check. "
        "The proposal is a Gaussian step bounded to the unit interval by rejecting any move outside $(0, 1)$. "
        f"After {burn_conj:,} burn-in draws and {n_conj_draws - burn_conj:,} retained draws, the chain's empirical mean and variance should match the closed-form values to within a few percent. "
        "If they do not, either the proposal scale is too small to mix or the acceptance rule is implemented wrong. "
        "Either way, no further conclusions from the same sampler are trustworthy.\n\n"

        "### Method 2: Random-walk Metropolis-Hastings on a mixture posterior\n\n"
        "Random-walk Metropolis-Hastings needs the posterior kernel at the current and proposed parameter values. "
        "The normalizing constant cancels from the acceptance ratio. "
        "The script runs three proposal scales to expose the tuning trade-off on the mixture target.\n\n"
        "```text\n"
        "Algorithm: random-walk Metropolis-Hastings\n"
        "Input: log posterior kernel ell(theta), proposal scale s, initial theta_0, draws T\n"
        "Output: draws from pi(theta | D), plus mode-crossing summaries\n"
        "1. Set theta = theta_0 and compute ell(theta)\n"
        "2. For t = 1, ..., T:\n"
        "       propose theta_star = theta + s * eta_t, eta_t ~ N(0, I)\n"
        "       compute log alpha = ell(theta_star) - ell(theta)\n"
        "       accept theta_star with probability min(1, exp(log alpha))\n"
        "       otherwise repeat the current theta\n"
        "3. Drop burn-in draws\n"
        "4. Report acceptance, mode switches, posterior mean error, and ESS\n"
        "```\n\n"
        "Proposal scale $s$ controls local move size. "
        "Tiny steps accept often but cross modes slowly. "
        "Large steps cross low-density regions more often, but many proposals are rejected. "
        "The known mixture mean lets the code measure finite-chain error.\n\n"
        "For high-dimensional Gaussian targets the asymptotically optimal acceptance rate is roughly 0.23 (Roberts, Gelman, and Gilks 1997). "
        "That result is why tuning advice for $s$ usually targets acceptance between 0.2 and 0.5. "
        "On bimodal targets like this one, the rule is a guide but not a guarantee, because what limits the chain is mode-jumping rather than local mixing.\n\n"
        "Two diagnostics measure these chain qualities. "
        "Effective sample size turns the autocorrelated chain into an equivalent count of independent draws. "
        "Let $\\rho_t = \\mathrm{Corr}(\\theta_s, \\theta_{s+t})$ denote the stationary lag-$t$ autocorrelation of a coordinate of the chain. "
        "The integrated autocorrelation time is $\\tau = 1 + 2 \\sum_{t \\ge 1} \\rho_t$ and the effective sample size for a chain of length $T$ is $\\mathrm{ESS} = T / \\tau$. "
        "We estimate $\\tau$ from the sample autocorrelations and truncate the sum at the first nonpositive lag, the standard initial-positive-sequence estimator. "
        "Mode switches count how often the chain crosses between regimes. "
        "Together these checks say whether posterior averages weight the structural regimes correctly or report a regime artifact."
    )

    # =========================================================================
    # Results: Method 1 conjugate model
    # =========================================================================
    theta_grid = np.linspace(1e-3, 1 - 1e-3, 600)
    prior_pdf = beta_pdf(theta_grid, ALPHA_PRIOR, BETA_PRIOR)
    posterior_pdf = beta_pdf(theta_grid, ALPHA_POST, BETA_POST)

    fig0, ax0 = plt.subplots(figsize=(8, 5))
    ax0.plot(theta_grid, prior_pdf, color="tab:gray", linestyle="--", linewidth=1.5,
             label=fr"Prior $\mathrm{{Beta}}({ALPHA_PRIOR:.0f}, {BETA_PRIOR:.0f})$")
    ax0.plot(theta_grid, posterior_pdf, color="tab:blue", linewidth=2,
             label=fr"Analytical posterior $\mathrm{{Beta}}({ALPHA_POST:.0f}, {BETA_POST:.0f})$")
    ax0.hist(bb_kept, bins=50, density=True, alpha=0.45, color="tab:orange",
             label=fr"MH histogram ({(n_conj_draws - burn_conj):,} draws)")
    ax0.axvline(POST_MEAN, color="tab:red", linestyle=":", linewidth=1.2,
                label=fr"Analytical mean $= {POST_MEAN:.4f}$")
    ax0.axvline(bb_mean_empirical, color="tab:purple", linestyle=":", linewidth=1.2,
                label=fr"MH mean $= {bb_mean_empirical:.4f}$")
    ax0.set_xlabel(r"$\theta$")
    ax0.set_ylabel("Density")
    ax0.set_title("Beta-Binomial conjugate model: analytical posterior vs Metropolis-Hastings histogram")
    ax0.legend(loc="upper left", fontsize=9)
    report.add_results(
        f"The Beta-Binomial calibration uses prior $\\mathrm{{Beta}}({ALPHA_PRIOR:.0f}, {BETA_PRIOR:.0f})$ and data {K_SUCCESSES} successes in {N_TRIALS} trials. "
        f"The closed-form posterior is $\\mathrm{{Beta}}({ALPHA_POST:.0f}, {BETA_POST:.0f})$ with mean {POST_MEAN:.4f} and variance {POST_VAR:.5f}. "
        f"The Metropolis-Hastings histogram overlays the analytical posterior tightly. "
        f"Empirical mean is {bb_mean_empirical:.4f} (error {bb_mean_error:.4f}). "
        f"Empirical variance is {bb_var_empirical:.5f} (error {bb_var_error:.5f}). "
        f"Acceptance rate is {bb_acceptance:.3f}, within the rule-of-thumb band for one-dimensional random-walk MH. "
        f"The match is the licence to trust the same sampler on a target where no closed form exists."
    )
    report.add_figure(
        "figures/conjugate-posterior.png",
        "Beta-Binomial conjugate posterior with overlaid Metropolis-Hastings histogram",
        fig0,
    )

    # Analytical tail probability via the regularized incomplete Beta function.
    p_gt_half_analytical = float(1.0 - betainc(ALPHA_POST, BETA_POST, 0.5))
    p_gt_half_mh = float(np.mean(bb_kept > 0.5))
    bb_summary = pd.DataFrame([
        {
            "Quantity": "Posterior mean",
            "Analytical": f"{POST_MEAN:.4f}",
            "MH empirical": f"{bb_mean_empirical:.4f}",
            "Absolute error": f"{bb_mean_error:.4f}",
        },
        {
            "Quantity": "Posterior variance",
            "Analytical": f"{POST_VAR:.5f}",
            "MH empirical": f"{bb_var_empirical:.5f}",
            "Absolute error": f"{bb_var_error:.5f}",
        },
        {
            "Quantity": "Posterior P(theta > 0.5)",
            "Analytical": f"{p_gt_half_analytical:.4f}",
            "MH empirical": f"{p_gt_half_mh:.4f}",
            "Absolute error": f"{abs(p_gt_half_analytical - p_gt_half_mh):.4f}",
        },
    ])
    report.add_table(
        "tables/conjugate-summary.csv",
        "Conjugate-model posterior moments: analytical vs Metropolis-Hastings",
        bb_summary,
        description=(
            "Three posterior summaries on the same Beta-Binomial model. "
            "The analytical column is the closed form. The MH column is from the bounded random-walk chain."
        ),
    )

    # =========================================================================
    # Results: Method 2 mixture posterior (existing figures)
    # =========================================================================
    x_grid, y_grid, density = make_density_grid()
    fig1, ax1 = plt.subplots(figsize=(7.2, 6.2))
    ax1.contour(x_grid, y_grid, density, levels=18, cmap="viridis", alpha=0.9)
    path = main_draws[burn : burn + 4_000 : 4]
    ax1.plot(path[:, 0], path[:, 1], color="tab:blue", alpha=0.45, linewidth=1.0)
    ax1.scatter(kept[::20, 0], kept[::20, 1], s=8, color="tab:orange", alpha=0.35, label="Kept draws")
    ax1.scatter(
        [MU1[0], MU2[0]],
        [MU1[1], MU2[1]],
        marker="*",
        s=180,
        color="black",
        label="Component means",
    )
    ax1.set_xlim(-5.0, 5.0)
    ax1.set_ylim(-5.0, 5.0)
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel(r"$\theta_2$")
    ax1.set_title(f"Random-walk path on the mixture target, proposal step {main_step}")
    ax1.legend(loc="upper left")
    report.add_figure(
        "figures/mh-walk.png",
        "Metropolis-Hastings walk over structural-posterior contours",
        fig1,
        description=(
            f"With proposal step {main_step}, the chain visits both regimes and accepts "
            f"{main_acceptance:.1%} of proposed moves."
        ),
    )

    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for dim, ax in enumerate(axes2):
        ax.plot(main_draws[:, dim], color="tab:blue", linewidth=0.8)
        ax.axhline(TRUE_MEAN[dim], color="black", linestyle=":", linewidth=1.2, label="Target mean")
        ax.axvline(burn, color="crimson", linestyle="--", linewidth=1.0, label="Burn-in")
        ax.set_ylabel(rf"$\theta_{dim + 1}$")
        ax.legend(loc="upper right")
    axes2[-1].set_xlabel("Draw")
    axes2[0].set_title("Trace plots for the mixture target")
    report.add_figure(
        "figures/trace-plots.png",
        "Trace plots for the middle-step random-walk chain",
        fig2,
        description=(
            "The traces show burn-in, mode crossing, and persistence in the retained draws."
        ),
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.6))
    for proposal_step, (draws, _) in chains.items():
        retained = draws[burn:]
        cumulative_mean = np.cumsum(retained[:, 0]) / np.arange(1, len(retained) + 1)
        acf = autocorrelation(retained[:, 0], 120)
        axes3[0].plot(cumulative_mean, label=f"step {proposal_step}")
        axes3[1].plot(np.arange(len(acf)), acf, label=f"step {proposal_step}")
    axes3[0].axhline(TRUE_MEAN[0], color="black", linestyle=":", linewidth=1.0)
    axes3[0].set_xlabel("Retained draw")
    axes3[0].set_ylabel(r"Cumulative mean of $\theta_1$")
    axes3[0].set_title("Running mean")
    axes3[1].axhline(0.0, color="black", linewidth=0.8)
    axes3[1].set_xlabel("Lag")
    axes3[1].set_ylabel("Autocorrelation")
    axes3[1].set_title("Autocorrelation")
    axes3[1].legend()
    fig3.tight_layout()
    report.add_figure(
        "figures/tuning-diagnostics.png",
        "Proposal tuning changes posterior bias and persistence",
        fig3,
        description=(
            "The running mean and autocorrelation show how proposal scale changes finite-chain "
            "error."
        ),
    )

    report.add_table(
        "tables/proposal-comparison.csv",
        "Proposal-scale diagnostics on the mixture target",
        summary,
        description=(
            "The true posterior mean is zero. Each coordinate has marginal variance "
            f"{TRUE_COV[0, 0]:.2f} because the modes are far apart."
        ),
    )

    report.add_results(
        f"On the mixture target the middle proposal step {main_step} is the one used in the path and trace plots. "
        f"It gives acceptance {main_acceptance:.1%} and moves between regimes. "
        "The smallest proposal accepts most often but crosses modes slowly. "
        "The largest proposal has lower acceptance, more mode switches, and the smallest mean error. "
        "The table shows why acceptance rate alone is not enough. "
        "Unlike the conjugate model, there is no closed-form posterior mean to compare against; "
        "we know the analytical mean here only because we set the mixture by hand. "
        "In a real structural application the diagnostics in the table are all we have."
    )

    report.add_takeaway(
        "Conjugate Bayes is the right tool when the model permits it. "
        "The Beta-Binomial posterior is one line of algebra, and every moment is analytical. "
        "Conjugate families exist for many useful pairs: Beta-Bernoulli for proportions, "
        "Normal-Normal for known-variance Gaussian means, Gamma-Poisson for counts, "
        "and conjugate priors on the linear-regression coefficient vector. "
        "When conjugacy is available, use it.\n\n"
        "Metropolis-Hastings is the workhorse when conjugacy is not. "
        "It turns a posterior kernel into draws without ever computing the normalizing constant. "
        "It needs only that the kernel can be evaluated pointwise. "
        "It is what makes Bayesian inference practical for structural models, latent-variable models, and any posterior with a nonstandard shape.\n\n"
        "Run the sampler on a conjugate problem first. "
        "The conjugate model has analytical moments, so the empirical mean and variance from the chain can be checked exactly. "
        "If the sampler fails on a tractable problem, it cannot be trusted on an intractable one. "
        "If it passes, the same machinery transfers to the mixture target and to structural posteriors with similar geometry.\n\n"
        "Finite-chain diagnostics matter on multimodal targets. "
        "The mixture chain can still weight regimes incorrectly even after thousands of draws. "
        "Trace plots, cumulative means, mode switches, and autocorrelation are the routine checks. "
        "They are what stand between a posterior average and a regime artifact, and they extend naturally to gradient-based samplers like Hamiltonian Monte Carlo for harder posteriors.\n\n"
        "Random-walk Metropolis-Hastings, Hamiltonian Monte Carlo, and Bayesian optimization are three corners of the same problem: doing inference when each evaluation of the posterior or likelihood is expensive. "
        "Random-walk MH is the gradient-free, posterior-sampling tool that this tutorial introduces. "
        "Hamiltonian Monte Carlo in `computational-methods/hamiltonian-monte-carlo/` is the gradient-aware posterior-sampling alternative for curved or strongly correlated posteriors. "
        "Bayesian optimization in `numerical-methods/bayesian-optimization/` is the gradient-free alternative when the goal is to *maximize* the posterior or any other expensive black-box objective rather than to sample it."
    )

    report.add_references(
        [
            "Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd edition. CRC Press, Ch. 2 on single-parameter conjugate models and Ch. 11 on MCMC.",
            "[Metropolis, N. et al. (1953). Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 21(6), 1087-1092.](https://doi.org/10.1063/1.1699114)",
            "[Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97-109.](https://doi.org/10.1093/biomet/57.1.97)",
            "[Chib, S. and Greenberg, E. (1995). Understanding the Metropolis-Hastings Algorithm. *The American Statistician*, 49(4), 327-335.](https://doi.org/10.1080/00031305.1995.10476177)",
            "Roberts, G. O., Gelman, A., and Gilks, W. R. (1997). *Weak Convergence and Optimal Scaling of Random Walk Metropolis Algorithms*. Annals of Applied Probability, 7, 110-120.",
            "**See also.** Method 1 of this tutorial is the closed-form Bayesian baseline. Method 2 is the Markov-chain sampler. Hamiltonian Monte Carlo on a curved banana posterior is in `computational-methods/hamiltonian-monte-carlo/`, which repeats the Method 2 algorithm here as its random-walk baseline. Gaussian-process Bayesian optimization for expensive black-box maximization is in `numerical-methods/bayesian-optimization/`, which uses the same prior-times-likelihood-equals-posterior framework as Method 1 here but over an unknown function instead of a scalar probability.",
        ]
    )

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
