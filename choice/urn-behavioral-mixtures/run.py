#!/usr/bin/env python3
"""Urn choices, Bayesian learning, and finite mixtures of behavioral rules."""
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


@dataclass(frozen=True)
class Rule:
    """Candidate latent decision rule."""

    name: str
    short: str
    description: str


RULES = [
    Rule("Bayesian likelihood-ratio", "Bayes", "Choose high if the posterior probability of the high urn is at least one half."),
    Rule("Conservative Bayes", "Conservative", "Choose high only when the posterior probability of the high urn is at least 0.75."),
    Rule("Red-share cutoff", "Share cutoff", "Choose high when at least half of sampled balls are red."),
    Rule("Red-count cutoff", "Count cutoff", "Choose high when at least four sampled balls are red, ignoring sample size."),
]


def log_likelihood_ratio(k_red: np.ndarray, n_draws: np.ndarray, p_red_h: float, p_red_l: float) -> np.ndarray:
    """Return log Pr(signals | H) / Pr(signals | L)."""
    return (
        k_red * np.log(p_red_h / p_red_l)
        + (n_draws - k_red) * np.log((1.0 - p_red_h) / (1.0 - p_red_l))
    )


def posterior_high(
    k_red: np.ndarray,
    n_draws: np.ndarray,
    prior_h: float,
    p_red_h: float,
    p_red_l: float,
) -> np.ndarray:
    """Exact posterior probability of the high-red urn."""
    log_prior_odds = np.log(prior_h / (1.0 - prior_h))
    return expit(log_prior_odds + log_likelihood_ratio(k_red, n_draws, p_red_h, p_red_l))


def deterministic_rule_choices(
    k_red: np.ndarray,
    n_draws: np.ndarray,
    prior_h: float,
    p_red_h: float,
    p_red_l: float,
) -> np.ndarray:
    """Return deterministic high-urn choices for each candidate rule."""
    posterior = posterior_high(k_red, n_draws, prior_h, p_red_h, p_red_l)
    choices = np.zeros((len(RULES), len(k_red)), dtype=int)
    choices[0] = posterior >= 0.5
    choices[1] = posterior >= 0.75
    choices[2] = k_red / n_draws >= 0.5
    choices[3] = k_red >= 4
    return choices


def rule_choice_probabilities(rule_choices: np.ndarray, tremble: float) -> np.ndarray:
    """Turn deterministic rules into choice probabilities with symmetric trembles."""
    return tremble + (1.0 - 2.0 * tremble) * rule_choices


def simulate_panel(
    rng: np.random.Generator,
    n_subjects: int,
    n_tasks: int,
    draw_counts: np.ndarray,
    draw_probabilities: np.ndarray,
    true_weights: np.ndarray,
    prior_h: float,
    p_red_h: float,
    p_red_l: float,
    tremble: float,
) -> dict[str, np.ndarray]:
    """Simulate repeated urn choices from a finite mixture of rules."""
    n_draws = rng.choice(draw_counts, size=n_tasks, p=draw_probabilities)
    state_high = rng.binomial(1, prior_h, size=n_tasks)
    p_red = np.where(state_high == 1, p_red_h, p_red_l)
    k_red = rng.binomial(n_draws, p_red)
    rule_choices = deterministic_rule_choices(k_red, n_draws, prior_h, p_red_h, p_red_l)
    rule_probs = rule_choice_probabilities(rule_choices, tremble)

    type_id = rng.choice(len(RULES), size=n_subjects, p=true_weights)
    choices = np.zeros((n_subjects, n_tasks), dtype=int)
    for i, m in enumerate(type_id):
        choices[i] = rng.binomial(1, rule_probs[m])

    return {
        "n_draws": n_draws,
        "k_red": k_red,
        "type_id": type_id,
        "choices": choices,
        "rule_choices": rule_choices,
        "rule_probs": rule_probs,
    }


def individual_log_likelihood(choices: np.ndarray, rule_probs: np.ndarray) -> np.ndarray:
    """Log likelihood for each individual under each latent rule."""
    p = np.clip(rule_probs, 1e-8, 1.0 - 1e-8)
    y = choices[:, None, :]
    return np.sum(y * np.log(p[None, :, :]) + (1 - y) * np.log(1.0 - p[None, :, :]), axis=2)


def estimate_mixture_weights(
    choices: np.ndarray,
    rule_probs: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1_000,
) -> dict[str, np.ndarray | float | int | bool]:
    """Estimate latent rule shares by EM with fixed rule likelihoods."""
    n_rules = rule_probs.shape[0]
    weights = np.full(n_rules, 1.0 / n_rules)
    log_like_by_rule = individual_log_likelihood(choices, rule_probs)
    old_ll = -np.inf

    for iteration in range(1, max_iter + 1):
        log_joint = log_like_by_rule + np.log(weights)[None, :]
        log_den = logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - log_den[:, None])
        weights = responsibilities.mean(axis=0)
        log_likelihood = float(np.sum(log_den))
        improvement = log_likelihood - old_ll
        if abs(improvement) < tol:
            break
        old_ll = log_likelihood

    return {
        "weights": weights,
        "responsibilities": responsibilities,
        "log_likelihood": log_likelihood,
        "iterations": iteration,
        "converged": abs(improvement) < tol,
    }


def allocation_table(true_type: np.ndarray, assigned_type: np.ndarray) -> pd.DataFrame:
    """Confusion matrix for true versus assigned latent rule."""
    matrix = np.zeros((len(RULES), len(RULES)), dtype=int)
    for t, a in zip(true_type, assigned_type):
        matrix[t, a] += 1
    df = pd.DataFrame(matrix, columns=[rule.short for rule in RULES])
    df.insert(0, "True rule", [rule.short for rule in RULES])
    return df


def main() -> None:
    rng = np.random.default_rng(1234)
    n_subjects = 600
    n_tasks = 60
    prior_h = 0.45
    p_red_h = 0.72
    p_red_l = 0.32
    tremble = 0.06
    draw_counts = np.array([3, 4, 5, 6, 7, 8, 9, 12])
    draw_probabilities = np.array([0.12, 0.10, 0.16, 0.10, 0.16, 0.10, 0.16, 0.10])
    true_weights = np.array([0.46, 0.24, 0.20, 0.10])

    panel = simulate_panel(
        rng,
        n_subjects,
        n_tasks,
        draw_counts,
        draw_probabilities,
        true_weights,
        prior_h,
        p_red_h,
        p_red_l,
        tremble,
    )
    estimates = estimate_mixture_weights(panel["choices"], panel["rule_probs"])
    weights_hat = np.asarray(estimates["weights"])
    responsibilities = np.asarray(estimates["responsibilities"])
    assigned = np.argmax(responsibilities, axis=1)
    max_posterior = np.max(responsibilities, axis=1)
    type_accuracy = float(np.mean(assigned == panel["type_id"]))
    weight_l1 = float(np.sum(np.abs(weights_hat - true_weights)))
    rule_choices = np.asarray(panel["rule_choices"])
    bayes_conservative_split = int(np.sum(rule_choices[0] != rule_choices[1]))
    bayes_share_split = int(np.sum(rule_choices[0] != rule_choices[2]))
    bayes_count_split = int(np.sum(rule_choices[0] != rule_choices[3]))

    print("Urn behavioral mixtures tutorial")
    print(f"  EM converged: {estimates['converged']} in {estimates['iterations']} iterations")
    print(f"  Type allocation accuracy: {type_accuracy:.3f}")
    print(f"  Weight L1 error: {weight_l1:.3f}")

    setup_style()
    report = ModelReport(
        "Are People Bayesian? Decision-Rule Mixtures via EM",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "In an urn experiment, a subject sees a small sample from one of two urns. "
        "The choice is whether the hidden state is high red.\n\n"
        "The object is a decision rule. It maps red counts and sample sizes into "
        "a high-urn choice.\n\n"
        "Bayes' rule gives a benchmark for each task. Repeated choices need EM "
        "because the researcher observes choices, not each subject's rule."
    )

    report.add_equations(
        r"""
Let $H$ denote the high-red urn and $L$ the low-red urn. A task draws $n$ balls
and observes $k$ red balls. Let $p_H$ and $p_L$ denote the red-ball probability
under urns $H$ and $L$. The likelihood-ratio statistic is

$$
\Lambda(k,n)
= \log \frac{\Pr(k\mid H,n)}{\Pr(k\mid L,n)}
= k\log\frac{p_H}{p_L} + (n-k)\log\frac{1-p_H}{1-p_L}.
$$

With prior $\pi_0=\Pr(H)$, Bayes' rule is

$$
\Pr(H\mid k,n)
= \frac{1}{1+\exp[-\{\log(\pi_0/(1-\pi_0))+\Lambda(k,n)\}]}.
$$

Rule $m$ maps the sufficient statistic and counts into a choice probability
$q_m(k,n)$. With subject $i$'s choices $d_{it}\in\{0,1\}$ (where $t$ indexes tasks), the panel likelihood
under rule $m$ is

$$
L_{im}
= \prod_t q_m(k_t,n_t)^{d_{it}}
[1-q_m(k_t,n_t)]^{1-d_{it}}.
$$

The finite-mixture likelihood is

$$
\ell(w)=\sum_i \log\left[\sum_m w_m L_{im}\right],
\qquad \sum_m w_m=1,\quad w_m\geq 0.
$$

The posterior probability that subject $i$ follows rule $m$ is

$$
\tau_{im}
= \frac{w_m L_{im}}{\sum_h w_h L_{ih}}.
$$

Here $h$ is a summation index ranging over the same rule set as $m$.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| Subjects | {n_subjects} | Repeated-choice panel units |\n"
        f"| Tasks per subject | {n_tasks} | Variation used to classify latent rules |\n"
        f"| Prior high urn | {prior_h:.2f} | Baseline probability of state $H$ |\n"
        f"| Red probability under $H$ | {p_red_h:.2f} | Signal distribution for high urn |\n"
        f"| Red probability under $L$ | {p_red_l:.2f} | Signal distribution for low urn |\n"
        f"| Draw counts | {', '.join(str(x) for x in draw_counts)} | Signal-size variation separates Bayes and cutoff rules |\n"
        f"| Bayes-conservative separating tasks | {bayes_conservative_split} | Tasks with posterior between the two decision cutoffs |\n"
        f"| Tremble rate | {tremble:.2f} | Symmetric error around each deterministic rule |\n"
        f"| Latent rules | {len(RULES)} | Bayesian and cutoff decision types |"
    )

    report.add_solution_method(
        "Each task is first reduced to the log likelihood ratio. This statistic is "
        "enough for the Bayesian posterior.\n\n"
        "EM then estimates the shares of fixed "
        "candidate rules.\n\n"
        "```text\n"
        "Algorithm: EM for latent decision rules\n"
        "Input: repeated choices d_it, task counts (k_t, n_t), candidate rules m=1,...,M\n"
        "1. For each task, compute Lambda(k_t,n_t)\n"
        "2. For each rule, compute q_m(k_t,n_t), the probability of choosing high\n"
        "3. Initialize weights w_m = 1/M\n"
        "4. Repeat until the log likelihood changes by less than the tolerance:\n"
        "   E step: tau_im = w_m L_im / sum_h w_h L_ih\n"
        "   M step: w_m = mean_i tau_im\n"
        "5. Assign each subject to argmax_m tau_im\n"
        "Output: mixture shares, posterior responsibilities, allocation accuracy\n"
        "```"
    )

    display_draws = 5
    k_grid = np.arange(0, display_draws + 1)
    n_grid = np.full_like(k_grid, display_draws)
    posterior_grid = posterior_high(k_grid, n_grid, prior_h, p_red_h, p_red_l)
    log_lr_grid = log_likelihood_ratio(k_grid, n_grid, p_red_h, p_red_l)

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(log_lr_grid, posterior_grid, marker="o")
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Bayes cutoff")
    ax1.axhline(0.75, color="tab:orange", linestyle=":", linewidth=1.5, label="Conservative cutoff")
    for k, x_pos, y_pos in zip(k_grid, log_lr_grid, posterior_grid):
        ax1.text(x_pos, y_pos + 0.025, str(k), ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("Log likelihood ratio")
    ax1.set_ylabel("Posterior probability of high urn")
    ax1.set_title("Bayesian State Variable")
    ax1.legend()
    report.add_results(
        "The likelihood ratio orders tasks by evidence for the high-red urn. "
        f"With {display_draws} draws, a count of three red balls crosses the Bayes threshold. "
        "It does not cross the conservative cutoff. Such tasks separate exact "
        "Bayesian updating from stricter rules."
    )
    report.add_figure(
        "figures/bayes-likelihood-ratio.png",
        "Bayesian posterior as a function of the likelihood ratio",
        fig1,
        description="Signal counts become posterior beliefs before classification.",
    )

    x_pos = np.arange(len(RULES))
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    ax2.bar(x_pos - width / 2, true_weights, width, label="True")
    ax2.bar(x_pos + width / 2, weights_hat, width, label="Estimated")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([rule.short for rule in RULES], rotation=20, ha="right")
    ax2.set_ylabel("Mixture weight")
    ax2.set_title("Latent Rule Shares")
    ax2.legend()
    report.add_results(
        "EM estimates the population share of each fixed rule. "
        f"The L1 distance between estimated and true weights is **{weight_l1:.3f}**. "
        "The estimate answers one heterogeneity question: how many subjects behave "
        "like each rule?"
    )
    report.add_figure(
        "figures/mixture-weights.png",
        "True and estimated latent rule shares",
        fig2,
        description="Repeated choices identify shares without observing rule labels.",
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.hist(max_posterior, bins=np.linspace(0, 1, 21), color="tab:green", alpha=0.85)
    ax3.set_xlabel("Largest posterior responsibility")
    ax3.set_ylabel("Subjects")
    ax3.set_title("Classification Confidence")
    report.add_results(
        "Responsibilities give subject-level rule probabilities. Diffuse responsibilities "
        "mark choice histories that several rules can explain. Hard allocation accuracy "
        f"is **{type_accuracy:.3f}**. Bayes differs from the conservative rule on "
        f"{bayes_conservative_split} tasks. It differs from the red-share rule on "
        f"{bayes_share_split} tasks and the raw-count rule on {bayes_count_split} tasks."
    )
    report.add_figure(
        "figures/classification-confidence.png",
        "Posterior confidence in assigned latent rule",
        fig3,
        description="Max responsibilities show how confident the type assignment is.",
    )

    weights_table = pd.DataFrame(
        {
            "Rule": [rule.short for rule in RULES],
            "Definition": [rule.description for rule in RULES],
            "True weight": true_weights,
            "Estimated weight": weights_hat,
            "Error": weights_hat - true_weights,
        }
    )
    report.add_table(
        "tables/mixture-weights.csv",
        "Latent rule weight recovery",
        weights_table.round({"True weight": 4, "Estimated weight": 4, "Error": 4}),
        description=(
            f"EM converges in {estimates['iterations']} iterations; log likelihood is "
            f"{float(estimates['log_likelihood']):.2f}."
        ),
    )

    report.add_table(
        "tables/type-allocation.csv",
        "True versus assigned latent rule counts",
        allocation_table(panel["type_id"], assigned),
        description=(
            "Rows are true simulated rules. Columns are posterior-modal assignments."
        ),
    )

    report.add_takeaway(
        "The likelihood ratio gives a task-level belief benchmark. EM uses repeated "
        "choices to estimate shares of latent decision rules. The method is useful "
        "when simple rules are meaningful but rule labels are unobserved."
    )

    report.add_references(
        [
            "[El-Gamal, M. A. and Grether, D. M. (1995). Are People Bayesian? Uncovering Behavioral Strategies. *Journal of the American Statistical Association*, 90(432), 1137-1145.](https://doi.org/10.1080/01621459.1995.10476622)",
            "[McLachlan, G. and Peel, D. (2000). *Finite Mixture Models*. Wiley.](https://doi.org/10.1002/0471721182)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
