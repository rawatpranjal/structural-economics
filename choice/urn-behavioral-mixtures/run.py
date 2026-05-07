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
        "state_high": state_high,
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


def bayes_diagnostics(
    n_draws: np.ndarray,
    k_red: np.ndarray,
    state_high: np.ndarray,
    prior_h: float,
    p_red_h: float,
    p_red_l: float,
) -> pd.DataFrame:
    """Task-level sufficient statistics and exact Bayesian classifications."""
    posterior = posterior_high(k_red, n_draws, prior_h, p_red_h, p_red_l)
    log_lr = log_likelihood_ratio(k_red, n_draws, p_red_h, p_red_l)
    bayes_choice = (posterior >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "Task": np.arange(1, len(n_draws) + 1),
            "Draws": n_draws,
            "Red count": k_red,
            "True high urn": state_high,
            "Log likelihood ratio": log_lr,
            "Posterior high": posterior,
            "Bayes choice high": bayes_choice,
            "Bayes correct": bayes_choice == state_high,
        }
    )


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
    bayes_table = bayes_diagnostics(
        panel["n_draws"],
        panel["k_red"],
        panel["state_high"],
        prior_h,
        p_red_h,
        p_red_l,
    )
    bayes_accuracy = float(bayes_table["Bayes correct"].mean())
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
        "Urn Choices and Latent Decision Rules",
        "Bayesian learning benchmarks and EM mixtures for repeated urn classifications.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Consider a lab task in which a subject sees a small sample from one of two urns "
        "and must decide whether the urn is the high-red state. The setting is attractive "
        "for studying belief-based choice because the Bayesian benchmark is fully pinned "
        "down. For any red count and sample size, Bayes' rule gives the posterior "
        "probability of the high state.\n\n"
        "Repeated choices make the inference problem more interesting. The researcher "
        "observes a sequence of high-urn choices, not the rule each subject used. The code "
        "first turns each signal into a likelihood-ratio state variable. It then estimates "
        "a finite mixture by EM, treating each person's decision rule as a latent class. "
        "That gives the researcher two outputs: population shares for the rules and "
        "subject-level probabilities over candidate behavioral types."
    )

    report.add_equations(
        r"""
Let $H$ denote the high-red urn and $L$ the low-red urn. A task draws $n$ balls
and observes $k$ red balls. The likelihood-ratio statistic is

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
$q_m(k,n)$. With subject $i$'s choices $d_{it}\in\{0,1\}$, the panel likelihood
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
        "The calculation has two layers. First, each urn task is reduced to a likelihood "
        "ratio, so the Bayesian benchmark depends on $(k,n)$ rather than the full signal "
        "history. Second, EM estimates the population shares of the candidate rules. Each "
        "E step computes the probability that a subject followed each rule, and each M "
        "step averages those probabilities into new mixture weights.\n\n"
        "```text\n"
        "Algorithm: EM for latent decision rules\n"
        "Input: repeated choices d_it, task counts (k_t, n_t), candidate rules m=1,...,M\n"
        "1. For each task, compute Lambda(k_t,n_t)\n"
        "2. For each rule, compute q_m(k_t,n_t), the probability of choosing high\n"
        "3. Initialize weights w_m = 1/M\n"
        "4. Repeat until the log likelihood changes by less than the tolerance:\n"
        "   E step: tau_im = w_m L_im / sum_h w_h L_ih\n"
        "   M step: w_m = mean_i tau_im\n"
        "5. Assign each subject to argmax_m tau_im for diagnostics\n"
        "Output: mixture shares, posterior responsibilities, allocation accuracy\n"
        "```\n\n"
        "Because the rule-specific choice probabilities are fixed here, the M step is just "
        "a normalized average of responsibilities. With unknown cutoff points or response "
        "noise, the same likelihood would optimize those rule-specific parameters inside "
        "the M step."
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
        "The likelihood ratio is the sufficient statistic for the Bayesian state "
        f"classification. With {display_draws} draws, three red balls put the posterior "
        "above one half but below the conservative 0.75 cutoff. Those middle signals "
        "help the repeated-choice panel distinguish exact Bayesian updating from a "
        "stricter decision rule."
    )
    report.add_figure(
        "figures/bayes-likelihood-ratio.png",
        "Bayesian posterior as a function of the likelihood ratio",
        fig1,
        description="Exact Bayesian updating converts the signal count into a posterior belief before classification.",
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
        "The EM estimator recovers the population shares of the fixed candidate rules. "
        f"The L1 distance between estimated and true weights is **{weight_l1:.3f}**. "
        "The exercise keeps the candidate rules fixed, so the estimate answers a concrete "
        "heterogeneity question: what share of subjects behave like each rule?"
    )
    report.add_figure(
        "figures/mixture-weights.png",
        "True and estimated latent rule shares",
        fig2,
        description="Mixture weights are recovered from repeated choices, not from observing rule labels.",
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.hist(max_posterior, bins=np.linspace(0, 1, 21), color="tab:green", alpha=0.85)
    ax3.set_xlabel("Largest posterior responsibility")
    ax3.set_ylabel("Subjects")
    ax3.set_title("Classification Confidence")
    report.add_results(
        "Posterior responsibilities measure subject-level classification confidence. "
        "A subject with choices that several rules can explain receives diffuse "
        "responsibilities, even if the aggregate mixture weights are accurate. The hard "
        f"allocation accuracy in this run is **{type_accuracy:.3f}**. In the simulated "
        f"task menu, Bayes differs from the conservative rule on {bayes_conservative_split} "
        f"tasks, from the red-share rule on {bayes_share_split} tasks, and from the "
        f"raw-count rule on {bayes_count_split} tasks."
    )
    report.add_figure(
        "figures/classification-confidence.png",
        "Posterior confidence in assigned latent rule",
        fig3,
        description="The distribution of max responsibilities separates confident rule assignments from ambiguous choice histories.",
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
            f"The EM likelihood converged in {estimates['iterations']} iterations with "
            f"log likelihood {float(estimates['log_likelihood']):.2f}."
        ),
    )

    report.add_table(
        "tables/type-allocation.csv",
        "True versus assigned latent rule counts",
        allocation_table(panel["type_id"], assigned),
        description=(
            "Rows are true simulated rules and columns are posterior-modal assignments. "
            "The labels are known only because this is a Monte Carlo tutorial."
        ),
    )

    task_table = bayes_table.head(12).copy()
    report.add_table(
        "tables/bayes-task-diagnostics.csv",
        "Bayesian classifier diagnostics for the first twelve tasks",
        task_table.round({"Log likelihood ratio": 4, "Posterior high": 4}),
        description=(
            f"The exact Bayes classifier is correct on {bayes_accuracy:.1%} of the task "
            "states in this simulated set. The table keeps the likelihood-ratio statistic "
            "visible because it is the state variable for the later rule estimator."
        ),
    )

    diagnostics = pd.DataFrame(
        {
            "Diagnostic": [
                "EM converged",
                "EM iterations",
                "Mixture weight L1 error",
                "Hard type allocation accuracy",
                "Mean max posterior responsibility",
                "Bayes task-state accuracy",
            ],
            "Value": [
                float(bool(estimates["converged"])),
                float(estimates["iterations"]),
                weight_l1,
                type_accuracy,
                float(np.mean(max_posterior)),
                bayes_accuracy,
            ],
        }
    )
    report.add_table(
        "tables/estimator-diagnostics.csv",
        "Estimator and known-truth diagnostics",
        diagnostics,
        description="The diagnostics separate aggregate share recovery from individual-level type classification.",
    )

    report.add_takeaway(
        "Likelihood ratios turn each urn signal into a belief benchmark. The EM mixture "
        "then turns repeated choices into estimates of latent behavioral-rule shares. "
        "This is the reusable lesson for nearby choice data: when several simple rules "
        "are economically meaningful, a finite mixture can estimate population shares "
        "and show which individuals are hard to classify."
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
