#!/usr/bin/env python3
"""Risk aversion and monotone stochastic choice in lottery panels."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def crra_utility(x: np.ndarray | float, rho: float) -> np.ndarray | float:
    """CRRA utility for positive lottery prizes."""
    if abs(rho - 1.0) < 1e-8:
        return np.log(x)
    return (np.asarray(x) ** (1.0 - rho) - 1.0) / (1.0 - rho)


def lottery_eu_difference(prob_high: np.ndarray, rho: float) -> np.ndarray:
    """Expected utility of risky lottery B minus safer lottery A."""
    a_high, a_low = 2.00, 1.60
    b_high, b_low = 3.85, 0.10
    eu_a = prob_high * crra_utility(a_high, rho) + (1.0 - prob_high) * crra_utility(a_low, rho)
    eu_b = prob_high * crra_utility(b_high, rho) + (1.0 - prob_high) * crra_utility(b_low, rho)
    return np.asarray(eu_b - eu_a, dtype=float)


def structural_probabilities(
    prob_high: np.ndarray,
    rho: float,
    scale: float,
    lapse: float,
) -> np.ndarray:
    """Risky choice probabilities from fixed scale structural logit."""
    latent = scale * lottery_eu_difference(prob_high, rho)
    return lapse + (1.0 - 2.0 * lapse) * expit(latent)


def log_likelihood(counts: np.ndarray, trials: np.ndarray, probabilities: np.ndarray) -> float:
    """Binomial log likelihood up to constants."""
    p = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return float(np.sum(counts * np.log(p) + (trials - counts) * np.log(1.0 - p)))


def estimate_unconstrained_logits(counts: np.ndarray, trials: np.ndarray) -> dict[str, np.ndarray | float]:
    """Saturated task-level logit estimates."""
    shares = np.clip(counts / trials, 1e-5, 1.0 - 1e-5)
    logits = logit(shares)
    return {
        "logits": logits,
        "probabilities": shares,
        "log_likelihood": log_likelihood(counts, trials, shares),
        "violations": float(np.sum(np.diff(shares) < -1e-12)),
    }


def estimate_fixed_scale_crra(
    counts: np.ndarray,
    trials: np.ndarray,
    prob_high: np.ndarray,
    scale: float,
    lapse: float,
) -> dict[str, float | np.ndarray | bool]:
    """Estimate the risk-aversion parameter with the scale fixed."""

    def objective(rho: float) -> float:
        probabilities = structural_probabilities(prob_high, rho, scale, lapse)
        return -log_likelihood(counts, trials, probabilities)

    result = minimize_scalar(objective, bounds=(-0.50, 2.00), method="bounded", options={"xatol": 1e-10})
    probabilities = structural_probabilities(prob_high, float(result.x), scale, lapse)
    return {
        "rho": float(result.x),
        "probabilities": probabilities,
        "log_likelihood": log_likelihood(counts, trials, probabilities),
        "success": bool(result.success),
        "violations": float(np.sum(np.diff(probabilities) < -1e-12)),
    }


def estimate_monotone_logits(counts: np.ndarray, trials: np.ndarray) -> dict[str, np.ndarray | float | bool | int]:
    """Constrained row logits with risky choice probabilities nondecreasing."""
    start = np.asarray(estimate_unconstrained_logits(counts, trials)["logits"], dtype=float)

    def objective(alpha: np.ndarray) -> float:
        return -log_likelihood(counts, trials, expit(alpha))

    constraints = [{"type": "ineq", "fun": lambda alpha, j=j: alpha[j + 1] - alpha[j]} for j in range(len(start) - 1)]
    result = minimize(
        objective,
        start,
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500, "disp": False},
    )
    probabilities = expit(result.x)
    residuals = np.diff(result.x)
    return {
        "logits": result.x,
        "probabilities": probabilities,
        "log_likelihood": log_likelihood(counts, trials, probabilities),
        "success": bool(result.success),
        "iterations": int(result.nit),
        "violations": float(np.sum(np.diff(probabilities) < -1e-10)),
        "min_spacing": float(np.min(residuals)),
    }


def simulate_choices(
    rng: np.random.Generator,
    prob_high: np.ndarray,
    trials_per_task: int,
    rho_true: float,
    scale_true: float,
    lapse: float,
) -> dict[str, np.ndarray]:
    """Simulate risky-lottery counts by task."""
    probabilities = structural_probabilities(prob_high, rho_true, scale_true, lapse)
    trials = np.full(len(prob_high), trials_per_task)
    counts = rng.binomial(trials, probabilities)
    return {"probabilities": probabilities, "counts": counts, "trials": trials}


def model_summary(
    true_prob: np.ndarray,
    counts: np.ndarray,
    trials: np.ndarray,
    unconstrained: dict[str, np.ndarray | float],
    fixed_scale: dict[str, np.ndarray | float | bool],
    monotone: dict[str, np.ndarray | float | bool | int],
    rho_true: float,
) -> pd.DataFrame:
    """Collect comparison metrics for the three estimators."""
    rows = []
    for name, result in [
        ("Unconstrained task logit", unconstrained),
        ("Fixed scale CRRA logit", fixed_scale),
        ("Monotone row logit", monotone),
    ]:
        probabilities = np.asarray(result["probabilities"], dtype=float)
        rows.append(
            {
                "Model": name,
                "Log likelihood": float(result["log_likelihood"]),
                "Monotonicity violations": int(result["violations"]),
                "Max probability error": float(np.max(np.abs(probabilities - true_prob))),
                "Estimated rho": float(result["rho"]) if "rho" in result else np.nan,
                "Rho error": float(result["rho"] - rho_true) if "rho" in result else np.nan,
            }
        )
    saturated_ll = float(unconstrained["log_likelihood"])
    for row in rows:
        row["LL loss vs saturated"] = saturated_ll - row["Log likelihood"]
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(1)
    prob_high = np.arange(0.10, 1.01, 0.10)
    trials_per_task = 80
    rho_true = 0.45
    scale_true = 5.00
    fixed_scale = 5.00
    lapse = 0.02

    simulated = simulate_choices(rng, prob_high, trials_per_task, rho_true, scale_true, lapse)
    counts = simulated["counts"]
    trials = simulated["trials"]
    true_prob = simulated["probabilities"]
    observed_share = counts / trials

    unconstrained = estimate_unconstrained_logits(counts, trials)
    fixed = estimate_fixed_scale_crra(counts, trials, prob_high, fixed_scale, lapse)
    monotone = estimate_monotone_logits(counts, trials)
    summary = model_summary(true_prob, counts, trials, unconstrained, fixed, monotone, rho_true)

    print("Risk aversion monotone choice tutorial")
    print(f"  True rho: {rho_true:.3f}")
    print(f"  Fixed scale rho estimate: {float(fixed['rho']):.3f}")
    print(f"  Observed monotonicity violations: {int(unconstrained['violations'])}")
    print(f"  Monotone optimizer success: {monotone['success']}")

    setup_style()
    report = ModelReport(
        "Lottery Risk Aversion with Monotone Choice",
        "Estimate risk aversion from a lottery ladder with ordered choice probabilities.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A Holt-Laury ladder gives subjects repeated choices between a safer lottery and a "
        "riskier lottery.\n\n"
        "Each row changes only the probability of the high payoff. The object is the "
        "risky choice curve across rows.\n\n"
        "Finite samples can make that curve fall in one row. We estimate CRRA risk aversion "
        "and enforce monotone row probabilities."
    )

    report.add_equations(
        r"""
At row $j$, the high payoff probability is $p_j$. The subject chooses between a
safer lottery $A$ and a riskier lottery $B$:

$$
A(p)=(2.00 \text{ with } p,\ 1.60 \text{ otherwise}),\qquad
B(p)=(3.85 \text{ with } p,\ 0.10 \text{ otherwise}).
$$

For CRRA utility,

$$
u(c;\rho)=\frac{c^{1-\rho}-1}{1-\rho},
\qquad \rho\neq 1.
$$

The expected utility index for choosing the risky lottery is

$$
\Delta EU(p;\rho)=E[u(B(p);\rho)]-E[u(A(p);\rho)].
$$

The fixed scale choice model is

$$
\Pr(d=1\mid p;\rho)
= \lambda + (1-2\lambda)\frac{1}{1+\exp[-s\,\Delta EU(p;\rho)]}.
$$

The monotone row logit estimates one logit $\alpha_j$ per row from binomial
counts $y_j$ out of $N_j$ choices:

$$
\ell(\alpha)=\sum_j y_j\log\Lambda(\alpha_j) + (N_j-y_j)\log[1-\Lambda(\alpha_j)].
$$

The shape restriction is

$$
\alpha_{j+1}\geq \alpha_j
\quad \text{for all adjacent rows }j.
$$

Since $\Lambda$ is monotone, probability ordering follows from logit ordering.
This gives
$\Pr(d=1\mid p_{j+1})\geq \Pr(d=1\mid p_j)$.
"""
    )

    report.add_model_setup(
        f"| Primitive | Value | Economic role |\n"
        f"|--------|-------|------|\n"
        f"| Lottery rows | {len(prob_high)} | Probability ladder from 0.10 to 1.00 |\n"
        f"| Choices per row | {trials_per_task} | Binomial observations for each lottery pair |\n"
        f"| True risk aversion | {rho_true:.2f} | Data-generating CRRA curvature |\n"
        f"| True scale | {scale_true:.2f} | Maps utility differences into stochastic choice |\n"
        f"| Lapse rate | {lapse:.2f} | Symmetric lower and upper error floor |\n"
        f"| Fixed scale estimator | scale = {fixed_scale:.2f} | Recovers $\\rho$ from the payoff index |\n"
        f"| Shape restriction | nondecreasing | Risky-choice probability cannot fall as $p$ rises |"
    )

    report.add_solution_method(
        "The CRRA logit is the structural fit. It searches over $\\rho$ after the stochastic "
        "scale is fixed.\n\n"
        "The monotone logit is the flexible fit. It maximizes the binomial likelihood "
        "subject to ordered row logits.\n\n"
        "```text\n"
        "Algorithm: monotone lottery-choice estimation\n"
        "Input: rows (p_j, y_j, N_j), fixed scale s, lapse rate lambda\n"
        "1. Simulate risky choice counts from CRRA probabilities.\n"
        "2. Estimate saturated logits for each row.\n"
        "3. Search over rho for the fixed scale CRRA likelihood.\n"
        "4. Estimate row logits subject to alpha_{j+1} >= alpha_j.\n"
        "5. Compare fitted curves and likelihood loss.\n"
        "Output: fitted risky choice curves and model comparisons\n"
        "```"
    )

    fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(prob_high, true_prob, marker="o", label="True")
    ax1.scatter(prob_high, observed_share, color="black", zorder=3, label="Observed shares")
    ax1.plot(prob_high, np.asarray(unconstrained["probabilities"]), "--", label="Unconstrained")
    ax1.plot(prob_high, np.asarray(fixed["probabilities"]), ":", linewidth=2.5, label="Fixed scale CRRA")
    ax1.plot(prob_high, np.asarray(monotone["probabilities"]), "-.", label="Monotone")
    ax1.set_xlabel("Probability of high payoff")
    ax1.set_ylabel("Probability of choosing risky lottery")
    ax1.set_title("Risky Choice Along the Lottery Ladder")
    ax1.legend()
    report.add_results(
        "At low high payoff probabilities, few subjects choose the risky lottery. "
        "The observed share falls from $p=0.20$ to $p=0.30$. "
        "The unconstrained fit repeats that step. "
        "The monotone fit pools the two rows."
    )
    report.add_figure(
        "figures/risky-choice-fits.png",
        "Observed and fitted risky choice probabilities",
        fig1,
        description="The constrained curve removes the sample reversal.",
    )

    rho_grid = np.linspace(-0.25, 1.25, 220)
    ll_grid = np.array([
        log_likelihood(counts, trials, structural_probabilities(prob_high, rho, fixed_scale, lapse))
        for rho in rho_grid
    ])
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(rho_grid, ll_grid)
    ax2.axvline(rho_true, color="black", linestyle="--", linewidth=1.2, label="True rho")
    ax2.axvline(float(fixed["rho"]), color="tab:red", linestyle=":", linewidth=2, label="Estimate")
    ax2.set_xlabel("Risk aversion parameter rho")
    ax2.set_ylabel("Log likelihood")
    ax2.set_title("Fixed Scale Structural Likelihood")
    ax2.legend()
    report.add_results(
        "The CRRA likelihood uses payoffs rather than row labels. "
        f"With scale fixed at {fixed_scale:.1f}, the estimate is "
        f"**{float(fixed['rho']):.3f}**. "
        f"The true value is **{rho_true:.3f}**."
    )
    report.add_figure(
        "figures/rho-likelihood.png",
        "Likelihood over CRRA risk aversion",
        fig2,
        description="The fixed scale turns the ladder into a likelihood for rho.",
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    eu_true = lottery_eu_difference(prob_high, rho_true)
    ax3.axhline(0, color="black", linewidth=1.0)
    ax3.plot(prob_high, eu_true, marker="o")
    ax3.set_xlabel("Probability of high payoff")
    ax3.set_ylabel("EU(B) minus EU(A)")
    ax3.set_title("Expected-Utility Index")
    report.add_results(
        "For the true $\\rho$, the expected utility difference rises with the high payoff "
        "probability. "
        "It crosses zero where the risky lottery first gives higher expected utility. "
        "The monotone fit uses only this ordering implication."
    )
    report.add_figure(
        "figures/eu-index.png",
        "Expected-utility difference for the true CRRA parameter",
        fig3,
        description="The risky lottery becomes more attractive as the high payoff probability rises.",
    )

    rows = pd.DataFrame(
        {
            "Row": np.arange(1, len(prob_high) + 1),
            "High payoff probability": prob_high,
            "Risky count": counts,
            "Trials": trials,
            "Observed share": observed_share,
            "True probability": true_prob,
            "Unconstrained fit": np.asarray(unconstrained["probabilities"]),
            "Fixed-scale fit": np.asarray(fixed["probabilities"]),
            "Monotone fit": np.asarray(monotone["probabilities"]),
        }
    )
    report.add_table(
        "tables/row-fits.csv",
        "Row fit diagnostics",
        rows.round(4),
        description="Equal adjacent monotone fits show where the inequality constraint binds.",
    )

    report.add_table(
        "tables/model-comparison.csv",
        "Estimator comparison",
        summary.round(5),
        description=(
            f"The monotone model gives up {summary.loc[summary['Model'] == 'Monotone row logit', 'LL loss vs saturated'].iloc[0]:.2f} "
            "likelihood points relative to the saturated fit. The CRRA model gives up "
            "more fit because one curve must explain all rows."
        ),
    )

    report.add_takeaway(
        "Risk aversion estimates need preference structure and noise control. "
        "The CRRA likelihood gives a single curvature estimate. "
        "The monotone logit keeps row flexibility while removing sample reversals. "
        "This helps when the ordering is credible but the CRRA curve is too tight."
    )

    report.add_references(
        [
            "[Holt, C. A. and Laury, S. K. (2002). Risk Aversion and Incentive Effects. *American Economic Review*, 92(5), 1644-1655.](https://doi.org/10.1257/000282802762024700)",
            "[Apesteguia, J. and Ballester, M. A. (2018). Monotone Stochastic Choice Models: The Case of Risk and Time Preferences. *Journal of Political Economy*, 126(1), 74-106.](https://doi.org/10.1086/695504)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
