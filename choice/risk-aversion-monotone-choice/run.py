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
from lib.plotting import save_figure, save_thumbnail, setup_style


# Shared sign threshold for counting monotonicity violations. One constant
# keeps the "Monotonicity violations" summary column to a single measurement
# definition across all three estimators.
MONOTONICITY_TOL = 1e-10


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
        "violations": float(np.sum(np.diff(shares) < -MONOTONICITY_TOL)),
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
        "violations": float(np.sum(np.diff(probabilities) < -MONOTONICITY_TOL)),
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
        "violations": float(np.sum(np.diff(probabilities) < -MONOTONICITY_TOL)),
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
    save_figure(fig1, "figures/risky-choice-fits.png", dpi=150)

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
    save_figure(fig2, "figures/rho-likelihood.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    eu_true = lottery_eu_difference(prob_high, rho_true)
    ax3.axhline(0, color="black", linewidth=1.0)
    ax3.plot(prob_high, eu_true, marker="o")
    ax3.set_xlabel("Probability of high payoff")
    ax3.set_ylabel("EU(B) minus EU(A)")
    ax3.set_title("Expected-Utility Index")
    save_figure(fig3, "figures/eu-index.png", dpi=150)

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
    Path("tables/row-fits.csv").parent.mkdir(parents=True, exist_ok=True)
    rows.round(4).to_csv("tables/row-fits.csv", index=False)

    Path("tables/model-comparison.csv").parent.mkdir(parents=True, exist_ok=True)
    summary.round(5).to_csv("tables/model-comparison.csv", index=False)

    save_thumbnail("figures/risky-choice-fits.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
