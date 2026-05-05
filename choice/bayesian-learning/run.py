#!/usr/bin/env python3
"""Bayesian learning from sequential signals.

A decision maker is uncertain about a binary state and observes noisy public
signals. The sufficient state is the posterior belief, which can be updated
recursively and used in both classification and finite-horizon stopping.

Reference: DeGroot (1970), Chamley (2004).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


# =============================================================================
# Bayesian updating engine
# =============================================================================

def bayesian_update(prior_H: float, signal: int, p_red_H: float, p_red_L: float) -> float:
    """Update belief P(H) after observing a signal (1=red, 0=blue).

    Applies Bayes' rule:
        P(H|s) = P(s|H) * P(H) / [P(s|H)*P(H) + P(s|L)*P(L)]
    """
    lik_H = p_red_H if signal == 1 else (1.0 - p_red_H)
    lik_L = p_red_L if signal == 1 else (1.0 - p_red_L)
    marginal = lik_H * prior_H + lik_L * (1.0 - prior_H)
    posterior = lik_H * prior_H / marginal
    return posterior


def posterior_from_counts(
    red_count: np.ndarray | int,
    T: int,
    prior_H: float,
    p_red_H: float,
    p_red_L: float,
) -> np.ndarray | float:
    """Return P(H | k red signals out of T draws)."""
    if T == 0:
        if np.isscalar(red_count):
            return prior_H
        return np.full_like(np.asarray(red_count, dtype=float), prior_H)

    k = np.asarray(red_count, dtype=float)
    log_prior_odds = np.log(prior_H / (1.0 - prior_H))
    log_lr = (
        k * np.log(p_red_H / p_red_L)
        + (T - k) * np.log((1.0 - p_red_H) / (1.0 - p_red_L))
    )
    posterior = 1.0 / (1.0 + np.exp(-(log_prior_odds + log_lr)))
    if np.isscalar(red_count):
        return float(posterior)
    return posterior


def exact_mean_posterior_path(
    true_state: str,
    T: int,
    prior_H: float,
    p_red_H: float,
    p_red_L: float,
) -> np.ndarray:
    """Integrate the posterior over the binomial signal distribution."""
    p_red = p_red_H if true_state == "H" else p_red_L
    means = np.zeros(T + 1)
    for t in range(T + 1):
        k_grid = np.arange(t + 1)
        posterior = posterior_from_counts(k_grid, t, prior_H, p_red_H, p_red_L)
        means[t] = np.sum(binom.pmf(k_grid, t, p_red) * posterior)
    return means


def exact_bayes_accuracy(
    T: int,
    prior_H: float,
    p_red_H: float,
    p_red_L: float,
) -> float:
    """Bayes classifier accuracy integrated over the signal distribution."""
    k_grid = np.arange(T + 1)
    posterior = posterior_from_counts(k_grid, T, prior_H, p_red_H, p_red_L)
    predict_H = posterior >= 0.5
    correct_if_H = np.sum(binom.pmf(k_grid[predict_H], T, p_red_H))
    correct_if_L = np.sum(binom.pmf(k_grid[~predict_H], T, p_red_L))
    return prior_H * correct_if_H + (1.0 - prior_H) * correct_if_L


def simulate_belief_path(true_state: str, T: int, prior_H: float,
                         p_red_H: float, p_red_L: float, rng: np.random.Generator):
    """Simulate a single path of posterior beliefs over T signals.

    Returns:
        beliefs: array of shape (T+1,) starting with the prior
        signals: array of shape (T,) with 1=red, 0=blue
    """
    p_red = p_red_H if true_state == "H" else p_red_L
    signals = rng.binomial(1, p_red, size=T)
    beliefs = np.zeros(T + 1)
    beliefs[0] = prior_H
    for t in range(T):
        beliefs[t + 1] = bayesian_update(beliefs[t], signals[t], p_red_H, p_red_L)
    return beliefs, signals


# =============================================================================
# Optimal stopping
# =============================================================================

def compute_optimal_stopping_boundary(T: int, payoff_invest_H: float,
                                      payoff_invest_L: float,
                                      payoff_wait: float,
                                      p_red_H: float, p_red_L: float):
    """Compute the optimal stopping boundary via backward induction.

    At each period, the agent can:
      - Invest: expected payoff = p*payoff_invest_H + (1-p)*payoff_invest_L
      - Don't invest (stop gathering info): payoff = payoff_wait (=0)
      - Continue: expected value of waiting one more period

    Returns upper and lower belief thresholds for each period.
    """
    # Value of investing at belief p
    def v_invest(p):
        return p * payoff_invest_H + (1.0 - p) * payoff_invest_L

    # Value of not investing
    v_not_invest = payoff_wait

    # At terminal period T, must decide: invest or not
    # Invest if v_invest(p) > v_not_invest => p > threshold
    p_threshold_invest = (payoff_wait - payoff_invest_L) / (payoff_invest_H - payoff_invest_L)

    # Backward induction: at each t, continuation value vs stopping
    # We discretize the belief space
    n_p = 1000
    p_grid = np.linspace(0.001, 0.999, n_p)

    # Terminal value
    V = np.maximum(v_invest(p_grid), v_not_invest)

    upper_bounds = np.zeros(T + 1)
    lower_bounds = np.zeros(T + 1)

    # At terminal period
    upper_bounds[T] = p_threshold_invest
    lower_bounds[T] = p_threshold_invest

    for t in range(T - 1, -1, -1):
        V_new = np.zeros(n_p)
        for i, p in enumerate(p_grid):
            # Value of stopping now
            v_stop = max(v_invest(p), v_not_invest)

            # Value of continuing: expected V(p') after one more signal
            # P(red) = p * p_red_H + (1-p) * p_red_L
            p_red = p * p_red_H + (1.0 - p) * p_red_L
            # Posterior after red signal
            p_after_red = p * p_red_H / p_red
            # Posterior after blue signal
            p_blue = 1.0 - p_red
            p_after_blue = p * (1.0 - p_red_H) / p_blue if p_blue > 0 else p

            v_red = np.interp(p_after_red, p_grid, V)
            v_blue = np.interp(p_after_blue, p_grid, V)
            v_continue = p_red * v_red + p_blue * v_blue

            V_new[i] = max(v_stop, v_continue)

        # Find boundaries where agent switches from continue to stop
        # Upper: invest region
        invest_val = v_invest(p_grid)
        stop_better = (np.maximum(invest_val, v_not_invest) >= V_new - 1e-10)
        # Upper boundary: highest p where continuing is still better
        continue_region = ~stop_better
        if np.any(continue_region):
            indices = np.where(continue_region)[0]
            # Upper bound: where continuation region ends (invest threshold)
            upper_bounds[t] = p_grid[indices[-1]] if len(indices) > 0 else p_threshold_invest
            # Lower bound: where continuation region starts (don't invest threshold)
            lower_bounds[t] = p_grid[indices[0]] if len(indices) > 0 else p_threshold_invest
        else:
            upper_bounds[t] = p_threshold_invest
            lower_bounds[t] = p_threshold_invest

        V = V_new

    return upper_bounds, lower_bounds


# =============================================================================
# ML classifier for comparison
# =============================================================================

def train_ml_classifier(n_train: int, T: int, p_red_H: float, p_red_L: float,
                        rng: np.random.Generator):
    """Train a logistic regression classifier on signal sequences.

    For each sequence, features = cumulative signal counts at each step.
    Labels = true state (1=H, 0=L).

    Returns the trained model and training data.
    """
    # Generate training data
    states = rng.binomial(1, 0.5, size=n_train)  # 1=H, 0=L
    X_all = np.zeros((n_train, T))

    for i in range(n_train):
        p_red = p_red_H if states[i] == 1 else p_red_L
        signals = rng.binomial(1, p_red, size=T)
        X_all[i, :] = np.cumsum(signals) / np.arange(1, T + 1)  # running fraction of red

    # Train logistic regression on full signal history
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_all, states)

    return model, X_all, states


def bayes_classifier_probs(X: np.ndarray, T: int, p_red_H: float, p_red_L: float):
    """Compute Bayesian posterior P(H) for each sample using sufficient statistics.

    Given the running fraction of reds at time T, the sufficient statistic is
    the total number of red signals k out of T draws.
    """
    # X[:, -1] is the running fraction of reds at the final period
    k = np.round(X[:, -1] * T).astype(int)  # number of red signals
    k = np.clip(k, 0, T)

    return posterior_from_counts(k, T, 0.5, p_red_H, p_red_L)


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    p_red_H = 0.7         # P(red | state=H)
    p_red_L = 0.3         # P(red | state=L)
    prior_H = 0.5         # Prior belief P(H)
    T = 50                # Number of signals
    n_paths = 200         # Number of simulation paths per state
    n_train = 5000        # Training samples for ML classifier
    n_test = 2000         # Test samples for comparison
    seed = 42

    # Optimal stopping payoffs
    payoff_invest_H = 1.0   # Payoff from investing when state is H
    payoff_invest_L = -0.5  # Payoff from investing when state is L
    payoff_wait = 0.0       # Payoff from not investing

    rng = np.random.default_rng(seed)

    # =========================================================================
    # Simulate belief paths
    # =========================================================================
    print("Simulating belief paths...")
    beliefs_H = np.zeros((n_paths, T + 1))  # Paths when true state is H
    beliefs_L = np.zeros((n_paths, T + 1))  # Paths when true state is L

    for i in range(n_paths):
        beliefs_H[i], _ = simulate_belief_path("H", T, prior_H, p_red_H, p_red_L, rng)
        beliefs_L[i], _ = simulate_belief_path("L", T, prior_H, p_red_H, p_red_L, rng)
    exact_mean_H = exact_mean_posterior_path("H", T, prior_H, p_red_H, p_red_L)
    exact_mean_L = exact_mean_posterior_path("L", T, prior_H, p_red_H, p_red_L)

    # =========================================================================
    # Speed of learning vs signal informativeness
    # =========================================================================
    print("Computing learning speed across informativeness levels...")
    informativeness_levels = np.linspace(0.55, 0.95, 20)
    convergence_times = np.zeros(len(informativeness_levels))
    threshold = 0.95  # Belief threshold for "learned"

    for idx, p_h in enumerate(informativeness_levels):
        p_l = 1.0 - p_h  # Symmetric: P(red|L) = 1 - P(red|H)
        times = []
        for _ in range(500):
            beliefs, _ = simulate_belief_path("H", T, prior_H, p_h, p_l, rng)
            # Find first time belief exceeds threshold
            above = np.where(beliefs > threshold)[0]
            if len(above) > 0:
                times.append(above[0])
            else:
                times.append(T)
        convergence_times[idx] = np.mean(times)

    # =========================================================================
    # Optimal stopping boundary
    # =========================================================================
    print("Computing optimal stopping boundary...")
    T_stop = 30  # Shorter horizon for stopping problem
    upper_bounds, lower_bounds = compute_optimal_stopping_boundary(
        T_stop, payoff_invest_H, payoff_invest_L, payoff_wait, p_red_H, p_red_L
    )

    # =========================================================================
    # ML comparison: train classifier and compare to Bayes
    # =========================================================================
    print("Training ML classifier...")
    model, X_train, y_train = train_ml_classifier(n_train, T, p_red_H, p_red_L, rng)

    # Generate test data
    print("Evaluating classifiers on test data...")
    y_test = rng.binomial(1, 0.5, size=n_test)
    X_test = np.zeros((n_test, T))
    for i in range(n_test):
        p_red = p_red_H if y_test[i] == 1 else p_red_L
        signals = rng.binomial(1, p_red, size=T)
        X_test[i, :] = np.cumsum(signals) / np.arange(1, T + 1)

    # ML predictions
    ml_probs = model.predict_proba(X_test)[:, 1]
    ml_preds = (ml_probs >= 0.5).astype(int)

    # Bayesian predictions
    bayes_probs = bayes_classifier_probs(X_test, T, p_red_H, p_red_L)
    bayes_preds = (bayes_probs >= 0.5).astype(int)

    # Accuracy
    ml_accuracy = accuracy_score(y_test, ml_preds)
    bayes_accuracy = accuracy_score(y_test, bayes_preds)

    # ROC curves
    fpr_ml, tpr_ml, _ = roc_curve(y_test, ml_probs)
    fpr_bayes, tpr_bayes, _ = roc_curve(y_test, bayes_probs)
    auc_ml = auc(fpr_ml, tpr_ml)
    auc_bayes = auc(fpr_bayes, tpr_bayes)

    # Accuracy at different time horizons
    horizons = [5, 10, 20, 30, 50]
    ml_acc_by_t = []
    bayes_acc_by_t = []
    exact_acc_by_t = []
    for t_eval in horizons:
        # Retrain ML on first t_eval signals
        model_t = LogisticRegression(max_iter=1000, random_state=42)
        model_t.fit(X_train[:, :t_eval], y_train)
        ml_pred_t = model_t.predict(X_test[:, :t_eval])
        ml_acc_by_t.append(accuracy_score(y_test, ml_pred_t))
        exact_acc_by_t.append(exact_bayes_accuracy(t_eval, prior_H, p_red_H, p_red_L))

        # Bayes at horizon t_eval
        k_t = np.round(X_test[:, t_eval - 1] * t_eval).astype(int)
        k_t = np.clip(k_t, 0, t_eval)
        posterior_t = posterior_from_counts(k_t, t_eval, prior_H, p_red_H, p_red_L)
        bayes_pred_t = (posterior_t >= 0.5).astype(int)
        bayes_acc_by_t.append(accuracy_score(y_test, bayes_pred_t))

    print(f"  Bayes accuracy (T={T}): {bayes_accuracy:.4f}")
    print(f"  ML accuracy    (T={T}): {ml_accuracy:.4f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Bayesian Learning and Sequential Investment",
        "Posterior beliefs, sufficient statistics, and the option value of waiting before investment.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A firm is deciding whether to invest in a project whose quality is not directly "
        "observed. Each signal is noisy, so no single draw settles the question. What matters "
        "for choice is the posterior probability that the project is good, not the full signal "
        "history.\n\n"
        "The two-state urn model makes that compression exact. "
        "Nature chooses state $H$ or $L$, signals arrive sequentially, and the agent updates "
        "$p_t=\\Pr(H \\mid s_1,\\ldots,s_t)$. The posterior then feeds into a finite-horizon "
        "stopping problem: invest when confidence is high, reject when confidence is low, and "
        "continue sampling when information still has option value.\n\n"
        "Within the choice section, this is a belief-state example rather than a demand model. "
        "It complements [plain logit demand](../logit-discrete-choice/) because both rely on "
        "log odds, but here the probability is a posterior about an unknown state rather than "
        "a market share."
    )

    report.add_equations(
        r"""
Let $\theta\in\{H,L\}$ denote the unknown state and let $s_t\in\{R,B\}$ denote
the period-$t$ signal. The maintained signal probabilities are

$$\Pr(R\mid H)=p_H,\qquad \Pr(R\mid L)=p_L,\qquad p_H>p_L.$$

The posterior after observing $s_{t+1}$ is

$$p_{t+1}
=\frac{f_H(s_{t+1})p_t}
{f_H(s_{t+1})p_t+f_L(s_{t+1})(1-p_t)},$$

where $f_\theta(s)=\Pr(s\mid \theta)$.

Equivalently, posterior odds evolve additively in log likelihood ratios:

$$\log\frac{p_{t+1}}{1-p_{t+1}}
=\log\frac{p_t}{1-p_t}
+\log\frac{f_H(s_{t+1})}{f_L(s_{t+1})}.$$

After $T$ signals, if $k_T$ of them are red, the sufficient statistic is

$$\Lambda_T
=k_T\log\frac{p_H}{p_L}
+(T-k_T)\log\frac{1-p_H}{1-p_L}.$$

For the stopping problem, investing gives payoff $\pi_H$ in state $H$ and
$\pi_L$ in state $L$; rejecting gives zero. At belief $p$, the current action
value is

$$A(p)=\max[p\pi_H+(1-p)\pi_L,\ 0].$$

With one more signal available, the continuation value is

$$C_t(p)=\Pr(R\mid p)V_{t+1}(p_R')+\Pr(B\mid p)V_{t+1}(p_B'),$$

where $p_R'$ and $p_B'$ are the Bayes-updated beliefs after a red or blue
signal. The finite-horizon recursion is

$$V_t(p)=\max[A(p),\ C_t(p)].$$
        """
    )

    report.add_model_setup(
        "The calibration is symmetric around an uninformative prior. A red signal is "
        "evidence for $H$; a blue signal is evidence for $L$. The classifier comparison "
        "uses synthetic data from the same signal structure, so the Bayesian rule is the "
        "known benchmark rather than an estimated model.\n\n"
        f"| Object | Value | Role |\n"
        f"|-----------|-------|-------------|\n"
        f"| $p_H$ | {p_red_H} | Probability of a red signal in state $H$ |\n"
        f"| $p_L$ | {p_red_L} | Probability of a red signal in state $L$ |\n"
        f"| Prior $p_0$ | {prior_H} | Initial belief $\\Pr(H)$ |\n"
        f"| Signal horizon | {T} | Draws used for belief paths and classification |\n"
        f"| Simulated paths | {n_paths} per state | Monte Carlo paths shown against exact means |\n"
        f"| Classifier training sample | {n_train} | Histories used to train logistic regression |\n"
        f"| Classifier test sample | {n_test} | Histories used for finite-sample accuracy |\n"
        f"| Investment payoff in $H$ | {payoff_invest_H} | Payoff if the project is good |\n"
        f"| Investment payoff in $L$ | {payoff_invest_L} | Payoff if the project is bad |\n"
        f"| Reject payoff | {payoff_wait} | Outside option after stopping |"
    )

    report.add_solution_method(
        "There are two computational objects. The first is the filtering recursion that "
        "maps a signal history into one posterior belief. The second is a stopping boundary "
        "on that belief state. The logistic-regression exercise is only a comparison: it asks "
        "how close a flexible statistical classifier gets when the true likelihoods are already "
        "known to the Bayesian agent.\n\n"
        "```text\n"
        "Algorithm: Bayesian filtering and finite-horizon stopping\n"
        "Input: prior p_0, likelihoods f_H and f_L, payoffs pi_H and pi_L, horizon T\n"
        "Output: posterior path p_t and stopping regions over beliefs\n"
        "\n"
        "Filtering:\n"
        "    for each incoming signal s_{t+1}:\n"
        "        multiply prior odds p_t / (1-p_t) by f_H(s_{t+1}) / f_L(s_{t+1})\n"
        "        convert odds back to p_{t+1}\n"
        "\n"
        "Stopping:\n"
        "    set terminal value V_T(p) = max[p*pi_H + (1-p)*pi_L, 0]\n"
        "    for t = T-1, ..., 0:\n"
        "        for each belief grid point p_i:\n"
        "            compute posteriors after red and blue signals\n"
        "            interpolate V_{t+1} at those two posteriors\n"
        "            compare action value A(p_i) with continuation value C_t(p_i)\n"
        "        record the reject, continue, and invest regions\n"
        "```\n\n"
        "The sufficient statistic for classification is the red-signal count $k_T$. For "
        "the figures and table below, the code uses the exact binomial distribution of "
        "$k_T$ to benchmark the finite Monte Carlo paths and the trained classifier."
    )

    # --- Figure 1: Posterior belief evolution ---
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))

    periods = np.arange(T + 1)
    # Plot subset of paths for clarity
    n_show = 30
    for i in range(n_show):
        ax1a.plot(periods, beliefs_H[i], color="steelblue", alpha=0.2, linewidth=0.8)
    ax1a.plot(periods, np.mean(beliefs_H, axis=0), color="darkblue", linewidth=2.5,
              label="Simulated mean")
    ax1a.plot(periods, exact_mean_H, color="black", linestyle="--", linewidth=1.6,
              label="Exact mean")
    ax1a.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax1a.set_xlabel("Number of signals")
    ax1a.set_ylabel("$P(H)$")
    ax1a.set_title("Project is good")
    ax1a.set_ylim(-0.05, 1.05)
    ax1a.legend()

    for i in range(n_show):
        ax1b.plot(periods, beliefs_L[i], color="indianred", alpha=0.2, linewidth=0.8)
    ax1b.plot(periods, np.mean(beliefs_L, axis=0), color="darkred", linewidth=2.5,
              label="Simulated mean")
    ax1b.plot(periods, exact_mean_L, color="black", linestyle="--", linewidth=1.6,
              label="Exact mean")
    ax1b.axhline(y=0.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax1b.set_xlabel("Number of signals")
    ax1b.set_ylabel("$P(H)$")
    ax1b.set_title("Project is bad")
    ax1b.set_ylim(-0.05, 1.05)
    ax1b.legend()

    fig1.suptitle("Posterior beliefs under repeated signals", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    report.add_figure(
        "figures/belief-evolution.png",
        "Posterior belief P(H) over time, with exact conditional means overlaid.",
        fig1,
        description=(
            "The figure separates pathwise uncertainty from the law of large numbers. "
            "Light traces are individual histories, the solid curve is the mean across the "
            f"{n_paths} simulated paths, and the dashed curve integrates the posterior exactly "
            "over the binomial signal distribution. With a good project, evidence drifts "
            "toward one; with a bad project, it drifts toward zero. The early dispersion is "
            "not numerical error. It is the information problem."
        ),
    )

    # --- Figure 2: Speed of learning vs signal informativeness ---
    fig2, ax2 = plt.subplots()
    ax2.plot(informativeness_levels, convergence_times, "b-o", markersize=4, linewidth=2)
    ax2.set_xlabel("Signal informativeness $P(\\mathrm{red} \\mid H)$")
    ax2.set_ylabel("Mean signals to reach 95% belief")
    ax2.set_title("Signal precision and speed of learning")
    ax2.axhline(y=T, color="gray", linestyle="--", alpha=0.5, label=f"Horizon T={T}")
    ax2.legend()
    report.add_figure(
        "figures/learning-speed.png",
        "Mean number of signals needed to reach a 95 percent posterior belief.",
        fig2,
        description=(
            "Signal quality varies while the prior is held fixed. As $p_H$ "
            "moves away from one half, each draw carries a larger likelihood-ratio increment. "
            "Learning is nonlinear in signal precision: weak signals can leave "
            "the firm undecided for most of the horizon, while precise signals settle the "
            "investment case quickly."
        ),
    )

    # --- Figure 3: ROC curve comparison ---
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr_bayes, tpr_bayes, "b-", linewidth=2.5,
             label=f"Bayes (AUC = {auc_bayes:.4f})")
    ax3.plot(fpr_ml, tpr_ml, "r--", linewidth=2,
             label=f"Logistic Regression (AUC = {auc_ml:.4f})")
    ax3.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5, label="Random")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("Bayes benchmark and trained classifier")
    ax3.legend(loc="lower right")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    report.add_figure(
        "figures/roc-comparison.png",
        "ROC curves for the Bayes rule and a trained logistic classifier.",
        fig3,
        description=(
            "The classifier comparison is favorable to Bayes by construction: the likelihoods used "
            "by the Bayesian rule are the true likelihoods. Logistic regression sees simulated "
            "histories and learns a similar monotone rule in the red-signal count. The near "
            "overlap is the point. Machine learning recovers the same "
            "sufficient statistic rather than discovering a different economic object."
        ),
    )

    # --- Figure 4: Optimal stopping boundary ---
    fig4, ax4 = plt.subplots()
    t_grid = np.arange(T_stop + 1)

    ax4.fill_between(t_grid, upper_bounds, 1.0, alpha=0.3, color="green", label="Invest")
    ax4.fill_between(t_grid, 0.0, lower_bounds, alpha=0.3, color="red", label="Don't invest")
    ax4.fill_between(t_grid, lower_bounds, upper_bounds, alpha=0.2, color="gray",
                     label="Continue observing")
    ax4.plot(t_grid, upper_bounds, "g-", linewidth=2)
    ax4.plot(t_grid, lower_bounds, "r-", linewidth=2)

    # Overlay a few belief paths
    for i in range(5):
        path, _ = simulate_belief_path("H", T_stop, prior_H, p_red_H, p_red_L, rng)
        ax4.plot(np.arange(T_stop + 1), path, "k-", alpha=0.3, linewidth=0.8)

    ax4.set_xlabel("Period")
    ax4.set_ylabel("Belief $P(H)$")
    ax4.set_title("Stopping regions over the belief state")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="center right")
    report.add_figure(
        "figures/stopping-boundary.png",
        "Invest, reject, and continue regions in the finite-horizon stopping problem.",
        fig4,
        description=(
            "The stopping boundary turns posterior beliefs into actions. High beliefs make "
            "investment attractive, low beliefs make rejection attractive, and intermediate "
            "beliefs preserve the option value of another signal. The continuation region "
            "shrinks as the deadline approaches because there are fewer future opportunities "
            "to use the information."
        ),
    )

    # --- Table: Classification accuracy comparison ---
    table_data = {
        "Signals observed": [str(h) for h in horizons],
        "Exact Bayes accuracy": [f"{a:.4f}" for a in exact_acc_by_t],
        "Bayes on test sample": [f"{a:.4f}" for a in bayes_acc_by_t],
        "Logistic accuracy": [f"{a:.4f}" for a in ml_acc_by_t],
        "Logistic minus exact": [f"{m - e:+.4f}" for m, e in zip(ml_acc_by_t, exact_acc_by_t)],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/accuracy-comparison.csv",
        "Classification Accuracy Against the Exact Bayes Benchmark",
        df,
        description=(
            "The table separates the population benchmark from finite-sample evaluation. "
            "The exact Bayes column integrates over the binomial distribution of signal counts. "
            "The test-sample columns use the same simulated histories as the logistic classifier, "
            "so small sign changes in the final column are Monte Carlo and training variation, "
            "not evidence that the learned classifier has beaten the Bayes rule."
        ),
    )

    report.add_takeaway(
        "Bayesian learning reduces a long signal history to the "
        "posterior belief relevant for choice. The same object classifies the state, "
        "sets the stopping boundary, and prices the value of one more signal.\n\n"
        "The relevant comparison is not Bayes versus machine learning as slogans. In this "
        "controlled experiment the likelihood is known, so Bayes is the population benchmark. "
        "A trained classifier can learn the same likelihood-ratio rule from data, but the "
        "economic state variable remains the posterior belief. Once that state is in hand, "
        "the investment decision is a standard dynamic choice problem: act at extreme beliefs, "
        "wait in the middle, and accept that the waiting region collapses as the deadline "
        "approaches."
    )

    report.add_references([
        "DeGroot, M. (1970). *Optimal Statistical Decisions*. McGraw-Hill.",
        "Chamley, C. (2004). *Rational Herds: Economic Models of Social Learning*. Cambridge University Press.",
        "El-Gamal, M. and Grether, D. (1995). Are People Bayesian? Uncovering Behavioral Strategies. *Journal of the American Statistical Association*, 90(432), 1137-1145.",
        "Berger, J. (1985). *Statistical Decision Theory and Bayesian Analysis*. Springer, 2nd edition.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
