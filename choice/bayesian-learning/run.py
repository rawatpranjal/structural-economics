#!/usr/bin/env python3
"""Bayesian Learning: Belief Updating from Signals.

Demonstrates how a rational agent updates beliefs about an unknown state using
Bayes' rule. The classic urn problem: the agent does not know which of two urns
(H or L) is the true state and must learn from sequential signal draws.

Compares the structural (Bayesian) approach to a reduced-form ML classifier,
showing that Bayes' rule is the optimal classifier for this problem.

Reference: DeGroot (1970), Chamley (2004).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
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

    # Log-likelihood ratio
    log_lr = k * np.log(p_red_H / p_red_L) + (T - k) * np.log((1 - p_red_H) / (1 - p_red_L))
    # Posterior with prior 0.5 => P(H) = sigmoid(log_lr)
    posterior = 1.0 / (1.0 + np.exp(-log_lr))
    return posterior


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
    for t_eval in horizons:
        # Retrain ML on first t_eval signals
        model_t = LogisticRegression(max_iter=1000, random_state=42)
        model_t.fit(X_train[:, :t_eval], y_train)
        ml_pred_t = model_t.predict(X_test[:, :t_eval])
        ml_acc_by_t.append(accuracy_score(y_test, ml_pred_t))

        # Bayes at horizon t_eval
        k_t = np.round(X_test[:, t_eval - 1] * t_eval).astype(int)
        k_t = np.clip(k_t, 0, t_eval)
        log_lr_t = k_t * np.log(p_red_H / p_red_L) + (t_eval - k_t) * np.log((1 - p_red_H) / (1 - p_red_L))
        bayes_pred_t = (log_lr_t > 0).astype(int)
        bayes_acc_by_t.append(accuracy_score(y_test, bayes_pred_t))

    print(f"  Bayes accuracy (T={T}): {bayes_accuracy:.4f}")
    print(f"  ML accuracy    (T={T}): {ml_accuracy:.4f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Bayesian Learning",
        "How rational agents update beliefs from sequential signals under uncertainty.",
    )

    report.add_overview(
        "The Bayesian learning model is the structural approach to information problems. "
        "An agent faces uncertainty about the true state of the world and observes noisy "
        "signals over time. Bayes' rule provides the optimal way to aggregate information "
        "from these signals into a posterior belief.\n\n"
        "We study the classic urn problem: nature selects one of two urns (H or L), and "
        "the agent draws balls sequentially, observing their color. Each draw provides "
        "partial information, and the agent's belief converges to the truth over time. "
        "This model is foundational for understanding financial markets, social learning, "
        "and optimal experimentation."
    )

    report.add_equations(
        r"""
**Bayes' rule (sequential updating):**

$$P(H \mid s_t) = \frac{P(s_t \mid H) \cdot P_t(H)}{P(s_t \mid H) \cdot P_t(H) + P(s_t \mid L) \cdot P_t(L)}$$

where $P_t(H)$ is the prior at time $t$ and $s_t \in \{\text{red}, \text{blue}\}$ is the signal.

**Signal structure (urn problem):**
- $P(\text{red} \mid H) = 0.7$, $P(\text{red} \mid L) = 0.3$
- Signals are i.i.d. conditional on the true state

**Log-likelihood ratio (sufficient statistic):**

$$\lambda_T = \sum_{t=1}^{T} \log \frac{P(s_t \mid H)}{P(s_t \mid L)} = k \log\frac{p_H}{p_L} + (T-k) \log\frac{1-p_H}{1-p_L}$$

where $k$ is the number of red signals out of $T$ draws.

**Optimal stopping:** The agent chooses when to stop gathering signals and act (invest or not).
The value of continuing to observe signals must exceed the value of acting now.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $p_H$ | {p_red_H} | P(red $\\mid$ H) |\n"
        f"| $p_L$ | {p_red_L} | P(red $\\mid$ L) |\n"
        f"| Prior $P(H)$ | {prior_H} | Initial belief |\n"
        f"| $T$ | {T} | Number of signals |\n"
        f"| Paths | {n_paths} | Simulations per state |\n"
        f"| Invest payoff (H) | {payoff_invest_H} | Payoff if invest and state=H |\n"
        f"| Invest payoff (L) | {payoff_invest_L} | Payoff if invest and state=L |\n"
        f"| Wait payoff | {payoff_wait} | Payoff from not investing |"
    )

    report.add_solution_method(
        "**Bayesian updating** is applied sequentially: each signal updates the prior "
        "into a posterior, which becomes the prior for the next signal. By the martingale "
        "convergence theorem, beliefs converge almost surely to the truth.\n\n"
        "**Optimal stopping** is solved by backward induction: at each period, the agent "
        "compares the value of acting now (invest or wait) against the expected continuation "
        "value from one more signal.\n\n"
        "**ML comparison:** A logistic regression classifier is trained on signal sequences "
        "and compared against the Bayesian classifier. Since the data-generating process "
        "matches the Bayesian model exactly, Bayes' rule is the *optimal* classifier "
        "(it achieves the Bayes error rate, which is the irreducible minimum)."
    )

    # --- Figure 1: Posterior belief evolution ---
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))

    periods = np.arange(T + 1)
    # Plot subset of paths for clarity
    n_show = 30
    for i in range(n_show):
        ax1a.plot(periods, beliefs_H[i], color="steelblue", alpha=0.2, linewidth=0.8)
    ax1a.plot(periods, np.mean(beliefs_H, axis=0), color="darkblue", linewidth=2.5,
              label="Mean belief")
    ax1a.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax1a.set_xlabel("Number of signals")
    ax1a.set_ylabel("$P(H)$")
    ax1a.set_title("True state = H")
    ax1a.set_ylim(-0.05, 1.05)
    ax1a.legend()

    for i in range(n_show):
        ax1b.plot(periods, beliefs_L[i], color="indianred", alpha=0.2, linewidth=0.8)
    ax1b.plot(periods, np.mean(beliefs_L, axis=0), color="darkred", linewidth=2.5,
              label="Mean belief")
    ax1b.axhline(y=0.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax1b.set_xlabel("Number of signals")
    ax1b.set_ylabel("$P(H)$")
    ax1b.set_title("True state = L")
    ax1b.set_ylim(-0.05, 1.05)
    ax1b.legend()

    fig1.suptitle("Posterior Belief Evolution", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    report.add_figure(
        "figures/belief-evolution.png",
        "Posterior belief P(H) over time for both true states. "
        "Beliefs converge to the truth as signals accumulate.",
        fig1,
        description="The martingale convergence theorem guarantees that beliefs converge to the truth almost surely. "
        "Individual paths (light traces) show considerable uncertainty early on, but the ensemble mean (bold line) converges smoothly. "
        "The spread of paths reflects the inherent difficulty of learning from noisy signals.",
    )

    # --- Figure 2: Speed of learning vs signal informativeness ---
    fig2, ax2 = plt.subplots()
    ax2.plot(informativeness_levels, convergence_times, "b-o", markersize=4, linewidth=2)
    ax2.set_xlabel("Signal informativeness $P(\\mathrm{red} \\mid H)$")
    ax2.set_ylabel("Mean signals to reach 95% belief")
    ax2.set_title("Speed of Learning vs Signal Informativeness")
    ax2.axhline(y=T, color="gray", linestyle="--", alpha=0.5, label=f"Horizon T={T}")
    ax2.legend()
    report.add_figure(
        "figures/learning-speed.png",
        "More informative signals (further from 0.5) lead to faster learning.",
        fig2,
        description="The convex decline shows that learning speed is highly nonlinear in signal quality. "
        "Signals near $p = 0.5$ are almost uninformative and require dozens of draws, while highly informative signals ($p > 0.8$) resolve uncertainty in just a few observations. "
        "This has practical implications for experiment design: slightly more precise instruments can dramatically reduce sample requirements.",
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
    ax3.set_title("ROC Curve: Bayes vs ML Classifier")
    ax3.legend(loc="lower right")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    report.add_figure(
        "figures/roc-comparison.png",
        "ROC curves for Bayesian and ML classifiers. "
        "Bayes achieves the theoretical optimum; ML approaches it asymptotically.",
        fig3,
        description="When the data-generating process matches the structural model, Bayes' rule achieves the irreducible minimum error rate. "
        "The near-overlap of the two ROC curves shows that logistic regression, trained on enough data, effectively learns the same decision boundary -- "
        "but it requires training data to do so, whereas Bayes' rule uses the known model directly.",
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
    ax4.set_title("Optimal Stopping Boundary")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="center right")
    report.add_figure(
        "figures/stopping-boundary.png",
        "Optimal stopping regions: the agent invests when beliefs are high enough, "
        "abstains when low enough, and continues observing in between.",
        fig4,
        description="The narrowing continuation region as the deadline approaches reflects the diminishing option value of information. "
        "Early on, the agent is willing to wait because more signals can substantially refine beliefs; near the terminal period, "
        "the agent must act even with moderate uncertainty. Sample belief paths (black traces) show typical trajectories through the decision space.",
    )

    # --- Table: Classification accuracy comparison ---
    table_data = {
        "Signals observed": [str(h) for h in horizons],
        "Bayes accuracy": [f"{a:.4f}" for a in bayes_acc_by_t],
        "ML accuracy": [f"{a:.4f}" for a in ml_acc_by_t],
        "Bayes advantage": [f"{b - m:+.4f}" for b, m in zip(bayes_acc_by_t, ml_acc_by_t)],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/accuracy-comparison.csv",
        "Classification Accuracy: Bayesian vs ML at Different Horizons",
        df,
        description="Bayes consistently matches or beats the ML classifier at every horizon. "
        "The advantage is largest with few signals, where the structural model's knowledge of the DGP compensates for limited data. "
        "As the horizon grows, both approaches converge because the log-likelihood ratio becomes the dominant feature.",
    )

    report.add_takeaway(
        "Bayesian learning is the structural approach to information problems. Rather than "
        "fitting a black-box classifier, the agent uses a model of how signals are generated "
        "to update beliefs optimally.\n\n"
        "**Key insights:**\n"
        "- **Convergence:** Beliefs converge to the truth almost surely (martingale convergence "
        "theorem). With symmetric signals ($p_H = 0.7$, $p_L = 0.3$), learning is equally "
        "fast in both states.\n"
        "- **Informativeness matters:** More informative signals (further from 0.5) dramatically "
        "speed up learning. Near-uninformative signals ($p \\approx 0.5$) require many draws.\n"
        "- **Bayes is optimal:** When the true data-generating process matches the model, "
        "Bayes' rule achieves the minimum possible classification error (the Bayes error rate). "
        "No ML method can beat it --- they can only approach it with enough data.\n"
        "- **Optimal stopping:** The value of information creates an option value of waiting. "
        "The agent continues gathering signals when beliefs are in an intermediate range, "
        "and acts only when sufficiently confident.\n"
        "- **Sufficient statistics:** The number of red draws $k$ out of $T$ is a sufficient "
        "statistic for the state. The full signal history can be compressed without information "
        "loss --- this is why logistic regression (which uses running averages) approximates "
        "Bayes well."
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
