#!/usr/bin/env python3
"""Stochastic choice and random consideration sets (Manzini-Mariotti 2014).

A boundedly-rational agent has a strict preference ranking over a finite
set of alternatives plus a per-alternative attention probability. From a
menu, each alternative independently enters the consideration set with
its attention probability. The agent picks the preference-best
alternative in the consideration set; if the consideration set is empty
the agent picks the default option.

The induced random choice rule has a closed-form choice probability,
identifies both the ranking and the attention parameters from menu
variation in stochastic choice data, and produces structured violations
of Luce's IIA that the Luce / multinomial-logit model cannot match.

Two estimators are compared:

- Method 1: joint maximum likelihood with brute-force enumeration of the
  J! candidate rankings. Given the ranking the attention MLE has a
  closed form because the log-likelihood factorises across alternatives.
- Method 2: a two-step revealed-preference procedure. Attention is
  recovered from singleton-with-default frequencies; the ranking is
  recovered from the asymmetric menu-removal pattern.

A Luce benchmark fitted by maximum likelihood gives the comparison
point.

Reference: Manzini, P., & Mariotti, M. (2014). Stochastic Choice and
Consideration Sets. Econometrica 82(3), 1153-1176.
"""
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


# ---------------------------------------------------------------------------
# Closed-form choice probability (Manzini-Mariotti Definition 2)
# ---------------------------------------------------------------------------
def choice_prob(menu_mask: np.ndarray, ranking: np.ndarray,
                gamma: np.ndarray) -> np.ndarray:
    """Closed-form Manzini-Mariotti choice probabilities for one menu.

    Parameters
    ----------
    menu_mask : ndarray, shape (J,), bool
        True for alternatives present in the menu.
    ranking : ndarray, shape (J,), int
        Permutation of {0, ..., J-1} from best to worst.
    gamma : ndarray, shape (J,)
        Attention probabilities, one per alternative.

    Returns
    -------
    probs : ndarray, shape (J + 1,)
        Probability of choosing each alternative; index J is the default.
    """
    J = len(gamma)
    probs = np.zeros(J + 1)
    one_minus_gamma = np.ones(J) - gamma
    higher_factor = 1.0
    for j in ranking:
        if menu_mask[j]:
            probs[j] = gamma[j] * higher_factor
            higher_factor *= one_minus_gamma[j]
    probs[J] = higher_factor
    return probs


def all_menu_probs(ranking: np.ndarray, gamma: np.ndarray,
                   menus: np.ndarray) -> np.ndarray:
    """Stack closed-form probabilities for an array of menus."""
    return np.array([choice_prob(m, ranking, gamma) for m in menus])


# ---------------------------------------------------------------------------
# Method 1: joint MLE with brute-force ranking
# ---------------------------------------------------------------------------
def mle_gamma_given_ranking(counts: np.ndarray, menus: np.ndarray,
                             ranking: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Closed-form attention MLE conditional on the preference ranking.

    For a fixed ranking, gamma_hat(j) = N_chose(j) / [N_chose(j) + N_blocked(j)],
    where N_blocked(j) counts observations in which j was in the menu and
    ranked above the chosen alternative or the default. The proof uses
    independence of the attention draws across alternatives.

    Parameters
    ----------
    counts : ndarray, shape (M, J + 1)
        Choice count per (menu, alternative-or-default) cell.
    menus : ndarray, shape (M, J), bool
        Menu indicator per row.
    ranking : ndarray, shape (J,), int
        Best-to-worst permutation.

    Returns
    -------
    gamma_hat : ndarray, shape (J,)
    """
    J = menus.shape[1]
    n_chose = counts[:, :J].sum(axis=0)
    n_blocked = np.zeros(J)
    rank_position = np.empty(J, dtype=int)
    rank_position[ranking] = np.arange(J)
    for m_idx, mask in enumerate(menus):
        for j in range(J):
            if not mask[j]:
                continue
            chosen_indices = np.where(counts[m_idx, :J] > 0)[0]
            for chosen in chosen_indices:
                if chosen == j:
                    continue
                if rank_position[j] < rank_position[chosen]:
                    n_blocked[j] += counts[m_idx, chosen]
            n_blocked[j] += counts[m_idx, J]
    denom = n_chose + n_blocked
    gamma_hat = np.where(denom > 0, n_chose / np.maximum(denom, eps), 0.5)
    return np.clip(gamma_hat, eps, 1.0 - eps)


def log_likelihood_at(ranking: np.ndarray, gamma: np.ndarray,
                      counts: np.ndarray, menus: np.ndarray,
                      eps: float = 1e-12) -> float:
    """Sample log-likelihood under the random consideration set rule."""
    probs = all_menu_probs(ranking, gamma, menus)
    return float(np.sum(counts * np.log(np.maximum(probs, eps))))


def fit_method1(counts: np.ndarray, menus: np.ndarray) -> tuple:
    """Brute-force MLE over all J! rankings; closed-form gamma per ranking."""
    J = menus.shape[1]
    best = (-np.inf, None, None)
    for ranking_tuple in permutations(range(J)):
        ranking = np.array(ranking_tuple, dtype=int)
        gamma_hat = mle_gamma_given_ranking(counts, menus, ranking)
        ll = log_likelihood_at(ranking, gamma_hat, counts, menus)
        if ll > best[0]:
            best = (ll, ranking, gamma_hat)
    return best[1], best[2], best[0]


# ---------------------------------------------------------------------------
# Method 2: two-step revealed-preference identification
# ---------------------------------------------------------------------------
def fit_method2(counts: np.ndarray, menus: np.ndarray) -> tuple:
    """Recover gamma from singleton-with-default; ranking from asymmetric impact.

    Step 1: gamma_hat(j) = 1 - p_hat(default, {j}) using the singleton menu.
    Step 2: For each pair (i, j), compute the impact of removing i on j's
    probability and vice versa, averaged across menus that contain both.
    Asymmetry pins down the ranking: the alternative whose removal raises
    the other's probability is the higher-ranked one.
    """
    J = menus.shape[1]
    menu_totals = counts.sum(axis=1)
    eps = 1e-9
    freqs = counts / np.maximum(menu_totals[:, None], eps)
    # Step 1: attention from singletons
    gamma_hat = np.zeros(J)
    for j in range(J):
        singleton_idx = next(
            (i for i, m in enumerate(menus) if m.sum() == 1 and m[j])
        )
        gamma_hat[j] = 1.0 - freqs[singleton_idx, J]
    gamma_hat = np.clip(gamma_hat, eps, 1.0 - eps)
    # Step 2: ranking from menu-removal asymmetric impact
    impact_when_removed = np.zeros((J, J))
    counts_when_removed = np.zeros((J, J), dtype=int)
    for m_idx, mask in enumerate(menus):
        present = np.where(mask)[0]
        if len(present) < 2:
            continue
        for j in present:
            for i in present:
                if i == j:
                    continue
                # Find the menu that drops i
                target_mask = mask.copy()
                target_mask[i] = False
                if not target_mask.any():
                    continue
                target_idx = next(
                    (k for k, m in enumerate(menus)
                     if np.array_equal(m, target_mask)), None,
                )
                if target_idx is None:
                    continue
                p_full = freqs[m_idx, j]
                p_removed = freqs[target_idx, j]
                impact_when_removed[i, j] += p_removed - p_full
                counts_when_removed[i, j] += 1
    avg_impact = np.where(
        counts_when_removed > 0,
        impact_when_removed / np.maximum(counts_when_removed, 1),
        0.0,
    )
    # Score: alternative j ranks high when removing j raises others a lot
    # (because j is the one preventing others from being chosen) and others'
    # removal does not raise j (because nothing is ranked above j).
    score = np.zeros(J)
    for j in range(J):
        score[j] = avg_impact[j, :].sum() - avg_impact[:, j].sum()
    ranking_hat = np.argsort(-score)
    return ranking_hat, gamma_hat


# ---------------------------------------------------------------------------
# Luce benchmark
# ---------------------------------------------------------------------------
def fit_luce(counts: np.ndarray, menus: np.ndarray) -> tuple:
    """MLE of Luce / multinomial-logit utilities on the same data.

    Luce: p_L(a, A) = u(a) / sum_{b in A} u(b). The default is treated as
    an additional alternative with utility u_default. Parameters are J
    log-utilities for inside alternatives plus one for the default.
    """
    J = menus.shape[1]
    menus_with_default = np.column_stack([menus, np.ones(menus.shape[0], dtype=bool)])

    def neg_ll(log_u: np.ndarray) -> float:
        u = np.exp(log_u)
        ll = 0.0
        for m_idx, mask in enumerate(menus_with_default):
            denom = u[mask].sum()
            for k in range(J + 1):
                if mask[k] and counts[m_idx, k] > 0:
                    ll += counts[m_idx, k] * (np.log(u[k]) - np.log(denom))
        return -ll

    res = minimize(neg_ll, x0=np.zeros(J + 1), method="L-BFGS-B")
    log_u_hat = res.x
    log_u_hat -= log_u_hat[J]  # normalise default to 1
    return np.exp(log_u_hat), -res.fun


def luce_probs(u: np.ndarray, menus: np.ndarray) -> np.ndarray:
    """Closed-form Luce choice probabilities including the default."""
    J = menus.shape[1]
    out = np.zeros((menus.shape[0], J + 1))
    for m_idx, mask in enumerate(menus):
        full_mask = np.concatenate([mask, [True]])
        denom = u[full_mask].sum()
        for k in range(J + 1):
            if full_mask[k]:
                out[m_idx, k] = u[k] / denom
    return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def enumerate_menus(J: int) -> np.ndarray:
    """All non-empty subset indicators of {0, ..., J-1}."""
    out = []
    for mask_int in range(1, 1 << J):
        mask = np.array([(mask_int >> j) & 1 for j in range(J)], dtype=bool)
        out.append(mask)
    return np.array(out)


def simulate_counts(menus: np.ndarray, ranking: np.ndarray, gamma: np.ndarray,
                    n_subjects: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial choice counts per menu under the closed-form probabilities."""
    M = menus.shape[0]
    J = menus.shape[1]
    counts = np.zeros((M, J + 1), dtype=int)
    for m_idx, mask in enumerate(menus):
        probs = choice_prob(mask, ranking, gamma)
        counts[m_idx] = rng.multinomial(n_subjects, probs)
    return counts


# ---------------------------------------------------------------------------
# Bootstrap standard errors
# ---------------------------------------------------------------------------
def bootstrap_se(menus: np.ndarray, ranking_true: np.ndarray, gamma_true: np.ndarray,
                 n_subjects: int, n_boot: int, rng: np.random.Generator,
                 fit_fn) -> tuple:
    """Subject-cluster bootstrap of attention and ranking accuracy."""
    J = menus.shape[1]
    gamma_draws = np.zeros((n_boot, J))
    ranking_correct = 0
    for b in range(n_boot):
        boot_counts = simulate_counts(menus, ranking_true, gamma_true,
                                       n_subjects, rng)
        out = fit_fn(boot_counts, menus)
        ranking_hat = out[0]
        gamma_hat = out[1]
        gamma_draws[b] = gamma_hat
        if np.array_equal(ranking_hat, ranking_true):
            ranking_correct += 1
    return gamma_draws.std(axis=0), ranking_correct / n_boot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(20260509)

    # True DGP. Attention is deliberately not co-monotone with the ranking.
    J = 5
    ranking_true = np.array([0, 1, 2, 3, 4], dtype=int)
    gamma_true = np.array([0.40, 0.70, 0.50, 0.80, 0.60])
    n_subjects = 500
    n_boot = 80

    menus = enumerate_menus(J)
    counts = simulate_counts(menus, ranking_true, gamma_true, n_subjects, rng)

    # Method 1: joint MLE with brute-force ranking
    ranking_m1, gamma_m1, ll_m1 = fit_method1(counts, menus)

    # Method 2: revealed-preference two-step
    ranking_m2, gamma_m2 = fit_method2(counts, menus)
    ll_m2 = log_likelihood_at(ranking_m2, gamma_m2, counts, menus)

    # Luce benchmark
    u_luce, ll_luce = fit_luce(counts, menus)
    luce_pred = luce_probs(u_luce, menus)

    # Log-likelihood at the true primitives: the ceiling every estimator
    # is chasing. Each fitted log-likelihood is read against this value.
    ll_true = log_likelihood_at(ranking_true, gamma_true, counts, menus)

    # Bootstrap (Method 1 only; Method 2 is faster but its discrete output
    # makes bootstrap ranking-recovery the headline rather than a SE per gamma)
    gamma_se_m1, ranking_acc_m1 = bootstrap_se(
        menus, ranking_true, gamma_true, n_subjects, n_boot, rng, fit_method1,
    )
    _, ranking_acc_m2 = bootstrap_se(
        menus, ranking_true, gamma_true, n_subjects, n_boot, rng, fit_method2,
    )

    # Replicate Manzini-Mariotti Example 2 (stochastic intransitivity)
    J_ex = 3
    ranking_ex = np.array([0, 1, 2], dtype=int)  # a, b, c
    gamma_ex = np.array([4.0 / 9.0, 0.5, 0.9])
    menus_ex = enumerate_menus(J_ex)
    probs_ex = all_menu_probs(ranking_ex, gamma_ex, menus_ex)
    # Map menus to readable labels
    label_for = {tuple(m.tolist()): "".join("abc"[j] for j in range(J_ex) if m[j])
                  for m in menus_ex}
    # Look up menu rows by membership mask rather than by hardcoded integer,
    # so the labels {a,b}, {b,c}, {a,c} always point at the right rows.
    def menu_row(members: list[bool]) -> int:
        return next(i for i, m in enumerate(menus_ex)
                    if np.array_equal(m, np.array(members, dtype=bool)))

    i_ab = menu_row([True, True, False])   # menu {a, b}
    i_bc = menu_row([False, True, True])   # menu {b, c}
    i_ac = menu_row([True, False, True])   # menu {a, c}
    p_ab = {"a": probs_ex[i_ab, 0], "b": probs_ex[i_ab, 1]}
    p_bc = {"b": probs_ex[i_bc, 1], "c": probs_ex[i_bc, 2]}
    p_ac = {"a": probs_ex[i_ac, 0], "c": probs_ex[i_ac, 2]}

    # =====================================================================
    # Figures and tables
    # =====================================================================
    setup_style()

    # ------------------------------------------------------------------
    # Figure 1: observed vs predicted choice frequency for every (menu, alt) cell
    # ------------------------------------------------------------------
    pred_m1 = all_menu_probs(ranking_m1, gamma_m1, menus)
    obs_freq = counts / counts.sum(axis=1, keepdims=True)
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    for m_idx in range(len(menus)):
        for j in range(J + 1):
            if menus[m_idx, j] if j < J else True:
                color = "tab:red" if j == J else "tab:blue"
                ax1.scatter(pred_m1[m_idx, j], obs_freq[m_idx, j],
                            color=color, s=20, alpha=0.6,
                            edgecolor="black", linewidth=0.3)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="45 degrees")
    ax1.set_xlabel("Method-1 predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Observed vs predicted choice frequencies (Method 1 fit)")
    ax1.scatter([], [], color="tab:blue", s=30, edgecolor="black",
                label="Inside alternative")
    ax1.scatter([], [], color="tab:red", s=30, edgecolor="black",
                label="Default option")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    within_one_se = bool(np.all(np.abs(gamma_m1 - gamma_true) <= gamma_se_m1))
    n_within_se = int(np.sum(np.abs(gamma_m1 - gamma_true) <= gamma_se_m1))
    if within_one_se:
        attention_sentence = (
            "and the recovered attention parameters lie within one bootstrap "
            "standard error of the truth on every alternative."
        )
    else:
        attention_sentence = (
            f"and the recovered attention parameters lie within one bootstrap "
            f"standard error of the truth on {n_within_se} of {J} "
            "alternatives."
        )
    save_figure(fig1, "figures/menu-fit.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 2: menu-removal asymmetry heat-map
    # ------------------------------------------------------------------
    impact_matrix = np.full((J, J), np.nan)
    for i in range(J):
        for j in range(J):
            if i == j:
                continue
            ratios = []
            for m_idx, mask in enumerate(menus):
                if not (mask[i] and mask[j]):
                    continue
                target_mask = mask.copy()
                target_mask[i] = False
                target_idx = next(
                    (k for k, m in enumerate(menus)
                     if np.array_equal(m, target_mask)), None,
                )
                if target_idx is None:
                    continue
                p_full = obs_freq[m_idx, j]
                p_removed = obs_freq[target_idx, j]
                if p_full > 1e-9:
                    ratios.append(p_removed / p_full)
            if ratios:
                impact_matrix[i, j] = float(np.mean(ratios))

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = ax2.imshow(impact_matrix, cmap="RdBu_r", vmin=0.5, vmax=2.0,
                    origin="upper")
    plt.colorbar(im, ax=ax2,
                  label=r"Mean $\hat p(j, A \setminus \{i\}) / \hat p(j, A)$")
    labels = [f"$a_{j+1}$" for j in range(J)]
    ax2.set_xticks(range(J))
    ax2.set_yticks(range(J))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Alternative whose probability we measure ($j$)")
    ax2.set_ylabel("Alternative removed from the menu ($i$)")
    ax2.set_title("Menu-removal impact ratios")
    for i in range(J):
        for j in range(J):
            if not np.isnan(impact_matrix[i, j]):
                color = "white" if abs(impact_matrix[i, j] - 1) > 0.4 else "black"
                ax2.text(j, i, f"{impact_matrix[i, j]:.2f}",
                          ha="center", va="center", color=color, fontsize=9)
    save_figure(fig2, "figures/menu-removal-asymmetry.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3: replication of Manzini-Mariotti Example 2 (intransitivity)
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    pairs = [
        (r"$p(a, \{a, b\})$", p_ab["a"]),
        (r"$p(b, \{b, c\})$", p_bc["b"]),
        (r"$p(a, \{a, c\})$", p_ac["a"]),
    ]
    labels_x = [pair[0] for pair in pairs]
    values = [pair[1] for pair in pairs]
    bars = ax3.bar(labels_x, values, color=["tab:blue", "tab:green", "tab:orange"],
                   edgecolor="black", linewidth=0.5)
    ax3.axhline(0.5, color="tab:red", linestyle="--", linewidth=1.5,
                label="Weak stochastic transitivity threshold = 0.5")
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                  f"{val:.3f}", ha="center", fontsize=10)
    ax3.set_ylabel("Choice probability")
    ax3.set_title(r"Example 2: $\gamma(a) = 4/9$, $\gamma(b) = 1/2$, $\gamma(c) = 9/10$, $a \succ b \succ c$")
    ax3.set_ylim(0, 0.7)
    ax3.legend(loc="upper right", fontsize=9)
    save_figure(fig3, "figures/intransitivity-replication.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 4: parameter recovery
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(J)
    width = 0.25
    ax4.bar(x_pos - width, gamma_true, width, color="tab:gray",
            edgecolor="black", linewidth=0.5, label="True")
    ax4.bar(x_pos, gamma_m1, width, yerr=gamma_se_m1, color="tab:blue",
            edgecolor="black", linewidth=0.5, capsize=3, label="Method 1 MLE")
    ax4.bar(x_pos + width, gamma_m2, width, color="tab:green",
            edgecolor="black", linewidth=0.5, label="Method 2 moments")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"$a_{j+1}$" for j in range(J)])
    ax4.set_ylabel(r"Attention probability $\gamma$")
    ax4.set_title("Attention recovery: true vs Method 1 MLE vs Method 2 moments")
    ax4.set_ylim(0, 1)
    ax4.legend(loc="upper left", fontsize=9)
    save_figure(fig4, "figures/parameter-recovery.png", dpi=150)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    rank_correct_m1 = sum(
        1 for i in range(J) for j in range(i + 1, J)
        if (np.where(ranking_m1 == i)[0][0] < np.where(ranking_m1 == j)[0][0]) ==
           (np.where(ranking_true == i)[0][0] < np.where(ranking_true == j)[0][0])
    )
    rank_correct_m2 = sum(
        1 for i in range(J) for j in range(i + 1, J)
        if (np.where(ranking_m2 == i)[0][0] < np.where(ranking_m2 == j)[0][0]) ==
           (np.where(ranking_true == i)[0][0] < np.where(ranking_true == j)[0][0])
    )
    n_pairs = J * (J - 1) // 2
    rank_table = pd.DataFrame({
        "Method": ["Method 1 MLE", "Method 2 moments"],
        "Estimated ranking (best to worst)": [
            " > ".join(f"a_{j+1}" for j in ranking_m1),
            " > ".join(f"a_{j+1}" for j in ranking_m2),
        ],
        "Correct pairs": [f"{rank_correct_m1} / {n_pairs}",
                           f"{rank_correct_m2} / {n_pairs}"],
        "Bootstrap exact-ranking rate": [f"{ranking_acc_m1:.0%}", f"{ranking_acc_m2:.0%}"],
    })
    if rank_correct_m1 == n_pairs and rank_correct_m2 == n_pairs:
        point_estimate_sentence = (
            "Both methods recover the full ranking on the point estimate at "
            "this sample size. "
        )
    elif rank_correct_m1 == n_pairs:
        point_estimate_sentence = (
            "Method 1 recovers the full ranking on the point estimate; "
            f"Method 2 gets {rank_correct_m2} of {n_pairs} pairs right and "
            "swaps one adjacent pair. "
        )
    else:
        point_estimate_sentence = (
            f"Method 1 gets {rank_correct_m1} of {n_pairs} pairs and "
            f"Method 2 gets {rank_correct_m2} of {n_pairs} on the point "
            "estimate. "
        )
    Path("tables").mkdir(parents=True, exist_ok=True)
    rank_table.to_csv("tables/ranking-recovery.csv", index=False)

    pred_m2 = all_menu_probs(ranking_m2, gamma_m2, menus)
    eps = 1e-12
    kl_true = lambda probs: float(np.sum(
        all_menu_probs(ranking_true, gamma_true, menus)
        * (np.log(np.maximum(all_menu_probs(ranking_true, gamma_true, menus), eps))
           - np.log(np.maximum(probs, eps)))
    ))
    method_table = pd.DataFrame({
        "Method": ["True DGP", "Method 1 MLE", "Method 2 moments",
                   "Luce / MNL benchmark"],
        "Log-likelihood": [f"{ll_true:.1f}", f"{ll_m1:.1f}", f"{ll_m2:.1f}",
                           f"{ll_luce:.1f}"],
        "KL divergence to true": ["0.0000",
                                  f"{kl_true(pred_m1):.4f}",
                                  f"{kl_true(pred_m2):.4f}",
                                  f"{kl_true(luce_pred):.4f}"],
        "Captures asymmetric impact": ["yes", "yes", "yes", "no"],
    })
    method_table.to_csv("tables/method-comparison.csv", index=False)

    save_thumbnail("figures/menu-fit.png", "figures/thumb.png")
    print("\nDone: 4 figures, 2 tables, thumb reproduced.")


if __name__ == "__main__":
    main()
