#!/usr/bin/env python3
"""Sequential consumer search in the Ursu-Seiler-Honka framework."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


PRODUCTS = ["Basic", "Comfort", "Sport", "Premium", "Boutique"]
COST_COMPLEXITY_SLOPE = 0.48
RESERVATION_BRACKET = (-8.0, 8.0)
MOMENT_SCALE_FLOOR = 0.08
SEARCH_START = np.array([0.95, np.log(0.09)])
SEARCH_MAXITER = 220
SEARCH_XATOL = 1e-4
SEARCH_FATOL = 1e-6
COUNTERFACTUAL_MULTIPLIERS = [0.50, 0.75, 1.00, 1.50, 2.00]


def expected_gain_over_threshold(k: float) -> float:
    """E[max(Z-k,0)] for Z distributed standard normal."""
    return norm.pdf(k) - k * (1.0 - norm.cdf(k))


def reservation_values(mean_utility: np.ndarray, search_cost: np.ndarray, match_sd: float) -> np.ndarray:
    """Compute Weitzman reservation values for normal match values."""
    values = np.zeros_like(mean_utility)
    for j, cost in enumerate(search_cost):
        target = cost / match_sd

        def equation(k: float) -> float:
            return expected_gain_over_threshold(k) - target

        k_star = brentq(equation, *RESERVATION_BRACKET)
        values[j] = mean_utility[j] + match_sd * k_star
    return values


def unpack_theta(theta: np.ndarray) -> tuple[float, float]:
    """Map optimizer parameters into structural primitives."""
    beta_quality = theta[0]
    base_cost = np.exp(theta[1])
    return beta_quality, base_cost


def primitives(theta: np.ndarray, quality: np.ndarray, price: np.ndarray, complexity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean utilities and search costs implied by theta."""
    beta_quality, base_cost = unpack_theta(theta)
    price_taste = 0.32
    mean_utility = beta_quality * quality - price_taste * price
    search_cost = base_cost * np.exp(COST_COMPLEXITY_SLOPE * complexity)
    return mean_utility, search_cost


def simulate_search(
    theta: np.ndarray,
    draws: np.ndarray,
    quality: np.ndarray,
    price: np.ndarray,
    complexity: np.ndarray,
    match_sd: float,
    cost_multiplier: float = 1.0,
) -> dict[str, np.ndarray]:
    """Simulate sequential search with perfect recall."""
    mean_utility, search_cost = primitives(theta, quality, price, complexity)
    search_cost = cost_multiplier * search_cost
    match_value = mean_utility[None, :] + match_sd * draws
    reservation = reservation_values(mean_utility, search_cost, match_sd)
    order = np.argsort(-reservation)

    n_consumers, n_products = match_value.shape
    search_flags = np.zeros((n_consumers, n_products), dtype=bool)
    search_order = np.full((n_consumers, n_products), -1, dtype=int)
    best_value = np.zeros(n_consumers)
    best_product = np.full(n_consumers, n_products, dtype=int)
    search_count = np.zeros(n_consumers, dtype=int)

    for product in order:
        active = reservation[product] > best_value
        if not np.any(active):
            continue
        positions = search_count[active]
        search_order[active, positions] = product
        search_flags[active, product] = True
        search_count[active] += 1
        improved = active & (match_value[:, product] > best_value)
        best_value[improved] = match_value[improved, product]
        best_product[improved] = product

    return {
        "purchase": best_product,
        "search_flags": search_flags,
        "search_order": search_order,
        "search_count": search_count,
        "reservation": reservation,
        "mean_utility": mean_utility,
        "search_cost": search_cost,
        "match_value": match_value,
    }


def full_information_choice(
    theta: np.ndarray,
    draws: np.ndarray,
    quality: np.ndarray,
    price: np.ndarray,
    complexity: np.ndarray,
    match_sd: float,
) -> np.ndarray:
    """Choices if consumers saw all match values before choosing."""
    mean_utility, _ = primitives(theta, quality, price, complexity)
    match_value = mean_utility[None, :] + match_sd * draws
    outside = np.zeros((match_value.shape[0], 1))
    all_values = np.column_stack([match_value, outside])
    return all_values.argmax(axis=1)


def shares(choice: np.ndarray, n_products: int) -> np.ndarray:
    """Inside-product and outside-option shares."""
    return np.bincount(choice, minlength=n_products + 1) / len(choice)


def moments(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Moment vector for simulated-moments estimation."""
    n_products = sample["search_flags"].shape[1]
    purchase_shares = shares(sample["purchase"], n_products)
    search_rates = sample["search_flags"].mean(axis=0)
    search_count = sample["search_count"]
    return np.r_[
        search_rates,
        purchase_shares,
        search_count.mean(),
        (search_count == 1).mean(),
    ]


def criterion(
    theta: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    draws: np.ndarray,
    quality: np.ndarray,
    price: np.ndarray,
    complexity: np.ndarray,
    match_sd: float,
) -> float:
    """Scaled simulated-moments criterion."""
    beta_quality, base_cost = unpack_theta(theta)
    if beta_quality < 0.2 or beta_quality > 2.2 or base_cost < 0.015 or base_cost > 0.70:
        return 1e6
    sample = simulate_search(theta, draws, quality, price, complexity, match_sd)
    diff = (moments(sample) - target) / scale
    return float(diff @ diff)


def estimate_by_moments(
    target: np.ndarray,
    draws: np.ndarray,
    quality: np.ndarray,
    price: np.ndarray,
    complexity: np.ndarray,
    match_sd: float,
) -> dict[str, object]:
    """Estimate preference and search-cost parameters."""
    scale = np.maximum(np.abs(target), MOMENT_SCALE_FLOOR)
    result = minimize(
        criterion,
        SEARCH_START,
        args=(target, scale, draws, quality, price, complexity, match_sd),
        method="Nelder-Mead",
        options={"maxiter": SEARCH_MAXITER, "xatol": SEARCH_XATOL, "fatol": SEARCH_FATOL, "disp": False},
    )
    theta_hat = np.asarray(result.x, dtype=float)
    fitted = simulate_search(theta_hat, draws, quality, price, complexity, match_sd)
    residual = (moments(fitted) - target) / scale
    return {
        "theta": theta_hat,
        "criterion": float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
        "sample": fitted,
        "residual": residual,
    }


def parameter_table(theta_true: np.ndarray, estimate: dict[str, object]) -> pd.DataFrame:
    """Known-truth parameter recovery."""
    beta_true, base_true = unpack_theta(theta_true)
    beta_hat, base_hat = unpack_theta(np.asarray(estimate["theta"], dtype=float))
    return pd.DataFrame(
        {
            "Parameter": ["Quality taste", "Base search cost", "Complexity cost slope"],
            "True": [beta_true, base_true, COST_COMPLEXITY_SLOPE],
            "Estimate": [beta_hat, base_hat, COST_COMPLEXITY_SLOPE],
            "Error": [beta_hat - beta_true, base_hat - base_true, 0.0],
            "Status": ["estimated", "estimated", "fixed"],
        }
    )


def moment_table(target: np.ndarray, estimate: dict[str, object], product_names: list[str]) -> pd.DataFrame:
    """Observed and fitted moment diagnostics."""
    fitted_moments = moments(estimate["sample"])
    names = [f"Search rate: {name}" for name in product_names]
    names += [f"Purchase share: {name}" for name in product_names] + ["Purchase share: Outside"]
    names += ["Average searches", "One-search rate"]
    return pd.DataFrame(
        {
            "Moment": names,
            "Observed target": target,
            "Simulated at estimate": fitted_moments,
            "Difference": fitted_moments - target,
        }
    )


def counterfactual_table(
    theta: np.ndarray,
    draws: np.ndarray,
    quality: np.ndarray,
    price: np.ndarray,
    complexity: np.ndarray,
    match_sd: float,
) -> pd.DataFrame:
    """Search-cost counterfactuals."""
    rows = []
    for multiplier in COUNTERFACTUAL_MULTIPLIERS:
        sample = simulate_search(theta, draws, quality, price, complexity, match_sd, cost_multiplier=multiplier)
        purchase_shares = shares(sample["purchase"], len(PRODUCTS))
        rows.append(
            {
                "Search cost multiplier": multiplier,
                "Average searches": sample["search_count"].mean(),
                "Inside purchase share": purchase_shares[:-1].sum(),
                "Outside share": purchase_shares[-1],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    quality = np.array([1.0, 1.45, 1.90, 2.35, 1.75])
    price = np.array([1.3, 1.8, 2.5, 3.4, 2.7])
    complexity = np.array([0.25, 0.45, 0.70, 1.05, 1.20])
    match_sd = 0.85
    theta_true = np.array([1.18, np.log(0.08)])
    n_observed = 4_000
    n_simulated = 9_000
    observed_draws = np.random.default_rng(711).normal(size=(n_observed, len(PRODUCTS)))
    simulation_draws = np.random.default_rng(712).normal(size=(n_simulated, len(PRODUCTS)))

    observed = simulate_search(theta_true, observed_draws, quality, price, complexity, match_sd)
    target = moments(observed)
    estimate = estimate_by_moments(target, simulation_draws, quality, price, complexity, match_sd)
    fitted = estimate["sample"]
    counterfactuals = counterfactual_table(np.asarray(estimate["theta"], dtype=float), simulation_draws, quality, price, complexity, match_sd)

    full_info = full_information_choice(np.asarray(estimate["theta"], dtype=float), simulation_draws, quality, price, complexity, match_sd)
    sequential_shares = shares(fitted["purchase"], len(PRODUCTS))
    full_info_shares = shares(full_info, len(PRODUCTS))

    print("Sequential-search tutorial")
    print(f"  True theta: {theta_true}")
    print(f"  Estimated theta: {np.asarray(estimate['theta'])}")
    print(f"  Criterion: {float(estimate['criterion']):.4f}")
    print(f"  Success: {estimate['success']}")

    setup_style()
    report = ModelReport(
        "Consumer Search with Sequential Inspection Costs",
        "Estimate search costs from observed search paths and purchases.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A consumer does not see every product match value before choosing. She can "
        "inspect products one at a time, pay a search cost, remember what she has seen, "
        "and stop when further inspection is no longer worth it.\n\n"
        "The tutorial implements the sequential search logic used in the empirical "
        "framework surveyed by Ursu, Seiler, and Honka. Search paths matter because "
        "they reveal more than final purchases. They help separate preference for a "
        "product from the cost of learning about it.\n\n"
        "The primitive object is a consideration process. A product can be valuable if "
        "inspected, but still rarely bought because consumers do not pay to inspect it. "
        "That is why the data include both searched products and final purchases."
    )

    report.add_equations(
        r"""
There are $J$ inside products and an outside option with value zero. Product
$j$ has observable quality $q_j$, price $p_j$, and complexity $x_j$. Before
search, the consumer knows the product's mean match value

$$
\begin{aligned}
\mu_j
&=
\beta q_j - \alpha p_j,
\end{aligned}
$$

The term $\mu_j$ is not the utility from buying product $j$. It is the expected
value before the consumer learns whether the product is a good personal fit.
Inspection reveals an idiosyncratic match component, so the realized value is

$$
\begin{aligned}
u_{ij}
&=
\mu_j + \sigma \varepsilon_{ij},
\qquad \varepsilon_{ij}\sim N(0,1).
\end{aligned}
$$

The shock $\varepsilon_{ij}$ is consumer-specific. Two consumers can face the
same price and quality but learn different match values after inspection. This
is why search paths carry information beyond final purchases.

Inspecting product $j$ costs

$$
\begin{aligned}
c_j
&=
c_0 \exp(\gamma x_j),
\end{aligned}
$$

where $x_j$ is product complexity. Higher complexity raises the cost of
learning about the product, not the utility from owning it. The consumer pays
$c_j$ before observing $u_{ij}$.

With perfect recall, the consumer keeps every inspected value. The Weitzman
reservation value $z_j$ is the cutoff that makes the option value of inspecting
product $j$ equal to its search cost:

$$
\begin{aligned}
c_j
&=
E[\max(u_{ij}-z_j,0)].
\end{aligned}
$$

The expectation is over the unknown match draw for product $j$. A high $z_j$
means the product is worth inspecting early because it has high mean utility,
low search cost, or enough upside risk.

After some inspections, the consumer has an inspected set $S_i$ and a current
best value

$$
\begin{aligned}
b_i
&=
\max\{0,\max_{j\in S_i} u_{ij}\}.
\end{aligned}
$$

The outside option enters through the zero in $b_i$. If every inspected product
has negative realized value, the best available action is not to buy.

The search rule is a threshold rule. Among uninspected products, the consumer
looks at the product with the highest reservation value. If that value exceeds
$b_i$, she searches it and updates $S_i$ and $b_i$. If the highest remaining
reservation value is below $b_i$, every other uninspected product has even lower
option value, so she stops.

The simulated-moments estimator chooses

$$
\begin{aligned}
\hat\theta
&=
\arg\min_\theta
\left[m_{sim}(\theta)-m_{obs}\right]^{\top}
W
\left[m_{sim}(\theta)-m_{obs}\right].
\end{aligned}
$$

The observed moment vector $m_{obs}$ summarizes search and purchase behavior.
The simulated vector $m_{sim}(\theta)$ is built by simulating complete search
paths under the same stopping rule. In this tutorial the moments include product
search rates, purchase shares, average searches, and the probability of
stopping after one search.

The weighting matrix $W$ is diagonal in the implementation. Each moment is
scaled so that a small purchase or search rate does not dominate the objective
only because it is measured in smaller units.

In this exercise, the price taste $\alpha$, match-value scale $\sigma$, and
complexity slope $\gamma$ are fixed. The estimator recovers the quality taste
$\beta$ and the base search-cost level $c_0$. That two-parameter target keeps
the tutorial focused on the central identification problem: separating products
that consumers dislike from products that consumers rarely inspect.
"""
    )

    beta_true, base_true = unpack_theta(theta_true)
    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| Products | {len(PRODUCTS)} | Alternatives that can be inspected |\n"
        f"| Observed consumers | {n_observed:,} | Synthetic search paths and purchases |\n"
        f"| Simulation consumers | {n_simulated:,} | Fixed draws for simulated moments |\n"
        f"| True quality taste | {beta_true:.2f} | Preference weight on product quality |\n"
        f"| True base search cost | {base_true:.3f} | Cost level before complexity adjustment |\n"
        f"| Complexity slope | {COST_COMPLEXITY_SLOPE:.2f} | Fixed search-cost increase with product complexity |\n"
        f"| Match-value sd | {match_sd:.2f} | Fixed uncertainty about product fit |\n\n"
        f"**Numerical settings**\n\n"
        f"| Setting | Value | Role |\n"
        f"|---------|-------|------|\n"
        f"| Optimizer | Nelder-Mead | Derivative-free search over quality taste and log base cost |\n"
        f"| Start | ({SEARCH_START[0]:.2f}, log {np.exp(SEARCH_START[1]):.2f}) | Initial quality taste and base search cost |\n"
        f"| Moment scale floor | {MOMENT_SCALE_FLOOR:.2f} | Prevents tiny moments from dominating the criterion |\n"
        f"| Reservation bracket | [{RESERVATION_BRACKET[0]:.0f}, {RESERVATION_BRACKET[1]:.0f}] | Root-search bracket for the normal reservation equation |\n"
        f"| Max iterations | {SEARCH_MAXITER} | Nelder-Mead iteration cap |\n"
        f"| Tolerances | xatol={SEARCH_XATOL:.0e}, fatol={SEARCH_FATOL:.0e} | Stopping rule for parameter moves and criterion changes |\n"
        f"| Counterfactual grid | {COUNTERFACTUAL_MULTIPLIERS} | Search-cost multipliers used in the policy experiment |"
    )

    report.add_solution_method(
        r"""
The search rule has two layers. First, reservation values rank products before the consumer knows her idiosyncratic match values. Second, after each search, the realized match value updates the current best option. Search continues only when the best remaining reservation value is above that current best value.

For a normal match distribution, the reservation equation can be solved as a one-dimensional root. A high search cost lowers $z_j$ because the product must offer more option value before inspection is worthwhile. A high mean utility raises $z_j$ because the product is likely to be useful if inspected.

### Algorithm 1. Reservation order

**Inputs.** Product primitives $\{q_j,p_j,x_j\}_{j=1}^J$, trial parameter $\theta=(\beta,\ell_c)$, and fixed $(\alpha,\gamma,\sigma)$.

**Outputs.** Reservation values $\{z_j(\theta)\}_{j=1}^J$ and priority order $\pi(\theta)$.

1. Convert the log cost into a positive base search cost:

$$
c_0=\exp(\ell_c).
$$

2. For each product $j$, compute mean utility and search cost:

$$
\mu_j(\theta)=\beta q_j-\alpha p_j,
\qquad
c_j(\theta)=c_0\exp(\gamma x_j).
$$

3. Solve the standardized reservation equation:

$$
G(k_j)=c_j(\theta)/\sigma,
\qquad
G(k)=\phi(k)-k[1-\Phi(k)],
$$

4. Recover the reservation value:

$$
z_j(\theta)=\mu_j(\theta)+\sigma k_j.
$$

5. Sort products by reservation values. The priority order $\pi(\theta)$ satisfies

$$
z_{\pi_1}(\theta)\geq z_{\pi_2}(\theta)\geq\cdots\geq z_{\pi_J}(\theta).
$$

### Algorithm 2. Simulate one search path

**Inputs.** Reservation values $\{z_j(\theta)\}$, order $\pi(\theta)$, shocks $\{\varepsilon_{ij}\}_{j=1}^J$, mean utilities $\{\mu_j(\theta)\}_{j=1}^J$, and match-value scale $\sigma$.

**Outputs.** Inspected set $S_i$, terminal best value $b_i$, and purchase $y_i$.

1. Initialize the path with no inspected products:

$$
S_i=\varnothing,
\qquad
b_i=0,
\qquad
y_i=0.
$$

2. For step $h=1,\ldots,J$, take the next product in reservation order: $j=\pi_h(\theta)$.

3. If $z_j(\theta)\leq b_i$, stop search.

4. If $z_j(\theta)>b_i$, inspect product $j$ and update the inspected set:

$$
S_i\leftarrow S_i\cup\{j\}.
$$

5. Reveal the match value:

$$
u_{ij}(\theta)=\mu_j(\theta)+\sigma\varepsilon_{ij}.
$$

6. If $u_{ij}(\theta)>b_i$, update the best option:

$$
b_i\leftarrow u_{ij}(\theta),
\qquad
y_i\leftarrow j.
$$

7. If no stopping condition has been met, return to step 2 for the next product.

8. Return $S_i$, $|S_i|$, $b_i$, and $y_i$.

The estimator simulates the full path for many consumers at each parameter vector. It matches search rates and purchase shares, so the same product can be identified as hard to discover rather than simply low quality.

### Algorithm 3. Estimate $\theta$ by simulated moments

**Inputs.** Observed moments $m_{obs}$, fixed simulation shocks $\{\varepsilon_{sj}\}_{s=1,j=1}^{S,J}$, starting value $\theta_0$, and scale floor $a_{min}$.

**Output.** Simulated-moments estimate $\hat\theta$.

1. For each moment $\ell$, compute the scale:

$$
a_\ell=\max\{|m_{obs,\ell}|,a_{min}\}.
$$

2. Let Nelder-Mead propose a candidate $\theta^m=(\beta^m,\ell_c^m)$.

3. At $\theta^m$, compute $z_j(\theta^m)$ and $\pi(\theta^m)$ using Algorithm 1.

4. For each simulated consumer $s$, simulate
$\{S_s(\theta^m), b_s(\theta^m), y_s(\theta^m)\}$ using Algorithm 2.

5. Build the simulated moment vector:

$$
m_{sim}(\theta^m)=
\left(
\Pr_{sim}\{j\in S_s(\theta^m)\}_{j=1}^J,\,
\Pr_{sim}\{y_s(\theta^m)=j\}_{j=1}^J,\,
E_{sim}|S_s(\theta^m)|,\,
\Pr_{sim}\{|S_s(\theta^m)|=1\}
\right).
$$

6. Evaluate the scaled criterion:

$$
Q_S(\theta^m)=
\sum_\ell
\left(
\frac{m_{sim,\ell}(\theta^m)-m_{obs,\ell}}{a_\ell}
\right)^2,
\qquad
\hat\theta=\arg\min_\theta Q_S(\theta).
$$

7. Continue until Nelder-Mead stops and return $\hat\theta$.

Purchase data alone confound low demand with high search costs. Search paths help because a product can be attractive among consumers who inspect it but rarely inspected when its search cost is high.
"""
    )

    observed_search = observed["search_flags"].mean(axis=0)
    fitted_search = fitted["search_flags"].mean(axis=0)
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(PRODUCTS))
    width = 0.35
    axes1[0].bar(x - width / 2, observed_search, width, label="Observed")
    axes1[0].bar(x + width / 2, fitted_search, width, label="Estimated")
    axes1[0].set_xticks(x)
    axes1[0].set_xticklabels(PRODUCTS, rotation=20)
    axes1[0].set_ylabel("Search rate")
    axes1[0].set_title("Product Inspection")
    axes1[0].legend()
    observed_purchase = shares(observed["purchase"], len(PRODUCTS))
    fitted_purchase = shares(fitted["purchase"], len(PRODUCTS))
    labels = PRODUCTS + ["Outside"]
    x2 = np.arange(len(labels))
    axes1[1].bar(x2 - width / 2, observed_purchase, width, label="Observed")
    axes1[1].bar(x2 + width / 2, fitted_purchase, width, label="Estimated")
    axes1[1].set_xticks(x2)
    axes1[1].set_xticklabels(labels, rotation=20)
    axes1[1].set_ylabel("Purchase share")
    axes1[1].set_title("Final Choices")
    axes1[1].legend()
    report.add_results(
        "The fitted model matches both product inspection and final choice. This matters "
        "because high-quality products can have low purchase shares either because they "
        "are unattractive or because few consumers pay to learn about them."
    )
    report.add_figure("figures/search-and-choice-fit.png", "Search and purchase fit", fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    x_all = np.arange(len(labels))
    ax2.bar(x_all - width / 2, sequential_shares, width, label="Sequential search")
    ax2.bar(x_all + width / 2, full_info_shares, width, label="Full information")
    ax2.set_xticks(x_all)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_ylabel("Choice share")
    ax2.set_title("Demand With and Without Search Frictions")
    ax2.legend()
    report.add_results(
        "Full-information demand is the benchmark where every match value is observed "
        "for free. Sequential search shifts demand because some products never enter a "
        "consumer's consideration set."
    )
    report.add_figure("figures/consideration-demand.png", "Sequential-search versus full-information demand", fig2)

    fig3, ax3 = plt.subplots(figsize=(7.5, 4.8))
    ax3.plot(counterfactuals["Search cost multiplier"], counterfactuals["Average searches"], marker="o", label="Average searches")
    ax3.set_xlabel("Search cost multiplier")
    ax3.set_ylabel("Average searches")
    ax3b = ax3.twinx()
    ax3b.plot(counterfactuals["Search cost multiplier"], counterfactuals["Inside purchase share"], marker="s", color="tab:red", label="Inside share")
    ax3b.set_ylabel("Inside purchase share")
    ax3.set_title("Search-Cost Counterfactual")
    lines, labels_left = ax3.get_legend_handles_labels()
    lines_b, labels_right = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines_b, labels_left + labels_right, loc="center right")
    report.add_results(
        "Increasing search costs lowers the amount of inspection and pushes some "
        "consumers to stop earlier. The inside purchase share falls because consumers "
        "are less likely to discover a product match that beats the outside option."
    )
    report.add_figure("figures/search-cost-counterfactual.png", "Search-cost counterfactual", fig3)

    report.add_results(
        "Known-truth recovery is approximate because the estimator matches moments, not "
        "the exact likelihood. The residual table shows which observed search and "
        "purchase summaries drive the fit."
    )
    report.add_table(
        "tables/parameter-recovery.csv",
        "Known-truth parameter recovery",
        parameter_table(theta_true, estimate).round(4),
    )
    report.add_table(
        "tables/moment-fit.csv",
        "Search and purchase moment fit",
        moment_table(target, estimate, PRODUCTS).round(4),
    )
    report.add_table(
        "tables/search-cost-counterfactual.csv",
        "Search-cost counterfactuals",
        counterfactuals.round(4),
    )

    report.add_takeaway(
        r"""
Sequential search turns observed demand into a joint outcome of preferences and information acquisition. Search-path data are valuable because they show which products entered consideration before the purchase. That is the key empirical distinction between a search model and a full-information discrete choice model.
"""
    )

    report.add_references(
        [
            "[Ursu, R., Seiler, S., and Honka, E. (2025). The sequential search model: A framework for empirical research. *Quantitative Marketing and Economics*, 23, 165-213.](https://doi.org/10.1007/s11129-024-09291-2)",
            "[Weitzman, M. L. (1979). Optimal Search for the Best Alternative. *Econometrica*, 47(3), 641-654.](https://doi.org/10.2307/1910412)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
