#!/usr/bin/env python3
"""Lucas-tree asset pricing with news shocks.

The Dynare file describes a representative-agent Lucas tree in which dividends
are also consumption. A news shock changes the expected dividend before the
dividend itself moves. The Python report solves the first-order pricing equation
directly and compares it with an exact nonlinear perfect-foresight transition
for the same deterministic shock path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


@dataclass(frozen=True)
class AssetNewsParams:
    """Calibration used by the Dynare source file."""

    beta: float = 0.99
    gamma: float = 2.0
    rho: float = 0.9
    sigma1: float = 0.1
    sigma2: float = 0.1


def read_mod_file(mod_path: Path) -> str:
    """Return the Dynare source text for documentation and sanity checks."""

    return mod_path.read_text()


def steady_state(params: AssetNewsParams) -> dict[str, float]:
    """Compute the deterministic steady state of the Lucas tree."""

    d_ss = 1.0
    p_ss = params.beta / (1.0 - params.beta)
    return {
        "d": d_ss,
        "p": p_ss,
        "pd_ratio": p_ss / d_ss,
        "gross_return": 1.0 / params.beta,
    }


def linear_price_coefficients(params: AssetNewsParams) -> dict[str, float]:
    """Solve q_t = A x_t + B n_t for the first-order price response.

    Here x_t = log d_t and q_t = log(p_t / p_ss). Current news n_t is known at
    date t and shifts x_{t+1} by sigma1.
    """

    beta = params.beta
    gamma = params.gamma
    rho = params.rho
    sigma1 = params.sigma1

    a_x = (gamma + rho * (1.0 - beta - gamma)) / (1.0 - beta * rho)
    b_news = sigma1 * (beta * a_x + 1.0 - beta - gamma)
    return {"A": a_x, "B": b_news}


def dividend_news_state(
    params: AssetNewsParams,
    shock_type: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the deterministic dividend and news-state paths for an IRF."""

    x = np.zeros(horizon)
    n = np.zeros(horizon)

    if shock_type == "surprise":
        x[0] = params.sigma2
    elif shock_type == "news":
        n[0] = 1.0
    else:
        raise ValueError(f"Unknown shock type: {shock_type}")

    for t in range(1, horizon):
        x[t] = params.rho * x[t - 1] + params.sigma1 * n[t - 1]

    return x, n


def nonlinear_price_benchmark(
    params: AssetNewsParams,
    x: np.ndarray,
    extra_periods: int = 260,
) -> np.ndarray:
    """Exact nonlinear perfect-foresight price path for a deterministic x path."""

    ss = steady_state(params)
    x_long = np.zeros(len(x) + extra_periods)
    x_long[: len(x)] = x
    for t in range(len(x), len(x_long)):
        x_long[t] = params.rho * x_long[t - 1]

    d = np.exp(x_long)
    p = np.zeros_like(d)
    p[-1] = ss["p"] * d[-1]

    for t in range(len(d) - 2, -1, -1):
        sdf = params.beta * (d[t + 1] / d[t]) ** (-params.gamma)
        p[t] = sdf * (p[t + 1] + d[t + 1])

    return np.log(p[: len(x)] / ss["p"])


def compute_irf(
    params: AssetNewsParams,
    shock_type: str,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Compute first-order and nonlinear benchmark responses to one shock."""

    ss = steady_state(params)
    coeffs = linear_price_coefficients(params)
    x, n = dividend_news_state(params, shock_type, horizon)
    q_linear = coeffs["A"] * x + coeffs["B"] * n
    q_nonlinear = nonlinear_price_benchmark(params, x)

    return {
        "x": x,
        "n": n,
        "q_linear": q_linear,
        "q_nonlinear": q_nonlinear,
        "d": np.exp(x),
        "p_linear": ss["p"] * np.exp(q_linear),
        "pd_linear": q_linear - x,
    }


def simulate_paths(
    params: AssetNewsParams,
    horizon: int = 220,
    seed: int = 20260504,
) -> dict[str, np.ndarray]:
    """Simulate the first-order model with both surprise and news shocks."""

    rng = np.random.default_rng(seed)
    z = rng.normal(size=horizon)
    n = rng.normal(size=horizon)
    x = np.zeros(horizon)
    coeffs = linear_price_coefficients(params)

    x[0] = params.sigma2 * z[0]
    for t in range(1, horizon):
        x[t] = params.rho * x[t - 1] + params.sigma1 * n[t - 1] + params.sigma2 * z[t]

    q = coeffs["A"] * x + coeffs["B"] * n
    return {
        "x": x,
        "q": q,
        "z": z,
        "n": n,
        "news_contribution": params.sigma1 * n,
        "surprise_contribution": params.sigma2 * z,
    }


def signed_percent(value: float) -> str:
    """Format a log deviation as percent."""

    return f"{100.0 * value:.3f}"


def main() -> None:
    mod_dir = Path(__file__).resolve().parent
    mod_text = read_mod_file(mod_dir / "model.mod")
    params = AssetNewsParams()
    ss = steady_state(params)
    coeffs = linear_price_coefficients(params)

    print("Parsed Dynare source for Lucas-tree news shocks.")
    print(f"  Source length: {len(mod_text.splitlines())} lines")
    print(f"  Steady-state price-dividend ratio: {ss['pd_ratio']:.2f}")
    print(f"  First-order price rule: q_t = {coeffs['A']:.4f} x_t + {coeffs['B']:.4f} n_t")

    horizon = 40
    surprise = compute_irf(params, "surprise", horizon)
    news = compute_irf(params, "news", horizon)
    sim = simulate_paths(params)

    setup_style()
    report = ModelReport(
        "Lucas-Tree News Shocks and Stochastic Discounting",
        "Anticipated dividend news in a representative-agent asset-pricing model where cash-flow news also moves marginal utility.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A news shock separates two dates that simple impulse responses tend to collapse: "
        "the date when agents learn something and the date when the cash flow actually "
        "changes. In this Lucas-tree example, a signal $n_t$ arrives today and shifts next "
        "period's dividend. A surprise shock $z_t$ instead moves today's dividend "
        "immediately.\n\n"
        "The economic wrinkle is that the dividend is also aggregate consumption. Good "
        "dividend news raises future payoffs, but it also implies lower future marginal "
        "utility. With the calibration in `model.mod`, $\\gamma=2$ makes this discount-rate "
        "channel slightly stronger on impact than the cash-flow channel. So the lesson is "
        "sharper than \"prices move before dividends\": anticipated shocks are priced "
        "before they realize, and the sign depends on the stochastic discount factor."
    )

    report.add_equations(
        rf"""
Let $d_t$ be the tree dividend and the representative household's consumption,
and define $x_t=\log d_t$. The Dynare file writes the dividend process as

```text
d = exp(rho*log(d(-1)) + sigma1*n(-1) + sigma2*z)
```

or, in log deviations,

$$
x_t = \rho x_{{t-1}} + \sigma_1 n_{{t-1}} + \sigma_2 z_t.
$$

The surprise innovation $z_t$ is contemporaneous. The news innovation $n_t$ is
known at date $t$ but enters dividends at date $t+1$. The asset-pricing equation is

$$
p_t d_t^{{-\gamma}}
=
\beta \mathbb{{E}}_t\left[
d_{{t+1}}^{{-\gamma}}(p_{{t+1}}+d_{{t+1}})
\right],
$$

which is equivalently

$$
p_t = \mathbb{{E}}_t\left[
M_{{t+1}}(p_{{t+1}}+d_{{t+1}})
\right],
\qquad
M_{{t+1}}=\beta\left(\frac{{d_{{t+1}}}}{{d_t}}\right)^{{-\gamma}}.
$$

At the deterministic steady state $d=1$,

$$
p = \beta(p+1), \qquad p=\frac{{\beta}}{{1-\beta}}={ss["p"]:.2f}.
$$

Write $q_t=\log(p_t/p)$. A first-order expansion of the Euler equation gives

$$
q_t =
\gamma x_t + \beta\mathbb{{E}}_t q_{{t+1}}
+(1-\beta-\gamma)\mathbb{{E}}_t x_{{t+1}}.
$$

Since $\mathbb{{E}}_t x_{{t+1}}=\rho x_t+\sigma_1 n_t$, the linear solution has
the form

$$
q_t = A x_t + B n_t,
$$

with

$$
A=\frac{{\gamma+\rho(1-\beta-\gamma)}}{{1-\beta\rho}},
\qquad
B=\sigma_1\left(\beta A+1-\beta-\gamma\right).
$$

For this calibration, $A={coeffs["A"]:.3f}$ and $B={coeffs["B"]:.3f}$.
"""
    )

    report.add_model_setup(
        "| Primitive | Value | Role |\n"
        "|---|---:|---|\n"
        f"| $\\beta$ | {params.beta:.2f} | Quarterly discount factor |\n"
        f"| $\\gamma$ | {params.gamma:.1f} | CRRA coefficient in marginal utility |\n"
        f"| $\\rho$ | {params.rho:.1f} | Persistence of log dividends |\n"
        f"| $\\sigma_1$ | {params.sigma1:.1f} | Effect of a unit news innovation on next period's log dividend |\n"
        f"| $\\sigma_2$ | {params.sigma2:.1f} | Effect of a unit surprise innovation on today's log dividend |\n"
        f"| IRF horizon | {horizon} quarters | Periods shown in the impulse-response figures |\n\n"
        "| Steady-state object | Value |\n"
        "|---|---:|\n"
        f"| Dividend $d$ | {ss['d']:.3f} |\n"
        f"| Asset price $p$ | {ss['p']:.3f} |\n"
        f"| Price-dividend ratio $p/d$ | {ss['pd_ratio']:.3f} |\n"
        f"| Gross return $1/\\beta$ | {ss['gross_return']:.4f} |"
    )

    report.add_solution_method(
        "The impulse responses use log deviations from steady state. The first-order "
        "solution is the closed-form pricing rule above. The comparison line is an "
        "exact nonlinear perfect-foresight transition for the same realized dividend "
        "path, computed by backward recursion on the level Euler equation. It is not "
        "a separate stochastic model. It is a local-solution check along the same "
        "one-shock experiment.\n\n"
        "```text\n"
        "Algorithm: Lucas-tree news and surprise IRFs\n"
        "Inputs: beta, gamma, rho, sigma1, sigma2, shock type, horizon T\n"
        "Outputs: x_t, q_t, p_t, and the price-dividend ratio\n\n"
        "1. Compute the steady state d=1 and p=beta/(1-beta).\n"
        "2. Linearize the Euler equation in x_t=log d_t and q_t=log(p_t/p).\n"
        "3. Use E_t x_{t+1}=rho x_t + sigma1 n_t to solve q_t=A x_t+B n_t.\n"
        "4. For a surprise shock, set x_0=sigma2 and n_t=0 for all t.\n"
        "5. For a news shock, set n_0=1, x_0=0, and let x_1=sigma1.\n"
        "6. Iterate x_t=rho x_{t-1} after the shock has entered dividends.\n"
        "7. Recover the first-order price response q_t=A x_t+B n_t.\n"
        "8. For the nonlinear benchmark, extend the same x_t path far into the\n"
        "   future and solve p_t=beta(d_{t+1}/d_t)^(-gamma)(p_{t+1}+d_{t+1})\n"
        "   backward from the terminal steady-state price.\n"
        "```\n\n"
        "The sign of $B$ is the diagnostic. Here $B<0$: a positive signal about future "
        "dividends slightly lowers today's price because the cash flow arrives in a "
        "future high-consumption state and is discounted at lower marginal utility."
    )

    periods = np.arange(horizon)

    fig1, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    shock_panels = [
        ("Surprise shock", surprise),
        ("News shock", news),
    ]
    for col, (title, irf) in enumerate(shock_panels):
        ax = axes[0, col]
        ax.plot(periods, 100.0 * irf["x"], color="#1f77b4", linewidth=2.5)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
        ax.set_title(title)
        ax.set_ylabel("Dividend, percent log dev.")
        if title == "News shock":
            ax.axvline(1, color="gray", linestyle=":", linewidth=1.1)

        ax = axes[1, col]
        ax.plot(
            periods,
            100.0 * irf["q_linear"],
            color="#b2182b",
            linewidth=2.5,
            label="First-order price",
        )
        ax.plot(
            periods,
            100.0 * irf["q_nonlinear"],
            color="black",
            linewidth=1.8,
            linestyle="--",
            label="Nonlinear benchmark",
        )
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
        if title == "News shock":
            ax.axvline(1, color="gray", linestyle=":", linewidth=1.1)
        ax.set_xlabel("Quarters after shock")
        ax.set_ylabel("Price, percent log dev.")

    axes[1, 0].legend(frameon=False, loc="upper right")
    fig1.suptitle("Dividend Timing and Asset Prices", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    report.add_results(
        "A surprise shock moves dividends immediately, so the price response is mostly "
        "a magnified version of the current dividend state. The nonlinear benchmark is "
        "nearly indistinguishable from the first-order rule at this scale. A news shock "
        "behaves differently: the dividend is still at steady state on impact, but the "
        "price moves because agents already know $x_1$ will be higher. In this "
        "calibration that impact movement is slightly negative, not positive, because "
        "the marginal-utility effect dominates until the dividend actually realizes."
    )
    report.add_figure(
        "figures/irf-surprise-vs-news.png",
        "Dividend and asset-price impulse responses under surprise and news shocks",
        fig1,
    )
    plt.close(fig1)

    component_values = pd.DataFrame(
        {
            "Channel": [
                "Continuation price",
                "Next dividend payoff",
                "Marginal utility discounting",
                "Net impact",
            ],
            "Contribution": [
                params.beta * coeffs["A"] * params.sigma1,
                (1.0 - params.beta) * params.sigma1,
                -params.gamma * params.sigma1,
                coeffs["B"],
            ],
        }
    )

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    colors = ["#1b6ca8" if v >= 0 else "#b2182b" for v in component_values["Contribution"]]
    ax2.bar(component_values["Channel"], 100.0 * component_values["Contribution"], color=colors)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Date-0 log price contribution, percent")
    ax2.set_title("Why Good Dividend News Need Not Raise Today's Price")
    ax2.tick_params(axis="x", labelrotation=18)
    for idx, value in enumerate(100.0 * component_values["Contribution"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.35 if value >= 0 else -0.35
        ax2.text(idx, value + offset, f"{value:.2f}", ha="center", va=va, fontsize=9)
    fig2.tight_layout()

    report.add_results(
        "The date-0 news response decomposes into three forces. Higher expected future "
        "prices raise today's value, and the next dividend payoff adds a small positive "
        "term. The stochastic discount factor moves the other way: future dividends are "
        "paid in a high-consumption state, where marginal utility is lower. With "
        "$\\gamma=2$, that discounting term is just large enough to make the net impact "
        "negative."
    )
    report.add_figure(
        "figures/price-dynamics.png",
        "Decomposition of the date-0 price response to a positive news shock",
        fig2,
    )
    plt.close(fig2)

    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t_sim = np.arange(len(sim["x"]))
    axes3[0].plot(
        t_sim,
        100.0 * sim["x"],
        color="#1f77b4",
        linewidth=1.4,
        label="Dividend",
    )
    axes3[0].plot(
        t_sim,
        100.0 * sim["q"],
        color="#b2182b",
        linewidth=1.4,
        label="Asset price",
    )
    axes3[0].axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    axes3[0].set_ylabel("Percent log dev.")
    axes3[0].set_title("Simulated Dividend and Price Deviations")
    axes3[0].legend(frameon=False, loc="upper right")

    axes3[1].plot(
        t_sim,
        100.0 * sim["surprise_contribution"],
        color="#4b8f29",
        linewidth=1.0,
        alpha=0.8,
        label=r"Surprise contribution $\sigma_2 z_t$",
    )
    axes3[1].plot(
        t_sim,
        100.0 * sim["news_contribution"],
        color="#6f4aa8",
        linewidth=1.0,
        alpha=0.8,
        label=r"News contribution $\sigma_1 n_t$ to $x_{t+1}$",
    )
    axes3[1].axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    axes3[1].set_xlabel("Quarter")
    axes3[1].set_ylabel("Percent log points")
    axes3[1].set_title("Innovations Feeding the Dividend Process")
    axes3[1].legend(frameon=False, loc="upper right")
    fig3.tight_layout()

    report.add_results(
        "In the simulated path, prices mostly track persistent dividends because the "
        "coefficient on the current dividend state is large. News still matters at the "
        "dates when signals arrive. It enters the price rule immediately through $B n_t$ "
        "and then enters the dividend process one period later through $\\sigma_1 n_t$."
    )
    report.add_figure(
        "figures/simulated-paths.png",
        "Simulated first-order dividend and asset-price paths with surprise and news innovations",
        fig3,
    )
    plt.close(fig3)

    impact_table = pd.DataFrame(
        [
            {
                "Object": "Dividend log deviation",
                "Surprise t=0": signed_percent(surprise["x"][0]),
                "News t=0": signed_percent(news["x"][0]),
                "News t=1": signed_percent(news["x"][1]),
            },
            {
                "Object": "Price log deviation, first order",
                "Surprise t=0": signed_percent(surprise["q_linear"][0]),
                "News t=0": signed_percent(news["q_linear"][0]),
                "News t=1": signed_percent(news["q_linear"][1]),
            },
            {
                "Object": "Price log deviation, nonlinear benchmark",
                "Surprise t=0": signed_percent(surprise["q_nonlinear"][0]),
                "News t=0": signed_percent(news["q_nonlinear"][0]),
                "News t=1": signed_percent(news["q_nonlinear"][1]),
            },
            {
                "Object": "Price-dividend ratio log deviation",
                "Surprise t=0": signed_percent(surprise["pd_linear"][0]),
                "News t=0": signed_percent(news["pd_linear"][0]),
                "News t=1": signed_percent(news["pd_linear"][1]),
            },
        ]
    )

    report.add_results(
        "The impact table is in percent log deviations. The news experiment has zero "
        "dividend movement at date 0 by construction, yet the price and price-dividend "
        "ratio already move. The date-1 column shows the delayed cash-flow realization. "
        "The nonlinear benchmark is close to the first-order solution, so the sign "
        "change is economic, not a plotting artifact."
    )
    report.add_table(
        "tables/impact-responses.csv",
        "Impact and Realization Responses",
        impact_table,
    )

    report.add_takeaway(
        "News shocks are about information timing, not mechanically about higher prices. "
        "The Lucas-tree Euler equation prices a future dividend with the future marginal "
        "utility of consumption. If the dividend is paid in a state where consumption is "
        "high, the stochastic discount factor can offset the cash-flow effect. In this "
        "calibration, positive dividend news moves the price before the dividend, but the "
        "impact sign is slightly negative.\n\n"
        "That makes this tutorial a companion to the "
        "[Lucas-tree dynamic-programming asset-pricing tutorial](../../dynamic-programming/asset-pricing/): "
        "both price payoffs with marginal utility, while this one isolates the timing "
        "distinction between surprise and anticipated shocks. The "
        "[Dynare RBC tutorial](../rbc/) uses the same local-solution logic for real "
        "quantities rather than asset prices."
    )

    report.add_references(
        [
            "Lucas, R. (1978). Asset Prices in an Exchange Economy. *Econometrica*, 46(6), 1429-1445.",
            "Cochrane, J. (2005). *Asset Pricing*. Princeton University Press.",
            "Beaudry, P. and Portier, F. (2006). Stock Prices, News, and Economic Fluctuations. *American Economic Review*, 96(4), 1293-1307.",
            "Schmitt-Grohe, S. and Uribe, M. (2012). What's News in Business Cycles. *Econometrica*, 80(6), 2733-2764.",
        ]
    )

    report.write("README.md")
    print(f"Generated README.md, {len(report._figures)} figures, and {len(report._tables)} table.")


if __name__ == "__main__":
    main()
