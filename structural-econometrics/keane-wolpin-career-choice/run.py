#!/usr/bin/env python3
"""Keane-Wolpin-style finite-horizon career choice tutorial.

The model is a compact structural labor example with schooling, blue-collar
work, white-collar work, and home choices. It compares exact backward
induction against a sampled Emax regression approximation, then simulates a
synthetic panel under the approximate policy.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


EULER_GAMMA = 0.5772156649015329
ACTIONS = ("school", "blue", "white", "home")


@dataclass(frozen=True)
class CareerPrimitives:
    """Calibration for the finite-horizon career problem."""

    beta: float = 0.94
    shock_scale: float = 0.22
    wage_sigma: float = 0.18
    start_age: int = 16
    horizon: int = 14
    initial_schooling: int = 10
    max_schooling: int = 18
    max_school_age: int = 23
    terminal_multiplier: float = 4.0
    approximation_points: int = 260
    simulation_agents: int = 6_000
    simulation_seed: int = 884


State = tuple[int, int, int]  # schooling, blue experience, white experience


def age_at(t: int, p: CareerPrimitives) -> int:
    """Map a model period into calendar age."""
    return p.start_age + t


def feasible_actions(t: int, state: State, p: CareerPrimitives) -> list[str]:
    """Return the choices available in a state."""
    schooling, _, _ = state
    age = age_at(t, p)
    actions = ["blue", "white", "home"]
    if schooling < p.max_schooling and age <= p.max_school_age:
        actions.insert(0, "school")
    return actions


def transition(t: int, state: State, action: str, p: CareerPrimitives) -> State:
    """Deterministic state transition after a discrete choice."""
    schooling, blue_exp, white_exp = state
    if action == "school":
        return min(schooling + 1, p.max_schooling), blue_exp, white_exp
    if action == "blue":
        return schooling, blue_exp + 1, white_exp
    if action == "white":
        return schooling, blue_exp, white_exp + 1
    if action == "home":
        return state
    raise ValueError(f"Unknown action: {action}")


def build_reachable_states(p: CareerPrimitives) -> list[list[State]]:
    """Enumerate all states reachable from the initial condition."""
    states: list[set[State]] = [set() for _ in range(p.horizon + 1)]
    states[0].add((p.initial_schooling, 0, 0))
    for t in range(p.horizon):
        for state in states[t]:
            for action in feasible_actions(t, state, p):
                states[t + 1].add(transition(t, state, action, p))
    return [sorted(layer) for layer in states]


def log_wage(action: str, state: State) -> float:
    """Choice-specific log wage offer before the transitory wage shock."""
    schooling, blue_exp, white_exp = state
    if action == "blue":
        return (
            0.46
            + 0.050 * schooling
            + 0.175 * np.sqrt(blue_exp + 1.0)
            - 0.010 * blue_exp
            + 0.012 * white_exp
        )
    if action == "white":
        college_years = max(schooling - 12, 0)
        return (
            -0.10
            + 0.108 * schooling
            + 0.065 * np.sqrt(white_exp + 1.0)
            + 0.055 * college_years
            - 0.006 * white_exp
            + 0.006 * blue_exp
        )
    raise ValueError(f"No wage for action: {action}")


def deterministic_flow(t: int, state: State, action: str, p: CareerPrimitives) -> float:
    """Current deterministic utility before the Type-I extreme value shock."""
    schooling, blue_exp, white_exp = state
    age = age_at(t, p)
    if action == "school":
        college_years = max(schooling - 12, 0)
        return (
            1.05
            + 0.12 * max(18 - age, 0)
            - 0.08 * college_years
            - 0.05 * max(age - 19, 0)
        )
    if action == "blue":
        return log_wage(action, state) + 0.18 - 0.010 * max(age - 24, 0)
    if action == "white":
        entry_penalty = 0.16 if white_exp == 0 and schooling < 13 else 0.0
        return log_wage(action, state) - entry_penalty
    if action == "home":
        return 1.04 - 0.018 * max(age - 20, 0) + 0.020 * (blue_exp + white_exp)
    raise ValueError(f"Unknown action: {action}")


def terminal_value(state: State, p: CareerPrimitives) -> float:
    """Continuation value from human capital after the last modeled age."""
    blue = deterministic_flow(p.horizon - 1, state, "blue", p)
    white = deterministic_flow(p.horizon - 1, state, "white", p)
    home = deterministic_flow(p.horizon - 1, state, "home", p)
    return p.terminal_multiplier * max(blue, white, home)


def choice_values(
    t: int,
    state: State,
    continuation: dict[State, float],
    p: CareerPrimitives,
) -> dict[str, float]:
    """Compute deterministic choice-specific values."""
    values = {}
    for action in feasible_actions(t, state, p):
        next_state = transition(t, state, action, p)
        values[action] = deterministic_flow(t, state, action, p) + p.beta * continuation[next_state]
    return values


def expected_max(values: dict[str, float], scale: float) -> float:
    """Logit expected maximum over choice-specific values."""
    arr = np.array(list(values.values()), dtype=float)
    return float(scale * (logsumexp(arr / scale) + EULER_GAMMA))


def solve_exact(
    states_by_t: list[list[State]],
    p: CareerPrimitives,
) -> tuple[list[dict[State, float]], list[dict[State, dict[str, float]]], float]:
    """Exact finite-horizon backward induction over all reachable states."""
    start = time.perf_counter()
    values: list[dict[State, float]] = [dict() for _ in range(p.horizon + 1)]
    choice_maps: list[dict[State, dict[str, float]]] = [dict() for _ in range(p.horizon)]
    values[p.horizon] = {state: terminal_value(state, p) for state in states_by_t[p.horizon]}
    for t in range(p.horizon - 1, -1, -1):
        for state in states_by_t[t]:
            cvals = choice_values(t, state, values[t + 1], p)
            choice_maps[t][state] = cvals
            values[t][state] = expected_max(cvals, p.shock_scale)
    return values, choice_maps, time.perf_counter() - start


def feature_matrix(states: list[State], t: int, p: CareerPrimitives) -> np.ndarray:
    """Polynomial state features for the Emax approximation."""
    if not states:
        return np.empty((0, 11))
    arr = np.asarray(states, dtype=float)
    schooling = (arr[:, 0] - p.initial_schooling) / (p.max_schooling - p.initial_schooling)
    blue = arr[:, 1] / max(p.horizon, 1)
    white = arr[:, 2] / max(p.horizon, 1)
    total_exp = blue + white
    age = np.full_like(schooling, age_at(t, p) / (p.start_age + p.horizon))
    return np.column_stack(
        [
            np.ones(len(states)),
            schooling,
            blue,
            white,
            total_exp,
            schooling**2,
            blue**2,
            white**2,
            schooling * blue,
            schooling * white,
            blue * white,
            age,
        ]
    )


def fit_ridge(features: np.ndarray, target: np.ndarray, penalty: float = 1.0e-6) -> np.ndarray:
    """Small ridge regression used for sampled Emax interpolation."""
    xtx = features.T @ features
    ridge = penalty * np.eye(xtx.shape[0])
    ridge[0, 0] = 0.0
    return np.linalg.solve(xtx + ridge, features.T @ target)


def solve_approximate_emax(
    states_by_t: list[list[State]],
    p: CareerPrimitives,
) -> tuple[list[dict[State, float]], list[dict[State, dict[str, float]]], pd.DataFrame, float]:
    """Backward induction with sampled Emax regression at each age."""
    rng = np.random.default_rng(p.simulation_seed + 17)
    start = time.perf_counter()
    values: list[dict[State, float]] = [dict() for _ in range(p.horizon + 1)]
    choice_maps: list[dict[State, dict[str, float]]] = [dict() for _ in range(p.horizon)]
    values[p.horizon] = {state: terminal_value(state, p) for state in states_by_t[p.horizon]}
    diagnostics = []

    for t in range(p.horizon - 1, -1, -1):
        states = states_by_t[t]
        n_states = len(states)
        sample_size = min(n_states, p.approximation_points)
        if sample_size < n_states:
            sample_idx = np.sort(rng.choice(n_states, size=sample_size, replace=False))
            sample_states = [states[i] for i in sample_idx]
        else:
            sample_states = states

        sample_targets = []
        for state in sample_states:
            cvals = choice_values(t, state, values[t + 1], p)
            sample_targets.append(expected_max(cvals, p.shock_scale))
        sample_targets_arr = np.asarray(sample_targets)

        x_sample = feature_matrix(sample_states, t, p)
        x_all = feature_matrix(states, t, p)
        coef = fit_ridge(x_sample, sample_targets_arr)
        predicted = x_all @ coef

        for state, value in zip(states, predicted, strict=True):
            values[t][state] = float(value)
            choice_maps[t][state] = choice_values(t, state, values[t + 1], p)

        in_sample_fit = x_sample @ coef
        diagnostics.append(
            {
                "age": age_at(t, p),
                "states": n_states,
                "sampled": sample_size,
                "in_sample_rmse": float(np.sqrt(np.mean((in_sample_fit - sample_targets_arr) ** 2))),
                "target_sd": float(np.std(sample_targets_arr)),
            }
        )

    diag = pd.DataFrame(diagnostics).sort_values("age").reset_index(drop=True)
    return values, choice_maps, diag, time.perf_counter() - start


def softmax_dict(values: dict[str, float], scale: float) -> dict[str, float]:
    """Logit probabilities over available actions."""
    keys = list(values)
    arr = np.array([values[k] for k in keys], dtype=float)
    probs = np.exp(arr / scale - logsumexp(arr / scale))
    return {key: float(prob) for key, prob in zip(keys, probs, strict=True)}


def simulate_panel(
    choice_maps: list[dict[State, dict[str, float]]],
    p: CareerPrimitives,
) -> pd.DataFrame:
    """Simulate a panel of choices from the approximate policy."""
    rng = np.random.default_rng(p.simulation_seed)
    rows = []
    for agent in range(p.simulation_agents):
        state: State = (p.initial_schooling, 0, 0)
        for t in range(p.horizon):
            values = choice_maps[t][state]
            probabilities = softmax_dict(values, p.shock_scale)
            actions = list(probabilities)
            probs = np.array([probabilities[action] for action in actions])
            action = str(rng.choice(actions, p=probs))
            wage = np.nan
            if action in {"blue", "white"}:
                wage = float(np.exp(log_wage(action, state) + rng.normal(0.0, p.wage_sigma)))
            rows.append(
                {
                    "agent": agent,
                    "age": age_at(t, p),
                    "schooling": state[0],
                    "blue_exp": state[1],
                    "white_exp": state[2],
                    "action": action,
                    "wage": wage,
                }
            )
            state = transition(t, state, action, p)
    return pd.DataFrame(rows)


def approximation_diagnostics(
    exact: list[dict[State, float]],
    approx: list[dict[State, float]],
    exact_choices: list[dict[State, dict[str, float]]],
    approx_choices: list[dict[State, dict[str, float]]],
    states_by_t: list[list[State]],
    p: CareerPrimitives,
) -> pd.DataFrame:
    """Compare exact and approximated Emax objects by age."""
    rows = []
    for t in range(p.horizon):
        states = states_by_t[t]
        exact_arr = np.array([exact[t][state] for state in states])
        approx_arr = np.array([approx[t][state] for state in states])
        error = approx_arr - exact_arr
        exact_sd = float(np.std(exact_arr))
        agreements = []
        for state in states:
            exact_action = max(exact_choices[t][state], key=exact_choices[t][state].get)
            approx_action = max(approx_choices[t][state], key=approx_choices[t][state].get)
            agreements.append(exact_action == approx_action)
        rows.append(
            {
                "age": age_at(t, p),
                "states": len(states),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "rmse_over_value_sd": (
                    float(np.sqrt(np.mean(error**2)) / exact_sd)
                    if exact_sd > 1.0e-10
                    else np.nan
                ),
                "p90_abs_error": float(np.quantile(np.abs(error), 0.90)),
                "max_abs_error": float(np.max(np.abs(error))),
                "policy_agreement": float(np.mean(agreements)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = CareerPrimitives()
    states_by_t = build_reachable_states(p)
    total_states = sum(len(states) for states in states_by_t[:-1])
    print(f"Reachable decision states: {total_states}")

    exact_values, exact_choices, exact_time = solve_exact(states_by_t, p)
    approx_values, approx_choices, emax_fit, approx_time = solve_approximate_emax(states_by_t, p)
    diagnostics = approximation_diagnostics(
        exact_values,
        approx_values,
        exact_choices,
        approx_choices,
        states_by_t,
        p,
    )
    panel = simulate_panel(approx_choices, p)
    max_normalized_rmse = float(np.nanmax(diagnostics["rmse_over_value_sd"]))
    min_policy_agreement = float(np.min(diagnostics["policy_agreement"]))

    choice_shares = (
        panel.pivot_table(index="age", columns="action", values="agent", aggfunc="count")
        .fillna(0.0)
        .reindex(columns=ACTIONS, fill_value=0.0)
    )
    choice_shares = choice_shares.div(choice_shares.sum(axis=1), axis=0)

    wage_profile = (
        panel.dropna(subset=["wage"])
        .groupby(["age", "action"], as_index=False)["wage"]
        .mean()
        .pivot(index="age", columns="action", values="wage")
    )

    terminal = panel.sort_values(["agent", "age"]).groupby("agent").tail(1)
    ever_school = (
        panel.assign(school_choice=panel["action"].eq("school"))
        .groupby("agent")["school_choice"]
        .sum()
    )
    moments = pd.DataFrame(
        {
            "Moment": [
                "Mean final schooling",
                "Share with at least 12 years",
                "Share with at least 16 years",
                "Mean blue experience at last observed age",
                "Mean white experience at last observed age",
                "Mean years spent in school during model",
                "Approximation runtime seconds",
                "Exact runtime seconds",
            ],
            "Value": [
                float(terminal["schooling"].mean()),
                float(np.mean(terminal["schooling"] >= 12)),
                float(np.mean(terminal["schooling"] >= 16)),
                float(terminal["blue_exp"].mean()),
                float(terminal["white_exp"].mean()),
                float(ever_school.mean()),
                approx_time,
                exact_time,
            ],
        }
    )

    setup_style()
    report = ModelReport(
        "Keane-Wolpin Career Choice by Emax Approximation",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A young worker chooses whether to stay in school, work blue collar, work white "
        "collar, or stay home. Schooling raises later white-collar wages. Work builds "
        "occupation-specific experience. The relevant state is the stock of human "
        "capital accumulated before the current age.\n\n"
        "The economic object is the life-cycle policy: when does the worker leave "
        "school, which occupation does she enter, and how does early experience shape "
        "later wages?\n\n"
        "The computational object is the Emax function. Exact backward induction is "
        "straightforward in this small version, but the number of states grows quickly. "
        "The Keane-Wolpin approximation computes exact Emax values at sampled states "
        "and predicts the rest with a regression."
    )

    report.add_equations(
        r"""
At age $t$, the state is

$$
s_t = (E_t, X^b_t, X^w_t),
$$

where $E_t$ is completed schooling and $X^b_t$, $X^w_t$ are occupation-specific
experience stocks. The action set is

$$
d_t \in D(s_t,t)
\subseteq \lbrace \mathrm{school}, \mathrm{blue}, \mathrm{white}, \mathrm{home} \rbrace.
$$

The state records accumulated human capital before the current choice.

Schooling is feasible only before the schooling cap and before the maximum
school age; blue-collar work, white-collar work, and home are always feasible.

The transition is deterministic conditional on the choice:

$$
\begin{aligned}
g(s,\mathrm{school}) &= (\min \lbrace E+1,\bar E \rbrace, X^b, X^w),\\
g(s,\mathrm{blue}) &= (E, X^b+1, X^w),\\
g(s,\mathrm{white}) &= (E, X^b, X^w+1),\\
g(s,\mathrm{home}) &= (E, X^b, X^w).
\end{aligned}
$$

School raises completed education, work raises occupation-specific experience,
and home leaves measured human capital unchanged.

Let age be $\alpha_t=16+t$ and college years be $C(E)=\max \lbrace E-12,0 \rbrace$.
The wage offers and deterministic flow payoffs used by the recursion are

$$
\begin{aligned}
\log w_b(s) &=
0.46 + 0.050E + 0.175\sqrt{X^b+1}
\quad -0.010X^b + 0.012X^w,\\
\log w_w(s) &=
{}-0.10 + 0.108E + 0.065\sqrt{X^w+1}
\quad +0.055C(E)-0.006X^w+0.006X^b,\\
u_{\mathrm{school}}(s,t) &=
1.05 + 0.12\max \lbrace 18-\alpha_t,0 \rbrace
\quad -0.08C(E)-0.05\max \lbrace \alpha_t-19,0 \rbrace,\\
u_{\mathrm{blue}}(s,t) &=
\log w_b(s)+0.18-0.010\max \lbrace \alpha_t-24,0 \rbrace,\\
u_{\mathrm{white}}(s,t) &=
\log w_w(s)-0.16\mathbb{1}\lbrace X^w=0 \text{ and } E<13 \rbrace,\\
u_{\mathrm{home}}(s,t) &=
1.04-0.018\max \lbrace \alpha_t-20,0 \rbrace +0.020(X^b+X^w).
\end{aligned}
$$

The terminal value prices the best post-horizon use of accumulated human
capital:

$$
\mathbb{E}_T(s)=4\max \lbrace
u_{\mathrm{blue}}(s,T-1),
u_{\mathrm{white}}(s,T-1),
u_{\mathrm{home}}(s,T-1)
\rbrace.
$$

The deterministic part of the choice-specific value is

$$
v_t(d,s) =
\underbrace{u_d(s,t)}_{\text{current school payoff, wage payoff, or home payoff}} +
\underbrace{\beta \mathbb{E}_{t+1}(g(s,d))}_{\text{discounted continuation value}}.
$$

Current payoffs and continuation values are the two objects backward induction
needs.

With Type-I extreme value taste shocks of scale $\sigma_\epsilon$, the Emax
function is

$$
\mathbb{E}_t(s) =
\underbrace{\sigma_\epsilon
\log \sum_{d \in D(s,t)} \exp(v_t(d,s)/\sigma_\epsilon)}_{\text{expected max over feasible discrete choices}} +
\underbrace{\sigma_\epsilon \gamma_E}_{\text{mean of the extreme-value shock}}.
$$

The log-sum-exp term integrates over the Type-I extreme-value taste shocks.
Here $\gamma_E$ is Euler's constant.

The same objects imply logit conditional choice probabilities:

$$
P_t(d \mid s)=
\frac{\exp(v_t(d,s)/\sigma_\epsilon)}
{\sum_{j \in D(s,t)} \exp(v_t(j,s)/\sigma_\epsilon)}.
$$

The exact recursion evaluates this expression at every reachable state. The
Keane-Wolpin approximation evaluates it only on a sampled set
$S_t^{sample}=\lbrace s_{t,1},\dots,s_{t,M_t} \rbrace$, then fits the regression

$$
Y_{t,i} = \mathbb{E}_t(s_{t,i}), \qquad
Y_{t,i} = \phi(s_{t,i},t)' b_t + \eta_{t,i}.
$$

Sampled exact Emax values become the training targets for the continuation-value
approximation.

In this tutorial the basis vector is

$$
\phi(s,t)=
(1, E, X^b, X^w, X^b+X^w, E^2, (X^b)^2, (X^w)^2,
E X^b, E X^w, X^b X^w, t).
$$

The fitted continuation surface is

$$
\widehat{\mathbb{E}}_t(s)=\phi(s,t)'\widehat b_t.
$$

This is the computational shortcut: unsampled states inherit continuation
values from the fitted Emax surface rather than from fresh exact integrations.
"""
    )

    report.add_model_setup(
        f"| Symbol | Calibration | Meaning |\n"
        f"|---|---:|---|\n"
        f"| $t$ | ages {p.start_age}-{p.start_age + p.horizon - 1} | Finite-horizon decision age |\n"
        f"| $s_t=(E_t,X^b_t,X^w_t)$ | starts at $({p.initial_schooling},0,0)$ | Schooling, blue-collar experience, white-collar experience |\n"
        f"| $D(s,t)$ | $\\lbrace \\mathrm{{school}},\\mathrm{{blue}},\\mathrm{{white}},\\mathrm{{home}} \\rbrace$ subject to feasibility | Discrete choice set |\n"
        f"| $g(s,d)$ | deterministic | Human-capital transition after choice $d$ |\n"
        f"| $\\beta$ | {p.beta:.2f} | Discount factor in the Emax recursion |\n"
        f"| $\\sigma_\\epsilon$ | {p.shock_scale:.2f} | Type-I extreme value scale for choice shocks |\n"
        f"| $\\gamma_E$ | {EULER_GAMMA:.4f} | Euler's constant in the Emax formula |\n"
        f"| $\\sigma_w$ | {p.wage_sigma:.2f} | Log wage shock used in simulated wage paths |\n"
        f"| $\\bar E$ | {p.max_schooling} years | Maximum completed schooling in the state grid |\n"
        f"| Max school age | {p.max_school_age} | Last age at which school is feasible |\n"
        f"| Terminal multiplier | {p.terminal_multiplier:.1f} | Weight on post-horizon human capital value |\n"
        f"| $\\phi(s,t)$ | 12 polynomial terms | Basis for the sampled Emax regression |\n"
        f"| $M_t$ | up to {p.approximation_points} sampled states | Exact Emax evaluations used to fit $\\widehat b_t$ |\n"
        f"| $\\lambda$ | $10^{{-6}}$ | Ridge penalty in the sampled Emax regression |\n"
        f"| $N_s$ | {total_states:,} pre-terminal states | Reachable state count in the exact benchmark |\n"
        f"| Synthetic panel | {p.simulation_agents:,} workers | Simulated from the approximate conditional choice probabilities |"
    )

    report.add_solution_method(
        r"""
Reachable states are generated forward from the initial state:

$$
S_0=\lbrace (E_0,0,0) \rbrace,\qquad
S_{t+1}=\lbrace g(s,d): s \in S_t,\ d \in D(s,t) \rbrace.
$$

The exact benchmark is the finite-horizon recursion

$$
Q_t(d,s)=u_d(s,t)+\beta \mathbb{E}_{t+1}(g(s,d)),
\qquad
\mathbb{E}_t(s)=\sigma_\epsilon \log \sum_{d\in D(s,t)}
\exp(Q_t(d,s)/\sigma_\epsilon)+\sigma_\epsilon\gamma_E.
$$

The exact recursion stores both the Emax value and the deterministic
choice-specific values $Q_t(d,s)$ at every reachable state.

The approximate solver keeps the same recursion but estimates
$\mathbb{E}_t(s)$ from sampled states. At each age, stack the sampled targets
in $Y_t$ and the basis functions in $\Phi_t$. The regression step is

$$
\widehat b_t=(\Phi_t'\Phi_t+\lambda I)^{-1}\Phi_t'Y_t,
\qquad
\widehat{\mathbb{E}}_t(s)=\phi(s,t)'\widehat b_t.
$$

For unsampled states, $\widehat{\mathbb{E}}_t(s)$ replaces a fresh exact Emax
integration. The small ridge term $\lambda$ only stabilizes the least-squares
fit when an early age has few reachable states.

```text
Algorithm: sampled Emax approximation
Input:
    horizon T, initial state s_0 = (E_0, 0, 0)
    feasible actions D(s,t), transition g(s,d), flow payoffs u_d(s,t)
    discount factor beta, taste-shock scale sigma_e, basis phi(s,t)
    sample cap M and ridge penalty lambda

Build reachable state sets:
    S_0 = {s_0}
    for t = 0, 1, ..., T-1:
        initialize S_{t+1} as empty
        for each state s in S_t:
            for each feasible action d in D(s,t):
                add g(s,d) to S_{t+1}

Terminal values:
    for each state s in S_T:
        E_T(s) = terminal_multiplier
                 * max{u_blue(s,T-1), u_white(s,T-1), u_home(s,T-1)}

Exact benchmark:
    for t = T-1, T-2, ..., 0:
        for each state s in S_t:
            for each feasible action d in D(s,t):
                s_next = g(s,d)
                Q_t(d,s) = u_d(s,t) + beta * E_{t+1}(s_next)
            E_t(s) = sigma_e * log sum_{d in D(s,t)}
                     exp(Q_t(d,s) / sigma_e) + sigma_e * EulerGamma
            store Q_t(d,s) for policy comparisons

Approximate Emax recursion:
    set Ehat_T(s) = E_T(s) for every terminal state
    for t = T-1, T-2, ..., 0:
        if |S_t| <= M:
            use every state in S_t as the sample
        else:
            draw M states without replacement from S_t

        for each sampled state s_i:
            for each feasible action d in D(s_i,t):
                s_next = g(s_i,d)
                Q_i(d) = u_d(s_i,t) + beta * Ehat_{t+1}(s_next)
            Y_i = sigma_e * log sum_{d in D(s_i,t)}
                  exp(Q_i(d) / sigma_e) + sigma_e * EulerGamma

        form Phi_t with row i equal to phi(s_i,t)'
        solve b_hat_t = (Phi_t' Phi_t + lambda I)^{-1} Phi_t' Y_t

        for each reachable state s in S_t:
            Ehat_t(s) = phi(s,t)' b_hat_t
            for each feasible action d in D(s,t):
                s_next = g(s,d)
                Qhat_t(d,s) = u_d(s,t) + beta * Ehat_{t+1}(s_next)
            store Qhat_t(d,s)
            store P_t(d | s) as the softmax of Qhat_t(d,s) / sigma_e

Diagnostics:
    for each age t:
        compare E_t(s) and Ehat_t(s) on every state in S_t
        compute MAE, RMSE, RMSE divided by the exact Emax standard deviation,
        upper-tail absolute errors, and deterministic argmax agreement

Simulation:
    start every worker at s_0
    for each worker and age:
        read the stored approximate values Qhat_t(d,s)
        convert them into probabilities P_t(d | s)
        draw a discrete action from those probabilities
        if the action is blue or white, draw the transitory wage shock
        record age, state, action, schooling, experience, wage, and probability
        update the state to g(s,d)
```

The approximation is deliberately auditable here. The exact solve is still run,
so the page can report whether the fitted Emax surface changes continuation
values or policies in this calibration.
"""
    )

    fig1, ax1 = plt.subplots(figsize=(9, 4.8))
    x_age = choice_shares.index.to_numpy()
    ax1.stackplot(
        x_age,
        [choice_shares[action].to_numpy() for action in ACTIONS],
        labels=["School", "Blue collar", "White collar", "Home"],
        alpha=0.88,
    )
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Choice share")
    ax1.set_title("Simulated Career Choices")
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="upper right", ncol=2)
    report.add_figure(
        "figures/choice-shares.png",
        "Simulated choice shares by age",
        fig1,
        description=(
            "Schooling is concentrated at early ages, when its option value is high. "
            "As the horizon shortens, workers move into blue- and white-collar jobs. "
            "White-collar work rises after schooling has accumulated."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    if "blue" in wage_profile:
        ax2.plot(wage_profile.index, wage_profile["blue"], marker="o", label="Blue collar")
    if "white" in wage_profile:
        ax2.plot(wage_profile.index, wage_profile["white"], marker="o", label="White collar")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Mean wage")
    ax2.set_title("Synthetic Wage Profiles")
    ax2.legend()
    report.add_figure(
        "figures/wage-profiles.png",
        "Mean wages in the simulated panel",
        fig2,
        description=(
            "The synthetic wage profiles separate the two human-capital margins. "
            "Blue-collar wages pay earlier experience. White-collar wages are more "
            "education intensive and rise after workers leave school."
        ),
    )

    fig3, ax3 = plt.subplots(figsize=(8, 4.8))
    ax3.plot(diagnostics["age"], diagnostics["rmse"], marker="o", label="Emax RMSE")
    ax3.plot(diagnostics["age"], diagnostics["p90_abs_error"], marker="s", label="90th pct. abs. error")
    ax3.set_xlabel("Age")
    ax3.set_ylabel("Approximation error")
    ax3.set_title("Emax Approximation Error")
    ax3.legend(loc="upper right")
    ax3b = ax3.twinx()
    ax3b.plot(
        diagnostics["age"],
        100.0 * diagnostics["policy_agreement"],
        color="black",
        linestyle="--",
        label="Policy agreement",
    )
    ax3b.set_ylabel("Policy agreement, percent")
    ax3b.set_ylim(0.0, 105.0)
    report.add_figure(
        "figures/emax-accuracy.png",
        "Approximation errors against exact backward induction",
        fig3,
        description=(
            "The exact solve provides a benchmark. Approximation errors are largest at "
            "young ages because early states carry many future option values. Policy "
            "agreement is the share of states where exact and approximate deterministic "
            "argmax choices match. In this run, the largest age-specific RMSE is "
            f"{max_normalized_rmse:.1%} of the exact Emax standard deviation, and "
            f"the lowest policy agreement is {min_policy_agreement:.1%}. That is "
            "acceptable for a teaching approximation, not a universal tolerance: in "
            "estimation, the relevant test is whether simulated moments and the "
            "likelihood are stable when the sample size or basis is enlarged."
        ),
    )

    output_diagnostics = diagnostics.copy()
    for col in [
        "mae",
        "rmse",
        "rmse_over_value_sd",
        "p90_abs_error",
        "max_abs_error",
        "policy_agreement",
    ]:
        output_diagnostics[col] = output_diagnostics[col].round(4)
    output_diagnostics = output_diagnostics.rename(
        columns={
            "age": "Age",
            "states": "States",
            "mae": "Mean abs. Emax error",
            "rmse": "Emax RMSE",
            "rmse_over_value_sd": "RMSE / exact Emax sd",
            "p90_abs_error": "90th pct. abs. error",
            "max_abs_error": "Max abs. error",
            "policy_agreement": "Policy agreement",
        }
    )
    report.add_table(
        "tables/emax-diagnostics.csv",
        "Emax approximation diagnostics by age",
        output_diagnostics,
        description=(
            "The table reports exact-vs-approximate Emax errors on every reachable "
            "state, not only on the sampled states used in the regression. The "
            "normalized RMSE divides by the exact Emax standard deviation at that age, "
            "so it is a scale check rather than a new objective."
        ),
    )

    fit_table = emax_fit.copy()
    fit_table[["in_sample_rmse", "target_sd"]] = fit_table[["in_sample_rmse", "target_sd"]].round(4)
    fit_table = fit_table.rename(
        columns={
            "age": "Age",
            "states": "States",
            "sampled": "Sampled states",
            "in_sample_rmse": "Regression RMSE on sampled states",
            "target_sd": "Sampled target sd",
        }
    )
    report.add_table(
        "tables/emax-fit.csv",
        "Sampled regression fit by age",
        fit_table,
        description=(
            "The Emax regression uses at most the sample cap at each age. Later ages "
            "have fewer continuation states, so the approximation becomes nearly exact."
        ),
    )

    moments_out = moments.copy()
    moments_out["Value"] = moments_out["Value"].round(4)
    report.add_table(
        "tables/lifecycle-moments.csv",
        "Synthetic life-cycle moments",
        moments_out,
        description=(
            "These are moments from the simulated panel, not estimates from real data. "
            "They make the dynamic policy visible in familiar labor outcomes."
        ),
    )

    report.add_takeaway(
        "Finite-horizon structural labor models are conceptually simple but expensive "
        "because schooling and experience create many continuation states. The "
        "Keane-Wolpin approximation keeps the dynamic logic intact while replacing "
        "most exact Emax evaluations with a fitted continuation-value surface. The "
        "tradeoff is visible: the method saves computation, but approximation error is "
        "largest where early human-capital choices have the most future consequences."
    )

    report.add_references(
        [
            "[Keane, M. P. and Wolpin, K. I. (1997). The Career Decisions of Young Men. *Journal of Political Economy*, 105(3), 473-522.](https://doi.org/10.1086/262080)",
            "[Keane, M. P. and Wolpin, K. I. (1994). The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence. *Review of Economics and Statistics*, 76(4), 648-672.](https://doi.org/10.2307/2109768)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
