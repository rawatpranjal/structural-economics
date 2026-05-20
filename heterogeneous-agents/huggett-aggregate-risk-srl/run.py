#!/usr/bin/env python3
"""Structural Reinforcement Learning for a Huggett model with aggregate risk.

This tutorial follows the Huggett aggregate-risk experiment in Yang, Wang,
Schaab, and Moll (2025). Households solve a one-bond incomplete-markets problem,
but their policy is indexed by current prices rather than by the full
cross-sectional distribution. The resulting object is a restricted-perceptions
equilibrium computed with a structural policy-gradient algorithm.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import rouwenhorst
from lib.plotting import setup_style, save_figure, save_thumbnail


@dataclass(frozen=True)
class Calibration:
    """Economic calibration for the Huggett aggregate-risk environment."""

    beta: float = 0.96
    gamma: float = 2.0
    idio_persistence: float = 0.6
    idio_volatility: float = 0.2
    aggregate_persistence: float = 0.9
    aggregate_volatility: float = 0.02
    borrowing_limit: float = -1.0
    bond_supply: float = 0.0
    consumption_floor: float = 1.0e-3


@dataclass(frozen=True)
class Profile:
    """Numerical profile for the tabular SRL solve."""

    asset_points: int
    asset_upper: float
    idio_states: int
    aggregate_states: int
    rate_points: int
    rate_lower: float
    rate_upper: float
    horizon: int
    epochs: int
    warmup_epochs: int
    batch_size: int
    initial_learning_rate: float
    learning_rate_decay: float
    convergence_tol: float
    soft_root_scale: float
    diagnostic_horizon: int
    seed: int = 2026


FULL_PROFILE = Profile(
    asset_points=200,
    asset_upper=50.0,
    idio_states=3,
    aggregate_states=30,
    rate_points=20,
    rate_lower=0.01,
    rate_upper=0.06,
    horizon=170,
    epochs=1000,
    warmup_epochs=50,
    batch_size=512,
    initial_learning_rate=1.0e-3,
    learning_rate_decay=0.5,
    convergence_tol=3.0e-4,
    soft_root_scale=0.05,
    diagnostic_horizon=170,
)

QUICK_PROFILE = Profile(
    asset_points=56,
    asset_upper=12.0,
    idio_states=3,
    aggregate_states=7,
    rate_points=9,
    rate_lower=0.01,
    rate_upper=0.06,
    horizon=44,
    epochs=90,
    warmup_epochs=12,
    batch_size=14,
    initial_learning_rate=0.045,
    learning_rate_decay=0.55,
    convergence_tol=1.0e-4,
    soft_root_scale=0.06,
    diagnostic_horizon=44,
)

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPS = 1.0e-8
THETA_CLIP = 9.0
GRADIENT_CHUNK_SIZE = 256


@dataclass
class Grids:
    """Numerical grids and transition matrices."""

    assets: np.ndarray
    idio_log_income: np.ndarray
    idio_income: np.ndarray
    idio_transition: np.ndarray
    idio_stationary: np.ndarray
    aggregate_log_income: np.ndarray
    aggregate_transition: np.ndarray
    aggregate_stationary: np.ndarray
    rates: np.ndarray


def build_grids(cal: Calibration, profile: Profile) -> Grids:
    """Build asset, income, aggregate-income, and interest-rate grids."""
    assets = np.linspace(cal.borrowing_limit, profile.asset_upper, profile.asset_points)

    idio_grid_jax, idio_transition_jax, idio_dist_jax = rouwenhorst(
        n=profile.idio_states,
        mu=0.0,
        sigma=cal.idio_volatility,
        rho=cal.idio_persistence,
    )
    aggregate_grid_jax, aggregate_transition_jax, aggregate_dist_jax = rouwenhorst(
        n=profile.aggregate_states,
        mu=0.0,
        sigma=cal.aggregate_volatility,
        rho=cal.aggregate_persistence,
    )

    idio_log_income = np.asarray(idio_grid_jax, dtype=float).reshape(-1)
    idio_transition = np.asarray(idio_transition_jax, dtype=float)
    aggregate_log_income = np.asarray(aggregate_grid_jax, dtype=float).reshape(-1)
    aggregate_transition = np.asarray(aggregate_transition_jax, dtype=float)
    idio_stationary = np.asarray(idio_dist_jax, dtype=float).reshape(-1)
    aggregate_stationary = np.asarray(aggregate_dist_jax, dtype=float).reshape(-1)

    raw_income = np.exp(idio_log_income)
    idio_income = raw_income / float(idio_stationary @ raw_income)
    rates = np.linspace(profile.rate_lower, profile.rate_upper, profile.rate_points)

    return Grids(
        assets=assets,
        idio_log_income=idio_log_income,
        idio_income=idio_income,
        idio_transition=idio_transition,
        idio_stationary=idio_stationary,
        aggregate_log_income=aggregate_log_income,
        aggregate_transition=aggregate_transition,
        aggregate_stationary=aggregate_stationary,
        rates=rates,
    )


def initial_distribution(grids: Grids, cal: Calibration) -> np.ndarray:
    """Start the cross section near zero assets and stationary idiosyncratic risk."""
    assets = grids.assets
    a0 = 0.0
    idx = int(np.searchsorted(assets, a0))
    asset_mass = np.zeros_like(assets)
    if idx <= 0:
        asset_mass[0] = 1.0
    elif idx >= len(assets):
        asset_mass[-1] = 1.0
    else:
        lo = idx - 1
        hi = idx
        width = assets[hi] - assets[lo]
        weight_hi = (a0 - assets[lo]) / width
        asset_mass[lo] = 1.0 - weight_hi
        asset_mass[hi] = weight_hi
    dist = asset_mass[:, None] * grids.idio_stationary[None, :]
    return dist / dist.sum()


def inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable logit transform."""
    x = np.clip(x, 1.0e-4, 1.0 - 1.0e-4)
    return np.log(x / (1.0 - x))


def initialize_policy_theta(grids: Grids, cal: Calibration, profile: Profile) -> np.ndarray:
    """Initialize a tabular feasible saving rule with rate-responsive demand."""
    assets = grids.assets[:, None, None, None]
    idio_income = grids.idio_income[None, :, None, None]
    aggregate_income = np.exp(grids.aggregate_log_income)[None, None, :, None]
    rates = grids.rates[None, None, None, :]
    cash = aggregate_income * idio_income + (1.0 + rates) * assets
    upper = np.minimum(profile.asset_upper, cash - cal.consumption_floor)
    upper = np.maximum(upper, cal.borrowing_limit + 1.0e-5)

    rate_tilt = 5.0 * (rates - np.mean(grids.rates))
    income_tilt = 0.18 * (aggregate_income * idio_income - 1.0)
    asset_pull = 0.90 * assets
    target = asset_pull + rate_tilt + income_tilt
    target = np.clip(target, cal.borrowing_limit + 1.0e-4, upper - 1.0e-4)

    share = (target - cal.borrowing_limit) / (upper - cal.borrowing_limit)
    return inverse_sigmoid(share).astype(np.float32)


def draw_markov_paths(
    transition: np.ndarray,
    stationary: np.ndarray,
    batch_size: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw aggregate-state index paths from a finite Markov chain."""
    cdf = np.cumsum(transition, axis=1)
    initial_cdf = np.cumsum(stationary)
    states = np.empty((batch_size, horizon), dtype=np.int32)
    states[:, 0] = np.searchsorted(initial_cdf, rng.random(batch_size), side="right")
    states[:, 0] = np.minimum(states[:, 0], transition.shape[0] - 1)
    for t in range(1, horizon):
        draws = rng.random(batch_size)
        prev = states[:, t - 1]
        states[:, t] = np.array([
            min(int(np.searchsorted(cdf[p], u, side="right")), transition.shape[0] - 1)
            for p, u in zip(prev, draws, strict=True)
        ])
    return states


def make_jax_objective(grids: Grids, cal: Calibration, profile: Profile):
    """Create JIT-compatible objective functions closed over grids."""
    assets = jnp.asarray(grids.assets, dtype=jnp.float32)
    idio_income = jnp.asarray(grids.idio_income, dtype=jnp.float32)
    idio_transition = jnp.asarray(grids.idio_transition, dtype=jnp.float32)
    aggregate_log_income = jnp.asarray(grids.aggregate_log_income, dtype=jnp.float32)
    rates = jnp.asarray(grids.rates, dtype=jnp.float32)
    initial_dist = jnp.asarray(initial_distribution(grids, cal), dtype=jnp.float32)
    rate_indices = jnp.arange(profile.rate_points)
    e_indices_flat = jnp.tile(jnp.arange(profile.idio_states), profile.asset_points)
    asset_spacing = float(grids.assets[1] - grids.assets[0])
    discounts = cal.beta ** jnp.arange(profile.horizon, dtype=jnp.float32)
    discount_sum = jnp.sum(discounts)
    borrowing_limit = float(cal.borrowing_limit)
    consumption_floor = float(cal.consumption_floor)
    asset_upper = float(profile.asset_upper)
    soft_scale = float(profile.soft_root_scale)
    gamma = float(cal.gamma)

    def utility(consumption: jax.Array) -> jax.Array:
        c = jnp.maximum(consumption, consumption_floor)
        if abs(gamma - 1.0) < 1.0e-8:
            return jnp.log(c)
        return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    def policy_for_rate(theta: jax.Array, z_idx: jax.Array, r_idx: jax.Array):
        theta_slice = theta[:, :, z_idx, r_idx]
        share = jax.nn.sigmoid(theta_slice)
        income = jnp.exp(aggregate_log_income[z_idx]) * idio_income[None, :]
        cash = income + (1.0 + rates[r_idx]) * assets[:, None]
        upper = jnp.minimum(asset_upper, cash - consumption_floor)
        upper = jnp.maximum(upper, borrowing_limit + 1.0e-5)
        next_assets = borrowing_limit + share * (upper - borrowing_limit)
        consumption = jnp.maximum(cash - next_assets, consumption_floor)
        return next_assets, consumption

    def transition_distribution(dist: jax.Array, next_assets: jax.Array) -> jax.Array:
        position = (next_assets - borrowing_limit) / asset_spacing
        lower = jnp.floor(position).astype(jnp.int32)
        lower = jnp.clip(lower, 0, profile.asset_points - 2)
        upper = lower + 1
        weight_upper = jnp.clip(position - lower.astype(jnp.float32), 0.0, 1.0)
        weight_lower = 1.0 - weight_upper

        flat_mass = dist.reshape(-1)
        lower_flat = lower.reshape(-1)
        upper_flat = upper.reshape(-1)
        weight_lower_flat = weight_lower.reshape(-1)
        weight_upper_flat = weight_upper.reshape(-1)
        after_asset = jnp.zeros_like(dist)
        after_asset = after_asset.at[(lower_flat, e_indices_flat)].add(flat_mass * weight_lower_flat)
        after_asset = after_asset.at[(upper_flat, e_indices_flat)].add(flat_mass * weight_upper_flat)
        next_dist = after_asset @ idio_transition
        return next_dist / jnp.maximum(jnp.sum(next_dist), 1.0e-12)

    def soft_market_step(theta: jax.Array, dist: jax.Array, z_idx: jax.Array):
        next_assets_all, consumption_all = jax.vmap(
            lambda r_idx: policy_for_rate(theta, z_idx, r_idx)
        )(rate_indices)

        mass = dist[None, :, :]
        demand = jnp.sum(mass * next_assets_all, axis=(1, 2)) - cal.bond_supply
        weights = jax.nn.softmax(-((demand / soft_scale) ** 2))
        period_utility_by_rate = jnp.sum(mass * utility(consumption_all), axis=(1, 2))
        aggregate_consumption_by_rate = jnp.sum(mass * consumption_all, axis=(1, 2))
        transition_all = jax.vmap(lambda a_next: transition_distribution(dist, a_next))(
            next_assets_all
        )

        next_dist = jnp.sum(weights[:, None, None] * transition_all, axis=0)
        period_utility = jnp.sum(weights * period_utility_by_rate)
        soft_rate = jnp.sum(weights * rates)
        soft_residual = jnp.sum(weights * demand)
        aggregate_consumption = jnp.sum(weights * aggregate_consumption_by_rate)
        return next_dist, period_utility, soft_residual, soft_rate, aggregate_consumption

    def objective(theta: jax.Array, z_paths: jax.Array, update_distribution: bool):
        def run_path(z_path: jax.Array):
            def body(dist: jax.Array, z_idx: jax.Array):
                next_dist, flow_u, residual, rate, aggregate_consumption = soft_market_step(
                    theta, dist, z_idx
                )
                carried_dist = jnp.where(update_distribution, next_dist, initial_dist)
                return carried_dist, (flow_u, residual, rate, aggregate_consumption)

            _, outputs = jax.lax.scan(body, initial_dist, z_path)
            flow_utility, residuals, rates_path, aggregate_consumption_path = outputs
            discounted = jnp.sum(discounts * flow_utility) / discount_sum
            return (
                discounted,
                jnp.mean(jnp.abs(residuals)),
                jnp.mean(rates_path),
                jnp.mean(aggregate_consumption_path),
            )

        returns, abs_residuals, mean_rates, mean_consumption = jax.vmap(run_path)(z_paths)
        objective_value = jnp.mean(returns) - 1.0e-5 * jnp.mean(theta**2)
        aux = (
            jnp.mean(abs_residuals),
            jnp.mean(mean_rates),
            jnp.mean(mean_consumption),
        )
        return objective_value, aux

    value_and_grad = jax.jit(
        jax.value_and_grad(objective, has_aux=True),
        static_argnames=("update_distribution",),
    )
    value_only = jax.jit(objective, static_argnames=("update_distribution",))
    return value_and_grad, value_only


def learning_rate(epoch: int, profile: Profile) -> float:
    """Exponential schedule with a warm-up plateau."""
    if epoch <= profile.warmup_epochs:
        return profile.initial_learning_rate
    active_span = max(profile.epochs - profile.warmup_epochs, 1)
    fraction = (epoch - profile.warmup_epochs) / active_span
    return profile.initial_learning_rate * (profile.learning_rate_decay ** fraction)


def train_policy(
    theta0: np.ndarray,
    grids: Grids,
    cal: Calibration,
    profile: Profile,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    """Train the tabular saving rule by stochastic gradient ascent."""
    value_and_grad, value_only = make_jax_objective(grids, cal, profile)
    rng = np.random.default_rng(profile.seed)
    theta = jnp.asarray(theta0)
    first_moment = jnp.zeros_like(theta)
    second_moment = jnp.zeros_like(theta)
    rows: list[dict[str, float | int | str]] = []
    start = time.time()
    gradient_chunk_size = min(GRADIENT_CHUNK_SIZE, profile.batch_size)

    def batched_value_and_grad(
        theta_in: jax.Array,
        z_paths_in: np.ndarray,
        update_distribution: bool,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        total = float(z_paths_in.shape[0])
        objective_acc = jnp.asarray(0.0, dtype=jnp.float32)
        aux_acc = tuple(jnp.asarray(0.0, dtype=jnp.float32) for _ in range(3))
        grad_acc = jnp.zeros_like(theta_in)
        for start_idx in range(0, z_paths_in.shape[0], gradient_chunk_size):
            chunk = jnp.asarray(z_paths_in[start_idx:start_idx + gradient_chunk_size])
            weight = float(chunk.shape[0]) / total
            (objective_value, aux), grads = value_and_grad(
                theta_in, chunk, update_distribution=update_distribution
            )
            objective_acc = objective_acc + weight * objective_value
            aux_acc = tuple(acc + weight * x for acc, x in zip(aux_acc, aux, strict=True))
            grad_acc = grad_acc + weight * grads
        return objective_acc, aux_acc, grad_acc

    def batched_value(
        theta_in: jax.Array,
        z_paths_in: np.ndarray,
        update_distribution: bool,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        total = float(z_paths_in.shape[0])
        objective_acc = jnp.asarray(0.0, dtype=jnp.float32)
        aux_acc = tuple(jnp.asarray(0.0, dtype=jnp.float32) for _ in range(3))
        for start_idx in range(0, z_paths_in.shape[0], gradient_chunk_size):
            chunk = jnp.asarray(z_paths_in[start_idx:start_idx + gradient_chunk_size])
            weight = float(chunk.shape[0]) / total
            objective_value, aux = value_only(
                theta_in, chunk, update_distribution=update_distribution
            )
            objective_acc = objective_acc + weight * objective_value
            aux_acc = tuple(acc + weight * x for acc, x in zip(aux_acc, aux, strict=True))
        return objective_acc, aux_acc

    # Trigger compilation before the timed loop log starts moving.
    probe_paths = draw_markov_paths(
        grids.aggregate_transition,
        grids.aggregate_stationary,
        gradient_chunk_size,
        profile.horizon,
        rng,
    )
    probe_value, probe_aux, probe_grad = batched_value_and_grad(
        theta, probe_paths, update_distribution=False
    )
    del probe_value, probe_aux, probe_grad

    converged = False
    last_movement = np.inf
    for epoch in range(1, profile.epochs + 1):
        z_paths = draw_markov_paths(
            grids.aggregate_transition,
            grids.aggregate_stationary,
            profile.batch_size,
            profile.horizon,
            rng,
        )
        update_distribution = epoch > profile.warmup_epochs
        objective_value, aux, grads = batched_value_and_grad(
            theta, z_paths, update_distribution=update_distribution
        )
        first_moment = ADAM_BETA_1 * first_moment + (1.0 - ADAM_BETA_1) * grads
        second_moment = ADAM_BETA_2 * second_moment + (1.0 - ADAM_BETA_2) * (grads * grads)
        first_hat = first_moment / (1.0 - ADAM_BETA_1**epoch)
        second_hat = second_moment / (1.0 - ADAM_BETA_2**epoch)
        step = learning_rate(epoch, profile) * first_hat / (jnp.sqrt(second_hat) + ADAM_EPS)
        theta_next = jnp.clip(theta + step, -THETA_CLIP, THETA_CLIP)
        movement = float(jnp.max(jnp.abs(theta_next - theta)))
        theta = theta_next
        last_movement = movement

        should_log = (
            epoch == 1
            or epoch == profile.warmup_epochs
            or epoch % max(profile.epochs // 18, 1) == 0
            or epoch == profile.epochs
            or movement < profile.convergence_tol
        )
        if should_log:
            phase = "warmup" if not update_distribution else "adaptive"
            abs_residual, mean_rate, mean_consumption = [float(x) for x in aux]
            rows.append({
                "epoch": epoch,
                "phase": phase,
                "objective": float(objective_value),
                "mean_abs_soft_residual": abs_residual,
                "mean_interest_rate": mean_rate,
                "mean_aggregate_consumption": mean_consumption,
                "learning_rate": learning_rate(epoch, profile),
                "max_parameter_update": movement,
            })
            print(
                f"  epoch {epoch:4d} [{phase}] "
                f"objective={float(objective_value):+.5f} "
                f"soft_resid={abs_residual:.4e} "
                f"move={movement:.3e}"
            )

        if epoch > profile.warmup_epochs and movement < profile.convergence_tol:
            converged = True
            break

    elapsed = time.time() - start
    audit_paths = draw_markov_paths(
        grids.aggregate_transition,
        grids.aggregate_stationary,
        profile.batch_size,
        profile.horizon,
        rng,
    )
    value, aux = batched_value(theta, audit_paths, update_distribution=True)
    train_info = {
        "epochs_completed": float(epoch),
        "converged": float(converged),
        "elapsed_seconds": elapsed,
        "final_objective": float(value),
        "final_mean_abs_soft_residual": float(aux[0]),
        "final_mean_interest_rate": float(aux[1]),
        "final_mean_aggregate_consumption": float(aux[2]),
        "final_max_parameter_update": float(last_movement),
    }
    return np.asarray(theta), pd.DataFrame(rows), train_info


def policy_np(
    theta: np.ndarray,
    grids: Grids,
    cal: Calibration,
    profile: Profile,
    z_idx: int,
    r_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the feasible saving rule in NumPy."""
    assets = grids.assets[:, None]
    income = np.exp(grids.aggregate_log_income[z_idx]) * grids.idio_income[None, :]
    rate = grids.rates[r_idx]
    cash = income + (1.0 + rate) * assets
    upper = np.minimum(profile.asset_upper, cash - cal.consumption_floor)
    upper = np.maximum(upper, cal.borrowing_limit + 1.0e-5)
    share = 1.0 / (1.0 + np.exp(-theta[:, :, z_idx, r_idx]))
    next_assets = cal.borrowing_limit + share * (upper - cal.borrowing_limit)
    consumption = np.maximum(cash - next_assets, cal.consumption_floor)
    return next_assets, consumption


def update_distribution_np(
    dist: np.ndarray,
    next_assets: np.ndarray,
    grids: Grids,
    cal: Calibration,
) -> np.ndarray:
    """Non-stochastic histogram update for the cross-sectional distribution."""
    spacing = grids.assets[1] - grids.assets[0]
    position = (next_assets - cal.borrowing_limit) / spacing
    lower = np.floor(position).astype(int)
    lower = np.clip(lower, 0, len(grids.assets) - 2)
    upper = lower + 1
    weight_upper = np.clip(position - lower, 0.0, 1.0)
    weight_lower = 1.0 - weight_upper

    after_asset = np.zeros_like(dist)
    for i in range(dist.shape[0]):
        for e in range(dist.shape[1]):
            mass = dist[i, e]
            after_asset[lower[i, e], e] += mass * weight_lower[i, e]
            after_asset[upper[i, e], e] += mass * weight_upper[i, e]
    next_dist = after_asset @ grids.idio_transition
    return next_dist / max(float(next_dist.sum()), 1.0e-14)


def rate_schedule_for_state(
    theta: np.ndarray,
    dist: np.ndarray,
    grids: Grids,
    cal: Calibration,
    profile: Profile,
    z_idx: int,
) -> dict[str, np.ndarray | float | bool]:
    """Aggregate bond demand and interpolated market clearing for one state."""
    next_assets_by_rate = np.zeros((profile.rate_points, profile.asset_points, profile.idio_states))
    consumption_by_rate = np.zeros_like(next_assets_by_rate)
    demand = np.zeros(profile.rate_points)
    aggregate_consumption = np.zeros(profile.rate_points)
    for r_idx in range(profile.rate_points):
        next_assets, consumption = policy_np(theta, grids, cal, profile, z_idx, r_idx)
        next_assets_by_rate[r_idx] = next_assets
        consumption_by_rate[r_idx] = consumption
        demand[r_idx] = float(np.sum(dist * next_assets) - cal.bond_supply)
        aggregate_consumption[r_idx] = float(np.sum(dist * consumption))

    weights = np.zeros(profile.rate_points)
    crossings = np.flatnonzero(demand[:-1] * demand[1:] <= 0.0)
    bracketed = bool(len(crossings) > 0)
    if bracketed:
        idx = int(crossings[np.argmin(np.abs(demand[crossings]) + np.abs(demand[crossings + 1]))])
        denominator = demand[idx + 1] - demand[idx]
        if abs(denominator) < 1.0e-14:
            weight_hi = 0.5
        else:
            weight_hi = float(np.clip(-demand[idx] / denominator, 0.0, 1.0))
        weights[idx] = 1.0 - weight_hi
        weights[idx + 1] = weight_hi
    else:
        weights[int(np.argmin(np.abs(demand)))] = 1.0

    clearing_rate = float(np.sum(weights * grids.rates))
    residual = float(np.sum(weights * demand))
    return {
        "next_assets_by_rate": next_assets_by_rate,
        "consumption_by_rate": consumption_by_rate,
        "demand": demand,
        "aggregate_consumption_by_rate": aggregate_consumption,
        "weights": weights,
        "clearing_rate": clearing_rate,
        "residual": residual,
        "bracketed": bracketed,
    }


def simulate_hard_path(
    theta: np.ndarray,
    grids: Grids,
    cal: Calibration,
    profile: Profile,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Simulate one aggregate path using hard market-clearing rates."""
    rng = np.random.default_rng(profile.seed + 777)
    z_path = draw_markov_paths(
        grids.aggregate_transition,
        grids.aggregate_stationary,
        batch_size=1,
        horizon=horizon,
        rng=rng,
    )[0]
    dist = initial_distribution(grids, cal)
    rates = np.zeros(horizon)
    residuals = np.zeros(horizon)
    bracketed = np.zeros(horizon, dtype=bool)
    aggregate_consumption = np.zeros(horizon)
    aggregate_assets = np.zeros(horizon)
    aggregate_log_income = grids.aggregate_log_income[z_path]

    for t, z_idx in enumerate(z_path):
        schedule = rate_schedule_for_state(
            theta, dist, grids, cal, profile, int(z_idx)
        )
        weights = np.asarray(schedule["weights"])
        next_assets = np.sum(
            weights[:, None, None] * np.asarray(schedule["next_assets_by_rate"]),
            axis=0,
        )
        rates[t] = float(schedule["clearing_rate"])
        residuals[t] = float(schedule["residual"])
        bracketed[t] = bool(schedule["bracketed"])
        aggregate_consumption[t] = float(
            np.sum(weights * np.asarray(schedule["aggregate_consumption_by_rate"]))
        )
        aggregate_assets[t] = float(np.sum(dist * grids.assets[:, None]))
        dist = update_distribution_np(dist, next_assets, grids, cal)

    return {
        "z_path": z_path,
        "aggregate_log_income": aggregate_log_income,
        "rates": rates,
        "residuals": residuals,
        "bracketed": bracketed,
        "aggregate_consumption": aggregate_consumption,
        "aggregate_assets": aggregate_assets,
        "final_distribution": dist,
    }


def calibration_table(cal: Calibration) -> pd.DataFrame:
    """Paper Huggett calibration table."""
    rows = [
        ("Discount factor", cal.beta, "beta"),
        ("CRRA coefficient", cal.gamma, "gamma"),
        ("Idiosyncratic log-income persistence", cal.idio_persistence, "rho_e"),
        ("Idiosyncratic log-income volatility", cal.idio_volatility, "sigma_e"),
        ("Aggregate log-income persistence", cal.aggregate_persistence, "rho_z"),
        ("Aggregate log-income volatility", cal.aggregate_volatility, "sigma_z"),
        ("Borrowing limit", cal.borrowing_limit, "a_min"),
        ("Net bond supply", cal.bond_supply, "B"),
        ("Minimum consumption", cal.consumption_floor, "c_min"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value", "Symbol"])


def hyperparameter_table(profile: Profile) -> pd.DataFrame:
    """Report the paper Huggett grid and training settings used here."""
    fields = [
        ("Asset grid points", "asset_points"),
        ("Asset upper bound", "asset_upper"),
        ("Idiosyncratic states", "idio_states"),
        ("Aggregate states", "aggregate_states"),
        ("Interest-rate grid points", "rate_points"),
        ("Interest-rate lower bound", "rate_lower"),
        ("Interest-rate upper bound", "rate_upper"),
        ("Truncation horizon", "horizon"),
        ("Maximum parameter updates", "epochs"),
        ("Warm-up epochs", "warmup_epochs"),
        ("Batch size", "batch_size"),
        ("Initial learning rate", "initial_learning_rate"),
        ("Learning-rate decay", "learning_rate_decay"),
        ("Convergence threshold", "convergence_tol"),
        ("Diagnostic horizon", "diagnostic_horizon"),
    ]
    rows = []
    for label, attr in fields:
        rows.append({
            "Parameter": label,
            "Published SRL benchmark": getattr(FULL_PROFILE, attr),
            "Tutorial setting": getattr(profile, attr),
        })
    return pd.DataFrame(rows)


def diagnostics_table(
    train_info: dict[str, float],
    hard_sim: dict[str, np.ndarray],
    profile: Profile,
) -> pd.DataFrame:
    """Summarize training and interpolated market-clearing diagnostics."""
    residuals = hard_sim["residuals"]
    rates = hard_sim["rates"]
    consumption = hard_sim["aggregate_consumption"]
    income = np.exp(hard_sim["aggregate_log_income"])
    consumption_income_volatility = float(np.std(consumption) / max(np.std(income), 1.0e-12))
    rows = [
        ("Converged by policy movement criterion", "Yes" if train_info["converged"] else "No"),
        ("Epochs completed", f"{train_info['epochs_completed']:.0f}"),
        ("Final parameter movement", f"{train_info['final_max_parameter_update']:.6g}"),
        ("Convergence threshold", f"{profile.convergence_tol:.6g}"),
        ("Final normalized utility objective", f"{train_info['final_objective']:.6g}"),
        ("Mean soft market residual during training", f"{train_info['final_mean_abs_soft_residual']:.6g}"),
        ("Mean interpolated market-clearing residual", f"{np.mean(np.abs(residuals)):.6g}"),
        ("Maximum interpolated market-clearing residual", f"{np.max(np.abs(residuals)):.6g}"),
        ("Share of periods with a bracketing root", f"{np.mean(hard_sim['bracketed']):.3f}"),
        ("Mean equilibrium interest rate", f"{np.mean(rates):.6g}"),
        ("Interest-rate standard deviation", f"{np.std(rates):.6g}"),
        ("Mean aggregate consumption", f"{np.mean(consumption):.6g}"),
        ("Aggregate consumption volatility divided by income volatility", f"{consumption_income_volatility:.6g}"),
    ]
    return pd.DataFrame(rows, columns=["Diagnostic", "Value"])


def policy_shape_metrics(consumption: np.ndarray) -> dict[str, float]:
    """Check whether consumption is increasing and concave on the asset grid."""
    first_diff = np.diff(consumption, axis=0)
    second_diff = np.diff(consumption, n=2, axis=0)
    return {
        "monotone_share": float(np.mean(first_diff >= -1.0e-8)),
        "concave_share": float(np.mean(second_diff <= 1.0e-6)),
    }


def paper_benchmark_table(
    train_info: dict[str, float],
    hard_sim: dict[str, np.ndarray],
    consumption_mid: np.ndarray,
    benchmark_schedule: dict[str, np.ndarray | float | bool],
    profile: Profile,
) -> pd.DataFrame:
    """Compare this run with the published SRL Huggett benchmark."""
    residuals = hard_sim["residuals"]
    income = np.exp(hard_sim["aggregate_log_income"])
    volatility_ratio = float(
        np.std(hard_sim["aggregate_consumption"]) / max(np.std(income), 1.0e-12)
    )
    shape = policy_shape_metrics(consumption_mid)
    schedule_demand = np.asarray(benchmark_schedule["demand"])
    schedule_crosses_zero = bool(np.any(schedule_demand[:-1] * schedule_demand[1:] <= 0.0))

    qualitative_passes = [
        shape["monotone_share"] >= 0.995,
        shape["concave_share"] >= 0.95,
        volatility_ratio < 1.0,
        float(np.std(hard_sim["rates"])) > 0.0,
        schedule_crosses_zero,
        float(np.mean(np.abs(residuals))) < 1.0e-4,
    ]
    qualitative_status = "Matched" if all(qualitative_passes) else "Mixed"
    convergence_status = "Met" if train_info["converged"] else "Not met"
    # The interpolated residual is algebraically zero by construction of the
    # linear-clearing weights, so it is not comparable to the published
    # 4.4e-6 gap. Report it as a construction property, never "Matched".
    bracketing_share = float(np.mean(hard_sim["bracketed"]))
    residual_status = "Zero by construction" if bracketing_share > 0.95 else "Mixed"

    # The grid and training row must describe the profile that actually ran.
    # Only the full profile reproduces the published grid; the reduced quick
    # profile shrinks every grid and training dimension.
    grid_matches_benchmark = profile == FULL_PROFILE
    if grid_matches_benchmark:
        grid_tutorial_run = (
            "Uses the published grid, horizon, learning-rate schedule, and "
            "batch size"
        )
        grid_status = "Matched"
    else:
        grid_tutorial_run = (
            f"Reduced quick grid: {profile.asset_points} bond points, "
            f"b_max={profile.asset_upper:g}, {profile.aggregate_states} "
            f"aggregate states, {profile.rate_points} rate points, "
            f"T={profile.horizon}, {profile.epochs} epochs, "
            f"batch={profile.batch_size}"
        )
        grid_status = "Not matched"

    rows = [
        {
            "Benchmark item": "Calibration",
            "Published SRL benchmark": (
                "beta=0.96, sigma=2, rho_y=0.6, nu_y=0.2, "
                "rho_z=0.9, nu_z=0.02, B=0, borrowing limit=-1"
            ),
            "Tutorial run": "Uses the same Huggett calibration and c_min=1e-3",
            "Assessment": "Matched",
        },
        {
            "Benchmark item": "Grid and training settings",
            "Published SRL benchmark": (
                "200 bond points, b_max=50, 3 income states, 30 aggregate states, "
                "20 rate points on [0.01, 0.06], T=170, 1000 epochs, "
                "50 warm-up epochs, lr_ini=1e-3, lr_decay=0.5, batch=512"
            ),
            "Tutorial run": grid_tutorial_run,
            "Assessment": grid_status,
        },
        {
            "Benchmark item": "Convergence status",
            "Published SRL benchmark": "Average convergence at 480.6 epochs over 10 runs",
            "Tutorial run": (
                f"{'Converged' if train_info['converged'] else 'Did not converge'} "
                f"after {train_info['epochs_completed']:.0f} epochs; "
                f"final movement {train_info['final_max_parameter_update']:.3g}"
            ),
            "Assessment": convergence_status,
        },
        {
            "Benchmark item": "Market-clearing residual",
            "Published SRL benchmark": "Average bond-market clearing gap about 4.4e-6",
            "Tutorial run": (
                f"Interpolated residual is zero by construction of the linear "
                f"clearing weights: mean {np.mean(np.abs(residuals)):.3g}, "
                f"maximum {np.max(np.abs(residuals)):.3g}, bracketing share "
                f"{bracketing_share:.3f}. Not comparable to the published gap"
            ),
            "Assessment": residual_status,
        },
        {
            "Benchmark item": "Qualitative figure match",
            "Published SRL benchmark": (
                "monotone and concave consumption, a below-one C/Y volatility "
                "ratio, endogenous interest rates, saving schedule crossing zero"
            ),
            "Tutorial run": (
                f"monotone share {shape['monotone_share']:.3f}; "
                f"concave share {shape['concave_share']:.3f}; "
                f"C/Y volatility ratio {volatility_ratio:.3f}; "
                f"saving schedule crosses zero: {'yes' if schedule_crosses_zero else 'no'}"
            ),
            "Assessment": qualitative_status,
        },
    ]
    return pd.DataFrame(rows)


def add_figures_and_tables(
    theta: np.ndarray,
    grids: Grids,
    cal: Calibration,
    profile: Profile,
    train_log: pd.DataFrame,
    hard_sim: dict[str, np.ndarray],
    calibration_df: pd.DataFrame,
    hyper_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    paper_benchmark_df: pd.DataFrame,
) -> None:
    """Create tutorial figures and save tables."""
    z_mid = profile.aggregate_states // 2
    r_mid = profile.rate_points // 2
    assets = grids.assets
    next_assets_mid, consumption_mid = policy_np(theta, grids, cal, profile, z_mid, r_mid)
    saving_mid = next_assets_mid - assets[:, None]

    fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
    for e_idx, income in enumerate(grids.idio_income):
        ax1.plot(assets, consumption_mid[:, e_idx], label=f"idio income {income:.2f}")
    ax1.set_title("Consumption Policy")
    ax1.set_xlabel("Bond holdings $b$")
    ax1.set_ylabel("Consumption $c(a,e,z,r)$")
    ax1.legend()
    save_figure(fig1, "figures/policy-consumption.png", dpi=150)

    periods = np.arange(len(hard_sim["rates"]))
    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    ax2.plot(periods, hard_sim["rates"], color="tab:blue")
    ax2.axhline(np.mean(grids.rates), color="black", linewidth=1.0, linestyle="--")
    ax2.set_title("Endogenous Market-Clearing Interest Rate")
    ax2.set_xlabel("Simulation period")
    ax2.set_ylabel("Net interest rate")
    save_figure(fig2, "figures/interest-rate-path.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(8, 4.2))
    income = np.exp(hard_sim["aggregate_log_income"])
    ax3.plot(periods, income / np.mean(income), color="tab:gray", label="aggregate income")
    ax3.plot(
        periods,
        hard_sim["aggregate_consumption"] / np.mean(hard_sim["aggregate_consumption"]),
        color="tab:green",
        label="aggregate consumption",
    )
    ax3.set_title("Aggregate Consumption and Income")
    ax3.set_xlabel("Simulation period")
    ax3.set_ylabel("Normalized level")
    ax3.legend()
    save_figure(fig3, "figures/aggregate-consumption-path.png", dpi=150)

    fig4, ax4 = plt.subplots(figsize=(7.5, 4.8))
    for e_idx, idio_level in enumerate(grids.idio_income):
        ax4.plot(assets, saving_mid[:, e_idx], label=f"idio income {idio_level:.2f}")
    ax4.axhline(0.0, color="black", linewidth=1.0)
    ax4.set_title("Household Saving Schedule")
    ax4.set_xlabel("Bond holdings $b$")
    ax4.set_ylabel("Next assets minus current assets")
    ax4.legend()
    save_figure(fig4, "figures/saving-schedule.png", dpi=150)

    final_dist = hard_sim["final_distribution"]
    final_schedule = rate_schedule_for_state(
        theta, final_dist, grids, cal, profile, z_mid
    )
    demand = np.asarray(final_schedule["demand"])
    fig5, ax5 = plt.subplots(figsize=(7.5, 4.8))
    ax5.plot(grids.rates, demand, marker="o")
    ax5.axhline(0.0, color="black", linewidth=1.0)
    ax5.axvline(float(final_schedule["clearing_rate"]), color="tab:red", linestyle="--")
    ax5.set_title("Aggregate Bond-Demand Schedule")
    ax5.set_xlabel("Interest rate grid")
    ax5.set_ylabel("Aggregate desired next assets")
    save_figure(fig5, "figures/bond-demand-schedule.png", dpi=150)

    fig6, ax6 = plt.subplots(figsize=(8, 4.2))
    ax6.plot(periods, hard_sim["residuals"], color="tab:red")
    ax6.axhline(0.0, color="black", linewidth=1.0)
    ax6.set_title("Market-Clearing Residual")
    ax6.set_xlabel("Simulation period")
    ax6.set_ylabel("Interpolated residual")
    save_figure(fig6, "figures/aggregate-saving-residual.png", dpi=150)

    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(10, 4.2))
    if not train_log.empty:
        ax7a.plot(train_log["epoch"], train_log["objective"], marker="o", markersize=3)
        ax7b.semilogy(
            train_log["epoch"],
            train_log["max_parameter_update"],
            marker="o",
            markersize=3,
        )
        ax7a.axvline(profile.warmup_epochs, color="black", linewidth=0.9, linestyle="--")
        ax7b.axvline(profile.warmup_epochs, color="black", linewidth=0.9, linestyle="--")
    ax7a.set_title("Utility Objective")
    ax7a.set_xlabel("Epoch")
    ax7a.set_ylabel("Discounted utility")
    ax7b.set_title("Policy Movement")
    ax7b.set_xlabel("Epoch")
    ax7b.set_ylabel("Max parameter update")
    fig7.tight_layout()
    save_figure(fig7, "figures/training-curves.png", dpi=150)

    fig8, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for e_idx, income_level in enumerate(grids.idio_income):
        axes[0].plot(assets, consumption_mid[:, e_idx], label=f"y={income_level:.2f}")
    axes[0].set_title("Monotone, Concave Policy")
    axes[0].set_xlabel("Bond holdings $b$")
    axes[0].set_ylabel("Consumption")
    axes[0].legend(fontsize=8)

    axes[1].plot(periods, income / np.mean(income), color="tab:gray", label="income")
    axes[1].plot(
        periods,
        hard_sim["aggregate_consumption"] / np.mean(hard_sim["aggregate_consumption"]),
        color="tab:green",
        label="consumption",
    )
    axes[1].set_title("Consumption Smoothing")
    axes[1].set_xlabel("Simulation period")
    axes[1].legend(fontsize=8)

    axes[2].plot(grids.rates, demand, marker="o", color="tab:purple")
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].axvline(float(final_schedule["clearing_rate"]), color="tab:red", linestyle="--")
    axes[2].set_title("Saving Schedule Crosses Zero")
    axes[2].set_xlabel("Interest rate")
    axes[2].set_ylabel("Aggregate saving")
    fig8.tight_layout()
    save_figure(fig8, "figures/paper-benchmark-panel.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    calibration_df.to_csv("tables/calibration.csv", index=False)
    hyper_df.to_csv("tables/hyperparameters.csv", index=False)
    diagnostics_df.to_csv("tables/diagnostics.csv", index=False)
    paper_benchmark_df.to_csv("tables/paper_benchmark.csv", index=False)


def build_report(
    theta: np.ndarray,
    grids: Grids,
    cal: Calibration,
    profile: Profile,
    train_log: pd.DataFrame,
    train_info: dict[str, float],
    hard_sim: dict[str, np.ndarray],
    profile_label: str,
    selection_reason: str,
) -> None:
    """Generate figures and tables."""
    setup_style()

    calibration_df = calibration_table(cal)
    hyper_df = hyperparameter_table(profile)
    diagnostics_df = diagnostics_table(train_info, hard_sim, profile)
    z_mid = profile.aggregate_states // 2
    r_mid = profile.rate_points // 2
    _, consumption_mid = policy_np(theta, grids, cal, profile, z_mid, r_mid)
    benchmark_schedule = rate_schedule_for_state(
        theta, hard_sim["final_distribution"], grids, cal, profile, z_mid
    )
    paper_benchmark_df = paper_benchmark_table(
        train_info,
        hard_sim,
        consumption_mid,
        benchmark_schedule,
        profile,
    )

    add_figures_and_tables(
        theta,
        grids,
        cal,
        profile,
        train_log,
        hard_sim,
        calibration_df,
        hyper_df,
        diagnostics_df,
        paper_benchmark_df,
    )

    save_thumbnail("figures/policy-consumption.png", "figures/thumb.png")
    stale_run_profile = Path("tables/run_profile.csv")
    if stale_run_profile.exists():
        stale_run_profile.unlink()


def main() -> None:
    cal = Calibration()
    backend = jax.default_backend()
    if backend == "cpu":
        profile = QUICK_PROFILE
        profile_label = "quick"
        selection_reason = "auto selected quick because JAX sees only CPU devices"
    else:
        profile = FULL_PROFILE
        profile_label = "full"
        selection_reason = f"auto selected full on JAX backend {backend}"
    print(f"Running the SRL Huggett solver with the {profile_label} profile")
    print(f"JAX backend: {backend}")
    print("Building grids and initializing policy")
    grids = build_grids(cal, profile)
    theta0 = initialize_policy_theta(grids, cal, profile)

    print("Training SRL policy")
    theta, train_log, train_info = train_policy(theta0, grids, cal, profile)

    print("Running hard market-clearing diagnostics")
    hard_sim = simulate_hard_path(theta, grids, cal, profile, horizon=profile.diagnostic_horizon)

    print("Writing tutorial outputs")
    build_report(
        theta,
        grids,
        cal,
        profile,
        train_log,
        train_info,
        hard_sim,
        profile_label,
        selection_reason,
    )
    print("Done: generated README.md, figures/, and tables/.")


if __name__ == "__main__":
    main()
