#!/usr/bin/env python3
"""Stochastic optimal growth solved by Q-learning.

Brock-Mirman planner with log utility, Cobb-Douglas production, full
depreciation, and AR(1) productivity. The savings rate equals alpha*beta
regardless of the productivity shock, so the closed-form policy
k'(k, z) = alpha * beta * z * A * k^alpha audits both value iteration and
tabular Q-learning. A small DQN appendix repeats the audit with neural
function approximation.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import rouwenhorst
from lib.output import ModelReport
from lib.plotting import setup_style


ALPHA = 0.36
BETA = 0.95
A_TFP = 1.0
RHO_Z = 0.7
SIGMA_Z = 0.1

N_K = 41
N_Z = 7
N_A = 21

K_LOWER_FRAC = 0.20
K_UPPER_FRAC = 1.80

QL_STEPS = 1_500_000
LOG_EVERY = 25_000
PESSIMISTIC_INIT = -80.0
N_QL_SEEDS = 4

DQN_STEPS = 250_000
DQN_BATCH = 128
DQN_BUFFER = 50_000
DQN_TARGET_EVERY = 500
DQN_LR = 5e-4
DQN_HIDDEN = 96
DQN_GAMMA = BETA
DQN_EPISODE_LEN = 80
DQN_INFEASIBLE_REWARD = -20.0


def deterministic_steady_state() -> float:
    """Capital steady state of the deterministic Brock-Mirman planner."""
    return (ALPHA * BETA * A_TFP) ** (1.0 / (1.0 - ALPHA))


def closed_form_policy(k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Optimal next-period capital k'(k, z) under log utility."""
    return ALPHA * BETA * z * A_TFP * k ** ALPHA


def closed_form_consumption(k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Optimal consumption c(k, z) under log utility."""
    return (1.0 - ALPHA * BETA) * z * A_TFP * k ** ALPHA


def build_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capital grid, productivity grid, and z transition matrix."""
    kss = deterministic_steady_state()
    k_grid = np.linspace(K_LOWER_FRAC * kss, K_UPPER_FRAC * kss, N_K)

    log_grid_jnp, trans_jnp, _ = rouwenhorst(N_Z, mu=0.0, sigma=SIGMA_Z, rho=RHO_Z)
    z_grid = np.exp(np.asarray(log_grid_jnp).reshape(-1))
    z_trans = np.asarray(trans_jnp)
    return k_grid, z_grid, z_trans


def feasible_reward_table(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    a_grid: np.ndarray,
    infeasible_penalty: float,
) -> np.ndarray:
    """Pre-compute reward[i_k, i_z, i_a] = log(z A k^alpha - a_grid[i_a])."""
    n_k = len(k_grid)
    n_z = len(z_grid)
    n_a = len(a_grid)
    output = z_grid[None, :] * A_TFP * k_grid[:, None] ** ALPHA  # (n_k, n_z)
    rewards = np.full((n_k, n_z, n_a), infeasible_penalty, dtype=np.float64)
    for i_k in range(n_k):
        for i_z in range(n_z):
            consumption = output[i_k, i_z] - a_grid
            mask = consumption > 1e-10
            rewards[i_k, i_z, mask] = np.log(consumption[mask])
    return rewards


def value_iteration(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    z_trans: np.ndarray,
    rewards: np.ndarray,
    a_to_k_index: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve the discrete Bellman equation by value iteration.

    The action index i_a maps to a next-period capital index a_to_k_index[i_a],
    so V[k', z'] is read directly off the (n_k, n_z) value grid.
    """
    n_k = len(k_grid)
    n_z = len(z_grid)
    v = np.zeros((n_k, n_z), dtype=np.float64)
    start = time.time()

    for iteration in range(1, max_iter + 1):
        # Continuation: ev[i_a, i_z] = sum_{i_zp} z_trans[i_z, i_zp] * v[a_to_k_index[i_a], i_zp]
        v_at_actions = v[a_to_k_index, :]  # (n_a, n_z)
        ev = v_at_actions @ z_trans.T  # (n_a, n_z)
        # Q[i_k, i_z, i_a] = rewards[i_k, i_z, i_a] + beta * ev[i_a, i_z]
        q = rewards + BETA * ev.T[None, :, :]
        v_new = q.max(axis=2)
        policy = q.argmax(axis=2)
        err = float(np.max(np.abs(v_new - v)))
        v = v_new
        if err < tol:
            break

    runtime = time.time() - start
    return v, policy, {"iterations": iteration, "error": err, "runtime": runtime}


def sample_next_z(rng: np.random.Generator, z_trans: np.ndarray, i_z: int) -> int:
    """Draw next productivity index from row z_trans[i_z]."""
    return int(rng.choice(z_trans.shape[0], p=z_trans[i_z]))


def tabular_q_learning(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    z_trans: np.ndarray,
    rewards: np.ndarray,
    a_to_k_index: np.ndarray,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Off-policy Q-learning with uniform exploration over feasible (s, a)
    pairs and pessimistic initialization. Each step samples a random state
    and a random feasible action, simulates one Markov transition in z, and
    applies the Bellman-error TD update."""
    n_k, n_z, n_a = rewards.shape
    rng = np.random.default_rng(seed)
    q = np.full((n_k, n_z, n_a), PESSIMISTIC_INIT, dtype=np.float64)
    infeasible = rewards <= -1e8
    q[infeasible] = -1e10
    visits = np.zeros((n_k, n_z, n_a), dtype=np.int64)

    log_steps: list[int] = []
    log_policy_err: list[float] = []
    closed_form_kp = closed_form_policy(k_grid[:, None], z_grid[None, :])

    feasible_indices_by_state: list[np.ndarray] = []
    for ik in range(n_k):
        for iz in range(n_z):
            feasible_indices_by_state.append(np.flatnonzero(~infeasible[ik, iz]))

    start = time.time()
    for step in range(1, QL_STEPS + 1):
        i_k = int(rng.integers(n_k))
        i_z = int(rng.integers(n_z))
        feas = feasible_indices_by_state[i_k * n_z + i_z]
        if len(feas) == 0:
            continue
        i_a = int(rng.choice(feas))

        reward = rewards[i_k, i_z, i_a]
        i_kp = int(a_to_k_index[i_a])
        i_zp = sample_next_z(rng, z_trans, i_z)
        target = reward + BETA * q[i_kp, i_zp].max()

        visits[i_k, i_z, i_a] += 1
        lr = 1.0 / visits[i_k, i_z, i_a] ** 0.6
        q[i_k, i_z, i_a] += lr * (target - q[i_k, i_z, i_a])

        if step % LOG_EVERY == 0:
            policy_idx = q.argmax(axis=2)
            policy_kp = k_grid[a_to_k_index[policy_idx]]
            policy_err = float(np.mean((policy_kp - closed_form_kp) ** 2) ** 0.5)
            log_steps.append(step)
            log_policy_err.append(policy_err)

    runtime = time.time() - start
    info = {
        "steps": QL_STEPS,
        "runtime": runtime,
        "log_steps": np.array(log_steps),
        "log_policy_rmse": np.array(log_policy_err),
    }
    return q, info


def try_dqn_policy(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    z_trans: np.ndarray,
    a_grid: np.ndarray,
    seed: int = 11,
) -> dict | None:
    """Train a small DQN over (k, z) -> Q over discrete actions.

    Returns None if PyTorch is unavailable, so the rest of the tutorial keeps
    running.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[dqn] torch not installed, skipping DQN appendix.")
        return None

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_a = len(a_grid)
    z_cdf = np.cumsum(z_trans, axis=1)

    def output_at(k: float, z: float) -> float:
        return z * A_TFP * k ** ALPHA

    class QNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, DQN_HIDDEN),
                nn.Tanh(),
                nn.Linear(DQN_HIDDEN, DQN_HIDDEN),
                nn.Tanh(),
                nn.Linear(DQN_HIDDEN, n_a),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: UP037
            return self.net(x)

    online = QNet()
    target = QNet()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=DQN_LR)
    loss_fn = nn.SmoothL1Loss()

    k_lo = float(k_grid.min())
    k_hi = float(k_grid.max())
    z_lo = float(z_grid.min())
    z_hi = float(z_grid.max())

    def normalize(k: float, z: float) -> np.ndarray:
        return np.array([
            2.0 * (k - k_lo) / (k_hi - k_lo) - 1.0,
            2.0 * (z - z_lo) / (z_hi - z_lo) - 1.0,
        ], dtype=np.float32)

    buf_s = np.zeros((DQN_BUFFER, 2), dtype=np.float32)
    buf_a = np.zeros(DQN_BUFFER, dtype=np.int64)
    buf_r = np.zeros(DQN_BUFFER, dtype=np.float32)
    buf_sp = np.zeros((DQN_BUFFER, 2), dtype=np.float32)
    buf_done = np.zeros(DQN_BUFFER, dtype=np.float32)
    fill = 0

    def reset() -> tuple[float, int, float]:
        i_k = int(rng.integers(N_K))
        i_z = int(rng.integers(N_Z))
        return float(k_grid[i_k]), i_z, float(z_grid[i_z])

    k, i_z, z = reset()
    ep_step = 0

    losses: list[float] = []
    start = time.time()
    for step in range(1, DQN_STEPS + 1):
        eps = max(0.05, 1.0 - step / (DQN_STEPS * 0.6))
        s_norm = normalize(k, z)
        # Mask infeasible actions when querying argmax.
        cons_all = output_at(k, z) - a_grid
        feasible = cons_all > 1e-8
        if rng.random() < eps:
            choices = np.flatnonzero(feasible)
            i_a = int(rng.choice(choices)) if len(choices) > 0 else 0
        else:
            with torch.no_grad():
                qv = online(torch.from_numpy(s_norm)).numpy()
            qv = np.where(feasible, qv, -1e9)
            i_a = int(qv.argmax())

        kp = float(a_grid[i_a])
        cons = output_at(k, z) - kp
        if cons <= 1e-8 or kp < k_lo or kp > k_hi:
            reward = DQN_INFEASIBLE_REWARD
            done = 1.0
        else:
            reward = float(np.log(cons))
            done = 0.0

        u = rng.random()
        i_zp = int(np.searchsorted(z_cdf[i_z], u))
        i_zp = min(i_zp, N_Z - 1)
        zp = float(z_grid[i_zp])

        sp_norm = normalize(kp, zp)
        idx = fill % DQN_BUFFER
        buf_s[idx] = s_norm
        buf_a[idx] = i_a
        buf_r[idx] = reward
        buf_sp[idx] = sp_norm
        buf_done[idx] = done
        fill += 1

        k, i_z, z = kp, i_zp, zp
        ep_step += 1
        if done or ep_step >= DQN_EPISODE_LEN:
            k, i_z, z = reset()
            ep_step = 0

        if fill >= DQN_BATCH:
            n_avail = min(fill, DQN_BUFFER)
            sample = rng.integers(0, n_avail, size=DQN_BATCH)
            s_b = torch.from_numpy(buf_s[sample])
            a_b = torch.from_numpy(buf_a[sample])
            r_b = torch.from_numpy(buf_r[sample])
            sp_b = torch.from_numpy(buf_sp[sample])
            d_b = torch.from_numpy(buf_done[sample])

            q_pred = online(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = target(sp_b).max(dim=1).values
                y = r_b + DQN_GAMMA * (1.0 - d_b) * q_next
            loss = loss_fn(q_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        if step % DQN_TARGET_EVERY == 0:
            target.load_state_dict(online.state_dict())

    runtime = time.time() - start

    eval_grid = np.zeros((N_K, N_Z), dtype=np.float32)
    with torch.no_grad():
        for i in range(N_K):
            for j in range(N_Z):
                s = normalize(float(k_grid[i]), float(z_grid[j]))
                qv = online(torch.from_numpy(s)).numpy()
                # Mask infeasible actions
                cons = z_grid[j] * A_TFP * k_grid[i] ** ALPHA - a_grid
                qv = np.where(cons > 1e-8, qv, -1e9)
                eval_grid[i, j] = a_grid[int(qv.argmax())]

    return {
        "policy_kp": eval_grid,
        "loss_history": np.array(losses),
        "runtime": runtime,
        "steps": DQN_STEPS,
    }


def policy_comparison_figure(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    vfi_kp: np.ndarray,
    ql_kp: np.ndarray,
    dqn_kp: np.ndarray | None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    cf_kp = closed_form_policy(k_grid[:, None], z_grid[None, :])
    for ax, idx, label in zip(axes, [0, len(z_grid) - 1], ["low", "high"]):
        ax.plot(k_grid, cf_kp[:, idx], color="black", lw=2.4, label="closed form")
        ax.plot(k_grid, vfi_kp[:, idx], color="tab:blue", lw=1.8, ls="--", label="VFI")
        ax.plot(k_grid, ql_kp[:, idx], color="tab:orange", lw=1.6, label="Q-learning")
        if dqn_kp is not None:
            ax.plot(k_grid, dqn_kp[:, idx], color="tab:green", lw=1.4, ls=":", label="DQN")
        ax.set_xlabel("capital $k$")
        ax.set_title(f"productivity {label} ($z$ = {z_grid[idx]:.2f})")
        ax.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel("next-period capital $k'$")
    fig.tight_layout()
    return fig


def learning_curve_figure(
    log_steps: np.ndarray, log_policy_rmse: np.ndarray
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(log_steps / 1000, log_policy_rmse, color="tab:orange", lw=1.8)
    ax.set_xlabel("learning steps (thousands)")
    ax.set_ylabel("policy RMSE vs closed form")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def value_surface_figure(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    v_q: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    extent = [z_grid.min(), z_grid.max(), k_grid.min(), k_grid.max()]
    im = ax.imshow(v_q, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Q-learning value $V^{\ast}(k,z)$")
    cf_policy = closed_form_policy(k_grid[:, None], z_grid[None, :])
    cs = ax.contour(
        z_grid, k_grid, cf_policy, levels=8, colors="white", linewidths=0.9, alpha=0.85
    )
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    ax.set_xlabel("productivity $z$")
    ax.set_ylabel("capital $k$")
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    folder = Path(__file__).resolve().parent

    print("Building grids ...")
    k_grid, z_grid, z_trans = build_grids()
    a_grid = np.linspace(k_grid.min(), k_grid.max(), N_A)
    a_to_k_index = np.array([np.abs(k_grid - a).argmin() for a in a_grid])
    rewards = feasible_reward_table(k_grid, z_grid, a_grid, infeasible_penalty=-1e9)

    print("Solving VFI baseline ...")
    v_vfi, policy_idx_vfi, vfi_info = value_iteration(
        k_grid, z_grid, z_trans, rewards, a_to_k_index
    )
    policy_kp_vfi = k_grid[a_to_k_index[policy_idx_vfi]]
    print(f"  VFI converged in {vfi_info['iterations']} iters, error = {vfi_info['error']:.2e}, "
          f"runtime = {vfi_info['runtime']:.2f}s")

    print(f"Running tabular Q-learning ({N_QL_SEEDS} seeds, averaged) ...")
    seed_policies = []
    seed_values = []
    total_ql_runtime = 0.0
    log_curves = []
    for s in range(N_QL_SEEDS):
        q_s, ql_info = tabular_q_learning(
            k_grid, z_grid, z_trans, rewards, a_to_k_index, seed=7 + s
        )
        seed_policies.append(k_grid[a_to_k_index[q_s.argmax(axis=2)]])
        seed_values.append(q_s.max(axis=2))
        log_curves.append(ql_info["log_policy_rmse"])
        total_ql_runtime += ql_info["runtime"]
    policy_kp_ql = np.mean(np.stack(seed_policies), axis=0)
    v_ql = np.mean(np.stack(seed_values), axis=0)
    ql_info["log_policy_rmse"] = np.mean(np.stack(log_curves), axis=0)
    print(f"  Q-learning {N_QL_SEEDS} runs x {ql_info['steps']} steps, "
          f"total runtime = {total_ql_runtime:.2f}s")

    print("Training DQN appendix ...")
    dqn_result = try_dqn_policy(k_grid, z_grid, z_trans, a_grid)
    dqn_kp = dqn_result["policy_kp"] if dqn_result is not None else None

    closed_form_kp = closed_form_policy(k_grid[:, None], z_grid[None, :])
    interior_mask = np.ones_like(closed_form_kp, dtype=bool)
    interior_mask[:3] = False
    interior_mask[-3:] = False

    def policy_mae(policy: np.ndarray) -> float:
        return float(np.mean(np.abs(policy - closed_form_kp)[interior_mask]))

    vfi_mae = policy_mae(policy_kp_vfi)
    ql_mae = policy_mae(policy_kp_ql)
    dqn_mae = policy_mae(dqn_kp) if dqn_kp is not None else float("nan")
    value_supnorm_ql = float(np.max(np.abs(v_ql - v_vfi)))

    # The "state-action evaluations" column counts (s, a) updates. For value
    # iteration these are deterministic sweep evaluations; for Q-learning and
    # DQN they are stochastic sampled transitions. The column name is kept
    # neutral so it does not imply that value iteration draws random samples.
    rows = [
        {
            "algorithm": "value iteration",
            "transition matrix": "yes",
            "policy MAE (interior)": round(vfi_mae, 4),
            "value sup-norm vs VFI": 0.0,
            "state-action evaluations": int(vfi_info["iterations"]) * int(N_K * N_Z * N_A),
            "evaluation type": "deterministic sweeps",
            "runtime sec": round(vfi_info["runtime"], 3),
        },
        {
            "algorithm": f"tabular Q-learning ({N_QL_SEEDS} seeds avg.)",
            "transition matrix": "no",
            "policy MAE (interior)": round(ql_mae, 4),
            "value sup-norm vs VFI": round(value_supnorm_ql, 4),
            "state-action evaluations": int(ql_info["steps"]) * N_QL_SEEDS,
            "evaluation type": "stochastic samples",
            "runtime sec": round(total_ql_runtime, 3),
        },
    ]
    if dqn_result is not None:
        rows.append({
            "algorithm": "DQN",
            "transition matrix": "no",
            "policy MAE (interior)": round(dqn_mae, 4),
            "value sup-norm vs VFI": float("nan"),
            "state-action evaluations": int(dqn_result["steps"]),
            "evaluation type": "stochastic samples",
            "runtime sec": round(dqn_result["runtime"], 3),
        })
    comparison_df = pd.DataFrame(rows)

    print("Building report ...")
    report = ModelReport(
        "Stochastic Optimal Growth by Q-Learning",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A planner allocates output between consumption and productive capital. "
        "Productivity moves stochastically each period. The saving choice "
        "carries today's shock into tomorrow's capital stock.\n\n"
        "The target object is the optimal saving rule $k'(k, z)$. Log utility, "
        "Cobb-Douglas production, and full depreciation pin down a closed "
        r"form, $k'(k, z) = \alpha\beta z A k^{\alpha}$. The closed form audits "
        "any numerical solver.\n\n"
        "Value iteration solves the Bellman equation through the productivity "
        "transition matrix. Q-learning replaces the matrix with sampled "
        "transitions. The same saving rule emerges from interaction alone."
    )

    report.add_equations(
        r"Let $k_t$ be capital and $z_t$ a productivity shock. Output is "
        r"$y_t = z_t A k_t^{\alpha}$, the resource constraint is "
        r"$c_t + k_{t+1} = y_t$, and productivity follows "
        r"$\log z_{t+1} = \rho \log z_t + \sigma \varepsilon_{t+1}$ with "
        r"$\varepsilon_{t+1} \sim N(0, 1)$." + "\n\n"
        "The planner's value function solves the Bellman equation:" + "\n\n"
        "$$"
        r"V(k, z) = \max_{k' \in [0, y]} \{\, \log(z A k^{\alpha} - k') + \beta\, \mathbb{E}[V(k', z') \mid z] \,\}."
        "$$" + "\n\n"
        "Tabular Q-learning stores an action-value $Q(s, a)$ for each state-action "
        "pair and updates it from observed transitions:" + "\n\n"
        "$$"
        r"Q(s, a) \leftarrow Q(s, a) + \alpha_t [\, r + \beta \max_{a'} Q(s', a') - Q(s, a) \,]."
        "$$" + "\n\n"
        r"Here $\alpha_t$ is the step size (learning rate) for update $t$." + "\n\n"
        "Exploration draws each transition uniformly over feasible "
        r"state-action pairs $(s, a)$, so every region of the grid receives "
        "updates regardless of the on-policy distribution. The greedy policy "
        r"is read off the table as $a^{\ast}(s) = \arg\max_a Q(s, a)$."
    )

    kss = deterministic_steady_state()
    setup_md = (
        "| Object | Value |\n"
        "|--------|-------|\n"
        f"| Capital state $k$ | {N_K} grid points on $[{K_LOWER_FRAC:.2f}, {K_UPPER_FRAC:.2f}] \\cdot k_{{ss}}$ |\n"
        f"| Action $k'$ | {N_A} grid points on the same capital range |\n"
        f"| Productivity $z$ | {N_Z}-state Rouwenhorst chain |\n"
        f"| Capital share $\\alpha$ | {ALPHA:.2f} |\n"
        f"| Discount $\\beta$ | {BETA:.2f} |\n"
        f"| Productivity persistence $\\rho$ | {RHO_Z:.2f} |\n"
        f"| Innovation std $\\sigma$ | {SIGMA_Z:.2f} |\n"
        f"| TFP parameter $A$ | {A_TFP:.1f} |\n"
        f"| Q-learning steps per seed | {QL_STEPS:,} |\n"
        f"| Q-learning seeds (averaged) | {N_QL_SEEDS} |\n"
        f"| DQN training steps | {DQN_STEPS:,} |\n"
        f"| Benchmark | $k'(k, z) = \\alpha\\beta z A k^{{\\alpha}}$ |\n"
        f"| Steady-state capital $k_{{ss}}$ | {kss:.3f} |"
    )
    report.add_model_setup(setup_md)

    solution_md = (
        "Value iteration sweeps the discrete Bellman operator until the value "
        "function stops moving. Each sweep evaluates expected continuation "
        "values through the productivity transition matrix.\n\n"
        "Tabular Q-learning sees one transition at a time. Each step samples "
        "a state and a feasible action uniformly at random. The productivity "
        "Markov chain delivers the next state. The Bellman temporal-difference "
        "error corrects the action-value estimate.\n\n"
        "Uniform sampling makes coverage of the grid independent of the "
        "steady-state distribution. A Robbins-Monro step size "
        r"$1 / n_{s,a}^{0.6}$ decays with visit counts. Independent runs are "
        "averaged to dampen the action-argmax variance left on individual "
        "seeds.\n\n"
        "```text\n"
        "Algorithm: tabular Q-learning with uniform exploration\n"
        "Input: feasible reward r(s, a), productivity transition, step budget\n"
        "Output: action-value Q(s, a) and greedy policy a*(s)\n"
        "Initialize Q(s, a) <- pessimistic constant for all feasible (s, a)\n"
        "for t = 1, ..., T:\n"
        "    sample state s_t = (i_k, i_z) uniformly over the grid\n"
        "    sample action a_t uniformly over feasible actions at s_t\n"
        "    receive reward r_t = log(z A k^alpha - k'(a_t))\n"
        "    sample next productivity from the transition row\n"
        "    Q(s_t, a_t) += alpha_t * (r_t + beta * max_a Q(s_{t+1}, a) - Q(s_t, a_t))\n"
        "```\n\n"
        "The deep-RL appendix replaces the table with a small two-layer MLP "
        r"$Q_\theta(k, z, \cdot)$. A replay buffer stores recent transitions. "
        "The loss is a Huber penalty against a slow-moving target network.\n\n"
        "```text\n"
        "Algorithm: deep Q-network on continuous (k, z)\n"
        "Input: discrete next-capital actions, replay buffer, minibatch size\n"
        "Output: parameters theta of Q_theta(k, z, .)\n"
        "Initialize online and target networks with the same weights\n"
        "for t = 1, ..., T_dqn:\n"
        "    select a_t with epsilon-greedy on Q_theta(s_t, .)\n"
        "    step the environment, store (s_t, a_t, r_t, s_{t+1}) in the buffer\n"
        "    sample a minibatch and form targets y = r + beta * max_a Q_target(s', a)\n"
        "    take a gradient step on Huber(Q_theta(s, a) - y)\n"
        "    every K steps copy the online weights into the target network\n"
        "```"
    )
    report.add_solution_method(solution_md)

    fig_policy = policy_comparison_figure(k_grid, z_grid, policy_kp_vfi, policy_kp_ql, dqn_kp)
    fig_curve = learning_curve_figure(ql_info["log_steps"], ql_info["log_policy_rmse"])
    fig_value = value_surface_figure(k_grid, z_grid, v_ql)

    report.add_figure(
        "figures/policy-comparison.png",
        "Q-learning saving policy compared with VFI and the closed-form rule",
        fig_policy,
        description=(
            "The greedy policy out of the Q-table tracks the closed-form saving "
            "rule across capital and productivity states. Both numerical methods "
            "reproduce the same proportional response to a productivity shock."
        ),
    )
    report.add_figure(
        "figures/learning-curve.png",
        "Policy RMSE versus number of Q-learning steps",
        fig_curve,
        description=(
            "Policy error against the closed form falls as the agent visits "
            "more states. The curve flattens once each region of the grid has "
            "enough samples to anchor the maximizer."
        ),
    )
    report.add_figure(
        "figures/value-surface.png",
        "Q-learning value surface with closed-form policy contours",
        fig_value,
        description=(
            "The learned value surface is monotone in capital and increasing in "
            "productivity. White contours mark the closed-form saving rule. "
            "The iso-policy curves rise with $z$."
        ),
    )
    report.add_table(
        "tables/algorithm-comparison.csv",
        "Algorithm comparison",
        comparison_df,
        description=(
            "The table compares the solvers on the same calibration. "
            "Q-learning uses no transition matrix. It matches the VFI policy "
            "and value to a few hundredths in capital units. "
            "The policy MAE column is computed on interior capital states "
            "only: the three lowest and three highest capital grid rows are "
            "excluded, since the closed-form rule can push next-period "
            "capital outside the discrete action grid at the boundary. "
            "All three solvers use the identical mask, so the comparison "
            "stays apples-to-apples; a full-grid MAE would be somewhat "
            "larger for every solver. The evaluation-count column counts "
            "deterministic sweep evaluations for value iteration and "
            "stochastic sampled transitions for Q-learning and DQN."
        ),
    )
    closing = (
        f"VFI converges in {vfi_info['iterations']} sweeps. "
        f"Q-learning hits an interior-grid policy MAE of {ql_mae:.4f} after "
        f"{ql_info['steps'] * N_QL_SEEDS:,} sampled transitions across "
        f"{N_QL_SEEDS} seeds"
    )
    if dqn_result is not None:
        closing += f". DQN reaches {dqn_mae:.4f} after {DQN_STEPS:,} steps"
    closing += (
        ". The MAE figures exclude the three lowest and three highest "
        "capital grid rows, where the closed-form rule can leave the "
        "discrete action grid; the same boundary mask is applied to every "
        "solver"
    )
    report.add_results(closing + ".")

    report.add_takeaway(
        "When the transition is unknown, the planner can still recover the "
        "saving rule. Sampled transitions are enough.\n\n"
        "Q-learning trades a model for data. The closed-form Brock-Mirman "
        "policy keeps both the model-based and the model-free solvers honest."
    )

    report.add_references([
        "[Brock, W. A. and Mirman, L. J. (1972). Optimal Economic Growth and Uncertainty: The Discounted Case. *Journal of Economic Theory*, 4(3), 479-513.](https://doi.org/10.1016/0022-0531(72)90135-4)",
        "[Watkins, C. J. C. H. and Dayan, P. (1992). Q-Learning. *Machine Learning*, 8(3), 279-292.](https://doi.org/10.1007/BF00992698)",
        "[Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.](http://incompleteideas.net/book/the-book-2nd.html)",
        "[Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-Level Control through Deep Reinforcement Learning. *Nature*, 518, 529-533.](https://doi.org/10.1038/nature14236)",
    ])

    report.write(str(folder / "README.md"))
    print("Wrote README.md")


if __name__ == "__main__":
    main()
