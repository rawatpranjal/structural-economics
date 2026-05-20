#!/usr/bin/env python3
"""Bus engine replacement solved by Q-learning.

Rust's dynamic discrete choice problem with Type-I extreme value shocks.
Standard NFXP needs the mileage transition matrix to iterate the Bellman
fixed point. Soft Q-learning learns the same replacement hazard from the
simulated bus panel alone, and a small DQN appendix repeats the audit with
neural function approximation. The DDC primitives match the calibration in
`industrial-organization/dynamic-discrete-choice/`.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


EULER_GAMMA = 0.5772156649015329

BETA = 0.90
THETA_TRUE = np.array([2.00, -0.15])
X_MIN = 0.0
X_MAX = 15.0
DELTA_X = 0.25

N_BUSES = 1_500
N_PERIODS = 35
SIM_SEED = 12

QL_EPOCHS = 30
QL_SEEDS = 4
QL_BUFFER_SCALE = 1.0  # use full panel each epoch

DQN_EPOCHS = 80
DQN_BATCH = 256
DQN_HIDDEN = 64
DQN_LR = 5e-4
DQN_TARGET_EVERY = 200
DQN_SEED = 21


# ----- DDC primitives (match `industrial-organization/dynamic-discrete-choice/`) ----------

def build_transition_matrices(x: np.ndarray, delta_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Replacement and no-replacement mileage transition matrices."""
    n = len(x)
    keep = np.zeros((n, n))
    for i, current in enumerate(x):
        increments = x - current
        valid = increments >= -1e-12
        keep[i, valid] = np.exp(-np.maximum(increments[valid], 0.0)) * (1.0 - np.exp(-delta_x))
        keep[i, -1] += max(0.0, 1.0 - keep[i].sum())
        keep[i] = keep[i] / keep[i].sum()
    replace = np.tile(keep[0], (n, 1))
    return replace, keep


def solve_ddc(
    theta: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    tol: float = 1e-10,
    max_iter: int = 5_000,
) -> dict:
    """Solve the conditional value contraction. Returns CCPs and values."""
    flow_replace = np.zeros_like(x)
    flow_keep = theta[0] + theta[1] * x
    values = np.zeros((len(x), 2))
    for iteration in range(1, max_iter + 1):
        inclusive = logsumexp(values, axis=1) + EULER_GAMMA
        next_replace = F_replace @ inclusive
        next_keep = F_keep @ inclusive
        values_new = np.column_stack([
            flow_replace + beta * next_replace,
            flow_keep + beta * next_keep,
        ])
        err = float(np.max(np.abs(values_new - values)))
        values = values_new
        if err < tol:
            break
    log_denom = logsumexp(values, axis=1)
    p_replace = np.exp(values[:, 0] - log_denom)
    return {
        "values": values,
        "p_replace": p_replace,
        "iterations": iteration,
        "error": err,
    }


def simulate_buses(
    theta: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    n_buses: int,
    n_periods: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate panel data from the DDC under the structural CCPs."""
    solution = solve_ddc(theta, x, F_replace, F_keep, beta)
    p_replace = np.asarray(solution["p_replace"])
    rng = np.random.default_rng(seed)
    n_x = len(x)
    cdf_replace = np.cumsum(F_replace, axis=1)
    cdf_keep = np.cumsum(F_keep, axis=1)

    x_index = np.zeros((n_buses, n_periods), dtype=int)
    mileage = np.zeros((n_buses, n_periods))
    replace = np.zeros((n_buses, n_periods), dtype=int)

    current = np.zeros(n_buses, dtype=int)
    for t in range(n_periods):
        x_index[:, t] = current
        mileage[:, t] = x[current]
        u = rng.uniform(size=n_buses)
        replace[:, t] = u < p_replace[current]

        v = rng.uniform(size=n_buses)
        cdf = np.where(replace[:, t, None].astype(bool), cdf_replace[current], cdf_keep[current])
        next_state = np.minimum((v[:, None] > cdf).sum(axis=1), n_x - 1)
        current = next_state.astype(int)

    return {
        "x_index": x_index,
        "mileage": mileage,
        "replace": replace,
        "p_replace_truth": p_replace,
    }


# ----- Soft Q-learning ---------------------------------------------------------------------

def flow_payoffs(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """flow[i, a] for a in {replace=0, keep=1}."""
    flow = np.zeros((len(x), 2))
    flow[:, 0] = 0.0
    flow[:, 1] = theta[0] + theta[1] * x
    return flow


def soft_q_learning(
    transitions: dict[str, np.ndarray],
    flow: np.ndarray,
    beta: float,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Soft Q-learning over an observed (x_t, a_t, x_{t+1}) panel.

    The target uses the log-sum-exp continuation that matches Type-I extreme
    value shocks, so the implied CCP equals the structural replacement hazard.
    """
    rng = np.random.default_rng(seed)
    n_x = flow.shape[0]
    q = np.zeros((n_x, 2), dtype=np.float64)
    visits = np.zeros((n_x, 2), dtype=np.int64)

    s = transitions["s"]
    a = transitions["a"]
    sp = transitions["sp"]
    n_samples = len(s)

    log_steps: list[int] = []
    log_hazard_err: list[float] = []
    truth_p = transitions["truth_p_replace"]

    start = time.time()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_samples)
        for idx in order:
            i = int(s[idx])
            ai = int(a[idx])
            j = int(sp[idx])
            r = flow[i, ai]
            target = r + beta * (logsumexp(q[j]) + EULER_GAMMA)

            visits[i, ai] += 1
            lr = 1.0 / visits[i, ai] ** 0.6
            q[i, ai] += lr * (target - q[i, ai])

        log_p = q - logsumexp(q, axis=1, keepdims=True)
        p_replace = np.exp(log_p[:, 0])
        log_steps.append(epoch * n_samples)
        log_hazard_err.append(float(np.mean(np.abs(p_replace - truth_p))))

    runtime = time.time() - start
    log_p = q - logsumexp(q, axis=1, keepdims=True)
    p_replace = np.exp(log_p[:, 0])
    info = {
        "epochs": epochs,
        "samples": n_samples * epochs,
        "runtime": runtime,
        "log_steps": np.array(log_steps),
        "log_hazard_mae": np.array(log_hazard_err),
        "p_replace": p_replace,
        "visits": visits,
    }
    return q, info


def build_panel_transitions(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten the simulated panel into (s, a, s') triples."""
    x_idx = data["x_index"]
    repl = data["replace"]
    s_t = x_idx[:, :-1].ravel()
    a_t = repl[:, :-1].ravel().astype(int)  # 1 = replace, 0 = keep
    # Action 0 = replace, 1 = keep in our convention
    a_idx = 1 - a_t
    sp_t = x_idx[:, 1:].ravel()
    return {"s": s_t, "a": a_idx, "sp": sp_t}


# ----- DQN appendix ------------------------------------------------------------------------

def try_dqn_hazard(
    transitions: dict[str, np.ndarray],
    x_grid: np.ndarray,
    flow: np.ndarray,
    beta: float,
) -> dict | None:
    """Soft DQN trained on the same observed panel."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[dqn] torch not installed, skipping DQN appendix.")
        return None

    torch.manual_seed(DQN_SEED)
    rng = np.random.default_rng(DQN_SEED)

    n_x = len(x_grid)
    s = torch.from_numpy(transitions["s"].astype(np.int64))
    a = torch.from_numpy(transitions["a"].astype(np.int64))
    sp = torch.from_numpy(transitions["sp"].astype(np.int64))

    x_lo = float(x_grid.min())
    x_hi = float(x_grid.max())

    def normalize(x_idx: "torch.Tensor") -> "torch.Tensor":
        x_val = torch.from_numpy(x_grid).float()[x_idx]
        return (2.0 * (x_val - x_lo) / (x_hi - x_lo) - 1.0).unsqueeze(1)

    s_norm = normalize(s)
    sp_norm = normalize(sp)

    flow_t = torch.from_numpy(flow).float()
    r_b = flow_t[s, a]

    class QNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, DQN_HIDDEN),
                nn.Tanh(),
                nn.Linear(DQN_HIDDEN, DQN_HIDDEN),
                nn.Tanh(),
                nn.Linear(DQN_HIDDEN, 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: UP037
            return self.net(x)

    online = QNet()
    target = QNet()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=DQN_LR)
    loss_fn = nn.SmoothL1Loss()

    n_samples = len(s)

    log_curve: list[float] = []
    truth_p = transitions["truth_p_replace"]

    start = time.time()
    step = 0
    for epoch in range(1, DQN_EPOCHS + 1):
        order = rng.permutation(n_samples)
        for batch_start in range(0, n_samples, DQN_BATCH):
            idx = order[batch_start : batch_start + DQN_BATCH]
            s_b = s_norm[idx]
            a_b = a[idx]
            sp_b = sp_norm[idx]
            r_batch = r_b[idx]

            with torch.no_grad():
                q_next = target(sp_b)
                v_next = torch.logsumexp(q_next, dim=1) + EULER_GAMMA
                y = r_batch + beta * v_next

            q_pred = online(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = loss_fn(q_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % DQN_TARGET_EVERY == 0:
                target.load_state_dict(online.state_dict())

        with torch.no_grad():
            full_x = torch.from_numpy(np.arange(n_x))
            full_norm = normalize(full_x)
            q_full = online(full_norm).numpy()
        log_p = q_full - logsumexp(q_full, axis=1, keepdims=True)
        p_replace = np.exp(log_p[:, 0])
        log_curve.append(float(np.mean(np.abs(p_replace - truth_p))))

    runtime = time.time() - start
    with torch.no_grad():
        full_x = torch.from_numpy(np.arange(n_x))
        full_norm = normalize(full_x)
        q_full = online(full_norm).numpy()
    log_p = q_full - logsumexp(q_full, axis=1, keepdims=True)
    p_replace = np.exp(log_p[:, 0])
    return {
        "p_replace": p_replace,
        "values": q_full,
        "runtime": runtime,
        "epochs": DQN_EPOCHS,
        "log_hazard_mae": np.array(log_curve),
    }


# ----- Figures -----------------------------------------------------------------------------

def hazard_figure(
    x_grid: np.ndarray,
    p_truth: np.ndarray,
    p_ql: np.ndarray,
    p_dqn: np.ndarray | None,
    visit_total: np.ndarray | None = None,
    visit_threshold: int = 5,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x_grid, p_truth, color="black", lw=2.4, label="NFXP (truth)")
    p_ql_masked = p_ql.copy()
    if visit_total is not None:
        p_ql_masked = np.where(visit_total >= visit_threshold, p_ql, np.nan)
    ax.plot(x_grid, p_ql_masked, color="tab:orange", lw=1.8, label="soft Q-learning")
    if p_dqn is not None:
        ax.plot(x_grid, p_dqn, color="tab:green", lw=1.6, ls="--", label="soft DQN")
    if visit_total is not None:
        last_visited = float(x_grid[np.where(visit_total >= visit_threshold)[0][-1]])
        ax.axvline(last_visited, color="grey", lw=0.8, ls=":", alpha=0.7)
        ax.text(
            last_visited + 0.1,
            0.05,
            "panel coverage\nruns out",
            fontsize=8,
            color="grey",
            va="bottom",
        )
    ax.set_xlabel("mileage $x$")
    ax.set_ylabel(r"replacement hazard $P(\mathrm{replace} \mid x)$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def value_figure(
    x_grid: np.ndarray,
    nfxp_values: np.ndarray,
    ql_values: np.ndarray,
    visit_total: np.ndarray | None = None,
    visit_threshold: int = 30,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x_grid, nfxp_values[:, 0], color="black", lw=2.0, label="NFXP $v(x, \\mathrm{replace})$")
    ax.plot(x_grid, nfxp_values[:, 1], color="black", lw=2.0, ls=":", label="NFXP $v(x, \\mathrm{keep})$")
    if visit_total is not None:
        mask = visit_total >= visit_threshold
        v_repl = np.where(mask, ql_values[:, 0], np.nan)
        v_keep = np.where(mask, ql_values[:, 1], np.nan)
    else:
        v_repl = ql_values[:, 0]
        v_keep = ql_values[:, 1]
    ax.plot(x_grid, v_repl, color="tab:orange", lw=1.6, label="Q-learning replace")
    ax.plot(x_grid, v_keep, color="tab:orange", lw=1.6, ls=":", label="Q-learning keep")
    ax.set_xlabel("mileage $x$")
    ax.set_ylabel("conditional value")
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def sample_efficiency_figure(
    bus_counts: list[int],
    hazard_mae_per_count: list[float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(bus_counts, hazard_mae_per_count, color="tab:orange", marker="o", lw=1.8)
    ax.set_xlabel("number of simulated buses")
    ax.set_ylabel("hazard MAE vs NFXP")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def trajectory_figure(
    bus_mileage: np.ndarray,
    bus_replace: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    t = np.arange(len(bus_mileage))
    ax.plot(t, bus_mileage, color="tab:blue", lw=1.6)
    replace_idx = np.where(bus_replace == 1)[0]
    ax.scatter(replace_idx, bus_mileage[replace_idx], color="tab:red", zorder=5, label="replacement")
    ax.set_xlabel("period")
    ax.set_ylabel("mileage")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    folder = Path(__file__).resolve().parent

    print("Building primitives ...")
    x_grid = np.arange(X_MIN, X_MAX + DELTA_X, DELTA_X)
    F_replace, F_keep = build_transition_matrices(x_grid, DELTA_X)
    flow = flow_payoffs(THETA_TRUE, x_grid)

    print("Solving NFXP ...")
    t0 = time.time()
    nfxp = solve_ddc(THETA_TRUE, x_grid, F_replace, F_keep, BETA)
    nfxp_runtime = time.time() - t0
    print(f"  NFXP iterations = {nfxp['iterations']}, runtime = {nfxp_runtime:.3f}s")

    print(f"Simulating panel ({N_BUSES} buses x {N_PERIODS} periods) ...")
    data = simulate_buses(THETA_TRUE, x_grid, F_replace, F_keep, BETA, N_BUSES, N_PERIODS, SIM_SEED)
    transitions = build_panel_transitions(data)
    transitions["truth_p_replace"] = nfxp["p_replace"]

    print(f"Soft Q-learning ({QL_SEEDS} seeds, {QL_EPOCHS} epochs each) ...")
    seed_p_replaces = []
    seed_q_tables = []
    log_curves = []
    total_ql_runtime = 0.0
    visits_per_action = None
    for s in range(QL_SEEDS):
        q_s, ql_info = soft_q_learning(transitions, flow, BETA, QL_EPOCHS, seed=100 + s)
        seed_p_replaces.append(ql_info["p_replace"])
        seed_q_tables.append(q_s)
        log_curves.append(ql_info["log_hazard_mae"])
        total_ql_runtime += ql_info["runtime"]
        if visits_per_action is None:
            visits_per_action = ql_info["visits"].copy()
    visit_min = visits_per_action.min(axis=1)
    visit_total = visits_per_action.sum(axis=1)
    p_replace_ql = np.mean(np.stack(seed_p_replaces), axis=0)
    q_table_ql = np.mean(np.stack(seed_q_tables), axis=0)
    avg_curve = np.mean(np.stack(log_curves), axis=0)
    avg_log_steps = ql_info["log_steps"]
    print(f"  Q-learning {QL_SEEDS} seeds, total runtime = {total_ql_runtime:.2f}s")

    print("Sample-efficiency sweep ...")
    bus_counts = [50, 150, 450, 1500]
    hazard_mae_per_count: list[float] = []
    for nb in bus_counts:
        sub = simulate_buses(THETA_TRUE, x_grid, F_replace, F_keep, BETA, nb, N_PERIODS, SIM_SEED + 1)
        sub_trans = build_panel_transitions(sub)
        sub_trans["truth_p_replace"] = nfxp["p_replace"]
        _, info = soft_q_learning(sub_trans, flow, BETA, QL_EPOCHS, seed=999)
        hazard_mae_per_count.append(float(info["log_hazard_mae"][-1]))
    sample_eff_df = pd.DataFrame({
        "buses": bus_counts,
        "hazard MAE": [round(m, 6) for m in hazard_mae_per_count],
    })
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    sample_eff_df.to_csv(folder / "tables" / "sample-efficiency.csv", index=False)

    print("Training DQN appendix ...")
    dqn_result = try_dqn_hazard(transitions, x_grid, flow, BETA)
    p_replace_dqn = dqn_result["p_replace"] if dqn_result is not None else None

    visited_mask = visit_min >= 5  # require both actions to have visits
    hazard_mae_ql = float(np.mean(
        np.abs(p_replace_ql[visited_mask] - nfxp["p_replace"][visited_mask])
    ))
    hazard_mae_dqn = (
        float(np.mean(np.abs(p_replace_dqn - nfxp["p_replace"])))
        if dqn_result is not None
        else float("nan")
    )

    def crossing_x(p: np.ndarray, mask: np.ndarray | None = None) -> float:
        if mask is None:
            idx = int(np.argmin(np.abs(p - 0.5)))
        else:
            sub = np.where(mask, np.abs(p - 0.5), np.inf)
            idx = int(np.argmin(sub))
        return float(x_grid[idx])

    threshold_truth = crossing_x(nfxp["p_replace"])
    threshold_ql = crossing_x(p_replace_ql, mask=visited_mask)
    threshold_dqn = crossing_x(p_replace_dqn) if dqn_result is not None else float("nan")

    rows = [
        {
            "method": "NFXP (model-based)",
            "transition matrix": "yes",
            "hazard MAE": 0.0,
            "P=0.5 mileage": round(threshold_truth, 2),
            "samples": int(nfxp["iterations"]),
            "runtime sec": round(nfxp_runtime, 4),
        },
        {
            "method": f"soft Q-learning ({QL_SEEDS} seeds avg.)",
            "transition matrix": "no",
            "hazard MAE": round(hazard_mae_ql, 4),
            "P=0.5 mileage": round(threshold_ql, 2),
            "samples": int(QL_EPOCHS) * int(len(transitions["s"])) * QL_SEEDS,
            "runtime sec": round(total_ql_runtime, 3),
        },
    ]
    if dqn_result is not None:
        rows.append({
            "method": "soft DQN",
            "transition matrix": "no",
            "hazard MAE": round(hazard_mae_dqn, 4),
            "P=0.5 mileage": round(threshold_dqn, 2),
            "samples": int(DQN_EPOCHS) * int(len(transitions["s"])),
            "runtime sec": round(dqn_result["runtime"], 3),
        })
    comparison_df = pd.DataFrame(rows)

    print("Saving figures and tables ...")
    n_samples = len(transitions["s"])

    visit_threshold = 5
    fig_hazard = hazard_figure(
        x_grid, nfxp["p_replace"], p_replace_ql, p_replace_dqn,
        visit_total=visit_min, visit_threshold=visit_threshold,
    )
    fig_values = value_figure(
        x_grid, nfxp["values"], q_table_ql,
        visit_total=visit_min, visit_threshold=visit_threshold,
    )
    fig_eff = sample_efficiency_figure(bus_counts, hazard_mae_per_count)

    bus0_mileage = data["mileage"][0]
    bus0_replace = data["replace"][0]
    fig_traj = trajectory_figure(bus0_mileage, bus0_replace)

    (folder / "figures").mkdir(parents=True, exist_ok=True)
    save_figure(fig_hazard, str(folder / "figures" / "replacement-hazard.png"), dpi=150)
    save_figure(fig_values, str(folder / "figures" / "value-comparison.png"), dpi=150)
    save_figure(fig_eff, str(folder / "figures" / "sample-efficiency.png"), dpi=150)
    save_figure(fig_traj, str(folder / "figures" / "trajectory.png"), dpi=150)

    Path(folder / "tables").mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(folder / "tables" / "method-comparison.csv", index=False)

    save_thumbnail(
        str(folder / "figures" / "replacement-hazard.png"),
        str(folder / "figures" / "thumb.png"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
