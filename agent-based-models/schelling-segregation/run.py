#!/usr/bin/env python3
"""Schelling's checkerboard model of residential segregation.

Simulates two groups on a spatial grid. Agents care only about the composition
of their local neighborhood, but the repeated movement of dissatisfied agents
can generate aggregate segregation.

Reference: Schelling (1971), "Dynamic Models of Segregation."
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


EMPTY = 0
GROUP_A = 1
GROUP_B = 2
NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
COLORS = {
    EMPTY: (242, 242, 239),
    GROUP_A: (52, 116, 180),
    GROUP_B: (219, 109, 63),
}


@dataclass
class SimulationResult:
    """A complete run for one threshold and seed."""

    tau: float
    seed: int
    history: list[np.ndarray]
    segregation: list[float]
    moved: list[int]
    converged: bool
    remaining_unhappy: int

    @property
    def iterations(self) -> int:
        return max(len(self.segregation) - 1, 0)

    @property
    def final_grid(self) -> np.ndarray:
        return self.history[-1]

    @property
    def final_segregation(self) -> float:
        return self.segregation[-1]


def initialize_city(
    n: int,
    vacancy_share: float,
    minority_share: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a random city with two groups and vacant cells."""
    total_cells = n * n
    vacant = int(round(total_cells * vacancy_share))
    occupied = total_cells - vacant
    group_b = int(round(occupied * minority_share))
    group_a = occupied - group_b

    cells = np.array(
        [EMPTY] * vacant + [GROUP_A] * group_a + [GROUP_B] * group_b,
        dtype=np.int8,
    )
    rng.shuffle(cells)
    return cells.reshape(n, n)


def neighbor_counts(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Count occupied, same-group, group-A, and group-B neighbors."""
    padded = np.pad(grid, 1, mode="constant", constant_values=EMPTY)
    occupied = np.zeros_like(grid, dtype=int)
    group_a = np.zeros_like(grid, dtype=int)
    group_b = np.zeros_like(grid, dtype=int)

    for dr, dc in NEIGHBOR_OFFSETS:
        block = padded[
            1 + dr: 1 + dr + grid.shape[0],
            1 + dc: 1 + dc + grid.shape[1],
        ]
        occupied += block != EMPTY
        group_a += block == GROUP_A
        group_b += block == GROUP_B

    same = np.zeros_like(grid, dtype=int)
    same[grid == GROUP_A] = group_a[grid == GROUP_A]
    same[grid == GROUP_B] = group_b[grid == GROUP_B]
    return occupied, same, group_a, group_b


def similar_share_grid(grid: np.ndarray) -> np.ndarray:
    """Return each occupied agent's share of same-group occupied neighbors."""
    occupied, same, _, _ = neighbor_counts(grid)
    share = np.ones_like(grid, dtype=float)
    observed = (grid != EMPTY) & (occupied > 0)
    share[observed] = same[observed] / occupied[observed]
    return share


def segregation_index(grid: np.ndarray) -> float:
    """Average same-group neighbor share among occupied agents."""
    share = similar_share_grid(grid)
    return float(share[grid != EMPTY].mean())


def similar_share_at(grid: np.ndarray, row: int, col: int, group: int) -> float:
    """Same-group neighbor share for a proposed group at one location."""
    n_rows, n_cols = grid.shape
    occupied = 0
    same = 0
    for dr, dc in NEIGHBOR_OFFSETS:
        rr = row + dr
        cc = col + dc
        if 0 <= rr < n_rows and 0 <= cc < n_cols:
            neighbor = grid[rr, cc]
            if neighbor != EMPTY:
                occupied += 1
                same += int(neighbor == group)
    return 1.0 if occupied == 0 else same / occupied


def is_satisfied(grid: np.ndarray, row: int, col: int, tau: float) -> bool:
    """Check whether the occupant at a location is content."""
    group = int(grid[row, col])
    if group == EMPTY:
        return True
    return similar_share_at(grid, row, col, group) >= tau


def move_dissatisfied_agents(
    grid: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Move dissatisfied agents to satisfying vacant cells when available."""
    next_grid = grid.copy()
    share = similar_share_grid(next_grid)
    unhappy = np.argwhere((next_grid != EMPTY) & (share < tau))
    rng.shuffle(unhappy)

    moved = 0
    for row, col in unhappy:
        group = int(next_grid[row, col])
        if group == EMPTY or is_satisfied(next_grid, row, col, tau):
            continue

        vacancies = np.argwhere(next_grid == EMPTY)
        rng.shuffle(vacancies)
        destination: tuple[int, int] | None = None
        for dest_row, dest_col in vacancies:
            if similar_share_at(next_grid, int(dest_row), int(dest_col), group) >= tau:
                destination = (int(dest_row), int(dest_col))
                break

        if destination is None:
            continue

        dest_row, dest_col = destination
        next_grid[dest_row, dest_col] = group
        next_grid[row, col] = EMPTY
        moved += 1

    return next_grid, moved


def run_schelling(
    tau: float,
    seed: int,
    n: int = 50,
    vacancy_share: float = 0.10,
    minority_share: float = 0.50,
    max_iter: int = 100,
    keep_history: bool = False,
) -> SimulationResult:
    """Run the classic Schelling checkerboard dynamics."""
    rng = np.random.default_rng(seed)
    grid = initialize_city(n, vacancy_share, minority_share, rng)
    history = [grid.copy()]
    segregation = [segregation_index(grid)]
    moved: list[int] = []
    converged = False
    remaining_unhappy = 0

    for _ in range(max_iter):
        share = similar_share_grid(grid)
        unhappy = np.argwhere((grid != EMPTY) & (share < tau))
        if len(unhappy) == 0:
            converged = True
            break

        grid, n_moved = move_dissatisfied_agents(grid, tau, rng)
        moved.append(n_moved)
        segregation.append(segregation_index(grid))
        if keep_history:
            history.append(grid.copy())
        else:
            history = [history[0], grid.copy()]

        if n_moved == 0:
            remaining_unhappy = int(len(unhappy))
            break
    else:
        share = similar_share_grid(grid)
        remaining_unhappy = int(np.sum((grid != EMPTY) & (share < tau)))

    if keep_history and len(history) != len(segregation):
        history.append(grid.copy())

    return SimulationResult(
        tau=tau,
        seed=seed,
        history=history,
        segregation=segregation,
        moved=moved,
        converged=converged,
        remaining_unhappy=remaining_unhappy,
    )


def sweep_thresholds(
    tau_grid: np.ndarray,
    seeds: list[int],
    n: int,
    vacancy_share: float,
    minority_share: float,
    max_iter: int,
) -> tuple[pd.DataFrame, dict[float, SimulationResult]]:
    """Run a threshold sweep and keep one representative path per threshold."""
    rows = []
    examples: dict[float, SimulationResult] = {}
    for tau in tau_grid:
        results = [
            run_schelling(
                tau=float(tau),
                seed=seed,
                n=n,
                vacancy_share=vacancy_share,
                minority_share=minority_share,
                max_iter=max_iter,
                keep_history=(seed == seeds[0]),
            )
            for seed in seeds
        ]
        examples[round(float(tau), 4)] = results[0]
        final_s = np.array([result.final_segregation for result in results])
        iterations = np.array([result.iterations for result in results])
        total_moved = np.array([sum(result.moved) for result in results])
        rows.append({
            "tau": float(tau),
            "mean_final_S": float(final_s.mean()),
            "sd_final_S": float(final_s.std(ddof=1)),
            "mean_iterations": float(iterations.mean()),
            "mean_moves": float(total_moved.mean()),
            "converged_runs": int(sum(result.converged for result in results)),
            "runs": len(results),
        })
    return pd.DataFrame(rows), examples


def city_image(grid: np.ndarray, cell_size: int = 8, top_pad: int = 46, label: str = "") -> Image.Image:
    """Render a city grid as a PIL image for GIF frames."""
    height, width = grid.shape
    image = Image.new("RGB", (width * cell_size, height * cell_size + top_pad), "white")
    draw = ImageDraw.Draw(image)
    if label:
        draw.text((8, 14), label, fill=(30, 30, 30))
    for row in range(height):
        for col in range(width):
            color = COLORS[int(grid[row, col])]
            x0 = col * cell_size
            y0 = top_pad + row * cell_size
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=color,
            )
    return image


def save_city_gif(result: SimulationResult, path: str, max_frames: int = 32) -> None:
    """Save an animated GIF of the checkerboard dynamics."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(result.history) <= max_frames:
        frame_indices = list(range(len(result.history)))
    else:
        frame_indices = sorted(set(np.linspace(0, len(result.history) - 1, max_frames, dtype=int)))

    frames = []
    for index in frame_indices:
        label = f"tau={result.tau:.3f}   step={index:02d}   S={result.segregation[index]:.3f}"
        frames.append(city_image(result.history[index], label=label))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        optimize=True,
    )


def plot_city(grid: np.ndarray, title: str) -> plt.Figure:
    """Plot one city grid with the two groups and vacancies."""
    color_array = np.zeros((*grid.shape, 3), dtype=float)
    for value, color in COLORS.items():
        color_array[grid == value] = np.array(color) / 255.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(color_array, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def plot_paths(path_results: list[SimulationResult]) -> plt.Figure:
    """Plot segregation-index paths for selected thresholds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in path_results:
        steps = np.arange(len(result.segregation))
        ax.plot(steps, result.segregation, marker="o", markersize=3, label=f"$\\tau={result.tau:.3f}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Segregation index $S(t)$")
    ax.set_title("Local Movement Raises Same-Group Exposure")
    ax.set_ylim(0.45, 0.92)
    ax.legend()
    return fig


def plot_phase_transition(summary: pd.DataFrame) -> plt.Figure:
    """Plot final segregation against the tolerance threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary["tau"],
        summary["mean_final_S"],
        yerr=summary["sd_final_S"],
        marker="o",
        linewidth=2.0,
        capsize=3,
    )
    ax.axvline(1.0 / 3.0, color="black", linestyle="--", linewidth=1.6, label="$1/3$")
    ax.set_xlabel("Minimum same-group neighbor share $\\tau$")
    ax.set_ylabel("Final segregation index")
    ax.set_title("A Small Change in Tolerance Can Change the Aggregate Pattern")
    ax.set_ylim(0.52, 0.91)
    ax.legend()
    return fig


def plot_move_counts(path_results: list[SimulationResult]) -> plt.Figure:
    """Plot how much relocation each threshold induces."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in path_results:
        steps = np.arange(1, len(result.moved) + 1)
        ax.plot(steps, result.moved, marker="o", markersize=3, label=f"$\\tau={result.tau:.3f}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Agents moved")
    ax.set_title("Relocation Is Endogenous to the Existing Pattern")
    ax.legend()
    return fig


def main() -> None:
    n = 50
    vacancy_share = 0.10
    minority_share = 0.50
    max_iter = 100
    seeds = [2021, 2022, 2023, 2024, 2025]
    tau_grid = np.array([0.20, 0.225, 0.25, 0.275, 0.30, 1.0 / 3.0, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50])
    path_taus = [0.25, 1.0 / 3.0, 0.40, 0.50]
    gif_tau = 0.35

    setup_style()
    summary, examples = sweep_thresholds(
        tau_grid=tau_grid,
        seeds=seeds,
        n=n,
        vacancy_share=vacancy_share,
        minority_share=minority_share,
        max_iter=max_iter,
    )
    gif_result = run_schelling(
        tau=gif_tau,
        seed=2031,
        n=n,
        vacancy_share=vacancy_share,
        minority_share=minority_share,
        max_iter=max_iter,
        keep_history=True,
    )
    selected_paths = [
        run_schelling(
            tau=tau,
            seed=2021,
            n=n,
            vacancy_share=vacancy_share,
            minority_share=minority_share,
            max_iter=max_iter,
            keep_history=True,
        )
        for tau in path_taus
    ]

    Path("figures").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)
    save_city_gif(gif_result, "figures/schelling-tau-035.gif")

    table = summary.rename(columns={
        "tau": "Threshold tau",
        "mean_final_S": "Mean final segregation S",
        "sd_final_S": "SD final segregation S",
        "mean_iterations": "Mean iterations",
        "mean_moves": "Mean moves",
        "converged_runs": "Converged runs",
        "runs": "Replications",
    }).copy()
    table["Threshold tau"] = table["Threshold tau"].map(lambda x: f"{x:.3f}")
    table["Mean final segregation S"] = table["Mean final segregation S"].map(lambda x: f"{x:.3f}")
    table["SD final segregation S"] = table["SD final segregation S"].map(lambda x: f"{x:.3f}")
    table["Mean iterations"] = table["Mean iterations"].map(lambda x: f"{x:.1f}")
    table["Mean moves"] = table["Mean moves"].map(lambda x: f"{x:.0f}")

    transition_row = summary.iloc[(summary["tau"] - 1.0 / 3.0).abs().argmin()]
    low_row = summary[summary["tau"] == 0.25].iloc[0]
    high_row = summary[summary["tau"] == 0.50].iloc[0]

    print("Schelling segregation sweep")
    print(f"  Grid: {n}x{n}, vacancy_share={vacancy_share:.2f}, group shares=0.50/0.50")
    print(f"  tau=0.250 mean final S={low_row['mean_final_S']:.3f}")
    print(f"  tau~1/3 mean final S={transition_row['mean_final_S']:.3f}")
    print(f"  tau=0.500 mean final S={high_row['mean_final_S']:.3f}")

    report = ModelReport(
        "Schelling Segregation on a Checkerboard",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Schelling's segregation model starts from a simple city. Two groups occupy "
        "a checkerboard with some vacant locations. Each person looks only at nearby "
        "neighbors and moves if too few are from the same group.\n\n"
        "The economic point is not that agents choose segregation as a social outcome. "
        "They choose local neighborhoods. The aggregate pattern is produced by the "
        "feedback from many moves.\n\n"
        "This tutorial keeps the model in its classic form. We simulate a 50 x 50 "
        "city, sweep the minimum same-group neighbor share $\\tau$, and track the "
        "segregation index $S(t)$ until the city stops moving."
    )

    report.add_equations(
        r"""
Let $G$ be the city grid. The state variable is the whole checkerboard, not an
aggregate stock. Cell $i$ at iteration $t$ is
$X_t(i)$. Empty cells have $X_t(i)=0$. Occupied cells have group
$g_i(t)=X_t(i)$, where $g_i(t)\in\lbrace A,B\rbrace$.

For each occupied cell, $N_i$ is the local Moore neighborhood: the adjacent
horizontal, vertical, and diagonal cells, up to eight in total. Empty cells are
possible destinations, but they are not neighbors whose type enters the local
composition. The occupied-neighbor count is

$$
O_i(t)=\sum_{j\in N_i} \mathbf{1}[X_t(j)\neq 0].
$$

The same-group neighbor count keeps only occupied neighbors with the same group
as the resident in cell $i$:

$$
m_i(t)=\sum_{j\in N_i} \mathbf{1}[X_t(j)=g_i(t)].
$$

When $O_i(t)>0$, the local same-group share is the object agents care about:

$$
s_i(t)=\frac{m_i(t)}{O_i(t)}.
$$

When an occupied cell has no occupied neighbors, set $s_i(t)=1$. This convention
treats isolation as not violating the local same-group requirement. The
threshold $\tau$ is the minimum acceptable same-group share. An agent is content
when

$$
s_i(t)\geq \tau.
$$

If $s_i(t)<\tau$, the agent is dissatisfied. Let $E_t$ be the set of vacant
cells. A move is allowed only if the destination would satisfy the same local
threshold for that agent's group. Thus agents do not choose a global segregation
target; they only search for an acceptable local neighborhood.

The aggregate segregation index averages local exposure among occupied cells:

$$
S(t)=\frac{1}{M}\sum_{i:X_t(i)\neq 0} s_i(t),
$$

where $M$ is the number of occupied cells, which stays fixed because moves only
swap an occupied cell with a vacancy. A random initial city with equal group
sizes has $S(t)$ near one half. Large values of $S(t)$ mean that the typical
person mostly sees same-group neighbors, even though each decision used only
the local rule above.
"""
    )

    report.add_model_setup(
        "The calibration follows the checkerboard spirit of Schelling's spatial "
        "proximity model. The numbers are artificial by design; they let us see the "
        "dynamic mechanism.\n\n"
        f"| Symbol | Value | Role |\n"
        f"|---|---|---|\n"
        f"| $G$ | {n} x {n} cells | City grid |\n"
        f"| $E_t$ | {vacancy_share:.0%} of cells initially vacant | Empty cells that permit movement |\n"
        f"| $g_i$ | $A$ or $B$ | Occupant group in cell $i$ |\n"
        f"| $N_i$ | Moore neighborhood, up to 8 cells | Local reference group |\n"
        f"| $O_i(t)$ | occupied neighbors in $N_i$ | Denominator for local exposure |\n"
        f"| $s_i(t)$ | between 0 and 1 | Same-group neighbor share for cell $i$ |\n"
        f"| $\\tau$ | 0.20 to 0.50 | Local tolerance threshold |\n"
        f"| $S(t)$ | average of $s_i(t)$ | Aggregate segregation index |\n"
        f"| $T$ | {max_iter} iterations | Stop rule cap |\n"
        f"| Replications | {len(seeds)} per threshold | Simulation noise check |"
    )

    report.add_solution_method(
        "The model is an agent-based simulation. There is no representative agent "
        "and no global optimization problem. The state is the whole checkerboard.\n\n"
        "```text\n"
        "Algorithm: Schelling checkerboard dynamics\n"
        "Input: initial grid X_0, vacancy set E_0, threshold tau, horizon T\n"
        "Output: city states X_t, segregation path S(t), move counts\n\n"
        "For t = 0, 1, ..., T - 1:\n"
        "  1. For every occupied cell i, compute O_i(t), m_i(t), and s_i(t).\n"
        "  2. Let D_t be occupied cells with s_i(t) < tau.\n"
        "  3. If D_t is empty, stop with a locally stable city.\n"
        "  4. Visit cells in D_t in random order.\n"
        "  5. For each still-dissatisfied i, form candidate vacant cells C_i(t):\n"
        "       vacant cells e in E_t where group g_i would have exposure at least tau.\n"
        "  6. If C_i(t) is nonempty, draw e from C_i(t), move g_i from i to e,\n"
        "       and update X_t and E_t before visiting the next agent.\n"
        "  7. Record X_{t+1}, S(t+1), and the number of moves.\n"
        "  8. Stop if no one moves or if t + 1 = T.\n"
        "```\n\n"
        "The random order matters because one move changes the neighborhoods of "
        "nearby agents. That dependence is the point of the model. Small local "
        "moves change the local incentives faced by others, and the city can tip "
        "toward a much more sorted pattern."
    )

    report.add_results(
        "The animation shows the focal run at $\\tau=0.35$, just above one third. "
        "Blue cells are one group, orange cells are the other group, and the light "
        "cells are empty locations. Each frame is a saved city state after a wave "
        "of relocation decisions. The city begins close to a random mix. "
        "Dissatisfied agents move into locations where their local threshold is "
        "met, and same-group clusters become self-reinforcing.\n\n"
        '<img src="figures/schelling-tau-035.gif" alt="Animated Schelling checkerboard at tau 0.35" width="80%">'
    )

    report.add_results(
        "The path plot tracks the segregation index $S(t)$ for four thresholds. "
        "The critical hyperparameter is $\\tau$, the local tolerance threshold. "
        "At low thresholds, the city settles after little sorting. Near the "
        "one-third region, the same local rule produces a visibly higher "
        "same-group exposure. Small integer neighborhoods make this region "
        "important: with only a few occupied neighbors, one additional same-group "
        "neighbor can move an agent across the threshold. The plateaus come from "
        "the integer number of neighbors on a finite checkerboard."
    )
    report.add_figure(
        "figures/segregation-paths.png",
        "Segregation-index paths for selected thresholds",
        plot_paths(selected_paths),
    )

    report.add_results(
        "The threshold sweep makes the nonlinearity clearer. Schelling emphasized "
        "that a demand around one third generated much less segregation than a "
        "demand near one half in his checkerboard examples. This run shows the "
        "same qualitative lesson: final segregation rises quickly as the local "
        "demand moves out of the low-tolerance range."
    )
    report.add_figure(
        "figures/phase-transition.png",
        "Final segregation index by same-group threshold",
        plot_phase_transition(summary),
    )

    report.add_results(
        "Movement is concentrated early. Once enough agents have relocated, many "
        "neighborhoods become locally stable even though the aggregate city is far "
        "more sorted than the initial draw."
    )
    report.add_figure(
        "figures/move-counts.png",
        "Moved agents by iteration",
        plot_move_counts(selected_paths),
    )

    report.add_results(
        "The final city at $\\tau=0.35$ has same-group clusters even though every "
        "agent used only local neighbor composition."
    )
    report.add_figure(
        "figures/final-city-tau-035.png",
        "Final checkerboard city for tau 0.35",
        plot_city(gif_result.final_grid, f"Final city at tau={gif_tau:.2f}, S={gif_result.final_segregation:.3f}"),
    )

    report.add_table(
        "tables/threshold-sweep.csv",
        "Threshold sweep summary",
        table,
        "Simulation detail behind the phase-transition figure. Each row averages "
        f"over {len(seeds)} random initial cities.",
    )

    report.add_takeaway(
        "The Schelling model is a warning about aggregation. Modest local tolerance "
        "rules need not preserve a mixed city. When movement changes the local "
        "environment faced by others, individual relocation decisions can create "
        "segregated aggregate patterns that are much stronger than the rule each "
        "agent follows."
    )

    report.add_references([
        "[Schelling, T. C. (1971). Dynamic Models of Segregation. *The Journal of Mathematical Sociology*, 1(2), 143-186.](https://doi.org/10.1080/0022250X.1971.9989794)",
        "[Schelling, T. C. (1978). *Micromotives and Macrobehavior*. W. W. Norton.]",
    ])

    report.write("README.md")


if __name__ == "__main__":
    main()
