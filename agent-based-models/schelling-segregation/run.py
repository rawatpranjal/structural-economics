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
from lib.plotting import save_figure, save_thumbnail, setup_style


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
    def tau_label(x: float) -> str:
        """Three-decimal label, widened to full precision when 3dp would not
        round-trip to the float actually used in the sweep (e.g. tau = 1/3)."""
        short = f"{x:.3f}"
        if abs(float(short) - x) < 1e-10:
            return short
        return repr(float(x))

    table["Threshold tau"] = table["Threshold tau"].map(tau_label)
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

    fig_paths = plot_paths(selected_paths)
    save_figure(fig_paths, "figures/segregation-paths.png", dpi=150)

    fig_phase = plot_phase_transition(summary)
    save_figure(fig_phase, "figures/phase-transition.png", dpi=150)

    fig_moves = plot_move_counts(selected_paths)
    save_figure(fig_moves, "figures/move-counts.png", dpi=150)

    fig_city = plot_city(gif_result.final_grid, f"Final city at tau={gif_tau:.2f}, S={gif_result.final_segregation:.3f}")
    save_figure(fig_city, "figures/final-city-tau-035.png", dpi=150)

    table.to_csv("tables/threshold-sweep.csv", index=False)

    save_thumbnail("figures/segregation-paths.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
