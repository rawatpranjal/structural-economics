"""Consistent matplotlib styling and save utilities."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


STYLE_CONFIG = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (7, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def setup_style() -> None:
    """Apply the library's consistent matplotlib style."""
    matplotlib.use("Agg")
    plt.rcParams.update(STYLE_CONFIG)


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a matplotlib figure, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_thumbnail(source_path: str | Path, thumb_path: str | Path, size: tuple[int, int] = (200, 150)) -> None:
    """Create a thumbnail from an existing image file."""
    from PIL import Image

    path = Path(thumb_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(source_path)
    img.thumbnail(size, Image.LANCZOS)
    img.save(path)
