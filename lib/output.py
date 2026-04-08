"""ModelReport: auto-generates standardized GitHub-flavored Markdown reports."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from lib.plotting import save_figure, save_thumbnail


class ModelReport:
    """Generates a standardized README.md with equations, figures, and tables."""

    def __init__(self, title: str, description: str = ""):
        self.title = title
        self.description = description
        self._overview: str = ""
        self._equations: str = ""
        self._model_setup: str = ""
        self._solution_method: str = ""
        self._results_items: list[tuple] = []  # mixed content: text, figures, tables in order
        self._takeaway: str = ""
        self._references: list[str] = []
        self._first_figure_path: Optional[str] = None

    @property
    def _figures(self) -> list[tuple[str, str]]:
        """Backward-compat: list of (path, caption) for figure items."""
        return [(i[1], i[2]) for i in self._results_items if i[0] == "figure"]

    @property
    def _tables(self) -> list[tuple[str, str, str]]:
        """Backward-compat: list of (path, caption, md) for table items."""
        return [(i[1], i[2], i[3]) for i in self._results_items if i[0] == "table"]

    def add_overview(self, text: str) -> None:
        self._overview = text.strip()

    def add_equations(self, latex: str, description: str = "") -> None:
        parts = []
        if description:
            parts.append(description.strip())
        parts.append(latex.strip())
        self._equations = "\n\n".join(parts)

    def add_model_setup(self, text: str) -> None:
        self._model_setup = text.strip()

    def add_solution_method(self, text: str) -> None:
        self._solution_method = text.strip()

    def add_figure(
        self, path: str, caption: str, fig: Figure,
        dpi: int = 150, description: str = "",
    ) -> None:
        """Save a matplotlib figure and register it for the report."""
        save_figure(fig, path, dpi=dpi)
        self._results_items.append(("figure", path, caption, description.strip()))
        if self._first_figure_path is None:
            self._first_figure_path = path

    def add_table(
        self, path: str, caption: str, df: pd.DataFrame, description: str = "",
    ) -> None:
        """Save a DataFrame as CSV and register it as a markdown table."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        md_table = df.to_markdown(index=False)
        self._results_items.append(("table", path, caption, md_table, description.strip()))

    def add_results(self, text: str) -> None:
        """Add a text block to the results section. Can be called multiple times."""
        self._results_items.append(("text", text.strip()))

    def add_takeaway(self, text: str) -> None:
        self._takeaway = text.strip()

    def add_references(self, refs: list[str]) -> None:
        self._references = refs

    def generate_thumbnail(self, thumb_path: str = "figures/thumb.png") -> None:
        """Create a 200x150 thumbnail from the first figure."""
        if self._first_figure_path:
            save_thumbnail(self._first_figure_path, thumb_path)

    def write(self, path: str = "README.md") -> None:
        """Assemble and write the full markdown report."""
        lines: list[str] = []

        # Title
        lines.append(f"# {self.title}")
        lines.append("")
        if self.description:
            lines.append(f"> {self.description}")
            lines.append("")

        # Overview
        if self._overview:
            lines.append("## Overview")
            lines.append("")
            lines.append(self._overview)
            lines.append("")

        # Equations
        if self._equations:
            lines.append("## Equations")
            lines.append("")
            lines.append(self._equations)
            lines.append("")

        # Model Setup
        if self._model_setup:
            lines.append("## Model Setup")
            lines.append("")
            lines.append(self._model_setup)
            lines.append("")

        # Solution Method
        if self._solution_method:
            lines.append("## Solution Method")
            lines.append("")
            lines.append(self._solution_method)
            lines.append("")

        # Results
        if self._results_items:
            lines.append("## Results")
            lines.append("")

            for item in self._results_items:
                if item[0] == "text":
                    lines.append(item[1])
                    lines.append("")
                elif item[0] == "figure":
                    _, fig_path, caption, desc = item
                    if desc:
                        lines.append(desc)
                        lines.append("")
                    lines.append(f"![{caption}]({fig_path})")
                    lines.append(f"*{caption}*")
                    lines.append("")
                elif item[0] == "table":
                    _, _, caption, md_table, desc = item
                    if desc:
                        lines.append(desc)
                        lines.append("")
                    lines.append(f"**{caption}**")
                    lines.append("")
                    lines.append(md_table)
                    lines.append("")

        # Economic Takeaway
        if self._takeaway:
            lines.append("## Economic Takeaway")
            lines.append("")
            lines.append(self._takeaway)
            lines.append("")

        # Reproduce
        lines.append("## Reproduce")
        lines.append("")
        lines.append("```bash")
        lines.append("python run.py")
        lines.append("```")
        lines.append("")

        # References
        if self._references:
            lines.append("## References")
            lines.append("")
            for ref in self._references:
                lines.append(f"- {ref}")
            lines.append("")

        # Write
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines))

        # Generate thumbnail
        self.generate_thumbnail()
