"""Faithfulness tests for the huggett-aggregate-risk-srl tutorial.

Each finding from bullshit-detector_huggett-aggregate-risk-srl_2026-05-20.md
gets two tests:

- ``test_<n>_violated_invariant`` encodes the bug. It PASSES on buggy code
  and must FAIL after the honest fix.
- ``test_<n>_honest_fix`` encodes the corrected behaviour. It FAILS on buggy
  code and must PASS after the honest fix.

Importing ``run.py`` would execute the whole tutorial, so prose/claim tests
read the generated artifacts (``README.md``, ``tables/*.csv``) and the
``run.py`` source text directly instead of importing.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
RUN_PY = TUTORIAL_DIR / "run.py"
README = TUTORIAL_DIR / "README.md"
DIAGNOSTICS_CSV = TUTORIAL_DIR / "tables" / "diagnostics.csv"
BENCHMARK_CSV = TUTORIAL_DIR / "tables" / "paper_benchmark.csv"


def _run_py_text() -> str:
    return RUN_PY.read_text(encoding="utf-8")


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _diagnostics_volatility_ratio() -> float:
    """Read the committed C/Y volatility ratio from the diagnostics table."""
    for row in csv.reader(DIAGNOSTICS_CSV.read_text(encoding="utf-8").splitlines()):
        if row and row[0].startswith("Aggregate consumption volatility divided by income"):
            return float(row[1])
    raise AssertionError("volatility ratio row not found in diagnostics.csv")


def _benchmark_row(item: str) -> list[str]:
    for row in csv.reader(BENCHMARK_CSV.read_text(encoding="utf-8").splitlines()):
        if row and row[0] == item:
            return row
    raise AssertionError(f"benchmark row {item!r} not found in paper_benchmark.csv")


# ---------------------------------------------------------------------------
# Finding 1: Takeaway claims "smoother aggregate consumption" when C/Y == 1.000
# ---------------------------------------------------------------------------

def test_1_violated_invariant() -> None:
    """Buggy code hardcodes the unconditional 'smoother' claim in run.py.

    The buggy Takeaway lists "smoother aggregate consumption" as a flat,
    unconditional clause; the README inherits it verbatim even though the
    committed C/Y volatility ratio is 1.000.
    """
    assert "concave consumption, endogenous prices, smoother aggregate" \
        in _run_py_text()
    assert "smoother aggregate consumption" in _readme_text()


def test_1_honest_fix() -> None:
    """Honest fix: README must not assert smoothing unless the ratio is < 1."""
    ratio = _diagnostics_volatility_ratio()
    readme = _readme_text()
    if ratio >= 1.0:
        assert "smoother aggregate consumption" not in readme
    else:
        assert "smoother aggregate consumption" in readme


# ---------------------------------------------------------------------------
# Finding 2: Benchmark table hardcodes "Grid and training settings: Matched"
# ---------------------------------------------------------------------------

def test_2_violated_invariant() -> None:
    """Buggy code emits the FULL-only 'same grid' string regardless of profile."""
    text = _run_py_text()
    assert (
        '"Tutorial run": "Uses the same grid, horizon, learning-rate schedule, '
        'and batch size"'
    ) in text


def test_2_honest_fix() -> None:
    """Honest fix: the grid row must be profile-conditional, not the FULL string.

    When the quick profile ran (asset grid != 200 points) the benchmark CSV
    must not claim the run used the same grid as the published benchmark.
    """
    grid_row = _benchmark_row("Grid and training settings")
    tutorial_run, assessment = grid_row[2], grid_row[3]
    quick_ran = "200 bond points" not in _readme_text() or "quick" in _readme_text()
    if quick_ran:
        assert "Uses the same grid, horizon, learning-rate schedule, and batch size" \
            != tutorial_run
        assert assessment != "Matched"


# ---------------------------------------------------------------------------
# Finding 3: diagnostics_table hardcodes FULL_PROFILE.convergence_tol
# ---------------------------------------------------------------------------

def test_3_violated_invariant() -> None:
    """Buggy code references FULL_PROFILE.convergence_tol in diagnostics_table."""
    assert "FULL_PROFILE.convergence_tol" in _run_py_text()


def test_3_honest_fix() -> None:
    """Honest fix: diagnostics_table uses the active profile's threshold."""
    assert "FULL_PROFILE.convergence_tol" not in _run_py_text()


# ---------------------------------------------------------------------------
# Finding 4: Market-clearing residual labeled "Matched" against 4.4e-6
# ---------------------------------------------------------------------------

def test_4_violated_invariant() -> None:
    """Buggy code labels the tautological residual 'Matched' in the benchmark.

    The interpolated residual is algebraically zero by construction, so the
    benchmark row claiming a 'Matched' assessment against the published
    4.4e-6 gap is misleading.
    """
    row = _benchmark_row("Market-clearing residual")
    assert row[3] == "Matched"


def test_4_honest_fix() -> None:
    """Honest fix: residual row must not be labeled 'Matched'.

    The interpolated residual is algebraically zero by construction of the
    linear-interpolation weights, so it is not comparable to the published
    4.4e-6 gap.
    """
    row = _benchmark_row("Market-clearing residual")
    assert row[3] != "Matched"


# ---------------------------------------------------------------------------
# Finding 5: Objective equation omits the L2 regularization term
# ---------------------------------------------------------------------------

def test_5_violated_invariant() -> None:
    """Buggy code: the L2 penalty exists in code but README never mentions it."""
    code_has_penalty = "1.0e-5 * jnp.mean(theta**2)" in _run_py_text()
    readme_mentions = bool(re.search(r"regulari[sz]", _readme_text(), re.IGNORECASE))
    assert code_has_penalty and not readme_mentions


def test_5_honest_fix() -> None:
    """Honest fix: the Equations section documents the L2 regularization term."""
    assert re.search(r"regulari[sz]", _readme_text(), re.IGNORECASE)
