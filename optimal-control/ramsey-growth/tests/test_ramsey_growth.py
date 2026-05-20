"""Faithfulness tests for the Ramsey growth tutorial.

Covers bullshit-detector findings 1 and 2 (2026-05-20 audit):
  - Finding 1 (DATA DRIFT): the committed README truncated c0 values to 5
    decimals while the CSV (and the {:.6f} format) keep 6.
  - Finding 2 (MISLABELED): the column "Terminal capital gap" actually
    holds the relative gap |k(T)-k*|/k*, since the code divides by k_star.
"""
import csv
import sys
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TUTORIAL_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

RUN_SRC = (TUTORIAL_DIR / "run.py").read_text()
README = (TUTORIAL_DIR / "README.md").read_text()
CSV_PATH = TUTORIAL_DIR / "tables" / "shooting-results.csv"


def _csv_rows():
    return list(csv.DictReader(CSV_PATH.open()))


def test_c0_formatted_six_decimals_in_run_py():
    """Violated invariant: run.py requests 6-decimal c0.

    Passes regardless of fix; documents the format contract.
    """
    assert '{c0:.6f}' in RUN_SRC


def test_readme_c0_matches_csv_to_six_decimals():
    """Honest fix: README c0 column matches the CSV to 1e-7.

    Fails on the stale README (c0 truncated to 5 decimals, diffs ~2-4e-6);
    passes after `python run.py` regenerates the README from the same
    {:.6f} format the CSV uses.
    """
    rows = _csv_rows()
    c0_col = [r["$c_0$ from shooting"] for r in rows]
    for c0 in c0_col:
        assert c0 in README, f"c0 {c0} missing from README table"


def test_terminal_gap_column_is_relative():
    """Violated invariant: the code computes a relative terminal gap.

    The residual divides by k_star, producing a dimensionless ratio.
    Passes on current code.
    """
    assert "k_star) / k_star" in RUN_SRC


def test_terminal_gap_column_label_says_relative():
    """Honest fix: the column label must say the gap is relative.

    Fails on the stale README/CSV ("Terminal capital gap"); passes once
    the label is corrected to name the relative gap.
    """
    header = CSV_PATH.open().readline()
    assert "Relative terminal capital gap" in header
    assert "Relative terminal capital gap" in README
    assert "| Terminal capital gap |" not in README
