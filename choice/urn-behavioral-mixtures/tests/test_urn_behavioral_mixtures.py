"""Faithfulness tests for the urn-behavioral-mixtures tutorial.

Covers the two non-HOLDS findings from
bullshit-detector_urn-behavioral-mixtures_2026-05-20.md:

  F8 (DATA DRIFT, LOW) -- the README states the EM iteration count and log
      likelihood, but neither value is persisted to any CSV artifact, so it
      cannot be re-grounded without a re-run.
  F9 (DATA DRIFT, LOW) -- the README states three rule-separation counts
      (Bayes vs conservative, vs red-share, vs raw-count), none of which are
      persisted to any CSV artifact.

Honest fix: run.py writes tables/diagnostics.csv carrying all five runtime
scalars, so every README number is backed by a committed artifact.

Violated-invariant tests check that the five values were absent from the
existing CSVs (the buggy state). Honest-fix tests check the new
diagnostics.csv exists and carries the five columns with the README values.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TUTORIAL = HERE.parent
TABLES = TUTORIAL / "tables"
README = (TUTORIAL / "README.md").read_text()


def _readme_int(label_phrase: str) -> int:
    """Pull the integer that follows a phrase in the README prose."""
    import re

    match = re.search(label_phrase + r"\D*(\d+)", README)
    assert match, f"phrase not found in README: {label_phrase}"
    return int(match.group(1))


# --- Violated invariant: the five scalars are absent from the old CSVs ------

def test_f8_violated_invariant_weights_csv_lacks_iteration_columns():
    """Violated invariant: passes while mixture-weights.csv has no iteration
    or log-likelihood column; the runtime scalars were unbacked."""
    header = (TABLES / "mixture-weights.csv").read_text().splitlines()[0]
    assert "iterations" not in header and "log_likelihood" not in header


def test_f9_violated_invariant_csvs_lack_split_columns():
    """Violated invariant: passes while neither committed CSV stores the
    rule-separation counts."""
    weights_header = (TABLES / "mixture-weights.csv").read_text().splitlines()[0]
    alloc_header = (TABLES / "type-allocation.csv").read_text().splitlines()[0]
    for col in ("bayes_conservative_split", "bayes_share_split", "bayes_count_split"):
        assert col not in weights_header and col not in alloc_header


# --- Honest fix: diagnostics.csv backs every runtime scalar -----------------

def test_f8_honest_fix_diagnostics_csv_has_em_scalars():
    """Honest fix: diagnostics.csv stores the EM iteration count and log
    likelihood, and they match the README prose."""
    diag = pd.read_csv(TABLES / "diagnostics.csv")
    assert "iterations" in diag.columns
    assert "log_likelihood" in diag.columns
    readme_iters = _readme_int("EM converges in")
    assert int(diag["iterations"].iloc[0]) == readme_iters
    # README rounds the log likelihood to 2 dp.
    assert f"{float(diag['log_likelihood'].iloc[0]):.2f}" in README


def test_f9_honest_fix_diagnostics_csv_has_split_counts():
    """Honest fix: diagnostics.csv stores the three rule-separation counts,
    and they match the README prose."""
    diag = pd.read_csv(TABLES / "diagnostics.csv")
    for col in ("bayes_conservative_split", "bayes_share_split", "bayes_count_split"):
        assert col in diag.columns
    assert int(diag["bayes_conservative_split"].iloc[0]) == _readme_int(
        "differs from the conservative rule on"
    )
    assert int(diag["bayes_share_split"].iloc[0]) == _readme_int(
        "differs from the red-share rule on"
    )
    assert int(diag["bayes_count_split"].iloc[0]) == _readme_int(
        "raw-count rule on"
    )
