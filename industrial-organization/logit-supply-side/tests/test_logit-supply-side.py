"""Faithfulness tests for the logit-supply-side tutorial.

Each finding from bullshit-detector_logit-supply-side_2026-05-20.md gets two
tests: a violated-invariant test (passed on the buggy code) and an honest-fix
test (failed on the buggy code). After the fix the violated-invariant test
fails and the honest-fix test passes.

The tutorial's run.py guards execution behind `if __name__ == "__main__"`, so
importing it does not run the pipeline. Tests inspect function source text and
the generated CSV artifacts.
"""
import csv
import importlib.util
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "logit_supply_side_run", TUTORIAL_DIR / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------------------------------------------------------
# Finding 1: Omega derivative subscript -- docstring vs README convention
# -------------------------------------------------------------------------

def test_finding1_violated_invariant_docstring_uses_old_convention():
    """Violated invariant: docstring used the `ds_j/dp_k` convention, which
    disagrees with the README equation `Omega_jk = -O_jk ds_k/dp_j`.

    PASSED on the buggy code; FAILS after the docstring is harmonized.
    """
    module = _load_run_module()
    doc = module.compute_share_derivatives.__doc__
    assert "ds_j/dp_k" in doc and "ds_k/dp_j" not in doc


def test_finding1_honest_fix_docstring_matches_readme_convention():
    """Honest fix: docstring uses the README `ds_k/dp_j` convention.

    FAILED on the buggy code; PASSES after the docstring is harmonized.
    """
    module = _load_run_module()
    doc = module.compute_share_derivatives.__doc__
    assert "ds_k/dp_j" in doc and doc.startswith("ds_k/dp_j")


# -------------------------------------------------------------------------
# Finding 2: MAE 0.455 dollars -- not present in any committed artifact
# -------------------------------------------------------------------------

def test_finding2_violated_invariant_mae_absent_from_estimation_csv():
    """Violated invariant: the MAE figure is not in estimation-results.csv,
    which holds only the four demand parameters.

    PASSES before and after the fix -- the estimation table is not where the
    cost-recovery MAE belongs. It documents the gap the fix closes.
    """
    rows = (TUTORIAL_DIR / "tables" / "estimation-results.csv").read_text()
    assert "0.455" not in rows


def test_finding2_honest_fix_mae_present_in_cost_recovery_csv():
    """Honest fix: a committed cost-recovery CSV records the market-0 MAE in
    a `Mean` footer row, so the README figure is verifiable without a re-run.

    FAILS on the buggy code (the CSV does not exist); PASSES after run.py
    regenerates it.
    """
    path = TUTORIAL_DIR / "tables" / "cost-recovery-market0.csv"
    assert path.exists()
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    footer = next(r for r in rows if r["Product"] == "Mean")
    mae = float(footer["Absolute error"])
    assert 0.0 < mae < 5.0
