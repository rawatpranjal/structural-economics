"""Faithfulness tests for the ridge-lasso-sparsity tutorial.

Generated from bullshit-detector_ridge-lasso-sparsity_2026-05-20.md.

Finding 1 (DILUTED, MED): "False inclusions by lasso = 0" is a DGP
tautology. The DGP draws every one of the 120 indicators with a nonzero
true coefficient, so ~true_nonzero is all-False and the false-inclusion
count is structurally zero regardless of lasso behaviour. The honest fix
discloses this in the selection-table prose.

run.py runs the whole tutorial on import, so the structural tautology is
tested against the run.py source text and the disclosure against README.
"""

import re
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
RUN_PY = (TUTORIAL_DIR / "run.py").read_text()
README = (TUTORIAL_DIR / "README.md").read_text()


# --- Finding 1: violated-invariant test --------------------------------------
# Proves the structural tautology: the DGP gives every indicator a nonzero
# true coefficient, so false inclusions cannot occur. This stays TRUE (the
# DGP is intentionally dense); the disclosure fix does not change it.

def test_violated_invariant_dgp_makes_all_indicators_nonzero():
    """Every indicator gets a per-concept weak addition >= 0.006 in the DGP."""
    # rng.uniform lower bound for the weak per-concept signal.
    assert "rng.uniform(0.006, 0.018" in RUN_PY
    # The false-inclusion count is computed as lasso_selected & ~true_nonzero,
    # i.e. it can only be nonzero if some true coefficient is exactly zero.
    assert "lasso_selected & ~true_nonzero" in RUN_PY


# --- Finding 1: honest-fix test ----------------------------------------------
# FAILS on current (pre-fix) code: README has no disclosure of the tautology.
# PASSES once the selection-table prose discloses it.

def test_honest_fix_readme_discloses_dgp_tautology():
    assert "always zero by DGP construction" in README


def test_honest_fix_disclosure_warns_against_misreading():
    """The disclosure must tell the reader the zero is not a lasso metric."""
    lowered = README.lower()
    assert "not" in lowered
    # The disclosure sentence mentions both the DGP and lasso precision.
    window = re.search(
        r"always zero by dgp construction.{0,400}", lowered, re.S
    )
    assert window is not None
    assert "lasso" in window.group(0)
