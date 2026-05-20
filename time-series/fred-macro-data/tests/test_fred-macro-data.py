"""Faithfulness tests for the fred-macro-data tutorial.

Covers bullshit-detector finding 1 (2026-05-20): the Equations section uses
the symbol sigma for two numerically distinct quantities -- the DGP scaling
vector (3.0, 1.5, 1.5, 3.0) and the HP-cycle standard deviation sd(c_{j,t}).
The prose warns of the overload, but the equations carry no distinguishing
notation. After the honest fix the DGP sigma and the cycle sigma get
distinct superscripts.

The README is generated from run.py; the run.py source text is checked.
"""

import re
from pathlib import Path

_TUTORIAL = Path(__file__).resolve().parents[1]
_RUN_PY = (_TUTORIAL / "run.py").read_text()

_SUPERSCRIPTED = re.compile(r"\\sigma\^[{]?[yc]")


def test_violated_invariant_sigma_not_distinguished():
    """Buggy state: equations carry one undistinguished sigma symbol.

    Fails once the honest fix adds a sigma^y / sigma^c superscript.
    """
    assert _SUPERSCRIPTED.search(_RUN_PY) is None


def test_honest_fix_sigma_distinguished():
    """Honest fix: DGP sigma and cycle sigma get distinct superscripts.

    Fails on the current buggy run.py.
    """
    assert _SUPERSCRIPTED.search(_RUN_PY) is not None


def test_honest_fix_both_variants_present():
    """Honest fix: both the DGP and the cycle superscripted symbols exist."""
    assert r"\sigma^{y}" in _RUN_PY and r"\sigma^{c}" in _RUN_PY
