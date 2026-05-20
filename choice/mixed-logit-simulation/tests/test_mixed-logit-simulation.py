"""Faithfulness tests for choice/mixed-logit-simulation/.

Audit: bullshit-detector_mixed-logit-simulation_2026-05-20.md
Finding 1 (DILUTED): the Model Setup SD bounds row said the bound is
"Applied to both random-coefficient standard deviations", omitting that the
constraint actually sits on log-sigma; the [0.03, 1.30] value is the
effective sigma range after exponentiation.
"""
import re
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent.parent
README = TUTORIAL_DIR / "README.md"


def _sd_bounds_row() -> str:
    for line in README.read_text().splitlines():
        if line.startswith("|") and "SD bounds" in line:
            return line
    raise AssertionError("SD bounds row not found in README")


def test_finding1_violated_invariant_sd_row_omits_log_qualifier():
    """Violated invariant: SD bounds row never named the log transform.

    PASSED on the buggy README; FAILS once the honest fix names log-sigma.
    """
    row = _sd_bounds_row()
    assert not re.search(r"log[- ]?sigma|log scale|log standard deviation", row)


def test_finding1_honest_fix_sd_row_names_log_space():
    """Honest fix: SD bounds row states the bound is enforced on log-sigma.

    FAILED on the buggy README; PASSES once the qualifier is present.
    """
    row = _sd_bounds_row()
    assert re.search(r"log[- ]?sigma|log scale|log standard deviation", row)
