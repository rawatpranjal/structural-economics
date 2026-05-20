"""Faithfulness tests for the numerical-optimization tutorial.

Finding 1 (DILUTED): the README calls the Newton line search a "backtracking
line search", which conventionally implies the Armijo sufficient-decrease
condition, but the original code accepted any pure decrease.

Finding 2 (DATA DRIFT): dual_annealing always runs its full annealing
schedule, so nit == maxiter and result.success only reports that the run
finished without error. The reported Success flag must instead certify that
the polished solution reached a posterior mode.

Each finding gets a violated-invariant test and an honest-fix test. The tests
parse run.py as text to avoid importing the tutorial (which would execute it).
"""

import inspect
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
RUN_PY = HERE / "run.py"


def newton_source() -> str:
    """Return the body of newton_with_backtracking by parsing run.py text."""
    src = RUN_PY.read_text()
    match = re.search(
        r"def newton_with_backtracking\(.*?\n(?=def |\Z)", src, re.DOTALL
    )
    assert match, "newton_with_backtracking not found in run.py"
    return match.group(0)


def dual_annealing_source() -> str:
    src = RUN_PY.read_text()
    match = re.search(r"def run_dual_annealing\(.*?\n(?=def |\Z)", src, re.DOTALL)
    assert match, "run_dual_annealing not found in run.py"
    return match.group(0)


# --- Finding 1: Newton backtracking should use Armijo sufficient-decrease ----

def test_f1_violated_invariant_line_search_compares_objective():
    """The Newton line search accepts a step by comparing objective values."""
    body = newton_source()
    assert "objective(candidate)" in body


def test_f1_honest_fix_line_search_uses_armijo_sufficient_decrease():
    """The line search must enforce an Armijo sufficient-decrease constant."""
    body = newton_source()
    assert any(
        "c1" in line or "sufficient" in line.lower()
        for line in body.splitlines()
    )


# --- Finding 2: dual annealing success flag must be guarded ------------------

def test_f2_violated_invariant_maxiter_is_eighty():
    """Dual annealing runs with maxiter=80, so nit always equals maxiter."""
    body = dual_annealing_source()
    assert "maxiter=80" in body


def test_f2_honest_fix_success_flag_certifies_mode_reached():
    """The reported Success flag must be conditioned on reaching a mode.

    scipy's result.success is True whenever the annealing schedule finished
    without error, even though nit always hits maxiter. The honest fix
    derives the reported success from an explicit distance-to-mode check.
    """
    body = dual_annealing_source()
    assert "reached_mode" in body
    assert "success=bool(result.success) and reached_mode" in body


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
