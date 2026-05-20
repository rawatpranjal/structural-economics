"""Faithfulness tests for the fixed-point-acceleration tutorial.

These tests encode the two findings of
``bullshit-detector_fixed-point-acceleration_2026-05-20.md``.

For each finding there are two tests:

* a ``violated_invariant`` test that PASSED on the original buggy code
  (it asserts the buggy behaviour) and must now FAIL on the fixed code;
* an ``honest_fix`` test that FAILED on the original buggy code and must
  now PASS on the fixed code.

Importing ``run.py`` would execute the whole tutorial (it has a bare
``main()`` body and writes README/figures/tables), so the prose and
hardcoded-status checks read the ``run.py`` source text directly. The
termination-status check exercises the real ``termination_status`` logic
by replicating its contract against the recorded table values.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
RUN_PY = TUTORIAL_DIR / "run.py"
METHOD_CSV = TUTORIAL_DIR / "tables" / "method_comparison.csv"
STRESS_CSV = TUTORIAL_DIR / "tables" / "stress_test.csv"

RUN_SRC = RUN_PY.read_text()

# Tolerance the tutorial declares (``tol = 1e-12`` in run.py).
TOL = 1e-12


def _method_table() -> pd.DataFrame:
    return pd.read_csv(METHOD_CSV)


def _stress_table() -> pd.DataFrame:
    return pd.read_csv(STRESS_CSV)


# ---------------------------------------------------------------------------
# Finding 1: Damped Picard labeled "converged" despite hitting max_iter.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Finding 1 fixed: Status is now derived from termination, "
    "not hardcoded. This violated-invariant test PASSED on the buggy "
    "code and must fail now.",
    strict=True,
)
def test_finding1_violated_invariant_status_hardcoded_converged():
    """Buggy code hardcoded the Status column as three "converged" strings.

    PASSED on the original buggy code. Must FAIL on the honest fix, which
    derives Status from the termination condition.
    """
    assert '"Status": ["converged", "converged", "converged"]' in RUN_SRC


def test_finding1_honest_fix_damped_picard_status_reflects_max_iter():
    """Damped Picard exhausted max_iter with a residual above tolerance.

    Its Status must report the max-iteration stop, not "converged".
    FAILED on the original buggy code; must PASS on the honest fix.
    """
    table = _method_table()
    row = table.loc[table["Method"] == "Damped Picard"]
    assert len(row) == 1
    status = row["Status"].values[0]
    residual = float(row["Final residual"].values[0])

    # The data the audit flagged: residual is far above tolerance.
    assert residual > TOL, "Damped Picard residual should exceed tolerance"
    # An honest status must not claim convergence for this row.
    assert status != "converged"
    assert "max_iter" in status or status in ("max_iter", "did not converge")


def test_finding1_status_is_consistent_with_residual_for_every_row():
    """Every method's Status must agree with its final residual.

    converged  <=>  final residual < tol. This is the invariant the
    hardcoded list violated; it must hold for all three rows now.
    """
    table = _method_table()
    for _, row in table.iterrows():
        residual = float(row["Final residual"])
        status = row["Status"]
        if residual < TOL:
            assert status == "converged", f"{row['Method']}: residual met tol but status={status!r}"
        else:
            assert status != "converged", f"{row['Method']}: residual above tol but status={status!r}"


# ---------------------------------------------------------------------------
# Finding 2: stress-test prose claimed a sweep "from a benign 0.5".
# ---------------------------------------------------------------------------

def _stress_sentence() -> str:
    m = re.search(
        r"The stress test sweeps the outside share from \d[\d.]* down to \d[\d.]*\.",
        RUN_SRC,
    )
    assert m, "stress-test sweep sentence not found in run.py"
    return m.group(0)


@pytest.mark.xfail(
    reason="Finding 2 fixed: stress-test prose now states the actual "
    "0.1 sweep start. This violated-invariant test PASSED on the buggy "
    "prose and must fail now.",
    strict=True,
)
def test_finding2_violated_invariant_prose_claims_0_5_start():
    """Buggy prose claimed the sweep starts at a benign 0.5.

    The feasibility check (e3 <= 0) silently skips 0.5 and 0.2, so the
    table starts at 0.1. PASSED on the buggy prose; must FAIL on the fix.
    """
    assert "from a benign 0.5" in RUN_SRC


def test_finding2_honest_fix_prose_states_actual_start():
    """Fixed prose must state the actual sweep start (0.1)."""
    sentence = _stress_sentence()
    assert "0.1" in sentence
    assert "0.5" not in sentence


def test_finding2_stress_table_starts_at_actual_first_share():
    """The stress table's largest outside share matches the prose claim.

    The four feasible rows are 0.10, 0.05, 0.02, 0.01; the largest is
    0.10. Prose and data must agree.
    """
    table = _stress_table()
    shares = sorted(table["Outside share"].astype(float), reverse=True)
    assert shares[0] == pytest.approx(0.10)
    assert 0.5 not in shares
    assert 0.2 not in shares
