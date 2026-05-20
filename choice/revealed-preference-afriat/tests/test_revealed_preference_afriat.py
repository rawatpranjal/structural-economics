"""Faithfulness tests for the revealed-preference-afriat tutorial.

These tests cover the three non-HOLDS findings from
bullshit-detector_revealed-preference-afriat_2026-05-20.md:

  F1 (MISLABELED, MED) -- title/prose say "Afriat's test" but the code
      implements GARP checking via Warshall transitive closure.
  F2 (DILUTED, LOW)    -- pseudocode omits the numerical tolerance TOL.
  F4 (DILUTED, LOW)    -- README does not disclose the 200-attempt retry
      loop and the hardcoded fallback in generate_inconsistent_data.

Each finding has a "violated-invariant" test (passes on the original buggy
text, fails after the honest fix) and an "honest-fix" test (fails on the
original buggy text, passes after the fix). Both pairs are kept so the
direction of each fix is pinned.

After the honest fix is applied, the honest-fix tests PASS and the
violated-invariant tests FAIL -- run this file with
``pytest -k violated`` to confirm the buggy state is gone.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUTORIAL = HERE.parent
RUN_PY = (TUTORIAL / "run.py").read_text()
README = (TUTORIAL / "README.md").read_text()


# --- Finding 1: "Afriat's test" label without Afriat inequalities -----------

def test_f1_violated_invariant_title_says_afriat_test():
    """Violated invariant: passes while the README title says 'Afriat's
    Test'; must FAIL after the honest fix renames the title to GARP."""
    title_line = README.splitlines()[0]
    assert "Afriat's Test" in title_line


def test_f1_honest_fix_code_implements_garp_not_afriat_inequalities():
    """The code never computes Afriat inequalities (lambda_t, CCEI, u_t)."""
    lowered = RUN_PY.lower()
    assert "afriat_inequalities" not in lowered
    assert "lambda_t" not in lowered
    assert "ccei" not in lowered
    assert "efficiency_index" not in lowered


def test_f1_honest_fix_title_and_method_name_garp():
    """Honest fix: title and Solution Method name the GARP test, not Afriat's
    test, since the code does GARP checking via Warshall closure."""
    title_line = README.splitlines()[0]
    assert "GARP" in title_line
    assert "Afriat's Test" not in title_line
    assert "graph version of Afriat's test" not in RUN_PY


# --- Finding 2: pseudocode omits the numerical tolerance --------------------

def test_f2_violated_invariant_readme_omits_tolerance():
    """Violated invariant: passes while the README omits the tolerance;
    must FAIL after the honest fix discloses TOL=1e-10."""
    assert "tolerance" not in README.lower() and "1e-10" not in README


def test_f2_honest_fix_readme_discloses_tolerance():
    """Honest fix: README discloses the 1e-10 numerical tolerance."""
    assert "tolerance" in README.lower() or "1e-10" in README


# --- Finding 4: undisclosed retry loop and fallback -------------------------

def test_f4_violated_invariant_readme_omits_fallback():
    """Violated invariant: passes while the README hides the retry loop and
    fallback; must FAIL after the honest fix discloses them."""
    lowered = README.lower()
    assert "fallback" not in lowered and "attempt" not in lowered


def test_f4_honest_fix_readme_discloses_retry_and_fallback():
    """Honest fix: README discloses the swap-retry loop and the fallback."""
    lowered = README.lower()
    assert "fallback" in lowered or "attempt" in lowered
