"""Faithfulness tests for the ar-processes tutorial.

Covers bullshit-detector findings (2026-05-20):
  Finding 1 (DILUTED/MED): pseudocode step 3 updates g_t with eps_t, the same
    symbol as the AR(1) shock, contradicting the Equations claim that the
    spending innovation eta_t is drawn independently of eps_t.
  Finding 2 (DATA DRIFT/LOW): Results prose says the half-life goes "from one
    to seven periods" while the table, Solution Method, and Takeaway all say
    6.6.

The README is generated from run.py; the README source text is checked.
"""

from pathlib import Path

_TUTORIAL = Path(__file__).resolve().parents[1]
_README = (_TUTORIAL / "README.md").read_text()


def _pseudocode_step3_line() -> str:
    return _README.split("3. Update")[1].split("\n")[0]


# --- Finding 1: pseudocode shock symbol ---------------------------------


def test_violated_invariant_pseudocode_reuses_eps_for_g():
    """Buggy state: pseudocode step 3 uses eps_t for the g_t update.

    The g_t update is the segment after 'g_t ='. On the buggy README it ends
    in eps_t; the honest fix changes it to eta_t, flipping this test.
    """
    g_update = _pseudocode_step3_line().split("g_t =")[1]
    assert "eps_t" in g_update


def test_honest_fix_pseudocode_uses_eta_for_g():
    """Honest fix: pseudocode step 3 uses eta_t for the g_t update.

    Fails on the current buggy pseudocode.
    """
    assert "eta_t" in _pseudocode_step3_line()


# --- Finding 2: half-life rounding drift ---------------------------------


def test_violated_invariant_prose_says_seven():
    """Buggy state: Results prose rounds the 6.6-period half-life to 'seven'.

    Fails once the honest fix removes 'seven'.
    """
    assert "seven" in _README


def test_honest_fix_prose_consistent_half_life():
    """Honest fix: 'seven' is gone and the prose uses 6.6.

    Fails on the current buggy README.
    """
    assert "seven" not in _README and "6.6" in _README
