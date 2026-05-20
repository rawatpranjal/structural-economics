"""Faithfulness tests for the consideration-set-estimation tutorial.

Each finding from bullshit-detector_consideration-set-estimation_2026-05-20.md
gets two tests:

- ``violated_invariant`` encodes the bug. It PASSES on buggy code and must
  FAIL once the finding is fixed.
- ``honest_fix`` encodes the faithful state. It FAILS on buggy code and must
  PASS once the finding is fixed.

``run.py`` is import-safe (``main()`` runs only under ``__main__``), so the
numeric helpers are imported directly. Prose findings are checked against the
``run.py`` source text and the generated ``README.md`` / committed CSVs.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TUT))
import run  # noqa: E402


# ---------------------------------------------------------------------------
# Finding 1: Example 2 menu indices
# ---------------------------------------------------------------------------
def _example2_probs():
    """Recompute the Example 2 menu-probability table the way main() does."""
    ranking_ex = np.array([0, 1, 2], dtype=int)
    gamma_ex = np.array([4.0 / 9.0, 0.5, 0.9])
    menus_ex = run.enumerate_menus(3)
    return run.all_menu_probs(ranking_ex, gamma_ex, menus_ex), menus_ex


def _menu_index(menus, members):
    """Index of the menu whose membership mask equals ``members``."""
    return next(i for i, m in enumerate(menus)
                if np.array_equal(m, np.array(members, dtype=bool)))


def test_f1_violated_invariant():
    """Buggy code reads menu {a,b} as ``probs_ex[1, 0]`` -- but index 1 is the
    singleton {b}, where a is absent, so the bar chart shows p=0.000.

    The math fact: row 1 of the Example 2 table is the singleton {b}, so
    p(a) there is exactly 0. The bug is main() pointing the {a,b} label at
    that row. This test passes while main() still does so."""
    probs_ex, _ = _example2_probs()
    assert probs_ex[1, 0] == 0.0  # row 1 = {b}: a absent
    src = inspect.getsource(run.main)
    assert 'probs_ex[1, 0]' in src  # main() mislabels this row as {a,b}


def test_f1_honest_fix():
    """The bar chart must pull p(a,{a,b}) and p(b,{b,c}) from the correct
    menus, giving 4/9 and 1/2 -- not 0.000."""
    src = inspect.getsource(run.main)
    # p_ab must be sourced from the {a,b} menu, p_bc from the {b,c} menu.
    assert 'probs_ex[1, 0]' not in src
    assert 'probs_ex[3, 1]' not in src
    probs_ex, menus_ex = _example2_probs()
    i_ab = _menu_index(menus_ex, [1, 1, 0])
    i_bc = _menu_index(menus_ex, [0, 1, 1])
    i_ac = _menu_index(menus_ex, [1, 0, 1])
    assert abs(probs_ex[i_ab, 0] - 4 / 9) < 1e-6
    assert abs(probs_ex[i_bc, 1] - 0.5) < 1e-6
    assert abs(probs_ex[i_ac, 0] - 4 / 9) < 1e-6


# ---------------------------------------------------------------------------
# Finding 2: "Both methods recover the full ranking" claim
# ---------------------------------------------------------------------------
def test_f2_violated_invariant():
    """Method 2's committed ranking gets only 9/10 pairs right, yet the buggy
    run.py hardcodes the unconditional 'Both methods recover the full ranking'
    string. The data fact and the bugged string together prove the claim is
    false. This test passes while that hardcoded string is still present."""
    csv = (TUT / "tables" / "ranking-recovery.csv").read_text()
    assert "9 / 10" in csv  # Method 2 misses one pair
    src = inspect.getsource(run.main)
    has_unconditional = (
        '"Both methods recover the full ranking on the point estimate at '
        'this sample size. "' in src
        and "point_estimate_sentence" not in src
    )
    assert has_unconditional


def test_f2_honest_fix():
    """The README must not assert both methods recover the full ranking."""
    readme = (TUT / "README.md").read_text()
    assert "Both methods recover the full ranking" not in readme


# ---------------------------------------------------------------------------
# Finding 3: KL "essentially zero" claim
# ---------------------------------------------------------------------------
def test_f3_violated_invariant():
    """Method 2's KL is 1.4908 in the committed CSV, far from zero, yet the
    buggy run.py hardcodes 'a Kullback-Leibler divergence of essentially
    zero'. The data fact and the bugged string together prove the claim is
    false. This test passes while that hardcoded phrase is still present."""
    df = pd.read_csv(TUT / "tables" / "method-comparison.csv")
    m2 = df[df["Method"] == "Method 2 moments"].iloc[0]
    assert float(m2["KL divergence to true"]) > 1.0  # not essentially zero
    src = inspect.getsource(run.main)
    assert "Kullback-Leibler divergence of essentially zero" in src


def test_f3_honest_fix():
    """The README must not call Method 2's KL 'essentially zero'."""
    readme = (TUT / "README.md").read_text()
    assert "essentially zero" not in readme


# ---------------------------------------------------------------------------
# Finding 4: true-DGP log-likelihood never disclosed
# ---------------------------------------------------------------------------
def _comparison_methods():
    return set(pd.read_csv(TUT / "tables" / "method-comparison.csv")["Method"])


def test_f4_violated_invariant():
    """The buggy method-comparison table has no true-DGP log-likelihood row,
    so the prose claim 'within one or two units of the true-DGP value' has no
    artifact to verify against. This test passes while no such row exists."""
    assert "True DGP" not in _comparison_methods()


def test_f4_honest_fix():
    """The method-comparison table must disclose the true-DGP log-likelihood."""
    csv = (TUT / "tables" / "method-comparison.csv").read_text()
    assert "True DGP" in csv or "true-DGP" in csv or "True-DGP" in csv


# ---------------------------------------------------------------------------
# Finding 5: attention-within-one-SE claim is hardcoded
# ---------------------------------------------------------------------------
def test_f5_violated_invariant():
    """The original prose asserts the within-one-SE claim unconditionally."""
    src = inspect.getsource(run.main)
    # In the buggy version the phrase sits in a plain (non f-string) literal
    # with no surrounding boolean check on gamma_m1 / gamma_se_m1.
    has_phrase = "within one standard error" in src
    # Honest fix conditions the phrase on an actual comparison.
    conditioned = "gamma_se_m1" in src and "abs(gamma_m1" in src
    assert has_phrase and not conditioned


def test_f5_honest_fix():
    """After the fix the within-SE statement is gated on the real check
    all(|gamma_m1 - gamma_true| <= gamma_se_m1)."""
    src = inspect.getsource(run.main)
    if "within one standard error" in (TUT / "README.md").read_text():
        # If the phrase still appears, the code must have verified it.
        assert "gamma_se_m1" in src and "gamma_true" in src
        assert "abs(gamma_m1" in src
    else:
        # Or the unverifiable phrase was dropped entirely.
        assert "within one standard error" not in src


# ---------------------------------------------------------------------------
# Finding 6: Method 2 score pseudocode sign vs code
# ---------------------------------------------------------------------------
def test_f6_violated_invariant():
    """Buggy README pseudocode writes score[j] as sum_i impact[i,j] -
    sum_i impact[j,i], the negative of what the code computes
    (score[j] = avg_impact[j,:].sum() - avg_impact[:,j].sum()). This test
    passes while the reversed-sign pseudocode line is still in run.py."""
    src = inspect.getsource(run.main)
    assert "score[j] <- sum_i impact[i, j] - sum_i impact[j, i]" in src


def test_f6_honest_fix():
    """README pseudocode must match the code's sign: score[j] is
    sum_i impact[j,i] - sum_i impact[i,j], sorted by argsort(-score)."""
    readme = (TUT / "README.md").read_text()
    assert "score[j] <- sum_i impact[j, i] - sum_i impact[i, j]" in readme
